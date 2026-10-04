"""CPU denormal-flush controls, isolated from the parent and the active package."""
import hashlib
from pathlib import Path

from test_registration import child


BASE = Path(__file__).resolve().parents[1]
OLD = BASE / 'tests/fixtures/zero_normalization_before.py'
assert hashlib.sha256(OLD.read_bytes()).hexdigest() == '14e303b87eeed6118949ea541c855ef77a71a85ba92d1fb584ec3af7fdd390fa'


def test_flush_reproduces_previous_zero_codes_and_preserves_new_zero_packing(tmp_path):
    got = child(f'''
import json,torch,types
from bitsandbytes_tpu import reference as r
old=types.ModuleType('bitsandbytes_tpu.old_reference');old.__package__='bitsandbytes_tpu'
exec(compile(open({str(OLD)!r}).read(),{str(OLD)!r},'exec'),old.__dict__)

import ast,copy,pathlib,bitsandbytes
from bitsandbytes.backends.utils import _get_4bit_code
from bitsandbytes_tpu.compatibility import validate_upstream
validate_upstream(bitsandbytes)
text=(pathlib.Path(bitsandbytes.__file__).parent/'backends/default/ops.py').read_text()
scope={{'torch':torch,'_get_4bit_code':_get_4bit_code}}
for node in ast.parse(text).body:
 if not isinstance(node,ast.FunctionDef):continue
 is_quant=any(isinstance(d,ast.Call) and isinstance(d.func,ast.Name) and d.func.id=='register_kernel' and d.args and isinstance(d.args[0],ast.Constant) and d.args[0].value=='bitsandbytes::quantize_4bit' for d in node.decorator_list)
 if node.name=='_get_4bit_quantize_bounds' or is_quant:
  n=copy.deepcopy(node);n.decorator_list=[]
  if is_quant:n.name='oracle_quantize'
  module=ast.fix_missing_locations(ast.Module(body=[n],type_ignores=[]))
  exec(compile(module,'pinned-default-quantizer','exec'),scope)
oracle=scope['oracle_quantize']
lengths=[1,2,63,64,65,127,128,129]
records=[]
assert torch.set_flush_denormal(False)
for dtype in [torch.float32,torch.bfloat16]:
 for n in lengths:
  a=torch.zeros(n,dtype=dtype)
  p,s=oracle(a,64,'nf4',torch.uint8)
  before,newscale=r.quantize_4bit(a,64,'nf4',torch.uint8)
  assert torch.equal(p,before) and torch.equal(s,newscale)
  assert torch.set_flush_denormal(True)
  try:
   bad,bads=old.quantize_4bit(a,64,'nf4',torch.uint8)
   fixed,fs=r.quantize_4bit(a,64,'nf4',torch.uint8)
   badcodes=torch.stack((bad[:,0]>>4,bad[:,0]&15),dim=1).reshape(-1)[:n]
   fixedcodes=torch.stack((fixed[:,0]>>4,fixed[:,0]&15),dim=1).reshape(-1)[:n]
   assert torch.equal(badcodes,torch.zeros(n,dtype=torch.uint8))
   assert torch.equal(fixedcodes,torch.full((n,),7,dtype=torch.uint8))
   assert torch.equal(fixed,p)
   if n%2:assert int(fixed[-1,0]&15)==7
   assert torch.equal(r.dequantize_4bit(fixed,fs,64,'nf4',[n],dtype),a)
   records.append({{'length':n,'dtype':str(dtype),'old_codes':0,'new_codes':7,'packed_exact':True,'unused_tail_code':7 if n%2 else None,'flushed_scale_values':fs.tolist()}})
  finally:assert torch.set_flush_denormal(False)
print(json.dumps({{'status':'LOCAL_CPU_FLUSH_EMULATION_ONLY','records':records,'new_exact_all':True,'prior_failure_reproduced':True,'subnormal_nonzero_tpu':'UNQUALIFIED'}}))
''')
    assert len(got['records']) == 16 and got['new_exact_all'] and got['prior_failure_reproduced']
    (tmp_path / 'zero-flush-control.json').write_text(__import__('json').dumps(got, indent=2)+'\n')


def test_flush_keeps_normal_values_midpoints_and_neighbor_packing(tmp_path):
    got = child(f'''
import json,torch,types
from bitsandbytes_tpu import reference as r
old=types.ModuleType('bitsandbytes_tpu.old_reference');old.__package__='bitsandbytes_tpu'
exec(compile(open({str(OLD)!r}).read(),{str(OLD)!r},'exec'),old.__dict__)

import ast,copy,pathlib,bitsandbytes
from bitsandbytes.backends.utils import _get_4bit_code
from bitsandbytes_tpu.compatibility import validate_upstream
validate_upstream(bitsandbytes)
text=(pathlib.Path(bitsandbytes.__file__).parent/'backends/default/ops.py').read_text()
scope={{'torch':torch,'_get_4bit_code':_get_4bit_code}}
for node in ast.parse(text).body:
 if not isinstance(node,ast.FunctionDef):continue
 is_quant=any(isinstance(d,ast.Call) and isinstance(d.func,ast.Name) and d.func.id=='register_kernel' and d.args and isinstance(d.args[0],ast.Constant) and d.args[0].value=='bitsandbytes::quantize_4bit' for d in node.decorator_list)
 if node.name=='_get_4bit_quantize_bounds' or is_quant:
  n=copy.deepcopy(node);n.decorator_list=[]
  if is_quant:n.name='oracle_quantize'
  module=ast.fix_missing_locations(ast.Module(body=[n],type_ignores=[]))
  exec(compile(module,'pinned-default-quantizer','exec'),scope)
oracle=scope['oracle_quantize']
assert torch.set_flush_denormal(False)
code=torch.tensor(r.NF4_CODE);bounds=(code[:-1]+code[1:])/2
lo=torch.nextafter(bounds,torch.full_like(bounds,-float('inf')))
hi=torch.nextafter(bounds,torch.full_like(bounds,float('inf')))
records=[]
for dtype in [torch.float32,torch.bfloat16]:
 for data,name in [(bounds,'midpoints'),(lo,'lower_neighbors'),(hi,'upper_neighbors'),(torch.linspace(-1,1,129),'normal_values')]:
  for n in [63,64,65,128,129]:
   a=torch.cat((torch.tensor([-1.,1.]),data.repeat(9)))[:n].to(dtype)
   expected,scales=oracle(a,64,'nf4',torch.uint8)
   assert torch.set_flush_denormal(True)
   try:
    fixed,got_scales=r.quantize_4bit(a,64,'nf4',torch.uint8)
    assert torch.equal(fixed,expected) and torch.equal(got_scales,scales)
   finally:assert torch.set_flush_denormal(False)
   records.append({{'length':n,'dtype':str(dtype),'pattern':name,'packed_exact':True,'scales_exact':True}})
print(json.dumps({{'status':'LOCAL_CPU_ONLY','records':records,'exact_all':True}}))
''')
    assert len(got['records']) == 40 and got['exact_all']
    (tmp_path / 'normal-boundary-control.json').write_text(__import__('json').dumps(got, indent=2)+'\n')
