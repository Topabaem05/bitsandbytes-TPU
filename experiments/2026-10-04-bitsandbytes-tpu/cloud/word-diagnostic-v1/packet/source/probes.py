"""Small exact word probes. Each output is retained, with its full shape."""
import importlib,json,sys,types
from pathlib import Path
import torch
PACKET=Path(__file__).resolve().parents[1]
WORDS=(0x00000000,0x80000000,0x00000001,0x80000001,0x007fffff,0x00800000,0x3f800000,0xbf800000,0x3f000000,0xbf000000,0x7f7fffff,0xff7fffff)
def signed(value):return value if value<0x80000000 else value-0x100000000

def load(variant):
 name='_m4_word_diag_'+variant
 if name not in sys.modules:
  pkg=types.ModuleType(name);pkg.__path__=[str(PACKET/'variants'/variant)];sys.modules[name]=pkg
 return importlib.import_module(name+'.exact'),importlib.import_module(name+'.functionalization')

def inputs(device):
 selected=json.loads((PACKET/'source/selected-inputs.json').read_bytes())
 # Make exact words on CPU first. Device inputs never use a float conversion.
 words=torch.tensor([signed(w)for w in WORDS],dtype=torch.int32)
 dynamic=torch.tensor(selected['dynamic_map'],dtype=torch.float32).view(torch.int32)
 return {k:v.to(device) for k,v in {
  'words':words,'dynamic':dynamic,'zero4':torch.zeros((1,4),dtype=torch.int32),
  'maxzero':torch.zeros(1,dtype=torch.int32),'tail1':torch.tensor([0x3f800000],dtype=torch.int32),
  'tail255':torch.arange(255,dtype=torch.int32),'counts':torch.tensor([0,1,23,30],dtype=torch.int32),
  'significands':torch.tensor([0,1,0x7fffff,0x800000,0xffffff,0x7fffffff,-1,-2147483648],dtype=torch.int32),
  'ones':torch.full_like(words,0x3f800000),'zeros':torch.zeros_like(words),
  'first8':torch.tensor([0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0],dtype=torch.float32).view(torch.int32),
 }.items()}

PRIMITIVES=('bitcast-raw-scalar','bitcast-raw-vector','bitcast-functional-scalar','bitcast-functional-vector','amax-1','amax-255','shift-left','shift-right','scalar-compare-select')
FAMILIES=('addsub','muldiv','midpoints','zero-normalize','tail-1-255')
def labels():return list(PRIMITIVES)+[v+'-'+f for v in('original','true_shift','true_shift_amax')for f in FAMILIES]

def primitive(name,x,device_type):
 E,F=load('original')
 if name.startswith('bitcast-'):
  w=x['words'][6]if name.endswith('scalar')else x['words'];functional='functional'in name
  if device_type=='cpu':
   source=w.view(torch.float32)
   if functional:source=torch._to_functional_tensor(source)
   middle=source.view(torch.int32);roundtrip=middle.view(torch.float32)
  else:
   source=E.floats(w)
   # Start the named case with the requested raw/native-functional state.
   if torch._is_functional_tensor(source):torch._functionalize_sync(source);source=torch._from_functional_tensor(source)
   if functional:source=torch._to_functional_tensor(source)
   with torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.Functionalize,False):
    middle=E.words(source);roundtrip=E.floats(middle)
  meta={'requested_functional':functional,'source_native_functional':torch._is_functional_tensor(source),'middle_native_functional':torch._is_functional_tensor(middle),'roundtrip_native_functional':torch._is_functional_tensor(roundtrip)}
  return {'words':middle,'roundtrip':roundtrip},meta
 def body(*args):
  y=dict(zip(x,args))
  if name.startswith('amax-'):
   value=y['tail1'if name=='amax-1'else'tail255'];return {'omitted':torch.amax(value),'dim0':torch.amax(value,dim=0),'dim_negative1':torch.amax(value,dim=-1)}
  if name.startswith('shift-'):
   words=y['significands'].unsqueeze(-1);counts=y['counts'].unsqueeze(0)
   if name=='shift-left':return {'legacy':words<<counts,'tensor':torch.ops.aten.bitwise_left_shift.Tensor(words,counts)}
   return {'legacy':words>>counts,'tensor':torch.ops.aten.bitwise_right_shift.Tensor(words,counts)}
  if name=='scalar-compare-select':
   words=y['significands'];zero=torch.zeros_like(words);one=torch.ones_like(words)
   return {'gt_python':words>0,'gt_s32':words>zero,'eq_python':words==0,'eq_s32':words==zero,'select_python':torch.where(words>0,1,words),'select_s32':torch.where(words>zero,one,words),'select_negative_python':torch.where(words==0,-149,words),'select_negative_s32':torch.where(words==zero,torch.full_like(words,-149),words)}
  raise ValueError('UNKNOWN_PRIMITIVE')
 return F.functionalized_kernel(body)(*x.values()),{}

def family(variant,name,x):
 E,F=load(variant);A=E.A;M=E.M;C=E.C
 def body(*args):
  y=dict(zip(x,args));w=y['words'];one=y['ones'];zero=y['zeros']
  if name=='addsub':
   summed=E.S.cpu_sum8_bits(y['first8'],A.add_bits)
   return {'add_zero':A.add_bits(w,zero),'subtract_zero':A.subtract_bits(w,zero),'cancellation':A.add_bits(w,w^C.SIGN),'shift_jam':A.shift_right_jam(y['significands']&0x7fffffff,torch.full_like(y['significands'],23)),'sum_first8':summed,'mean_first8':C.divide_count_bits(summed,8)}
  if name=='muldiv':
   mantissa,power,iszero=M.decode(w);om,op,oz=M.decode(one);high,low=M.product48(mantissa,om)
   return {'mantissa':mantissa,'power':power,'iszero':iszero,'product_high':high,'product_low':low,'multiply_one':M.multiply_bits(w,one),'divide_one':M.divide_bits(w,one),'divide_count2':C.divide_count_bits(w,2),'clamped':C.clamp_bits(w,E.NEGATIVE_ONE,E.ONE)}
  if name=='midpoints':
   added=A.add_bits(y['dynamic'][:-1],y['dynamic'][1:]);mid=E.midpoints(y['dynamic']);keys=E.finite_key(mid)
   return {'pair_added':added,'midpoint_words':mid,'finite_keys':keys,'negative_count':(keys<0).sum(dtype=torch.int32)}
  if name=='zero-normalize':
   denominator=C.clamp_bits(y['maxzero'],E.FLOOR);reciprocal=M.divide_bits(torch.full_like(denominator,E.ONE),denominator);normalized=E.normalize(y['zero4'],y['maxzero'],full=True);bounds=E.finite_key(E.midpoints(y['dynamic']));predicate=E.finite_key(normalized).reshape(-1,1)>bounds
   return {'denominator':denominator,'reciprocal':reciprocal,'normalized_words':normalized,'bounds':bounds,'predicate':predicate,'codes':predicate.sum(dim=-1,dtype=torch.int32).to(torch.uint8)}
  if name=='tail-1-255':
   c1,s1=E.quantize_words(y['tail1'],y['dynamic'],256);c255,s255=E.quantize_words(y['tail255'],y['dynamic'],256)
   return {'tail1_codes':c1,'tail1_scales':s1,'tail255_codes':c255,'tail255_scales':s255}
  raise ValueError('UNKNOWN_FAMILY')
 return F.functionalized_kernel(body)(*x.values()),{}

def execute(label,x,device_type):
 if label in PRIMITIVES:return primitive(label,x,device_type)
 for variant in('true_shift_amax','true_shift','original'):
  if label.startswith(variant+'-'):return family(variant,label[len(variant)+1:],x)
 raise ValueError('UNKNOWN_PROBE_LABEL')
