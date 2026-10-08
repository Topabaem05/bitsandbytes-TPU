"""Portable CPU/Meta transfer controls with explicit base and patched source paths.

The source fixtures disable backend entry-point discovery during import.
They do not replace Params4bit or Linear4bit methods at runtime.
Only Linux torch 2.9.0+cpu supplies qualified source-mechanics controls.
CPU/Meta controls never prove TPU execution or whole-module transactionality.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SHA = "e745fbf21aac10ed9118167a131d6505bf6dbe03e1fab0a669c663b26c5a5732"
SOURCE = ORIGINAL = BASE = None
RECORDS = []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    result = {}
    for path in root.rglob('*'):
        if path.is_symlink(): raise ValueError('SOURCE_SYMLINK')
        if path.is_file(): result[path.relative_to(root).as_posix()] = digest(path)
    return result


def prepare(source, work, patched_source=None):
    global SOURCE, ORIGINAL, BASE
    BASE = Path(work)
    manifest_path = ROOT/'patches/params4bit-xla-v1.json'
    assert digest(manifest_path) == MANIFEST_SHA, 'UNREVIEWED_MANIFEST'
    manifest = json.loads(manifest_path.read_text())
    patch = ROOT/manifest['patch_path']
    assert digest(patch) == manifest['patch_sha256'], 'PATCH_IDENTITY'
    ORIGINAL = BASE/'original'; SOURCE = BASE/'patched'
    ORIGINAL.mkdir()
    # A Git checkout supplies an immutable archive; an extracted package supplies exact files.
    if (source/'.git').exists():
        archive = subprocess.check_output(['git','-C',str(source),'archive','--format=tar',manifest['base_revision']])
        assert hashlib.sha256(archive).hexdigest() == manifest['base_archive_sha256'], 'BASE_ARCHIVE'
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            for member in bundle.getmembers():
                if member.name.startswith('bitsandbytes/') and member.isfile():
                    relative = Path(member.name).relative_to('bitsandbytes')
                    if relative.as_posix() in manifest['base_package_files']:
                        target = ORIGINAL/'bitsandbytes'/relative; target.parent.mkdir(parents=True,exist_ok=True)
                        target.write_bytes(bundle.extractfile(member).read())
    else:
        package = source/'bitsandbytes' if (source/'bitsandbytes').is_dir() else source
        assert inventory(package) == manifest['base_package_files'], 'BASE_INVENTORY'
        shutil.copytree(package, ORIGINAL/'bitsandbytes')
    assert inventory(ORIGINAL/'bitsandbytes') == manifest['base_package_files'], 'BASE_INVENTORY'
    if patched_source is None:
        shutil.copytree(ORIGINAL, SOURCE)
        subprocess.run(['git','apply','--check',str(patch)],cwd=SOURCE,check=True,capture_output=True)
        subprocess.run(['git','apply',str(patch)],cwd=SOURCE,check=True,capture_output=True)
    else:
        package = patched_source/'bitsandbytes' if (patched_source/'bitsandbytes').is_dir() else patched_source
        assert inventory(package) == manifest['post_patch_package_files'], 'POST_INVENTORY'
        shutil.copytree(package, SOURCE/'bitsandbytes')
    assert inventory(SOURCE/'bitsandbytes') == manifest['post_patch_package_files'], 'POST_INVENTORY'
    return manifest


def child(code, patched=True):
    tree = SOURCE if patched else ORIGINAL
    source_before = inventory(tree/'bitsandbytes')
    env = os.environ.copy(); env['PYTHONPATH'] = str(tree)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['TORCHINDUCTOR_CACHE_DIR'] = str(BASE/'compiler-cache'); env['XDG_CACHE_HOME'] = str(BASE/'cache')
    setup = f"""
import importlib.metadata,sys
original_entry_points=importlib.metadata.entry_points
def fixture_entry_points(*args,**kwargs):
 if kwargs.get('group')=='bitsandbytes.backends':return []
 return original_entry_points(*args,**kwargs)
# Isolate source mechanics from installed plugin entry points, then restore discovery.
importlib.metadata.entry_points=fixture_entry_points
try:
 import torch,bitsandbytes as b,pathlib,json,platform
finally:importlib.metadata.entry_points=original_entry_points
assert pathlib.Path(b.__file__).is_relative_to({str(tree)!r})
assert not torch.__future__.get_overwrite_module_params_on_conversion()
assert not torch.__future__.get_swap_module_params_on_conversion()
original_classes=(b.nn.Params4bit,b.nn.Linear4bit,b.functional.QuantState)
original_methods=(b.nn.Params4bit.to,b.nn.Params4bit._quantize,b.nn.Linear4bit.forward)
"""
    source_assertions = '''
assert original_classes==(b.nn.Params4bit,b.nn.Linear4bit,b.functional.QuantState)
assert original_methods==(b.nn.Params4bit.to,b.nn.Params4bit._quantize,b.nn.Linear4bit.forward)
assert not any(n=='torch_xla' or n.startswith('torch_xla.') or n=='jax' or n.startswith('jax.') for n in sys.modules)
'''
    process = subprocess.run([sys.executable,'-B','-c',setup+code+source_assertions],env=env,capture_output=True,text=True,timeout=60)
    assert process.returncode == 0, process.stdout+process.stderr
    assert inventory(tree/'bitsandbytes') == source_before, 'SOURCE_CHANGED_DURING_CONTROL'
    record = json.loads(process.stdout.splitlines()[-1]); RECORDS.append(record)
    return record


META_FIXTURE = """
original_quant=b.functional.quantize_4bit
produced_states=[]
def meta_quant(w,**kw):
 result=original_quant(w.to('meta'),**kw);produced_states.append(result[1]);return result
# The quantizer is a child-private result fixture; class methods remain source-defined.
b.functional.quantize_4bit=meta_quant
"""


def control_incompatible_result_keeps_identity_class_attrs_and_module_alias(method):
    got=child(META_FIXTURE+f'''
m=b.nn.Linear4bit(7,2,bias=True,quant_type='nf4',compress_statistics=False)
p=m.weight;p.custom_token={{'keep':[1,2]}};token=p.custom_token
identity=id(p);cls=type(p)
if {method!r}=='direct_to_fixture':result=p.to('cpu')
elif {method!r}=='module_to_fixture':result=m.to('cpu').weight
elif {method!r}=='direct_quantize_meta':result=p._quantize(torch.device('meta'))
else:
 def convert(t):
  return t._quantize(torch.device('meta')) if isinstance(t,b.nn.Params4bit) else t.to('meta')
 result=m._apply(convert).weight
assert result is p and id(m.weight)==identity and type(p) is cls is b.nn.Params4bit
assert p.custom_token is token and p.module is m
assert p.device.type=='meta' and p.dtype==torch.uint8 and p.bnb_quantized
assert p.quant_state is m.quant_state and p.quant_state.dtype==torch.float32
assert p.quant_state is produced_states[-1]
assert p.quant_state.shape==torch.Size([2,7]) and p.shape==torch.Size([7,1])
expected_bias='meta' if {method!r}=='module_apply_meta' else 'cpu'
assert m.bias.device.type==expected_bias and type(m.bias) is torch.nn.Parameter
print(json.dumps({{'method':{method!r},'scope':'CPU_META_TYPE_MECHANICS_ONLY','identity':True,'class':True,'attrs':True,'module_alias':True,'bias_device':expected_bias}}))
''')
    assert all(got[k] for k in ['identity','class','attrs','module_alias'])

def control_original_source_preserves_incompatible_failure(method):
    got=child(META_FIXTURE+f'''
m=b.nn.Linear4bit(7,2,bias=True,quant_type='nf4',compress_statistics=False)
p=m.weight;p.custom_token={{'keep':[1]}};old_dict=p.__dict__;old_ptr=p._cdata;before=p.detach().clone()
try:
 if {method!r}=='direct_to_fixture':p.to('cpu')
 elif {method!r}=='module_to_fixture':m.to('cpu')
 else:p._quantize(torch.device('meta'))
except RuntimeError as e:assert 'incompatible tensor type' in str(e)
else:raise AssertionError('Original failure was not reproduced')
assert p._cdata==old_ptr and p.__dict__ is old_dict and torch.equal(p.detach(),before)
assert not p.bnb_quantized and p.quant_state is None and m.quant_state is None
print(json.dumps({{'failure':'ORIGINAL_INCOMPATIBLE_TYPE','original_unchanged':True}}))
''',patched=False)
    assert got['original_unchanged']

def control_swap_rejection_keeps_tensorimpl_dict_values_and_module_state(negative):
    got=child(META_FIXTURE+f'''
import weakref
m=b.nn.Linear4bit(7,2,bias=True,quant_type='nf4',compress_statistics=False)
p=m.weight
if {negative!r}=='requires_grad':p.requires_grad_(True)
if {negative!r}=='subclass':
 class Other(b.nn.Params4bit):pass
 p=Other(torch.ones(2,7),quant_type='nf4',compress_statistics=False,module=m);m.weight=p
p.custom_token={{'keep':[1,2]}};token=p.custom_token
before=p.detach().clone();old_ptr=p._cdata;old_dict=p.__dict__;old_class=type(p);old_requires=p.requires_grad
if {negative!r}=='python_weakref':held=weakref.ref(p)
elif {negative!r}=='cpp_weakref':held=torch._C._WeakTensorRef(p)
elif {negative!r}=='held_impl':
 held=[torch.ops.aten.alias.default(p),torch.ops.aten.alias.default(p)]
 assert p._use_count()>2
try:p.to('cpu')
except (RuntimeError,NotImplementedError) as e:error=str(e)
else:raise AssertionError('Incorrect conversion was accepted')
assert p._cdata==old_ptr and p.__dict__ is old_dict and type(p) is old_class
assert p.device.type=='cpu' and p.dtype==torch.float32 and p.requires_grad==old_requires
assert torch.equal(p.detach(),before) and p.custom_token is token
assert not p.bnb_quantized and p.quant_state is None and m.quant_state is None and m.weight is p
if {negative!r}=='python_weakref':assert 'weakref associated' in error
if {negative!r}=='cpp_weakref':assert 'Expected no weakrefs' in error
if {negative!r}=='held_impl':assert 'use_count' in error
if {negative!r}=='requires_grad':assert 'floating point' in error
if {negative!r}=='subclass':assert 'original Params4bit' in error
print(json.dumps({{'negative':{negative!r},'rejected':True,'tensorimpl_unchanged':True,'dict_identity_unchanged':True,'data_unchanged':True,'module_unchanged':True,'error':error}}))
''')
    assert all(got[k] for k in ['rejected','tensorimpl_unchanged','dict_identity_unchanged','data_unchanged','module_unchanged'])

def control_compatible_cpu_branch_and_plain_state_format_weights_only(tmp_path):
    got=child(f'''
# This is genuine original CPU quantization, not the Meta result fixture.
m=b.nn.Linear4bit(7,2,bias=True,quant_type='nf4',compress_statistics=False)
p=m.weight;p.custom_token={{'keep':[1]}};token=p.custom_token;identity=id(p)
m.to('cpu')
assert m.weight is p and id(p)==identity and p.custom_token is token and p.module is m
assert p.dtype==torch.uint8 and p.quant_state is m.quant_state and p.bnb_quantized
state=m.state_dict()
assert all(type(v) is torch.Tensor for v in state.values())
expected={{k:(v.shape,v.dtype,type(v)) for k,v in state.items()}}
original_quant=b.functional.quantize_4bit
# Target type fixture, separate module. No numerical result is claimed for Meta tensors.
def meta_quant(w,**kw):return original_quant(w.to('meta'),**kw)
b.functional.quantize_4bit=meta_quant
other=b.nn.Linear4bit(7,2,bias=True,quant_type='nf4',compress_statistics=False)
other.to('cpu')
state2=other.state_dict()
assert {{k:(v.shape,v.dtype,type(v)) for k,v in state2.items()}}==expected
path={str(tmp_path/'state.pt')!r}
torch.save(state2,path)
loaded=torch.load(path,weights_only=True)
assert {{k:(v.shape,v.dtype,type(v)) for k,v in loaded.items()}}==expected
assert 'weight.quant_state.bitsandbytes__nf4' in loaded
assert all(type(v) is torch.Tensor for v in loaded.values())
assert other.weight.detach().__class__ is torch.Tensor
print(json.dumps({{'compatible_cpu':True,'weight_identity':True,'custom_attrs':True,'state_format':True,'weights_only_load':True,'keys':sorted(loaded)}}))
''')
    assert all(got[k] for k in ['compatible_cpu','weight_identity','custom_attrs','state_format','weights_only_load'])

def control_new_rejection_preserves_parameter_and_state(rejection):
    observed = child(META_FIXTURE + f'''
m=b.nn.Linear4bit(7,2,bias=True,quant_type='nf4',compress_statistics=False)
p=m.weight
if {rejection!r}=='existing_gradient':p.grad=torch.arange(14,dtype=torch.float32).reshape(2,7)
if {rejection!r}=='overwrite_flag':torch.__future__.set_overwrite_module_params_on_conversion(True)
if {rejection!r}=='swap_flag':torch.__future__.set_swap_module_params_on_conversion(True)
p.custom_token={{'keep':[1,2]}};token=p.custom_token
original_dict=p.__dict__;original_ptr=p._cdata;original_class=type(p)
original_state=p.quant_state;original_module_state=m.quant_state;original_grad=p.grad
original_bias=m.bias;original_requires=p.requires_grad;before=p.detach().clone()
try:m.to('cpu')
except RuntimeError as e:error=str(e)
else:raise AssertionError('Expected isolated rejection')
if {rejection!r}=='existing_gradient':assert 'without an existing gradient' in error
else:assert 'default module conversion flags' in error
assert m.weight is p and p._cdata==original_ptr and p.__dict__ is original_dict
assert type(p) is original_class is b.nn.Params4bit and torch.equal(p.detach(),before)
assert p.grad is original_grad and p.requires_grad==original_requires
assert p.quant_state is original_state and m.quant_state is original_module_state
assert p.quant_state is m.quant_state and not p.bnb_quantized
assert p.module is m and p.custom_token is token and m.bias is original_bias
print(json.dumps({{'case':{rejection!r},'scope':'CPU_META_TYPE_MECHANICS_ONLY','rejected_before_swap':True,'parameter_identity':True,'tensorimpl_identity':True,'dict_identity':True,'class_identity':True,'values_unchanged':True,'quant_state_identity':True,'module_alias_unchanged':True,'gradient_identity':True,'bias_identity':True}}))
''')
    assert all(value is True for key, value in observed.items() if key not in {"case", "scope"})

class TransferSourceControls(unittest.TestCase):
    def test_four_success_routes(self):
        if SOURCE is None: self.skipTest('Run this file with --source')
        for method in ('direct_to_fixture','module_to_fixture','direct_quantize_meta','module_apply_meta'):
            with self.subTest(method=method): control_incompatible_result_keeps_identity_class_attrs_and_module_alias(method)

    def test_three_original_failures(self):
        if SOURCE is None: self.skipTest('Run this file with --source')
        for method in ('direct_to_fixture','module_to_fixture','direct_quantize_meta'):
            with self.subTest(method=method): control_original_source_preserves_incompatible_failure(method)

    def test_five_swap_rejections(self):
        if SOURCE is None: self.skipTest('Run this file with --source')
        for negative in ('python_weakref','cpp_weakref','held_impl','requires_grad','subclass'):
            with self.subTest(negative=negative): control_swap_rejection_keeps_tensorimpl_dict_values_and_module_state(negative)

    def test_cpu_state_format(self):
        if SOURCE is None: self.skipTest('Run this file with --source')
        control_compatible_cpu_branch_and_plain_state_format_weights_only(BASE)

    def test_three_new_rejections(self):
        if SOURCE is None: self.skipTest('Run this file with --source')
        for rejection in ('existing_gradient','overwrite_flag','swap_flag'):
            with self.subTest(rejection=rejection): control_new_rejection_preserves_parameter_and_state(rejection)

    def test_cpu_forward_gradients_and_fresh_state(self):
        if SOURCE is None: self.skipTest('Run this file with explicit source paths')
        path=BASE/'cpu-state.pt'
        first=child('''
m=b.nn.Linear4bit(7,2,bias=True,quant_type='nf4',compress_statistics=False,compute_dtype=torch.float32)
with torch.no_grad():
 m.weight.copy_(torch.arange(14,dtype=torch.float32).reshape(2,7)/14-0.5)
 m.bias.copy_(torch.tensor([0.125,-0.25]))
p=m.weight;m.to('cpu')
x=(torch.arange(21,dtype=torch.float32).reshape(3,7)/21).requires_grad_()
y=m(x);decoded=b.functional.dequantize_4bit(p,quant_state=p.quant_state)
torch.testing.assert_close(y,torch.nn.functional.linear(x,decoded,m.bias),rtol=0,atol=0)
y.sum().backward()
torch.testing.assert_close(x.grad,torch.ones_like(y)@decoded,rtol=0,atol=0)
torch.testing.assert_close(m.bias.grad,torch.tensor([3.,3.]),rtol=0,atol=0)
assert m.weight is p and not p.requires_grad and p.grad is None and p.quant_state is m.quant_state
state=m.state_dict();assert all(type(v) is torch.Tensor for v in state.values())
torch.save(state,'''+repr(str(path))+''')
print(json.dumps({'case':'genuine_cpu_forward_backward','forward_exact':True,'dx_exact':True,'db_exact':True,'frozen_base':True,'state_plain_tensors':True,'forward_values':y.detach().tolist()}))
''')
        second=child('''
loaded=torch.load('''+repr(str(path))+''',weights_only=True)
m=b.nn.Linear4bit(7,2,bias=True,quant_type='nf4',compress_statistics=False,compute_dtype=torch.float32)
stats={k.removeprefix('weight.'):v for k,v in loaded.items() if k.startswith('weight.')}
m.weight=b.nn.Params4bit.from_prequantized(loaded['weight'],stats,device='cpu',module=m)
m.load_state_dict({k:loaded[k] for k in ['weight','bias']})
x=torch.arange(21,dtype=torch.float32).reshape(3,7)/21
torch.testing.assert_close(m(x),torch.tensor('''+repr(first['forward_values'])+'''),rtol=0,atol=0)
assert type(m.weight) is b.nn.Params4bit and type(m.weight.quant_state) is b.functional.QuantState
assert m.weight.quant_state is m.quant_state and not m.weight.requires_grad and m.weight.grad is None
assert all(type(v) is torch.Tensor for v in loaded.values())
print(json.dumps({'case':'fresh_process_cpu_public_restore','forward_exact':True,'weights_only':True,'original_classes':True,'module_state_alias':True,'frozen_base':True}))
''')
        self.assertTrue(first['forward_exact'] and second['forward_exact'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,help='Pristine Git checkout or extracted package; build isolated patch fixture')
    parser.add_argument('--base-source',type=Path)
    parser.add_argument('--patched-source',type=Path)
    parser.add_argument('--output','--report',dest='output',type=Path)
    args=parser.parse_args()
    if args.source is not None:
        if args.base_source is not None or args.patched_source is not None:parser.error('Choose --source or both explicit source paths')
    elif args.base_source is None or args.patched_source is None:
        parser.error('Both --base-source and --patched-source are required')
    import torch,platform
    report={'record_validation':'FAIL',
            'runtime':{'torch':torch.__version__,'python':platform.python_version(),'platform':platform.system()},
            'qualified_source_controls':platform.system()=='Linux' and sys.version_info[:2]==(3,12) and torch.__version__=='2.9.0+cpu',
            'tpu':'NOT_RUN','xla':'NOT_RUN','whole_module_transactionality':'NOT_PROMISED',
            'patch_manifest_sha256':MANIFEST_SHA,'controls':RECORDS}
    try:
        with tempfile.TemporaryDirectory() as work:
            prepare((args.source or args.base_source).resolve(),work,args.patched_source.resolve() if args.patched_source else None)
            result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TransferSourceControls))
            report['record_validation']='PASS' if result.wasSuccessful() else 'FAIL'
            report['failures']=[{'control':str(test),'traceback':trace} for test,trace in result.failures+result.errors]
    except Exception as error:
        report['reason']=type(error).__name__+': '+str(error)
    print(json.dumps(report))
    if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['record_validation']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
