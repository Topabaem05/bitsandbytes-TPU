"""CPU functionalization controls. These tests do not use an XLA client."""
import importlib

import pytest
import torch

from test_registration import child


def test_backend_context_wraps_all_kernels_and_restores_excluded_key():
    f = importlib.import_module('bitsandbytes_tpu.functionalization')
    from bitsandbytes_tpu import reference as r
    key = torch._C.DispatchKey.Functionalize
    p, s = r.quantize_4bit(torch.linspace(-1, 1, 14), 64, 'nf4', torch.uint8)
    ep, es = r.quantize_4bit(torch.cat((-torch.ones(7), torch.ones(7))), 64, 'nf4', torch.uint8)
    calls = [
        (r.quantize_4bit, (torch.linspace(-1, 1, 14), 64, 'nf4', torch.uint8), True),
        (r.dequantize_4bit, (p, s, 64, 'nf4', [2, 7], torch.float32), True),
        (r.gemm_4bit, (torch.ones(3, 7), p, [2, 7], s, 64, 'nf4', torch.zeros(2)), False),
        (r.gemm_4bit, (torch.ones(3, 7), ep, [2, 7], es, 64, 'nf4', torch.tensor([-.5, .5])), True),
    ]
    for function, args, exact in calls:
        expected = function(*args)
        def observed(*a, **kw):
            assert not torch._C._dispatch_tls_is_dispatch_key_excluded(key)
            assert torch._is_functional_tensor(a[0])
            factory = torch.tensor([1., 2., 3.])
            assert torch._is_functional_tensor(factory)
            assert factory[:-1].shape == (2,)
            return function(*a, **kw)
        wrapped = f.functionalized_kernel(observed)
        with torch._C._SetExcludeDispatchKeyGuard(key, True):
            got = wrapped(*args)
            assert torch._C._dispatch_tls_is_dispatch_key_excluded(key)
        if isinstance(expected, tuple):
            assert all(torch.equal(x, y) and not torch._is_functional_tensor(x) for x, y in zip(got, expected))
        elif exact:
            assert torch.equal(got, expected) and not torch._is_functional_tensor(got)
        else:
            # Existing fixed42 fp32_forward_gradient profile, not a new threshold.
            torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)
            assert not torch._is_functional_tensor(got)
    out = torch.full((2, 7), -99.)
    with torch._C._SetExcludeDispatchKeyGuard(key, True):
        assert f.functionalized_kernel(r.dequantize_4bit_out)(p, s, 64, 'nf4', [2, 7], torch.float32, out) is None
        assert torch._C._dispatch_tls_is_dispatch_key_excluded(key)
    assert torch.equal(out, r.dequantize_4bit(p, s, 64, 'nf4', [2, 7], torch.float32))


@pytest.mark.parametrize('excluded', [False, True])
def test_backend_context_exception_restores_key_and_output(excluded):
    f = importlib.import_module('bitsandbytes_tpu.functionalization')
    from bitsandbytes_tpu import reference as r
    key = torch._C.DispatchKey.Functionalize
    p, s = r.quantize_4bit(torch.ones(14), 64, 'nf4', torch.uint8)
    out = torch.full((2, 7), -99.)
    with torch._C._SetExcludeDispatchKeyGuard(key, excluded):
        before = (str(torch._C._dispatch_tls_local_include_set()), str(torch._C._dispatch_tls_local_exclude_set()))
        with pytest.raises(ValueError, match='output'):
            f.functionalized_kernel(r.dequantize_4bit_out)(p, s, 64, 'nf4', [7, 2], torch.float32, out)
        assert torch._C._dispatch_tls_is_dispatch_key_excluded(key) == excluded
        assert before == (str(torch._C._dispatch_tls_local_include_set()), str(torch._C._dispatch_tls_local_exclude_set()))
    assert torch.equal(out, torch.full_like(out, -99.))
    assert not torch._C._dispatch_tls_is_dispatch_key_excluded(key)
    assert torch.equal(f.functionalized_kernel(r.dequantize_4bit)(p, s, 64, 'nf4', [2, 7], torch.float32), torch.ones(2, 7))


def test_registered_functional_out_preserves_backend_and_alias_mutation():
    result = child('''
import json, torch, bitsandbytes
from bitsandbytes_tpu import register
register()
op=torch.ops.bitsandbytes.dequantize_4bit
p,s=torch.ops.bitsandbytes.quantize_4bit.default(torch.arange(-7.,7.),64,'nf4',torch.uint8)
expected=op.default(p,s,64,'nf4',[2,7],torch.float32)
storage=torch.full((4,7),-99.);out=storage[1:3]
def call(p,s,out):
 result=op.out(p,s,64,'nf4',[2,7],torch.float32,out=out)
 assert result is None
 return out
got=torch.func.functionalize(call,remove='mutations_and_views')(p,s,out)
assert torch.equal(got,expected) and torch.equal(out,expected)
assert torch.equal(storage[0],torch.full((7,),-99.)) and torch.equal(storage[3],torch.full((7,),-99.))
# This CPU-only backend control must keep options outside the XLA NF4 contract.
fp,fs=torch.ops.bitsandbytes.quantize_4bit.default(torch.arange(6.).half(),64,'fp4',torch.uint8)
want=op.default(fp,fs,64,'fp4',[2,3],torch.float16)
def other(p,s,out):
 op.out(p,s,64,'fp4',[2,3],torch.float16,out=out)
 return out
target=torch.full((2,3),-99.,dtype=torch.float16)
assert torch.equal(torch.func.functionalize(other)(fp,fs,target),want)
assert torch.equal(target,want)
# Output may alias an input scale. Read all scales before publishing mutation.
ap,as_=torch.ops.bitsandbytes.quantize_4bit.default(torch.tensor([-1.]),64,'nf4',torch.uint8)
def alias(p,s):
 op.out(p,s,64,'nf4',[1],torch.float32,out=s)
 return s
assert torch.equal(torch.func.functionalize(alias)(ap,as_),torch.tensor([-1.]))
assert torch.equal(as_,torch.tensor([-1.]))
bad=torch.full((7,2),-99.)
try: torch.func.functionalize(call)(p,s,bad)
except ValueError as e: assert 'output' in str(e)
else: raise AssertionError('Wrong output shape was accepted')
assert torch.equal(bad,torch.full_like(bad,-99.))
print(json.dumps({'view_and_alias_mutation':True,'fp4_cpu_backend_kept':True,'invalid_output_rejected':True}))
''')
    assert all(result.values())


def test_existing_functionalize_kernel_rejects_before_xla_registration():
    got = child('''
import json,torch,bitsandbytes as b
lib=torch.library.Library('bitsandbytes','IMPL','Functionalize')
lib.impl('dequantize_4bit.out',lambda *a,**k:None)
from bitsandbytes_tpu import register
try: register()
except RuntimeError as e: assert 'existing Functionalize' in str(e)
else: raise AssertionError('Functionalize collision was hidden')
assert not torch._C._dispatch_has_kernel_for_dispatch_key('bitsandbytes::quantize_4bit','XLA')
assert 'xla' not in b.supported_torch_devices
print(json.dumps({'rejected_before_partial_registration':True}))
''')
    assert got['rejected_before_partial_registration']


def test_functionalize_registration_failure_rolls_back_and_retries():
    got = child('''
import json,torch,bitsandbytes as b
from bitsandbytes_tpu import register
old=torch.library.Library.impl
def fail(self,name,fn,key='',**kw):
 if key=='Functionalize': raise RuntimeError('Injected Functionalize registration failure')
 return old(self,name,fn,key,**kw)
torch.library.Library.impl=fail
try:
 try: register()
 except RuntimeError as e: assert 'Injected' in str(e)
 else: raise AssertionError('Registration failure was hidden')
finally: torch.library.Library.impl=old
for name in ['quantize_4bit','dequantize_4bit','dequantize_4bit.out','gemm_4bit']:
 assert not torch._C._dispatch_has_kernel_for_dispatch_key('bitsandbytes::'+name,'XLA')
assert not torch._C._dispatch_has_kernel_for_dispatch_key('bitsandbytes::dequantize_4bit.out','Functionalize')
assert 'xla' not in b.supported_torch_devices
register();register()
assert torch._C._dispatch_has_kernel_for_dispatch_key('bitsandbytes::dequantize_4bit.out','Functionalize')
print(json.dumps({'rollback':True,'retry':True,'idempotent':True}))
''')
    assert all(got.values())
