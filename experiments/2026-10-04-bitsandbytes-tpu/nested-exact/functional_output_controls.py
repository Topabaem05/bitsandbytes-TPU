"""Real CPU functional wrappers with a stub XLA builder and device facade.

These isolated controls do not execute XLA or qualify the pinned Linux runtime.
The tested bitcast function is compiled unchanged from the selected source AST.
"""
import argparse,ast,hashlib,json,sys,tempfile,types
from pathlib import Path
from unittest.mock import patch
import torch

HERE=Path(__file__).resolve().parent
EXACT=HERE/'integration/plugin/src/bitsandbytes_tpu/exact.py'
class Facade:
    def __init__(self,tensor,device='xla:0'):self.tensor=tensor;self.device=torch.device(device)
    @property
    def dtype(self):return self.tensor.dtype
class TorchBoundary:
    float32=torch.float32;int32=torch.int32
    def __init__(self):self.wrap_calls=0;self.sync_calls=0;self.unwrap_calls=0
    def _is_functional_tensor(self,value):return torch._is_functional_tensor(value.tensor)
    def _functionalize_sync(self,value):self.sync_calls+=1;torch._functionalize_sync(value.tensor)
    def _from_functional_tensor(self,value):self.unwrap_calls+=1;return Facade(torch._from_functional_tensor(value.tensor),str(value.device))
    def _to_functional_tensor(self,value):self.wrap_calls+=1;return Facade(torch._to_functional_tensor(value.tensor),str(value.device))
def function(path,boundary,builder_sha):
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='bitcast')
    code=compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec')
    namespace={'torch':boundary,'hashlib':hashlib,'Path':Path,'BUILDER_SHA':builder_sha}
    exec(code,namespace);return namespace['bitcast']
def raw(value):
    if torch._is_functional_tensor(value):torch._functionalize_sync(value);return torch._from_functional_tensor(value)
    return value

def run(original):
    rows=[];key=torch._C.DispatchKey.Functionalize
    tls=lambda:torch._C._dispatch_tls_is_dispatch_key_excluded(key)
    def control(name,callback):
        before=tls();callback();assert tls()==before,'TLS_CHANGED';rows.append({'name':name,'status':'PASS'})
    with tempfile.TemporaryDirectory(prefix='m4-functional-controls-',dir='/tmp') as temporary:
        builder_file=Path(temporary)/'xla_builder.py';builder_file.write_bytes(b'STUB_XLA_BUILDER_CONTROL_ONLY\n')
        builder_sha=hashlib.sha256(builder_file.read_bytes()).hexdigest();state={};boundary=TorchBoundary()
        xb=types.ModuleType('torch_xla.core.xla_builder');xb.__file__=str(builder_file);xb.Type=types.SimpleNamespace(S32='S32',F32='F32')
        def shape(value):
            assert not torch._is_functional_tensor(value.tensor),'FUNCTIONAL_INPUT_REACHED_CPP_BRIDGE'
            state['bridge_input']=value;return tuple(value.tensor.shape)
        xb.tensor_shape=shape
        def create(name,callback,shapes):
            target=callback(types.SimpleNamespace(bitcast=lambda value:value));state['target']=target;return target
        xb.create_computation=create
        def call(computation,name):
            def execute(values):
                assert len(values)==1
                if state.get('raise'):raise ValueError('STUB_BUILDER_EXCEPTION')
                dtype=torch.int32 if computation=='S32' else torch.float32
                tensor=values[0].tensor.view(dtype)
                if state.get('wrong_dtype'):tensor=tensor.to(torch.float64)
                if state.get('output_functional'):tensor=torch._to_functional_tensor(tensor)
                output=Facade(tensor,'xla:1' if state.get('wrong_device') else str(values[0].device))
                state['output']=output
                return [] if state.get('arity')==0 else [output,output] if state.get('arity')==2 else [output]
            return execute
        xb.xla_computation_as_func=call
        modules={'torch_xla':types.ModuleType('torch_xla'),'torch_xla.core':types.ModuleType('torch_xla.core'),'torch_xla.core.xla_builder':xb}
        modules['torch_xla'].core=modules['torch_xla.core'];modules['torch_xla.core'].xla_builder=xb
        fixed=function(EXACT,boundary,builder_sha);old=function(original,boundary,builder_sha)
        signed=lambda x:x if x<2**31 else x-2**32
        bits=torch.tensor([signed(v) for v in [0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x3f800000,0xbf800000,0x7f7fffff,0xff7fffff]],dtype=torch.int32)
        def reset(**options):
            state.clear();state.update(options);boundary.wrap_calls=boundary.sync_calls=boundary.unwrap_calls=0
        def matrix(input_functional,output_functional,input_dtype):
            reset(output_functional=output_functional)
            source=bits.clone().view(input_dtype);source=torch._to_functional_tensor(source) if input_functional else source
            value=Facade(source);target=torch.int32 if input_dtype==torch.float32 else torch.float32
            result=fixed(value,target);expected=bits.clone().view(target)
            assert torch._is_functional_tensor(result.tensor)==(input_functional or output_functional)
            assert torch.equal(raw(result.tensor).view(torch.int32),expected.view(torch.int32))
            assert boundary.wrap_calls==int(input_functional and not output_functional)
            assert boundary.sync_calls==boundary.unwrap_calls==int(input_functional)
            if output_functional or not input_functional:assert result is state['output'],'OUTPUT_IDENTITY_CHANGED'
            out=torch._to_functional_tensor(torch.empty_like(expected));out.copy_(result.tensor);torch._functionalize_sync(out)
            assert torch.equal(torch._from_functional_tensor(out).view(torch.int32),expected.view(torch.int32))
        with patch.dict(sys.modules,modules):
            def old_failure():
                reset(output_functional=True)
                value=Facade(torch._to_functional_tensor(torch.tensor([1.],dtype=torch.float32)))
                try:old(value,torch.int32)
                except RuntimeError as exc:assert 'isFunctionalTensor(base_)' in str(exc) and 'FunctionalStorageImpl.cpp' in str(exc)
                else:raise AssertionError('OLD_DOUBLE_WRAP_DID_NOT_FAIL')
                assert boundary.wrap_calls==1
            control('old_unconditional_wrap_reproduces_real_torch_assertion',old_failure)
            for first in (False,True):
                for second in (False,True):
                    for dtype in (torch.float32,torch.int32):
                        control(f'matrix_input_{first}_output_{second}_{dtype}',lambda first=first,second=second,dtype=dtype:matrix(first,second,dtype))
            def mutation():
                reset(output_functional=True);value=Facade(torch._to_functional_tensor(torch.tensor([1.],dtype=torch.float32)))
                value.tensor.add_(2.);result=fixed(value,torch.int32)
                assert torch.equal(raw(result.tensor),torch.tensor([3.],dtype=torch.float32).view(torch.int32));assert boundary.sync_calls==1
            control('pending_input_mutation_syncs_before_bridge',mutation)
            def rejection(options,reason,*,dtype=torch.float32,target=torch.int32,device='xla:0',functional=True):
                reset(**options);tensor=torch.tensor([1.],dtype=dtype)
                value=Facade(torch._to_functional_tensor(tensor) if functional else tensor,device)
                try:fixed(value,target)
                except ValueError as exc:assert str(exc)==reason,(str(exc),reason)
                else:raise AssertionError('INCORRECT_BOUNDARY_ACCEPTED')
                assert boundary.wrap_calls==0
            for functional in (False,True):
                for option in ({'arity':0},{'arity':2},{'wrong_dtype':True},{'wrong_device':True}):
                    control(f'result_gate_{option}_functional_{functional}',lambda option=option,functional=functional:rejection(option,'DEVICE_BITCAST_RESULT',functional=functional))
            control('cpu_input_rejected',lambda:rejection({},'XLA_ONLY_NO_CPU_FALLBACK',device='cpu'))
            control('wrong_input_dtype_rejected',lambda:rejection({},'FP32_INT32_REQUIRED',dtype=torch.float64))
            control('wrong_target_dtype_rejected',lambda:rejection({},'FP32_INT32_REQUIRED',target=torch.int64))
            def wrong_builder():
                builder_file.write_bytes(b'WRONG_BUILDER\n')
                try:rejection({},'PINNED_XLA_BUILDER_REQUIRED')
                finally:builder_file.write_bytes(b'STUB_XLA_BUILDER_CONTROL_ONLY\n')
            control('builder_sha_rejected',wrong_builder)
            control('builder_exception_preserves_tls',lambda:rejection({'raise':True},'STUB_BUILDER_EXCEPTION'))
        assert not any(p.suffix=='.pyc' for p in Path(temporary).rglob('*'))
    assert not Path(temporary).exists()
    return {'kind':'M4_FUNCTIONAL_OUTPUT_CPU_BOUNDARY_CONTROLS_V1','status':'PASS','control_count':len(rows),'controls':rows,'torch_version':torch.__version__,'platform':sys.platform,'qualified_pinned_runtime':False,'boundary':'REAL_CPU_FUNCTIONAL_WRAPPERS_STUB_XLA_BUILDER_DEVICE_FACADE','exact_sha256':hashlib.sha256(EXACT.read_bytes()).hexdigest(),'original_exact_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'temporary_fixture_removed':True,'provider_calls':0,'XLA':'NOT_RUN','TPU':'NOT_RUN','m4_status':'NOT_QUALIFIED'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original-exact',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=run(a.original_exact);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':result['status'],'controls':result['control_count'],'TPU':'NOT_RUN'}))
