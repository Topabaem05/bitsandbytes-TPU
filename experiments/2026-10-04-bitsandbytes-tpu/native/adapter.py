"""Private native Torch/XLA Pallas adapter. Import has no JAX or XLA side effect."""
import hashlib
import importlib
import importlib.util
import math
from pathlib import Path
import sys

KERNEL_SHA='ca2d28ea0857913115d97c4484e5d97eec60aefaa862c0c2eadf1e9eafdf4d87'
BRIDGE_SHA='c347f8fcb4844fa8849109680ad81b94e221a2c8b011462553b48ec9f67e6338'
GUARD_SHA='fd8c53eb24fb373fba35c938e00b1ef36caadf3eaf8a74d266528ec6602ad42f'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)

def guarded_kernel(kernel_path,*,import_module=importlib.import_module,module_table=None,loader=None):
    """Injectable importer is for offline fixtures; device diagnostics use defaults."""
    table=sys.modules if module_table is None else module_table
    require('jax' not in table,'JAX_ALREADY_IMPORTED_REQUIRE_FRESH_PROCESS')
    require(sha(kernel_path)==KERNEL_SHA,'FROZEN_PALLAS_SOURCE')
    xla=import_module('torch_xla')
    guard=import_module('torch_xla._internal.jax_workarounds')
    require(sha(guard.__file__)==GUARD_SHA,'GUARD_SOURCE')
    guard.jax_import_guard()
    # First explicit JAX/Pallas import happens in the frozen kernel after the guard.
    if loader is None:
        spec=importlib.util.spec_from_file_location('r6_frozen_pallas',kernel_path);kernel=importlib.util.module_from_spec(spec);spec.loader.exec_module(kernel)
    else:kernel=loader(kernel_path)
    kernel.runtime_guard()
    bridge=import_module('torch_xla.experimental.custom_kernel')
    require(sha(bridge.__file__)==BRIDGE_SHA,'BRIDGE_SOURCE')
    return kernel,bridge,xla


def output_spec(activation,packed,scales):
    require(activation.ndim==2 and packed.ndim==3 and scales.ndim==3,'POSITIONAL_METADATA_RANK')
    return [((activation.shape[0],packed.shape[1]),activation.dtype)]


def native_boundary(call,torch,*operands):
    """Pure output bridge boundary; do not change or replace any original input."""
    require(len(operands)==3 and all(isinstance(value,torch.Tensor) for value in operands),'THREE_POSITIONAL_TENSORS')
    flags=[torch._is_functional_tensor(value) for value in operands];raw=[]
    for value,functional in zip(operands,flags):
        if functional:
            torch._functionalize_sync(value);value=torch._from_functional_tensor(value)
        raw.append(value)
    result=call(*raw)
    require(isinstance(result,torch.Tensor),'SINGLE_TENSOR_OUTPUT')
    if any(flags) and not torch._is_functional_tensor(result):result=torch._to_functional_tensor(result)
    return result


class ForwardAdapter:
    def __init__(self,torch,compatibility,reference,call,*,require_xla=True,events=None):
        self.torch=torch;self.compatibility=compatibility;self.reference=reference;self.call=call;self.require_xla=require_xla;self.events=[] if events is None else events
    def validate(self,A,B,shapeB,absmax,blocksize,quant_type,bias,nested):
        t=self.torch;self.compatibility.check_kind(blocksize,quant_type);self.compatibility.check_dtype(A.dtype)
        require(len(shapeB)==2 and all(type(size) is int and size>0 for size in shapeB),'POSITIVE_WEIGHT_SHAPE')
        require(A.ndim in (2,3) and A.shape[-1]==shapeB[1] and all(size>0 for size in A.shape),'ACTIVATION_SHAPE')
        self.compatibility.check_packed(B,absmax,shapeB,A.dtype)
        require(A.device==B.device==absmax.device,'OPERAND_DEVICE')
        if self.require_xla:require(A.device.type=='xla','XLA_DEVICE_REQUIRED')
        require(not any(value is not None for value in nested),'NESTED_NOT_ADMITTED_PENDING_M4')
        if bias is not None:require(tuple(bias.shape)==(shapeB[0],) and bias.dtype==A.dtype and bias.device==A.device,'BIAS_CONTRACT')
    def __call__(self,A,B,shapeB,absmax,blocksize,quant_type,bias=None,absmax_8bit=None,absmax_code=None,absmax_offset=None):
        self.validate(A,B,shapeB,absmax,blocksize,quant_type,bias,(absmax_8bit,absmax_code,absmax_offset))
        m=math.prod(A.shape[:-1]);n,k=shapeB
        reason='BIAS_REFERENCE' if bias is not None else 'UNSUPPORTED_TILE_REFERENCE' if m%128 or n%128 or k%256 else 'NONCONTIGUOUS_ACTIVATION_REFERENCE' if not A.is_contiguous() else None
        if reason:
            self.events.append({'path':'SAME_DEVICE_REFERENCE','reason':reason,'activation_shape':list(A.shape),'weight_shape':list(shapeB)})
            return self.reference.gemm_4bit(A,B,shapeB,absmax,blocksize,quant_type,bias)
        q=k//256
        flat=A.reshape(m,k)
        grouped=B.reshape(n,q,128).permute(1,0,2)
        scales=absmax.reshape(n,q,4).permute(1,0,2)
        # No .cpu(), .item(), JAX conversion, container operands or tensor kwargs.
        self.events.append({'path':'PALLAS_CANDIDATE','reason':'SUPPORTED','positional_operands':[{'shape':list(value.shape),'dtype':str(value.dtype),'stride':list(value.stride())} for value in (flat,grouped,scales)],'grouped_preparation':'TORCH_RESHAPE_PERMUTE','configuration_binding':'NOT_QUALIFIED'})
        result=native_boundary(self.call,self.torch,flat,grouped,scales)
        require(result.shape==(m,n) and result.dtype==A.dtype and result.device==A.device,'BRIDGE_OUTPUT_CONTRACT')
        return result.reshape(*A.shape[:-1],n)


def create_adapter(kernel_path,torch,compatibility,reference,*,events=None):
    kernel,bridge,xla=guarded_kernel(kernel_path)
    def entry(activation,packed,scales):return kernel.forward_grouped(activation,packed,scales,interpret=False)
    call=bridge.make_kernel_from_pallas(entry,output_spec)
    return ForwardAdapter(torch,compatibility,reference,call,events=events),entry,bridge


def implementation_map(adapter,reference,functionalization):
    """Unadopted replacement map for a future explicit plugin build; no registration."""
    wrap=functionalization.functionalized_kernel
    return {'quantize_4bit':wrap(reference.quantize_4bit),'dequantize_4bit':wrap(reference.dequantize_4bit),'dequantize_4bit.out':wrap(reference.dequantize_4bit_out),'gemm_4bit':wrap(adapter)}
