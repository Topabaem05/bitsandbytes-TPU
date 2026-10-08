"""Private arithmetic-probe kernels. No public method/dispatch replacement.

The XLA bitcast path is source-backed but has NOT executed on a TPU.
The explicit sum graph is a candidate, not a guarantee against reassociation.
"""
import hashlib
from pathlib import Path
import torch

BUILDER_SHA='ee315ac0cfa9f052a5a437d693bc1d65aae7ea7bf7c3d0b661384bc6723a8772'


def xla_bitcast(value, dtype):
    if value.device.type!='xla': raise ValueError('XLA tensors required; no CPU fallback')
    if (value.dtype,dtype) not in ((torch.float32,torch.int32),(torch.int32,torch.float32)):
        raise ValueError('Only float32/int32 bit reinterpretation is supported')
    import torch_xla.core.xla_builder as xb
    if hashlib.sha256(Path(xb.__file__).read_bytes()).hexdigest()!=BUILDER_SHA:
        raise ValueError('Pinned Torch/XLA2.9 builder source required')
    to_type=xb.Type.S32 if dtype==torch.int32 else xb.Type.F32
    name='R4PrimitiveBitcast'+to_type
    computation=xb.create_computation(name,lambda operand:operand.bitcast(to_type),(xb.tensor_shape(value),))
    outputs=xb.xla_computation_as_func(computation,name=name)([value])
    if len(outputs)!=1 or outputs[0].dtype!=dtype or outputs[0].device!=value.device:
        raise ValueError('Unexpected device bitcast result')
    return outputs[0],xb.get_computation_hlo(computation)


def row_sum(values):
    """Source-derived ILP4 cascade. Tensor math only; sizes are static metadata."""
    count=values.shape[0];ilp=count//4;shape=values.shape[1:]
    acc=[[torch.zeros(shape,dtype=values.dtype,device=values.device) for _ in range(4)] for _ in range(4)]
    power=max(4,((ilp-1).bit_length() if ilp else 0)//4);step=1<<power;mask=step-1;i=0
    while i+step<=ilp:
        for _ in range(step):
            for k in range(4):acc[0][k]=acc[0][k]+values[4*i+k]
            i+=1
        for level in range(1,4):
            for k in range(4):
                acc[level][k]=acc[level][k]+acc[level-1][k]
                acc[level-1][k]=torch.zeros(shape,dtype=values.dtype,device=values.device)
            if i&(mask<<(level*power)):break
    while i<ilp:
        for k in range(4):acc[0][k]=acc[0][k]+values[4*i+k]
        i+=1
    for level in range(1,4):
        for k in range(4):acc[0][k]=acc[0][k]+acc[level][k]
    for i in range(ilp*4,count):acc[0][0]=acc[0][0]+values[i]
    for k in range(1,4):acc[0][0]=acc[0][0]+acc[0][k]
    return acc[0][0]


def ordered_sum8(value):
    if value.dtype!=torch.float32 or value.ndim!=1 or value.numel()==0:
        raise ValueError('Nonempty rank1 float32 is required')
    if value.numel()<8:return row_sum(value)
    vectors=value.numel()//8
    partials=row_sum(value[:vectors*8].reshape(vectors,8))
    total=torch.zeros((),dtype=value.dtype,device=value.device)
    for i in range(vectors*8,value.numel()):total=total+value[i]
    for i in range(8):total=total+partials[i]
    return total


def arithmetic_outputs(value,denominator):
    if denominator.dtype!=torch.float32 or denominator.ndim!=0 or denominator.device!=value.device:
        raise ValueError('Rank0 float32 device denominator is required')
    total=ordered_sum8(value)
    return {'mean':value.mean(),'sum_div':value.sum()/denominator,
            'sum_reciprocal':value.sum()*(1./denominator),
            'ordered_sum8':total,'ordered_div8':total/denominator}
