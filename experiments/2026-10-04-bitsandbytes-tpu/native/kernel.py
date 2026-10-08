"""Private JAX0.7.1 Pallas NF4 forward candidate. CPU interpretation is not TPU qualification."""
from functools import partial
import importlib.metadata
import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BM=BN=BK=128
NF4_CODE=(-1.0,-0.6961928009986877,-0.5250730514526367,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.09105003625154495,0.0,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.5626170039176941,0.7229568362236023,1.0)


def runtime_guard():
    if jax.__version__!='0.7.1' or importlib.metadata.version('jaxlib')!='0.7.1':raise RuntimeError('JAX071_PAIR_REQUIRED')


def shape_policy(activation_shape,weight_shape):
    if len(activation_shape)!=2 or len(weight_shape)!=2:raise ValueError('RANK2_ONLY')
    m,k=activation_shape;n,wk=weight_shape
    if any(type(v) is not int or v<=0 for v in (m,n,k,wk)):raise ValueError('POSITIVE_STATIC_SHAPES')
    if k!=wk:raise ValueError('CONTRACTION_MISMATCH')
    if m%BM or n%BN or k%256:raise ValueError('UNSUPPORTED_TILE_SHAPE')
    return m,n,k,k//256


def prepare_operands(packed,scales,weight_shape):
    """Canonical public high-nibble-first [N*K/2,1] bytes to grouped operands."""
    if len(weight_shape)!=2:raise ValueError('WEIGHT_RANK')
    n,k=weight_shape
    if type(n) is not int or type(k) is not int or n<=0 or k<=0 or n%BN or k%256:raise ValueError('UNSUPPORTED_WEIGHT_SHAPE')
    if packed.shape!=(n*k//2,1) or packed.dtype!=jnp.uint8:raise ValueError('CANONICAL_PACKED_SHAPE_DTYPE')
    if scales.shape!=(n*k//64,) or scales.dtype!=jnp.float32:raise ValueError('SCALE_SHAPE_DTYPE')
    q=k//256
    return packed.reshape(n,q,128).transpose(1,0,2),scales.reshape(n,q,4).transpose(1,0,2)


def decode_tile(packed_group,scale_group,half):
    """Only a bounded [128,128] decoded weight tile; all NF4 literals are static."""
    byte_half=lax.dynamic_slice(packed_group,(0,half*64),(BN,64))
    high=(byte_half>>jnp.uint8(4)).astype(jnp.int32);low=(byte_half&jnp.uint8(15)).astype(jnp.int32)
    codes=jnp.stack((high,low),axis=-1).reshape(BN,BK)
    values=jnp.zeros((BN,BK),jnp.float32)
    for index,value in enumerate(NF4_CODE):values=jnp.where(codes==index,jnp.float32(value),values)
    selected=lax.dynamic_slice(scale_group,(0,half*2),(BN,2))
    scaled=jnp.repeat(selected,64,axis=1)
    return values*scaled


def nf4_kernel(activation_ref,packed_ref,scales_ref,out_ref,acc_ref):
    rk=pl.program_id(2)
    @pl.when(rk==0)
    def initialize():acc_ref[...]=jnp.zeros((BM,BN),jnp.float32)
    weight=decode_tile(packed_ref[...],scales_ref[...],rk%2).astype(activation_ref.dtype)
    product=lax.dot_general(activation_ref[...],weight,dimension_numbers=(((1,),(1,)),((),())),precision=lax.Precision.HIGHEST,preferred_element_type=jnp.float32)
    acc_ref[...]=acc_ref[...]+product
    @pl.when(rk==pl.num_programs(2)-1)
    def store():out_ref[...]=acc_ref[...].astype(out_ref.dtype)


def forward_grouped(activation,packed,scales,*,interpret=True):
    """One pallas_call result. Preparation stays outside this bridge candidate."""
    runtime_guard()
    if len(packed.shape)!=3:raise ValueError('PACKED_GROUPED_RANK')
    q,n,byte_group=packed.shape
    m,n,k,q_expected=shape_policy(activation.shape,(n,q*256))
    if q!=q_expected or byte_group!=128 or packed.dtype!=jnp.uint8:raise ValueError('PACKED_GROUPED_SHAPE_DTYPE')
    if scales.shape!=(q,n,4) or scales.dtype!=jnp.float32:raise ValueError('SCALES_GROUPED_SHAPE_DTYPE')
    if activation.dtype not in (jnp.float32,jnp.bfloat16):raise ValueError('ACTIVATION_DTYPE')
    return pl.pallas_call(nf4_kernel,out_shape=jax.ShapeDtypeStruct((m,n),activation.dtype),grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=0,in_specs=[pl.BlockSpec((BM,BK),lambda mi,ni,rk:(mi,rk)),pl.BlockSpec((None,BN,128),lambda mi,ni,rk:(rk//2,ni,0)),pl.BlockSpec((None,BN,4),lambda mi,ni,rk:(rk//2,ni,0))],out_specs=pl.BlockSpec((BM,BN),lambda mi,ni,rk:(mi,ni)),grid=(m//BM,n//BN,k//BK),scratch_shapes=[pltpu.VMEM((BM,BN),jnp.float32)]),compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','parallel','arbitrary')),interpret=interpret)(activation,packed,scales)


def forward(activation,canonical_packed,canonical_scales,weight_shape,*,interpret=True):
    shape_policy(activation.shape,weight_shape)
    packed,scales=prepare_operands(canonical_packed,canonical_scales,weight_shape)
    return forward_grouped(activation,packed,scales,interpret=interpret)
