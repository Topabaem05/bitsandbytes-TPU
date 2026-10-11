"""Private finite FP32 multiply/general divide using only int32/bool tensors.

Caller admits finite operands and a nonzero divisor. RNE overflow returns
signed infinity; underflow retains the XOR sign. No device value host read.
"""
import torch
from ._integer_clamp import left_shift, right_shift, scalar_s32
SIGN=-2147483648
FRACTION=0x7fffff
LIMB=0xffffff

def require_pair(left,right):
    if left.dtype!=torch.int32 or right.dtype!=torch.int32:raise ValueError('INT32_WORDS_REQUIRED')
    if left.device!=right.device:raise ValueError('SAME_DEVICE_REQUIRED')

def decode(words):
    magnitude=words&0x7fffffff;exponent=right_shift(magnitude, 23)
    mantissa=(magnitude&FRACTION)|(left_shift((torch.ops.aten.ne.Tensor(exponent, scalar_s32(words, 0))).to(torch.int32), 23))
    power=torch.where(torch.ops.aten.eq.Tensor(exponent, scalar_s32(words, 0)),-149,exponent-150)
    for _ in range(23):
        active=(torch.ops.aten.ne.Tensor(mantissa, scalar_s32(words, 0)))&(torch.ops.aten.lt.Tensor(mantissa, scalar_s32(words, 1<<23)))
        mantissa=torch.where(active,left_shift(mantissa, 1),mantissa)
        power=torch.where(active,power-1,power)
    return mantissa,power,torch.ops.aten.eq.Tensor(magnitude, scalar_s32(words, 0))

def encode(sign,rounded,power,normal,zero_input):
    carry=torch.ops.aten.ge.Tensor(rounded, scalar_s32(rounded, 1<<24))
    mantissa=torch.where(carry,right_shift(rounded, 1),rounded)
    final_power=power+carry.to(torch.int32)
    safe_power=torch.where(torch.ops.aten.gt.Tensor(final_power, scalar_s32(rounded, 127)),127,final_power)
    exponent=torch.where(normal,safe_power+127,0)
    magnitude=torch.where(normal,(left_shift(exponent, 23))|(mantissa&FRACTION),rounded)
    magnitude=torch.where(normal&(torch.ops.aten.ge.Tensor(final_power, scalar_s32(rounded, 128))),torch.full_like(magnitude,0x7f800000),magnitude)
    return sign|torch.where(zero_input,torch.zeros_like(magnitude),magnitude)

def product48(left,right):
    """Exact 24x24 product, stored as two base2**24 limbs. No int64 needed."""
    low=torch.zeros_like(left);high=torch.zeros_like(left)
    for bit in range(24):
        take=torch.ops.aten.ne.Tensor((right_shift(right, bit))&1, scalar_s32(left, 0))
        part_low=left_shift(left&((1<<(24-bit))-1), bit)
        part_high=right_shift(left, 24-bit)
        trial=low+torch.where(take,part_low,0)
        high=high+torch.where(take,part_high,0)+(right_shift(trial, 24))
        low=trial&LIMB
    return high,low

def rounded_shift48(high,low,shift):
    """RNE 48-bit positive integer right shift; admitted shift >=23."""
    low_shift=torch.where(torch.ops.aten.lt.Tensor(shift, scalar_s32(high, 24)),torch.full_like(shift,23),torch.full_like(shift,24))
    low_truncated=(left_shift(high, 24-low_shift))|(right_shift(low, low_shift))
    high_shift=torch.where(torch.ops.aten.lt.Tensor(shift-24, scalar_s32(high, 0)),0,torch.where(torch.ops.aten.gt.Tensor(shift-24, scalar_s32(high, 24)),24,shift-24))
    high_truncated=right_shift(high, high_shift)
    low_half=left_shift(torch.ones_like(low), low_shift-1)
    high_half=left_shift(torch.ones_like(high), torch.where(torch.ops.aten.gt.Tensor(high_shift, scalar_s32(high, 0)),high_shift-1,0))
    low_half_set=torch.ops.aten.ne.Tensor(low&low_half, scalar_s32(high, 0))
    high_half_set=torch.ops.aten.ne.Tensor(high&high_half, scalar_s32(high, 0))
    low_below=torch.ops.aten.ne.Tensor(low&(low_half-1), scalar_s32(high, 0))
    high_below=(torch.ops.aten.ne.Tensor(high&(high_half-1), scalar_s32(high, 0)))|(torch.ops.aten.ne.Tensor(low, scalar_s32(high, 0)))
    use_high=torch.ops.aten.gt.Tensor(shift, scalar_s32(high, 24))
    truncated=torch.where(use_high,high_truncated,low_truncated)
    half_set=torch.where(use_high,high_half_set,low_half_set)
    below=torch.where(use_high,high_below,low_below)
    increment=half_set&(below|(torch.ops.aten.ne.Tensor(truncated&1, scalar_s32(high, 0))))
    rounded=truncated+increment.to(torch.int32)
    return torch.where(torch.ops.aten.gt.Tensor(shift, scalar_s32(high, 48)),torch.zeros_like(rounded),rounded)

def multiply_bits(left,right):
    require_pair(left,right)
    lm,lp,lzero=decode(left);rm,rp,rzero=decode(right)
    high,low=product48(lm,rm)
    high_bit=torch.ops.aten.ge.Tensor(high, scalar_s32(left, 1<<23))
    power=lp+rp+46+high_bit.to(torch.int32)
    normal=torch.ops.aten.ge.Tensor(power, scalar_s32(left, -126))
    shift=torch.where(normal,23+high_bit.to(torch.int32),-(lp+rp)-149)
    rounded=rounded_shift48(high,low,shift)
    return encode((left^right)&SIGN,rounded,power,normal,lzero|rzero)

def divide_bits(numerator,denominator):
    """Finite numerator / finite nonzero denominator, rounded once to FP32."""
    require_pair(numerator,denominator)
    nm,np,nzero=decode(numerator);dm,dp,_=decode(denominator)
    integer_bit=nm>=dm
    quotient=integer_bit.to(torch.int32)
    remainder=torch.where(integer_bit,nm-dm,nm)
    low_ratio=nm<dm
    precision_shift=23+low_ratio.to(torch.int32)
    for bit in range(24):
        active=torch.ops.aten.gt.Tensor(precision_shift, scalar_s32(numerator, bit))
        trial=left_shift(remainder, 1);take=trial>=dm
        remainder=torch.where(active,torch.where(take,trial-dm,trial),remainder)
        quotient=torch.where(active,(left_shift(quotient, 1))|take.to(torch.int32),quotient)
    power=np-dp-low_ratio.to(torch.int32)
    normal=torch.ops.aten.ge.Tensor(power, scalar_s32(numerator, -126))
    twice=left_shift(remainder, 1)
    rounded=quotient+((twice>dm)|((twice==dm)&(torch.ops.aten.ne.Tensor(quotient&1, scalar_s32(numerator, 0))))).to(torch.int32)
    right=precision_shift-(np-dp+149)
    bounded=torch.where(torch.ops.aten.lt.Tensor(right, scalar_s32(numerator, 0)),0,torch.where(torch.ops.aten.gt.Tensor(right, scalar_s32(numerator, 24)),24,right))
    half=left_shift(torch.ones_like(quotient), torch.where(torch.ops.aten.gt.Tensor(bounded, scalar_s32(numerator, 0)),bounded-1,0))
    truncated=right_shift(quotient, bounded)
    half_set=torch.ops.aten.ne.Tensor(quotient&half, scalar_s32(numerator, 0))
    below=(torch.ops.aten.ne.Tensor(quotient&(half-1), scalar_s32(numerator, 0)))|(torch.ops.aten.ne.Tensor(remainder, scalar_s32(numerator, 0)))
    right_rounded=truncated+(half_set&(below|(torch.ops.aten.ne.Tensor(truncated&1, scalar_s32(numerator, 0))))).to(torch.int32)
    right_rounded=torch.where(torch.ops.aten.gt.Tensor(right, scalar_s32(numerator, 24)),torch.zeros_like(right_rounded),right_rounded)
    rounded=torch.where(normal,rounded,right_rounded)
    return encode((numerator^denominator)&SIGN,rounded,power,normal,nzero)
