"""Private tensor integer FP32 word operations; finite inputs are a precondition.

No float arithmetic, integer division, host value read, or expected-output table.
The caller admits finite words before device execution. Count is shape metadata.
CPU int32 controls are preparation only; TPU execution is NOT_RUN.
"""
import torch

SIGN=-2147483648
FRACTION=0x7fffff

def require_words(words):
    if words.dtype!=torch.int32: raise ValueError('INT32_WORDS_REQUIRED')

def scalar_s32(reference, value):
    """Create one checked S32 comparison operand on the reference device."""
    if not isinstance(reference, torch.Tensor) or reference.dtype != torch.int32:
        raise ValueError('INT32_SCALAR_REFERENCE_REQUIRED')
    if type(value) is not int or not -(1 << 31) <= value < (1 << 31):
        raise ValueError('SIGNED32_SCALAR_REQUIRED')
    return torch.tensor(value, dtype=torch.int32, device=reference.device)

def _shift_count(value, count):
    """Keep both tensor shift operands S32 on the same device."""
    require_words(value)
    if type(count) is int:
        return torch.tensor(count, dtype=torch.int32, device=value.device)
    if not isinstance(count, torch.Tensor):
        raise ValueError('INT32_SHIFT_COUNT_REQUIRED')
    require_words(count)
    if count.device != value.device:
        raise ValueError('SAME_DEVICE_SHIFT_COUNT_REQUIRED')
    return count

def left_shift(value, count):
    """Use the Tensor overload that lowers directly to XLA ShiftLeft."""
    return torch.ops.aten.bitwise_left_shift.Tensor(value, _shift_count(value, count))

def right_shift(value, count):
    """Use the Tensor overload with signed arithmetic right-shift semantics."""
    return torch.ops.aten.bitwise_right_shift.Tensor(value, _shift_count(value, count))

def scalar_word(word):
    if type(word) is not int or not -(1<<31)<=word<(1<<31): raise ValueError('SIGNED32_BOUND_REQUIRED')
    if ((word&0x7fffffff)>>23)==255: raise ValueError('FINITE_BOUND_REQUIRED')
    return word

def scalar_key(word):
    return SIGN-word if word<0 else word

def clamp_bits(words,minimum,maximum=0x7f7fffff):
    """Finite FP32 clamp, with input bits retained on equal-value comparisons."""
    require_words(words); minimum=scalar_word(minimum); maximum=scalar_word(maximum)
    low=scalar_key(minimum); high=scalar_key(maximum)
    if low>high: raise ValueError('ORDERED_BOUNDS_REQUIRED')
    keys=torch.where(torch.ops.aten.lt.Tensor(words, scalar_s32(words, 0)),-(words&0x7fffffff),words)
    result=torch.where(torch.ops.aten.lt.Tensor(keys, scalar_s32(words, low)),torch.full_like(words,minimum),words)
    return torch.where(torch.ops.aten.gt.Tensor(keys, scalar_s32(words, high)),torch.full_like(words,maximum),result)

def divide_count_bits(words,count):
    """Round finite FP32/count to nearest, ties to even; preserve zero signs.

    Count is an exact positive integer in [1, 2**24]. All arithmetic is int32.
    The restoring division has 24 input bits and at most 24 appended zeros.
    Remainders stay below count; shifted remainders stay below 2**25.
    """
    require_words(words)
    if type(count) is not int or not 1<=count<=1<<24: raise ValueError('EXACT_POSITIVE_COUNT_REQUIRED')
    zero=torch.zeros_like(words); one=torch.ones_like(words)
    magnitude=words&0x7fffffff; exponent=right_shift(magnitude, 23); fraction=magnitude&FRACTION
    mantissa=torch.where(torch.ops.aten.eq.Tensor(exponent, scalar_s32(words, 0)),fraction,fraction|(1<<23))
    power=torch.where(torch.ops.aten.eq.Tensor(exponent, scalar_s32(words, 0)),torch.full_like(words,-149),exponent-150)
    for _ in range(23):
        active=(torch.ops.aten.gt.Tensor(mantissa, scalar_s32(words, 0)))&(torch.ops.aten.lt.Tensor(mantissa, scalar_s32(words, 1<<23)))
        mantissa=torch.where(active,left_shift(mantissa, 1),mantissa)
        power=torch.where(active,power-1,power)
    log_count=count.bit_length()-1
    threshold=count<<(23-log_count) if log_count<=23 else count>>1
    low_ratio=torch.ops.aten.lt.Tensor(mantissa, scalar_s32(words, threshold))
    result_power=power+23-log_count-low_ratio.to(torch.int32)
    normal=torch.ops.aten.ge.Tensor(result_power, scalar_s32(words, -126))
    shift=torch.where(normal,torch.full_like(words,log_count)+low_ratio.to(torch.int32),power+149)
    quotient=zero; remainder=zero
    for bit in range(23,-1,-1):
        remainder=(left_shift(remainder, 1))|((right_shift(mantissa, bit))&1)
        take=torch.ops.aten.ge.Tensor(remainder, scalar_s32(words, count))
        remainder=torch.where(take,remainder-count,remainder)
        quotient=(left_shift(quotient, 1))|take.to(torch.int32)
    for bit in range(24):
        active=torch.ops.aten.gt.Tensor(shift, scalar_s32(words, bit))
        trial=left_shift(remainder, 1); take=torch.ops.aten.ge.Tensor(trial, scalar_s32(words, count))
        remainder=torch.where(active,torch.where(take,trial-count,trial),remainder)
        quotient=torch.where(active,(left_shift(quotient, 1))|take.to(torch.int32),quotient)
    twice=left_shift(remainder, 1)
    rounded=quotient+((torch.ops.aten.gt.Tensor(twice, scalar_s32(words, count)))|((torch.ops.aten.eq.Tensor(twice, scalar_s32(words, count)))&(torch.ops.aten.ne.Tensor(quotient&1, scalar_s32(words, 0))))).to(torch.int32)
    right=torch.where(torch.ops.aten.lt.Tensor(shift, scalar_s32(words, 0)),-shift,zero)
    half_shift=torch.where(torch.ops.aten.gt.Tensor(right, scalar_s32(words, 0)),right-1,zero)
    half=left_shift(one, half_shift); lower_mask=half-1
    truncated=right_shift(quotient, right)
    half_set=torch.ops.aten.ne.Tensor(quotient&half, scalar_s32(words, 0))
    above_half=(torch.ops.aten.ne.Tensor(quotient&lower_mask, scalar_s32(words, 0)))|(torch.ops.aten.ne.Tensor(remainder, scalar_s32(words, 0)))
    right_rounded=truncated+(half_set&(above_half|(torch.ops.aten.ne.Tensor(truncated&1, scalar_s32(words, 0))))).to(torch.int32)
    rounded=torch.where(torch.ops.aten.lt.Tensor(shift, scalar_s32(words, 0)),right_rounded,rounded)
    carry=torch.ops.aten.eq.Tensor(rounded, scalar_s32(words, 1<<24))
    normal_mantissa=torch.where(carry,right_shift(rounded, 1),rounded)
    normal_power=result_power+carry.to(torch.int32)
    encoded_normal=(left_shift(normal_power+127, 23))|(normal_mantissa&FRACTION)
    encoded=torch.where(normal,encoded_normal,rounded)
    encoded=torch.where(torch.ops.aten.eq.Tensor(magnitude, scalar_s32(words, 0)),zero,encoded)
    return (words&SIGN)|encoded
