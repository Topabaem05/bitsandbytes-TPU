"""Private finite FP32 multiply/general divide using only int32/bool tensors.

Caller admits finite operands and a nonzero divisor. RNE overflow returns
signed infinity; underflow retains the XOR sign. No device value host read.
"""
import torch
SIGN = -2147483648
FRACTION = 8388607
LIMB = 16777215

def require_pair(left, right):
    if left.dtype != torch.int32 or right.dtype != torch.int32:
        raise ValueError('INT32_WORDS_REQUIRED')
    if left.device != right.device:
        raise ValueError('SAME_DEVICE_REQUIRED')

def decode(words):
    magnitude = words & 2147483647
    exponent = _diagnostic_tensor_shift(magnitude, 23, 'right')
    mantissa = magnitude & FRACTION | _diagnostic_tensor_shift((exponent != 0).to(torch.int32), 23, 'left')
    power = torch.where(exponent == 0, -149, exponent - 150)
    for _ in range(23):
        active = (mantissa != 0) & (mantissa < _diagnostic_tensor_shift(1, 23, 'left'))
        mantissa = torch.where(active, _diagnostic_tensor_shift(mantissa, 1, 'left'), mantissa)
        power = torch.where(active, power - 1, power)
    return (mantissa, power, magnitude == 0)

def encode(sign, rounded, power, normal, zero_input):
    carry = rounded >= _diagnostic_tensor_shift(1, 24, 'left')
    mantissa = torch.where(carry, _diagnostic_tensor_shift(rounded, 1, 'right'), rounded)
    final_power = power + carry.to(torch.int32)
    safe_power = torch.where(final_power > 127, 127, final_power)
    exponent = torch.where(normal, safe_power + 127, 0)
    magnitude = torch.where(normal, _diagnostic_tensor_shift(exponent, 23, 'left') | mantissa & FRACTION, rounded)
    magnitude = torch.where(normal & (final_power >= 128), torch.full_like(magnitude, 2139095040), magnitude)
    return sign | torch.where(zero_input, torch.zeros_like(magnitude), magnitude)

def product48(left, right):
    """Exact 24x24 product, stored as two base2**24 limbs. No int64 needed."""
    low = torch.zeros_like(left)
    high = torch.zeros_like(left)
    for bit in range(24):
        take = _diagnostic_tensor_shift(right, bit, 'right') & 1 != 0
        part_low = _diagnostic_tensor_shift(left & _diagnostic_tensor_shift(1, 24 - bit, 'left') - 1, bit, 'left')
        part_high = _diagnostic_tensor_shift(left, 24 - bit, 'right')
        trial = low + torch.where(take, part_low, 0)
        high = high + torch.where(take, part_high, 0) + _diagnostic_tensor_shift(trial, 24, 'right')
        low = trial & LIMB
    return (high, low)

def rounded_shift48(high, low, shift):
    """RNE 48-bit positive integer right shift; admitted shift >=23."""
    low_shift = torch.where(shift < 24, torch.full_like(shift, 23), torch.full_like(shift, 24))
    low_truncated = _diagnostic_tensor_shift(high, 24 - low_shift, 'left') | _diagnostic_tensor_shift(low, low_shift, 'right')
    high_shift = torch.where(shift - 24 < 0, 0, torch.where(shift - 24 > 24, 24, shift - 24))
    high_truncated = _diagnostic_tensor_shift(high, high_shift, 'right')
    low_half = _diagnostic_tensor_shift(torch.ones_like(low), low_shift - 1, 'left')
    high_half = _diagnostic_tensor_shift(torch.ones_like(high), torch.where(high_shift > 0, high_shift - 1, 0), 'left')
    low_half_set = low & low_half != 0
    high_half_set = high & high_half != 0
    low_below = low & low_half - 1 != 0
    high_below = (high & high_half - 1 != 0) | (low != 0)
    use_high = shift > 24
    truncated = torch.where(use_high, high_truncated, low_truncated)
    half_set = torch.where(use_high, high_half_set, low_half_set)
    below = torch.where(use_high, high_below, low_below)
    increment = half_set & (below | (truncated & 1 != 0))
    rounded = truncated + increment.to(torch.int32)
    return torch.where(shift > 48, torch.zeros_like(rounded), rounded)

def multiply_bits(left, right):
    require_pair(left, right)
    lm, lp, lzero = decode(left)
    rm, rp, rzero = decode(right)
    high, low = product48(lm, rm)
    high_bit = high >= _diagnostic_tensor_shift(1, 23, 'left')
    power = lp + rp + 46 + high_bit.to(torch.int32)
    normal = power >= -126
    shift = torch.where(normal, 23 + high_bit.to(torch.int32), -(lp + rp) - 149)
    rounded = rounded_shift48(high, low, shift)
    return encode((left ^ right) & SIGN, rounded, power, normal, lzero | rzero)

def divide_bits(numerator, denominator):
    """Finite numerator / finite nonzero denominator, rounded once to FP32."""
    require_pair(numerator, denominator)
    nm, np, nzero = decode(numerator)
    dm, dp, _ = decode(denominator)
    integer_bit = nm >= dm
    quotient = integer_bit.to(torch.int32)
    remainder = torch.where(integer_bit, nm - dm, nm)
    low_ratio = nm < dm
    precision_shift = 23 + low_ratio.to(torch.int32)
    for bit in range(24):
        active = precision_shift > bit
        trial = _diagnostic_tensor_shift(remainder, 1, 'left')
        take = trial >= dm
        remainder = torch.where(active, torch.where(take, trial - dm, trial), remainder)
        quotient = torch.where(active, _diagnostic_tensor_shift(quotient, 1, 'left') | take.to(torch.int32), quotient)
    power = np - dp - low_ratio.to(torch.int32)
    normal = power >= -126
    twice = _diagnostic_tensor_shift(remainder, 1, 'left')
    rounded = quotient + ((twice > dm) | (twice == dm) & (quotient & 1 != 0)).to(torch.int32)
    right = precision_shift - (np - dp + 149)
    bounded = torch.where(right < 0, 0, torch.where(right > 24, 24, right))
    half = _diagnostic_tensor_shift(torch.ones_like(quotient), torch.where(bounded > 0, bounded - 1, 0), 'left')
    truncated = _diagnostic_tensor_shift(quotient, bounded, 'right')
    half_set = quotient & half != 0
    below = (quotient & half - 1 != 0) | (remainder != 0)
    right_rounded = truncated + (half_set & (below | (truncated & 1 != 0))).to(torch.int32)
    right_rounded = torch.where(right > 24, torch.zeros_like(right_rounded), right_rounded)
    rounded = torch.where(normal, rounded, right_rounded)
    return encode((numerator ^ denominator) & SIGN, rounded, power, normal, nzero)

from diagnostic_ops import tensor_shift as _diagnostic_tensor_shift
