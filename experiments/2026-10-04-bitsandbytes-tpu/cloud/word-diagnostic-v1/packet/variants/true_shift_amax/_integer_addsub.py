"""Private finite FP32 word addition/subtraction; no float arithmetic.

Callers admit finite inputs. IEEE RNE overflow returns signed infinity.
Local integer controls do not qualify XLA or TPU execution.
"""
import torch
SIGN = -2147483648
FRACTION = 8388607

def require_pair(left, right):
    if left.dtype != torch.int32 or right.dtype != torch.int32:
        raise ValueError('INT32_WORDS_REQUIRED')
    if left.device != right.device:
        raise ValueError('SAME_DEVICE_REQUIRED')

def shift_right_jam(value, count):
    """Positive int32 significands, count >= 0, with bounded shift operands."""
    zero = torch.zeros_like(count)
    bounded = torch.where(count < 31, count, torch.full_like(count, 30))
    mask = _diagnostic_tensor_shift(torch.ones_like(value), bounded, 'left') - 1
    shifted = _diagnostic_tensor_shift(value, bounded, 'right') | (value & mask != 0).to(torch.int32)
    return torch.where(count >= 31, (value != 0).to(torch.int32), torch.where(count == zero, value, shifted))

def add_bits(left, right):
    """Correctly rounded finite binary32 sum, with RNE signed-zero rules.

    Effective exponent 1 covers both subnormal and smallest-normal numbers.
    Three low bits hold guard, round, and sticky information. Intermediate
    significands remain nonnegative and below 2**28, inside signed int32.
    """
    require_pair(left, right)
    lm = left & 2147483647
    rm = right & 2147483647
    left_big = lm >= rm
    large = torch.where(left_big, left, right)
    small = torch.where(left_big, right, left)
    large_mag = large & 2147483647
    small_mag = small & 2147483647
    le = _diagnostic_tensor_shift(large_mag, 23, 'right')
    se = _diagnostic_tensor_shift(small_mag, 23, 'right')
    large_sig = large_mag & FRACTION | _diagnostic_tensor_shift((le != 0).to(torch.int32), 23, 'left')
    small_sig = small_mag & FRACTION | _diagnostic_tensor_shift((se != 0).to(torch.int32), 23, 'left')
    exponent = torch.where(le == 0, 1, le)
    small_exp = torch.where(se == 0, 1, se)
    aligned = shift_right_jam(_diagnostic_tensor_shift(small_sig, 3, 'left'), exponent - small_exp)
    same_sign = (large ^ small) & SIGN == 0
    extended = torch.where(same_sign, _diagnostic_tensor_shift(large_sig, 3, 'left') + aligned, _diagnostic_tensor_shift(large_sig, 3, 'left') - aligned)
    carry = extended >= _diagnostic_tensor_shift(1, 27, 'left')
    extended = torch.where(carry, shift_right_jam(extended, torch.ones_like(exponent)), extended)
    exponent = exponent + carry.to(torch.int32)
    for _ in range(26):
        active = (extended != 0) & (extended < _diagnostic_tensor_shift(1, 26, 'left')) & (exponent > 1)
        extended = torch.where(active, _diagnostic_tensor_shift(extended, 1, 'left'), extended)
        exponent = torch.where(active, exponent - 1, exponent)
    mantissa = _diagnostic_tensor_shift(extended, 3, 'right')
    tail = extended & 7
    increment = (tail > 4) | (tail == 4) & (mantissa & 1 != 0)
    rounded = mantissa + increment.to(torch.int32)
    round_carry = rounded >= _diagnostic_tensor_shift(1, 24, 'left')
    rounded = torch.where(round_carry, _diagnostic_tensor_shift(rounded, 1, 'right'), rounded)
    exponent = exponent + round_carry.to(torch.int32)
    normal = rounded >= _diagnostic_tensor_shift(1, 23, 'left')
    magnitude = torch.where(normal, _diagnostic_tensor_shift(exponent, 23, 'left') | rounded & FRACTION, rounded)
    magnitude = torch.where(exponent >= 255, torch.full_like(magnitude, 2139095040), magnitude)
    sign = large & SIGN
    sign = torch.where((magnitude == 0) & ~same_sign, torch.zeros_like(sign), sign)
    return sign | magnitude

def subtract_bits(left, right):
    require_pair(left, right)
    return add_bits(left, right ^ SIGN)

from diagnostic_ops import tensor_shift as _diagnostic_tensor_shift
