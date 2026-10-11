"""Private finite FP32 word addition/subtraction; no float arithmetic.

Callers admit finite inputs. IEEE RNE overflow returns signed infinity.
Local integer controls do not qualify XLA or TPU execution.
"""
import torch

SIGN = -2147483648
FRACTION = 0x7fffff

def require_pair(left, right):
    if left.dtype != torch.int32 or right.dtype != torch.int32:
        raise ValueError('INT32_WORDS_REQUIRED')
    if left.device != right.device:
        raise ValueError('SAME_DEVICE_REQUIRED')

def shift_right_jam(value, count):
    """Positive int32 significands, count >= 0, with bounded shift operands."""
    zero = torch.zeros_like(count)
    bounded = torch.where(count < 31, count, torch.full_like(count, 30))
    mask = (torch.ones_like(value) << bounded) - 1
    shifted = (value >> bounded) | ((value & mask) != 0).to(torch.int32)
    return torch.where(count >= 31, (value != 0).to(torch.int32),
                       torch.where(count == zero, value, shifted))

def add_bits(left, right):
    """Correctly rounded finite binary32 sum, with RNE signed-zero rules.

    Effective exponent 1 covers both subnormal and smallest-normal numbers.
    Three low bits hold guard, round, and sticky information. Intermediate
    significands remain nonnegative and below 2**28, inside signed int32.
    """
    require_pair(left, right)
    lm = left & 0x7fffffff
    rm = right & 0x7fffffff
    left_big = lm >= rm
    large = torch.where(left_big, left, right)
    small = torch.where(left_big, right, left)
    large_mag = large & 0x7fffffff
    small_mag = small & 0x7fffffff
    le = large_mag >> 23
    se = small_mag >> 23
    large_sig = (large_mag & FRACTION) | ((le != 0).to(torch.int32) << 23)
    small_sig = (small_mag & FRACTION) | ((se != 0).to(torch.int32) << 23)
    exponent = torch.where(le == 0, 1, le)
    small_exp = torch.where(se == 0, 1, se)
    aligned = shift_right_jam(small_sig << 3, exponent - small_exp)
    same_sign = ((large ^ small) & SIGN) == 0
    extended = torch.where(same_sign, (large_sig << 3) + aligned,
                           (large_sig << 3) - aligned)
    carry = extended >= (1 << 27)
    extended = torch.where(carry, shift_right_jam(extended, torch.ones_like(exponent)), extended)
    exponent = exponent + carry.to(torch.int32)
    for _ in range(26):
        active = (extended != 0) & (extended < (1 << 26)) & (exponent > 1)
        extended = torch.where(active, extended << 1, extended)
        exponent = torch.where(active, exponent - 1, exponent)
    mantissa = extended >> 3
    tail = extended & 7
    increment = (tail > 4) | ((tail == 4) & ((mantissa & 1) != 0))
    rounded = mantissa + increment.to(torch.int32)
    round_carry = rounded >= (1 << 24)
    rounded = torch.where(round_carry, rounded >> 1, rounded)
    exponent = exponent + round_carry.to(torch.int32)
    normal = rounded >= (1 << 23)
    magnitude = torch.where(normal, (exponent << 23) | (rounded & FRACTION), rounded)
    magnitude = torch.where(exponent >= 255, torch.full_like(magnitude, 0x7f800000), magnitude)
    sign = large & SIGN
    # Exact cancellation has +0 in RNE; -0 + -0 retains -0.
    sign = torch.where((magnitude == 0) & ~same_sign, torch.zeros_like(sign), sign)
    return sign | magnitude

def subtract_bits(left, right):
    require_pair(left, right)
    return add_bits(left, right ^ SIGN)
