"""Private finite FP32 word addition/subtraction; no float arithmetic.

Callers admit finite inputs. IEEE RNE overflow returns signed infinity.
Local integer controls do not qualify XLA or TPU execution.
"""
import torch
from ._integer_clamp import left_shift, right_shift, scalar_s32

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
    bounded = torch.where(torch.ops.aten.lt.Tensor(count, scalar_s32(value, 31)), count, torch.full_like(count, 30))
    mask = (left_shift(torch.ones_like(value), bounded)) - 1
    shifted = (right_shift(value, bounded)) | (torch.ops.aten.ne.Tensor(value & mask, scalar_s32(value, 0))).to(torch.int32)
    return torch.where(torch.ops.aten.ge.Tensor(count, scalar_s32(value, 31)), (torch.ops.aten.ne.Tensor(value, scalar_s32(value, 0))).to(torch.int32),
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
    le = right_shift(large_mag, 23)
    se = right_shift(small_mag, 23)
    large_sig = (large_mag & FRACTION) | (left_shift((torch.ops.aten.ne.Tensor(le, scalar_s32(left, 0))).to(torch.int32), 23))
    small_sig = (small_mag & FRACTION) | (left_shift((torch.ops.aten.ne.Tensor(se, scalar_s32(left, 0))).to(torch.int32), 23))
    exponent = torch.where(torch.ops.aten.eq.Tensor(le, scalar_s32(left, 0)), 1, le)
    small_exp = torch.where(torch.ops.aten.eq.Tensor(se, scalar_s32(left, 0)), 1, se)
    aligned = shift_right_jam(left_shift(small_sig, 3), exponent - small_exp)
    same_sign = torch.ops.aten.eq.Tensor((large ^ small) & SIGN, scalar_s32(left, 0))
    extended = torch.where(same_sign, (left_shift(large_sig, 3)) + aligned,
                           (left_shift(large_sig, 3)) - aligned)
    carry = torch.ops.aten.ge.Tensor(extended, scalar_s32(left, 1 << 27))
    extended = torch.where(carry, shift_right_jam(extended, torch.ones_like(exponent)), extended)
    exponent = exponent + carry.to(torch.int32)
    for _ in range(26):
        active = (torch.ops.aten.ne.Tensor(extended, scalar_s32(left, 0))) & (torch.ops.aten.lt.Tensor(extended, scalar_s32(left, 1 << 26))) & (torch.ops.aten.gt.Tensor(exponent, scalar_s32(left, 1)))
        extended = torch.where(active, left_shift(extended, 1), extended)
        exponent = torch.where(active, exponent - 1, exponent)
    mantissa = right_shift(extended, 3)
    tail = extended & 7
    increment = (torch.ops.aten.gt.Tensor(tail, scalar_s32(left, 4))) | ((torch.ops.aten.eq.Tensor(tail, scalar_s32(left, 4))) & (torch.ops.aten.ne.Tensor(mantissa & 1, scalar_s32(left, 0))))
    rounded = mantissa + increment.to(torch.int32)
    round_carry = torch.ops.aten.ge.Tensor(rounded, scalar_s32(left, 1 << 24))
    rounded = torch.where(round_carry, right_shift(rounded, 1), rounded)
    exponent = exponent + round_carry.to(torch.int32)
    normal = torch.ops.aten.ge.Tensor(rounded, scalar_s32(left, 1 << 23))
    magnitude = torch.where(normal, (left_shift(exponent, 23)) | (rounded & FRACTION), rounded)
    magnitude = torch.where(torch.ops.aten.ge.Tensor(exponent, scalar_s32(left, 255)), torch.full_like(magnitude, 0x7f800000), magnitude)
    sign = large & SIGN
    # Exact cancellation has +0 in RNE; -0 + -0 retains -0.
    sign = torch.where((torch.ops.aten.eq.Tensor(magnitude, scalar_s32(left, 0))) & ~same_sign, torch.zeros_like(sign), sign)
    return sign | magnitude

def subtract_bits(left, right):
    require_pair(left, right)
    return add_bits(left, right ^ SIGN)
