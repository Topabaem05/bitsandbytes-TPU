"""Finite word arithmetic for an explicit nested-exact source generation.

Only the XLA boundary handles float tensors. Word kernels are also callable
directly by isolated CPU controls; this is not a registered CPU fallback.
Finite operands/intermediates are admitted caller preconditions.
"""
import hashlib
from pathlib import Path

import torch

from . import _integer_clamp as C, _integer_addsub as A
from . import _integer_muldiv as M, _source_order as S

BUILDER_SHA = 'ee315ac0cfa9f052a5a437d693bc1d65aae7ea7bf7c3d0b661384bc6723a8772'
FLOOR = 0x006ce3ee
NEGATIVE_ONE = -1082130432
ONE = 0x3f800000


def bitcast(value, dtype):
    if value.device.type != 'xla':
        raise ValueError('XLA_ONLY_NO_CPU_FALLBACK')
    if (value.dtype, dtype) not in ((torch.float32, torch.int32), (torch.int32, torch.float32)):
        raise ValueError('FP32_INT32_REQUIRED')
    # A backend wrapper may own a native functional tensor. The builder's
    # C++ tensor bridge receives its unwrapped tensor, without a second wrapper.
    functional = torch._is_functional_tensor(value)
    if functional:
        torch._functionalize_sync(value)
        value = torch._from_functional_tensor(value)
    import torch_xla.core.xla_builder as xb
    if hashlib.sha256(Path(xb.__file__).read_bytes()).hexdigest() != BUILDER_SHA:
        raise ValueError('PINNED_XLA_BUILDER_REQUIRED')
    target = xb.Type.S32 if dtype == torch.int32 else xb.Type.F32
    name = 'R4NestedExactBits' + target
    computation = xb.create_computation(name, lambda x: x.bitcast(target), (xb.tensor_shape(value),))
    outputs = xb.xla_computation_as_func(computation, name=name)([value])
    if len(outputs) != 1 or outputs[0].dtype != dtype or outputs[0].device != value.device:
        raise ValueError('DEVICE_BITCAST_RESULT')
    return torch._to_functional_tensor(outputs[0]) if functional else outputs[0]


def words(value):
    return bitcast(value.to(torch.float32), torch.int32)


def floats(value):
    return bitcast(value, torch.float32)


def finite_key(words):
    return torch.where(words < 0, -(words & 0x7fffffff), words)


def midpoints(code):
    return C.divide_count_bits(A.add_bits(code[:-1], code[1:]), 2)


def normalize(blocks, maximum, *, full):
    denominator = C.clamp_bits(maximum, FLOOR)
    if full:
        reciprocal = M.divide_bits(torch.full_like(denominator, ONE), denominator)
        result = M.multiply_bits(blocks, reciprocal.unsqueeze(-1))
    else:
        result = M.divide_bits(blocks, denominator)
    return C.clamp_bits(result, NEGATIVE_ONE, ONE)


def quantize_words(value, code, blocksize):
    if value.dtype != torch.int32 or code.dtype != torch.int32 or value.numel() == 0:
        raise ValueError('NONEMPTY_INT32_WORDS_REQUIRED')
    flat = value.reshape(-1)
    tail = flat.numel() % blocksize
    full = flat.numel() - tail
    blocks = flat[:full].reshape(-1, blocksize)
    maximum = torch.amax(blocks & 0x7fffffff, dim=-1)
    scaled = normalize(blocks, maximum, full=True).reshape(-1)
    if tail:
        remaining = flat[full:]
        tail_maximum = C.clamp_bits(torch.amax(remaining & 0x7fffffff), FLOOR)
        maximum = torch.cat((maximum, tail_maximum.unsqueeze(0)))
        scaled = torch.cat((scaled, normalize(remaining, tail_maximum, full=False)))
    bounds = finite_key(midpoints(code))
    indices = (finite_key(scaled).unsqueeze(-1) > bounds).sum(dim=-1, dtype=torch.int32)
    return indices.to(torch.uint8), maximum


def decode_words(codes, scales, code, blocksize):
    flat = codes.reshape(-1).to(torch.int32)
    # repeat_interleave has an explicit static output extent. No int64 indices,
    # integer division or value-dependent host extraction is used.
    extent = scales.repeat_interleave(blocksize)[:flat.numel()]
    return M.multiply_bits(code[flat], extent).reshape(codes.shape)


def compress_words(first, dynamic):
    if first.dtype != torch.int32 or first.ndim != 1 or not 1 <= first.numel() <= 513:
        raise ValueError('FIXED_FIRST_SCALE_DOMAIN')
    total = S.cpu_sum8_bits(first, A.add_bits)
    offset = C.divide_count_bits(total, first.numel())
    centered = A.subtract_bits(first, offset)
    codes, second = quantize_words(centered, dynamic, 256)
    return codes, offset, second


def restore_words(codes, second, dynamic, offset):
    return A.add_bits(decode_words(codes, second, dynamic, 256), offset)


def compress_statistics(first, dynamic):
    if first.dtype != torch.float32 or first.ndim != 1 or not 1 <= first.numel() <= 513:
        raise ValueError('FIXED_FIRST_SCALE_DOMAIN')
    if dynamic.dtype != torch.float32 or tuple(dynamic.shape) != (256,) or dynamic.device != first.device:
        raise ValueError('DYNAMIC_MAP_REQUIRED')
    codes, offset, second = compress_words(words(first), words(dynamic))
    return codes, floats(offset), floats(second)


def restore_scales(codes, second, dynamic, offset):
    if codes.dtype != torch.uint8 or codes.ndim != 1 or not 1 <= codes.numel() <= 513:
        raise ValueError('FIXED_SCALE_CODE_DOMAIN')
    if second.dtype != torch.float32 or tuple(second.shape) != ((codes.numel()+255)//256,):
        raise ValueError('SECOND_SCALES_REQUIRED')
    if offset.dtype != torch.float32 or offset.ndim != 0:
        raise ValueError('FP32_SCALAR_OFFSET_REQUIRED')
    if dynamic.dtype != torch.float32 or tuple(dynamic.shape) != (256,):
        raise ValueError('DYNAMIC_MAP_REQUIRED')
    if any(t.device != codes.device for t in (second, dynamic, offset)):
        raise ValueError('SAME_DEVICE_REQUIRED')
    return floats(restore_words(codes, words(second), words(dynamic), words(offset)))
