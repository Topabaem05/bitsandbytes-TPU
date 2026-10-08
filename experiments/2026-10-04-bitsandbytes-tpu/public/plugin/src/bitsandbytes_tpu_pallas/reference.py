"""Device-preserving Torch reference kernels for non-nested NF4.

The quantization and decoding rules derive from bitsandbytes default/ops.py.
Copyright (c) Facebook, Inc. and its affiliates. MIT license.
See THIRD_PARTY_NOTICES.md. No fused kernel or memory benefit is claimed.
Inputs must be finite. Scales must be finite and nonnegative.
These value preconditions are not tested inside the kernels.
"""
import torch
import torch.nn.functional as F

from .compatibility import check_dtype, check_kind, check_packed

# Python values only. Import does not allocate a CPU or TPU tensor.
NF4_CODE = (-1.0, -0.6961928009986877, -0.5250730514526367,
            -0.39491748809814453, -0.28444138169288635,
            -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
            0.33791524171829224, 0.44070982933044434,
            0.5626170039176941, 0.7229568362236023, 1.0)


def quantize_4bit(A, blocksize, quant_type, quant_storage):
    check_kind(blocksize, quant_type)
    check_dtype(A.dtype)
    if quant_storage != torch.uint8:
        raise NotImplementedError("Only uint8 packed storage is supported")
    if A.numel() == 0:
        raise ValueError("Empty weights are not supported")
    code = torch.tensor(NF4_CODE, dtype=torch.float32, device=A.device)
    bounds = (code[:-1] + code[1:]) / 2
    flat = A.reshape(-1).float()
    remainder = flat.numel() % 64
    full = flat.numel() - remainder
    blocks = flat[:full].reshape(-1, 64)
    scales = blocks.abs().max(dim=-1)[0]
    # Match upstream multiplication by a reciprocal, including small scales.
    # TPU may flush the small clamp value to zero. A zero block needs no scaling.
    zero_blocks = scales == 0
    normalizer = torch.where(zero_blocks, torch.ones_like(scales), scales.clamp(min=1e-38))
    scaled_blocks = blocks * (1.0 / normalizer.view(-1, 1))
    scaled = torch.where(zero_blocks.view(-1, 1), torch.zeros_like(blocks), scaled_blocks).clamp(-1, 1).reshape(-1)
    if remainder:
        tail_max = flat[full:].abs().max()
        tail_scale = tail_max.clamp(min=1e-38)
        scales = torch.cat((scales, tail_scale.unsqueeze(0)))
        tail_normalizer = torch.where(tail_max == 0, torch.ones_like(tail_max), tail_scale)
        tail_values = torch.where(tail_max == 0, torch.zeros_like(flat[full:]), flat[full:] / tail_normalizer)
        scaled = torch.cat((scaled, tail_values.clamp(-1, 1)))
    if scaled.numel() % 2:
        scaled = F.pad(scaled, (0, 1))
    # right=False: a value equal to a midpoint enters the lower bucket.
    codes = (scaled.unsqueeze(-1) > bounds).sum(dim=-1, dtype=torch.int32).to(torch.uint8)
    packed = ((codes[::2] << 4) | codes[1::2]).unsqueeze(1)
    return packed, scales


def dequantize_4bit(A, absmax, blocksize, quant_type, shape, dtype):
    check_kind(blocksize, quant_type)
    count = check_packed(A, absmax, shape, dtype)
    packed = A.reshape(-1)
    indices = torch.stack((packed >> 4, packed & 15), dim=1).reshape(-1)[:count].long()
    code = torch.tensor(NF4_CODE, dtype=torch.float32, device=A.device)
    values = code[indices]
    block_ids = torch.arange(count, device=A.device, dtype=torch.int64) // 64
    return (values * absmax[block_ids]).reshape(shape).to(dtype)


def dequantize_4bit_out(A, absmax, blocksize, quant_type, shape, dtype, out):
    if tuple(out.shape) != tuple(shape) or out.dtype != dtype or out.device != A.device:
        raise ValueError("The output shape, dtype, and device must match")
    result = dequantize_4bit(A, absmax, blocksize, quant_type, shape, dtype)
    out.copy_(result)
    return None


def gemm_4bit(A, B, shapeB, absmax, blocksize, quant_type, bias=None,
              absmax_8bit=None, absmax_code=None, absmax_offset=None):
    if any(x is not None for x in (absmax_8bit, absmax_code, absmax_offset)):
        raise NotImplementedError("Nested NF4 is not supported in this initial slice")
    check_kind(blocksize, quant_type)
    check_dtype(A.dtype)
    if len(shapeB) != 2 or A.ndim not in (2, 3) or A.shape[-1] != shapeB[1]:
        raise ValueError("GEMM needs rank2 or rank3 input and matching [N, K] weights")
    if A.device != B.device:
        raise ValueError("Input and packed weights must be on the same device")
    if bias is not None:
        if tuple(bias.shape) != (shapeB[0],) or bias.dtype != A.dtype or bias.device != A.device:
            raise ValueError("Bias must have shape [N] and the input dtype and device")
    weights = dequantize_4bit(B, absmax, blocksize, quant_type, shapeB, A.dtype)
    return F.linear(A, weights, bias)
