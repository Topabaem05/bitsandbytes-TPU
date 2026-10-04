"""Previous normalization for the isolated failure control.

Adapted from pinned bitsandbytes default/ops.py and the accepted plugin reference.
Copyright (c) Facebook, Inc. and its affiliates. MIT license.
See the package THIRD_PARTY_NOTICES.md for the license text.
Upstream: 833649043474794b8fe7a4136e0c40faf077b2e0.
The fixture keeps the previous compare-sum behavior under denormal flushing.
"""
import torch
import torch.nn.functional as F
from bitsandbytes_tpu.compatibility import check_dtype, check_kind

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
    scaled = (blocks * (1.0 / scales.clamp(min=1e-38).view(-1, 1))).clamp(-1, 1).reshape(-1)
    if remainder:
        tail_scale = flat[full:].abs().max().clamp(min=1e-38)
        scales = torch.cat((scales, tail_scale.unsqueeze(0)))
        scaled = torch.cat((scaled, (flat[full:] / tail_scale).clamp(-1, 1)))
    if scaled.numel() % 2:
        scaled = F.pad(scaled, (0, 1))
    # right=False: a value equal to a midpoint enters the lower bucket.
    codes = (scaled.unsqueeze(-1) > bounds).sum(dim=-1, dtype=torch.int32).to(torch.uint8)
    packed = ((codes[::2] << 4) | codes[1::2]).unsqueeze(1)
    return packed, scales
