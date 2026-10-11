"""Private blocksize-256 prototype from the pinned default Python algorithm.

The finite, sorted dynamic map and finite input are caller preconditions.
No input-value check or host tensor extraction is used in these kernels.
Native CPU LUT/zero behavior is a separate reference and is not reproduced here.
Copyright (c) Facebook, Inc. and its affiliates. MIT license.
"""
import torch
from . import exact


def check_blockwise(A, code, blocksize):
    if blocksize != 256:
        raise NotImplementedError('Only blocksize 256 is supported for nested scales')
    if A.numel() == 0:
        raise ValueError('Empty scales are not supported')
    if code.dtype != torch.float32 or tuple(code.shape) != (256,) or not code.is_contiguous():
        raise ValueError('The map must be contiguous float32 [256]')
    if A.device != code.device:
        raise ValueError('Scale data and map must have the same device')


def quantize_blockwise(A, code, blocksize):
    check_blockwise(A, code, blocksize)
    if A.dtype != torch.float32:
        raise NotImplementedError('Nested centered scales must use float32')
    indices, scales = exact.quantize_words(exact.words(A), exact.words(code), blocksize)
    return indices.reshape(A.shape), exact.floats(scales)


def dequantize_blockwise(A, absmax, code, blocksize, dtype):
    check_blockwise(A, code, blocksize)
    if A.dtype != torch.uint8:
        raise ValueError('Centered scale codes must use uint8')
    if dtype != torch.float32:
        raise NotImplementedError('Restored nested scales must use float32')
    if absmax.dtype != torch.float32 or tuple(absmax.shape) != ((A.numel() + 255) // 256,):
        raise ValueError('Second scales must have one float32 value per 256 codes')
    if absmax.device != A.device:
        raise ValueError('Codes and second scales must have the same device')
    result = exact.decode_words(A, exact.words(absmax), exact.words(code), 256)
    return exact.floats(result)


def dequantize_blockwise_out(A, absmax, code, blocksize, dtype, out):
    if out.shape != A.shape or out.dtype != dtype or out.device != A.device:
        raise ValueError('Output shape, dtype, and device must match')
    result = dequantize_blockwise(A, absmax, code, blocksize, dtype)
    out.copy_(result)
    return None


def restored_scales(B, shape, absmax, codes, code, offset):
    if any(x is None for x in (codes, code, offset)):
        raise ValueError('All three nested arguments are required')
    count = 1
    for size in shape: count *= size
    if tuple(codes.shape) != ((count + 63) // 64,):
        raise ValueError('Nested codes must have one value per NF4 block')
    if offset.dtype != torch.float32 or offset.ndim != 0 or offset.device != B.device or codes.device != B.device:
        raise ValueError('Offset must be a scalar float32 on the packed-weight device')
    return exact.restore_scales(codes, absmax, code, offset)
