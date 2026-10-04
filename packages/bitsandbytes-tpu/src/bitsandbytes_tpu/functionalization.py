"""Local functionalization context for Python backend kernels.

Functionalization replaces intermediate views and mutations with tensor operations.
The context keeps input mutations, including output tensor updates, visible to callers.
"""
from functools import wraps

import torch


def functionalized_kernel(function):
    """Run a complete kernel with functionalization locally enabled."""
    transformed = torch.func.functionalize(function, remove="mutations_and_views")

    @wraps(function)
    def wrapped(*args, **kwargs):
        # Custom-op fallback excludes this key before entering a backend kernel.
        # Reentrant composites must restore it for new tensors and their views.
        with torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.Functionalize, False):
            return transformed(*args, **kwargs)

    return wrapped


def dequantize_4bit_out_functionalize(A, absmax, blocksize, quant_type, shape, dtype, out):
    """Decompose the mutable overload while keeping device dispatch and mutation."""
    if tuple(out.shape) != tuple(shape) or out.dtype != dtype or out.device != A.device:
        raise ValueError("The output shape, dtype, and device must match")
    with torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.Functionalize, False):
        result = torch.ops.bitsandbytes.dequantize_4bit.default(
            A, absmax, blocksize, quant_type, shape, dtype
        )
        out.copy_(result)
    return None
