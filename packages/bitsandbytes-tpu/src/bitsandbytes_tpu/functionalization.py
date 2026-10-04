"""Local native functionalization context for Python backend kernels.

Native wrappers remove intermediate views and mutations without a functorch layer.
The context keeps output tensor updates visible to callers.
"""
from functools import wraps

import torch
from torch.utils._pytree import tree_map_only


def functionalized_kernel(function):
    """Use native input wrappers and keep the caller's included dispatch keys."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        inputs = {}

        def wrap(tensor):
            if torch._is_functional_tensor(tensor):
                raise RuntimeError("Backend kernel inputs must be unwrapped tensors")
            # Keep repeated arguments attached to the same native wrapper.
            if id(tensor) not in inputs:
                inputs[id(tensor)] = (tensor, torch._to_functional_tensor(tensor))
            return inputs[id(tensor)][1]

        wrapped_args, wrapped_kwargs = tree_map_only(torch.Tensor, wrap, (args, kwargs))
        # Match the native composite helper: remove exclusion, do not force inclusion.
        with torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.Functionalize, False):
            result = function(*wrapped_args, **wrapped_kwargs)

        def unwrap(tensor):
            if torch._is_functional_tensor(tensor):
                torch._functionalize_sync(tensor)
                return torch._from_functional_tensor(tensor)
            return tensor

        result = tree_map_only(torch.Tensor, unwrap, result)
        updates = []
        for original, functional in inputs.values():
            updated = unwrap(functional)
            if updated is not original:
                if updated.shape != original.shape or updated.stride() != original.stride():
                    raise RuntimeError("Backend kernels must not change input metadata")
                updates.append((original, updated))
        # The mutable output overload needs its fixed-size data update published.
        with torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.Functionalize, True):
            for original, updated in updates:
                original.copy_(updated)
        return result

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
