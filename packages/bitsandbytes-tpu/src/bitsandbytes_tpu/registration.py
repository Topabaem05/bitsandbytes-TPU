"""Register exact upstream operators for XLA without changing CPU kernels."""
import sys
import threading

import torch

from .compatibility import SCHEMAS, validate_upstream
from . import reference
from .functionalization import functionalized_kernel, dequantize_4bit_out_functionalize

_LIBRARY = None
_LOCK = threading.Lock()
IMPLEMENTATIONS = {
    "quantize_4bit": functionalized_kernel(reference.quantize_4bit),
    "dequantize_4bit": functionalized_kernel(reference.dequantize_4bit),
    "dequantize_4bit.out": functionalized_kernel(reference.dequantize_4bit_out),
    "gemm_4bit": functionalized_kernel(reference.gemm_4bit),
}
FUNCTIONALIZE_OUT = "bitsandbytes::dequantize_4bit.out"


def register():
    """Register once, or fail before changing an existing XLA implementation."""
    global _LIBRARY
    with _LOCK:
        bnb = sys.modules.get("bitsandbytes")
        if bnb is None:
            raise RuntimeError("Import bitsandbytes to load its backend entrypoints")
        validate_upstream(bnb)
        if _LIBRARY is not None:
            if not all(torch._C._dispatch_has_kernel_for_dispatch_key("bitsandbytes::" + name, "XLA") for name in SCHEMAS):
                raise RuntimeError("The plugin registration is incomplete")
            if not torch._C._dispatch_has_kernel_for_dispatch_key(FUNCTIONALIZE_OUT, "Functionalize"):
                raise RuntimeError("The plugin functionalization registration is incomplete")
            return
        for name in SCHEMAS:
            if torch._C._dispatch_has_kernel_for_dispatch_key("bitsandbytes::" + name, "XLA"):
                raise RuntimeError(f"An existing XLA implementation will not be overwritten: {name}")
        if torch._C._dispatch_has_kernel_for_dispatch_key(FUNCTIONALIZE_OUT, "Functionalize"):
            raise RuntimeError("An existing Functionalize implementation will not be overwritten: dequantize_4bit.out")
        library = torch.library.Library("bitsandbytes", "IMPL")
        try:
            for name, implementation in IMPLEMENTATIONS.items():
                torch.library.register_kernel("bitsandbytes::" + name, "xla", implementation, lib=library)
            library.impl("dequantize_4bit.out", dequantize_4bit_out_functionalize, "Functionalize")
        except BaseException:
            # Remove partial registrations if a later registration fails.
            library._destroy()
            raise
        _LIBRARY = library
        bnb.supported_torch_devices.add("xla")
