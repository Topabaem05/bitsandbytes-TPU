"""Register exact upstream operators for XLA without changing CPU kernels."""
import sys
import threading

import torch

from .compatibility import SCHEMAS, validate_upstream
from . import reference, blockwise, exact
from .functionalization import functionalized_kernel, dequantize_4bit_out_functionalize, dequantize_blockwise_out_functionalize

_LIBRARY = None
_COMPOSITES = None
_LOCK = threading.Lock()
IMPLEMENTATIONS = {
    "quantize_blockwise": functionalized_kernel(blockwise.quantize_blockwise),
    "dequantize_blockwise": functionalized_kernel(blockwise.dequantize_blockwise),
    "dequantize_blockwise.out": functionalized_kernel(blockwise.dequantize_blockwise_out),
    "quantize_4bit": functionalized_kernel(reference.quantize_4bit),
    "dequantize_4bit": functionalized_kernel(reference.dequantize_4bit),
    "dequantize_4bit.out": functionalized_kernel(reference.dequantize_4bit_out),
    "gemm_4bit": functionalized_kernel(reference.gemm_4bit),
}
COMPOSITE_SCHEMAS = {
    "compress_statistics": "(Tensor first_scales, Tensor dynamic_code) -> (Tensor, Tensor, Tensor)",
    "restore_scales": "(Tensor scale_codes, Tensor second_scales, Tensor dynamic_code, Tensor offset) -> Tensor",
}
COMPOSITE_BODIES = {
    "compress_statistics": functionalized_kernel(exact.compress_statistics),
    "restore_scales": functionalized_kernel(exact.restore_scales),
}
FUNCTIONALIZE_OUTS = {"dequantize_4bit.out": dequantize_4bit_out_functionalize,
                     "dequantize_blockwise.out": dequantize_blockwise_out_functionalize}


def register():
    """Register once, or fail before changing an existing XLA implementation."""
    global _LIBRARY, _COMPOSITES
    with _LOCK:
        bnb = sys.modules.get("bitsandbytes")
        if bnb is None:
            raise RuntimeError("Import bitsandbytes to load its backend entrypoints")
        validate_upstream(bnb)
        if _LIBRARY is not None:
            if not all(torch._C._dispatch_has_kernel_for_dispatch_key("bitsandbytes::" + name, "XLA") for name in SCHEMAS):
                raise RuntimeError("The plugin registration is incomplete")
            if not all(torch._C._dispatch_has_kernel_for_dispatch_key("bitsandbytes::" + name, "Functionalize") for name in FUNCTIONALIZE_OUTS):
                raise RuntimeError("The plugin functionalization registration is incomplete")
            for name in COMPOSITE_SCHEMAS:
                if not torch._C._dispatch_has_kernel_for_dispatch_key("bnb_tpu_exact::"+name,"XLA"):
                    raise RuntimeError("Composite registration is incomplete")
            return
        for name in COMPOSITE_SCHEMAS:
            try:
                torch._C._dispatch_find_schema_or_throw("bnb_tpu_exact::"+name, "")
            except RuntimeError:
                continue
            raise RuntimeError("An existing composite schema will not be overwritten: "+name)
        for name in SCHEMAS:
            if torch._C._dispatch_has_kernel_for_dispatch_key("bitsandbytes::" + name, "XLA"):
                raise RuntimeError(f"An existing XLA implementation will not be overwritten: {name}")
        for name in FUNCTIONALIZE_OUTS:
            if torch._C._dispatch_has_kernel_for_dispatch_key("bitsandbytes::" + name, "Functionalize"):
                raise RuntimeError("An existing Functionalize implementation will not be overwritten: " + name)
        library = torch.library.Library("bitsandbytes", "IMPL")
        composites = torch.library.Library("bnb_tpu_exact", "DEF")
        try:
            for name, schema in COMPOSITE_SCHEMAS.items():
                composites.define(name+schema)
                torch.library.register_kernel("bnb_tpu_exact::"+name,"xla",COMPOSITE_BODIES[name],lib=composites)
            for name, implementation in IMPLEMENTATIONS.items():
                torch.library.register_kernel("bitsandbytes::" + name, "xla", implementation, lib=library)
            for name, implementation in FUNCTIONALIZE_OUTS.items():
                library.impl(name, implementation, "Functionalize")
        except BaseException:
            # Remove partial registrations if a later registration fails.
            library._destroy()
            composites._destroy()
            raise
        _LIBRARY = library
        _COMPOSITES = composites
        bnb.supported_torch_devices.add("xla")
