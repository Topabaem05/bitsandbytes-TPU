"""Lazy native forward. The upstream public API owns autograd and module state."""
from collections import deque
import importlib.metadata
from pathlib import Path
import platform
import threading

import torch

from . import adapter, compatibility, reference, source_admission, native_factory

RUNTIME = {
    "torch": "2.9.0+cpu", "torch-xla": "2.9.0", "libtpu": "0.0.21",
    "jax": "0.7.1", "jaxlib": "0.7.1",
}


def validate_runtime():
    # Metadata inspection does not import JAX or initialize an XLA client.
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("PALLAS_RUNTIME_PLATFORM")
    if platform.python_version() != "3.12.14" or torch.__version__ != RUNTIME["torch"]:
        raise RuntimeError("PALLAS_RUNTIME_PYTHON_TORCH")
    for distribution, version in RUNTIME.items():
        if importlib.metadata.version(distribution) != version:
            raise RuntimeError("PALLAS_RUNTIME_DISTRIBUTION:" + distribution)


class LazyForward:
    """One admitted factory per registered backend; no recovery to CPU/reference."""
    def __init__(self):
        self.events = deque(maxlen=64)
        self._lock = threading.Lock()
        self._call = None
        self._error = None
        self.native_calls = []
        self._adapter = adapter.ForwardAdapter(
            torch, compatibility, reference, self._native_call, events=self.events,
        )

    def _native_call(self, *operands):
        if self._call is None:
            with self._lock:
                if self._error is not None:
                    raise RuntimeError("PALLAS_INITIALIZATION_FAILED_NO_RETRY") from self._error
                if self._call is None:
                    try:
                        source_admission.validate_plugin()
                        validate_runtime()
                        kernel, bridge, xla = adapter.guarded_kernel(Path(__file__).with_name("pallas_kernel.py"))
                        def entry(activation, packed, scales):
                            return kernel.forward_grouped(activation, packed, scales, interpret=False)
                        self._call = native_factory.make_call(entry, adapter.output_spec, bridge, xla, torch, self.native_calls)
                    except Exception as exc:
                        self._error = exc
                        raise
        return self._call(*operands)

    def __call__(self, A, B, shapeB, absmax, blocksize, quant_type, bias=None,
                 absmax_8bit=None, absmax_code=None, absmax_offset=None):
        # Validation and same-device reference selection precede the native factory.
        if A.ndim in (2, 3) and A.numel() == 0:
            if self._adapter.require_xla and A.device.type != "xla":
                raise ValueError("XLA_DEVICE_REQUIRED")
            result = reference.gemm_4bit(
                A, B, shapeB, absmax, blocksize, quant_type, bias,
                absmax_8bit, absmax_code, absmax_offset,
            )
            self.events.append({"path": "SAME_DEVICE_REFERENCE", "reason": "EMPTY_ACTIVATION_REFERENCE",
                                "activation_shape": list(A.shape), "weight_shape": list(shapeB)})
            return result
        return self._adapter(A, B, shapeB, absmax, blocksize, quant_type, bias,
                             absmax_8bit, absmax_code, absmax_offset)


gemm_4bit = LazyForward()
