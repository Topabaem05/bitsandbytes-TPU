"""Make sure the source, ABI, and NF4 input contract agree."""
import hashlib
from pathlib import Path

import torch

UPSTREAM_REVISION = "833649043474794b8fe7a4136e0c40faf077b2e0"
SOURCE_SHA256 = {
    "__init__.py": "026aeaf979736dd15f80b145880dc4de03ebbe131b47b9f066645da1772255b8",
    "_ops.py": "e6d9ed4c268e52044a8ff5f895f94ca88bf02d156b25c4616141cfa9daaf924c",
    "functional.py": "520a9e9d3afba11124734df60e48a27e38b6389305df888834405b435e285817",
    "nn/modules.py": "f7ab2160681e2f089caae80d6fd283ac6a9b9f1b82269e6f395c3b046faf606e",
    "autograd/_functions.py": "77ca54c87b69ad470ac73e5d7dde94948f20de9de8b8764957728bc29831f37f",
    "backends/default/ops.py": "e39afd16dca6e6f34a14305ade0ba4acbb49dc2c7a9baf937341d7e33b682819",
    "backends/utils.py": "dc564f2fbf13dba81388a23d4c87167dd54da04af8f17b0db52685fe42b11c84",
}
SCHEMAS = {
    "quantize_4bit": "(Tensor A, int blocksize, str quant_type, ScalarType quant_storage) -> (Tensor, Tensor)",
    "dequantize_4bit": "(Tensor A, Tensor absmax, int blocksize, str quant_type, int[] shape, ScalarType dtype) -> Tensor",
    "dequantize_4bit.out": "(Tensor A, Tensor absmax, int blocksize, str quant_type, int[] shape, ScalarType dtype, Tensor! out) -> ()",
    "gemm_4bit": "(Tensor A, Tensor B, int[] shapeB, Tensor absmax, int blocksize, str quant_type, Tensor? bias=None, Tensor? absmax_8bit=None, Tensor? absmax_code=None, Tensor? absmax_offset=None) -> Tensor",
}
DTYPES = (torch.float32, torch.bfloat16)


def validate_upstream(bnb):
    """Fail before registration if the installed source or operator ABI differs.

    The upstream version attribute is not set until after backend autoload.
    Source hashes work during that import and do not import another runtime.
    """
    root = Path(bnb.__file__).resolve().parent
    for relative, expected in SOURCE_SHA256.items():
        try:
            actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        except OSError as exc:
            raise RuntimeError(f"bitsandbytes source pin is unavailable: {relative}") from exc
        if actual != expected:
            raise RuntimeError(f"bitsandbytes source pin mismatch: {relative}")
    for name, schema in SCHEMAS.items():
        base, _, overload = name.partition(".")
        actual = torch._C._dispatch_find_schema_or_throw("bitsandbytes::" + base, overload).schema()
        expected = torch._C.parse_schema("bitsandbytes::" + name + schema)
        if str(actual) != str(expected):
            raise RuntimeError(f"bitsandbytes schema mismatch: {name}")
    if not isinstance(bnb.supported_torch_devices, set):
        raise RuntimeError("bitsandbytes device declaration is incompatible")


def check_kind(blocksize, quant_type):
    if blocksize != 64 or quant_type != "nf4":
        raise NotImplementedError("Only NF4 with blocksize 64 is supported")


def check_dtype(dtype):
    if dtype not in DTYPES:
        raise NotImplementedError("Only float32 and bfloat16 compute are supported")


def check_finite(tensor):
    # Keep the assertion on the input device. Do not use item(), cpu(), or numpy().
    torch._assert_async(torch.isfinite(tensor).all(), "NF4 input must be finite")


def check_packed(A, absmax, shape, dtype):
    check_dtype(dtype)
    if not shape or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in shape):
        raise ValueError("The logical shape must contain positive integers")
    count = 1
    for n in shape:
        count *= n
    if A.dtype != torch.uint8 or A.ndim != 2 or tuple(A.shape) != ((count + 1) // 2, 1) or not A.is_contiguous():
        raise ValueError("Packed weights must be contiguous uint8 in canonical column layout")
    if absmax.dtype != torch.float32 or tuple(absmax.shape) != ((count + 63) // 64,):
        raise ValueError("Scales must be one float32 value per block; nested scales are not supported")
    if A.device != absmax.device:
        raise ValueError("Packed weights and scales must be on the same device")
    check_finite(absmax)
    torch._assert_async((absmax >= 0).all(), "NF4 scales must be nonnegative")
    return count
