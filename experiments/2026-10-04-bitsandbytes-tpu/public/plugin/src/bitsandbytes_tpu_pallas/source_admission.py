"""Exact separate source generation. External payload pins bind this manifest."""
import hashlib
import json
from pathlib import Path
import types

SOURCE_VARIANT = "pallas-forward-mosaic7-gather-bf16-fp32-v1"


def validate_plugin(root=None, expected_manifest_sha256=None):
    root = Path(__file__).resolve().parent if root is None else Path(root)
    path = root / "source_manifest.json"
    if path.is_symlink() or root.is_symlink():
        raise RuntimeError("PALLAS_SOURCE_SYMLINK")
    data = path.read_bytes()
    if expected_manifest_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_manifest_sha256:
        raise RuntimeError("PALLAS_MANIFEST_SHA256")
    manifest = json.loads(data)
    if manifest.get("format") != "bnb-tpu.pallas-forward-source.v1" or manifest.get("source_variant") != SOURCE_VARIANT:
        raise RuntimeError("PALLAS_SOURCE_VARIANT")
    files = manifest["python_files"]
    names = {p.relative_to(root).as_posix() for p in root.rglob("*.py")}
    if names != set(files):
        raise RuntimeError("PALLAS_SOURCE_INVENTORY")
    for name, expected in files.items():
        current = root / name
        if current.is_symlink() or any(p.is_symlink() for p in current.parents):
            raise RuntimeError("PALLAS_SOURCE_SYMLINK:" + name)
        if hashlib.sha256(current.read_bytes()).hexdigest() != expected:
            raise RuntimeError("PALLAS_SOURCE_HASH:" + name)
    return manifest


def validate_registration(bnb):
    from . import compatibility
    validate_plugin()
    # Existing ABI inspection does not replace any original method or schema.
    compatibility.validate_upstream(bnb)
    validate_upstream_source(Path(bnb.__file__).resolve().parent)
    validate_public_methods(bnb)


def validate_upstream_source(root, torch_version=None):
    """Read-only file control; optional version is for isolated source fixtures."""
    from . import compatibility
    variant = compatibility.validate_source_tree(root, torch_version=torch_version)
    if variant != "params4bit-xla-v1":
        raise RuntimeError("PALLAS_REQUIRES_EXACT_M2_PATCH")
    return variant


def validate_public_methods(bnb):
    """Compare loaded functions with the admitted original source, without replacement."""
    root = Path(bnb.__file__).resolve().parent
    groups = {
        "nn/modules.py": [(bnb.nn.Params4bit, ("__new__", "_quantize", "to")),
                          (bnb.nn.Linear4bit, ("__init__", "forward"))],
        "autograd/_functions.py": [(bnb.autograd._functions.MatMul4Bit, ("forward", "backward"))],
    }
    for relative, classes in groups.items():
        path = root / relative
        compiled = compile(path.read_text(), str(path), "exec")
        for cls, methods in classes:
            candidates = [c for c in compiled.co_consts if isinstance(c, types.CodeType) and c.co_name == cls.__name__]
            if len(candidates) != 1:
                raise RuntimeError("PALLAS_PUBLIC_CLASS")
            for name in methods:
                expected = [c for c in candidates[0].co_consts if isinstance(c, types.CodeType) and c.co_name == name]
                actual = getattr(cls, name)
                if not expected or not isinstance(actual, types.FunctionType) or actual.__code__ != expected[-1] or Path(actual.__code__.co_filename).resolve() != path:
                    raise RuntimeError("PALLAS_PUBLIC_METHOD_REPLACEMENT:" + name)
        if relative == "autograd/_functions.py":
            expected = [c for c in compiled.co_consts if isinstance(c, types.CodeType) and c.co_name == "matmul_4bit"]
            actual = bnb.matmul_4bit
            if len(expected) != 1 or not isinstance(actual, types.FunctionType) or actual.__code__ != expected[0] or Path(actual.__code__.co_filename).resolve() != path:
                raise RuntimeError("PALLAS_PUBLIC_METHOD_REPLACEMENT:matmul_4bit")
