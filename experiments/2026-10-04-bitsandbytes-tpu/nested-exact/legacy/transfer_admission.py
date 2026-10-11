"""Admit one reviewed source overlay. No package import or runtime replacement."""
import importlib.util
from pathlib import Path
import sys
import types

PATCH_MANIFEST_SHA = 'e745fbf21aac10ed9118167a131d6505bf6dbe03e1fab0a669c663b26c5a5732'
PATCH_SHA = '4554fa99293441c1916dca93d47c36c54ccf25cd8728314311aecb734185f240'
POST_MODULE_SHA = '690987fda1219e91b8b72930402c424ea1875b66b853a9b5472e00de18314e4a'


def python_files(files):
    return {path: value for path, value in files.items() if path.endswith('.py')}


def load_manifest(B, path):
    B.require(B.sha(path) == PATCH_MANIFEST_SHA, 'PATCH_MANIFEST_IDENTITY')
    manifest = B.read(path); profile, _ = B.load_spec()
    B.require(manifest['format'] == 'bnb-tpu.upstream-patch.v1' and manifest['id'] == 'params4bit-xla-v1', 'PATCH_FORMAT')
    B.require(manifest['base_revision'] == profile['source_commit'] and manifest['runtime_lock_sha256'] == profile['runtime_lock_sha256'], 'PATCH_BASE_RUNTIME')
    B.require(manifest['patch_sha256'] == PATCH_SHA and manifest['post_patch_package_files']['nn/modules.py'] == POST_MODULE_SHA, 'PATCH_SOURCE')
    base = python_files(manifest['base_package_files']); post = python_files(manifest['post_patch_package_files'])
    B.require(len(base) == len(post) == 47 and set(base) == set(post), 'PATCH_INVENTORY')
    B.require([p for p in base if base[p] != post[p]] == ['nn/modules.py'], 'PATCH_CHANGE_SCOPE')
    B.require(all(base.get(p) == value for p, value in profile['bnb_files'].items()), 'PATCH_CANONICAL_BASE')
    return manifest


def validate_admission(B, admission, manifest):
    profile, _ = B.load_spec()
    B.require(admission.get('format') == 'bnb-tpu.probe-source-admission.v1' and admission.get('runtime_lock_sha256') == profile['runtime_lock_sha256'], 'SOURCE_ADMISSION_FORMAT_RUNTIME')
    upstream = admission.get('bitsandbytes', {})
    B.require(upstream.get('commit') == profile['source_commit'], 'SOURCE_BASE_COMMIT')
    B.require(upstream.get('patch_manifest_sha256') == PATCH_MANIFEST_SHA, 'SOURCE_OVERLAY_LINK')
    B.require(upstream.get('files') == python_files(manifest['post_patch_package_files']), 'SOURCE_EXACT_OVERLAY')
    plugin = admission.get('bitsandbytes_tpu', {}).get('files')
    B.require(isinstance(plugin, dict) and plugin and all(isinstance(p, str) and p.endswith('.py') and not Path(p).is_absolute() and '..' not in Path(p).parts and isinstance(h, str) and len(h) == 64 for p, h in plugin.items()), 'PLUGIN_FULL_MAP')
    return admission


def verify_installed(B, admission, roots):
    for name in ('bitsandbytes', 'bitsandbytes_tpu'):
        B.verify_source_tree(roots[name], admission[name]['files'])


def admit(B, path, expected_sha, manifest_path):
    manifest = load_manifest(B, manifest_path)
    B.require(B.sha(path) == expected_sha, 'SOURCE_ADMISSION_HASH')
    admission = validate_admission(B, B.read(path), manifest)
    roots = {}
    for name in ('bitsandbytes', 'bitsandbytes_tpu'):
        B.require(name not in sys.modules, 'SOURCE_ALREADY_IMPORTED')
        spec = importlib.util.find_spec(name)
        B.require(spec is not None and spec.origin is not None, 'SOURCE_PACKAGE_MISSING_' + name)
        roots[name] = Path(spec.origin).parent
    verify_installed(B, admission, roots)
    return admission, roots


def public_methods(B, bnb, roots):
    """Bind critical loaded methods to compiled, admitted source before and after execution."""
    identity = B.public_identity(bnb, roots)
    path = (roots['bitsandbytes'] / 'nn/modules.py').resolve()
    code = compile(path.read_text(), str(path), 'exec')
    for name, methods in {'Params4bit': ('__new__', '_quantize', 'to'), 'Linear4bit': ('__init__', 'forward')}.items():
        candidates = [c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == name]
        B.require(len(candidates) == 1, 'PUBLIC_SOURCE_CLASS')
        cls = getattr(bnb.nn, name)
        for method in methods:
            expected = [c for c in candidates[0].co_consts if isinstance(c, types.CodeType) and c.co_name == method]
            function = getattr(cls, method)
            B.require(expected and isinstance(function, types.FunctionType) and function.__code__ == expected[-1] and Path(function.__code__.co_filename).resolve() == path, 'PUBLIC_METHOD_REPLACEMENT')
    return dict(identity, original_method_identity=True)
