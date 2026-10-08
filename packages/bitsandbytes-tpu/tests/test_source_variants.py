"""Exact source variants and runtime rejection. These are local gate fixtures."""
import hashlib
import json
from pathlib import Path
import shutil

import bitsandbytes as bnb
import pytest

from bitsandbytes_tpu import compatibility as C

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / 'patches/params4bit-xla-v1.json'
PATCH = ROOT / 'patches/params4bit-xla-v1.patch'


def source_fixture(tmp_path, patched=False):
    root = tmp_path / 'bitsandbytes'
    source = Path(bnb.__file__).resolve().parent
    for relative in C.SOURCE_SHA256:
        target = root / relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    if patched:
        import subprocess
        subprocess.run(['git', 'apply', '--check', str(PATCH)], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(['git', 'apply', str(PATCH)], cwd=tmp_path, check=True, capture_output=True)
    return root


def test_reviewed_manifest_matches_plugin_inventory_and_patch():
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == C.PATCH_MANIFEST_SHA256
    record = json.loads(MANIFEST.read_text())
    assert record['base_revision'] == C.UPSTREAM_REVISION
    assert record['base_package_files'] == C.SOURCE_SHA256
    post = dict(C.SOURCE_SHA256, **{'nn/modules.py': C.PATCHED_MODULE_SHA256})
    assert record['post_patch_package_files'] == post
    assert record['plugin_runtime_requirement'] == {'torch': C.PATCH_TORCH_VERSION}
    assert hashlib.sha256(PATCH.read_bytes()).hexdigest() == record['patch_sha256']
    assert not any(key in record for key in ('status', 'qualified', 'local_control_runtime'))


def test_pristine_source_retains_existing_runtime_admission(tmp_path):
    assert C.validate_source_tree(source_fixture(tmp_path)) == 'pristine'


def test_exact_patch_inventory_accepts_only_pinned_runtime_fixture(tmp_path):
    root = source_fixture(tmp_path, patched=True)
    # Explicit input fixture validates the gate; it does not install or qualify this runtime.
    assert C.validate_source_tree(root, torch_version='2.9.0+cpu') == 'params4bit-xla-v1'
    for version in ('2.14.1', '2.9.0', '2.9.0+cu128', '2.10.0+cpu'):
        with pytest.raises(RuntimeError, match='patched source requires torch'):
            C.validate_source_tree(root, torch_version=version)


def test_patch_rejects_actual_local_runtime_when_unqualified(tmp_path):
    root = source_fixture(tmp_path, patched=True)
    if C.torch.__version__ == C.PATCH_TORCH_VERSION:
        assert C.validate_source_tree(root) == 'params4bit-xla-v1'
    else:
        with pytest.raises(RuntimeError, match='patched source requires torch'):
            C.validate_source_tree(root)


@pytest.mark.parametrize('mutation', ['extra_python', 'changed_other', 'changed_patch', 'missing_file', 'symlink'])
def test_unknown_source_variants_reject(tmp_path, mutation):
    root = source_fixture(tmp_path, patched=True)
    if mutation == 'extra_python': (root/'unreviewed.py').write_text('# fixture\n')
    elif mutation == 'changed_other': (root/'optim/adam.py').write_text('# fixture\n')
    elif mutation == 'changed_patch':
        with (root/'nn/modules.py').open('a') as stream: stream.write('\n# fixture\n')
    elif mutation == 'missing_file': (root/'backends/hpu/ops.py').unlink()
    else:
        original = root/'utils.py'; data = original.read_bytes(); original.unlink()
        target = tmp_path/'outside.py'; target.write_bytes(data); original.symlink_to(target)
    with pytest.raises(RuntimeError, match='source pin'):
        C.validate_source_tree(root, torch_version='2.9.0+cpu')
