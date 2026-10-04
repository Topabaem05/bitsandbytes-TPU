import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('runtime_resolve', ROOT / 'resolve.py')
r = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = r
spec.loader.exec_module(r)


def wheel(tmp_path, metadata='Name: demo\nVersion: 1.0\nRequires-Python: >=3.10\n'):
    p = tmp_path / 'demo-1.0-py3-none-any.whl'
    with zipfile.ZipFile(p, 'w') as z:
        z.writestr('demo-1.0.dist-info/METADATA', metadata)
        z.writestr('demo-1.0.dist-info/WHEEL', 'Wheel-Version: 1.0\nTag: py3-none-any\n')
        z.writestr('demo/__init__.py', '')
    return p, {'name': 'demo', 'version': '1.0', 'filename': p.name,
               'size': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def test_marker_environment():
    assert r.active('demo; sys_platform == "linux"')
    assert not r.active('demo; sys_platform == "darwin"')
    assert r.active('setuptools; python_version >= "3.12"')
    assert not r.active('old; python_version < "3.10"')
    assert not r.active('demo; extra == "tpu"')
    assert r.active('demo; extra == "tpu"', ['tpu'])


def test_wheel_positive(tmp_path):
    p, rec = wheel(tmp_path)
    assert r.verify_wheel(p, rec)['name'] == 'demo'


def test_wrong_hash(tmp_path):
    p, rec = wheel(tmp_path)
    rec['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='WHEEL_SHA'):
        r.verify_wheel(p, rec)


def test_wrong_metadata(tmp_path):
    p, rec = wheel(tmp_path, 'Name: other\nVersion: 1.0\n')
    with pytest.raises(ValueError, match='WHEEL_IDENTITY'):
        r.verify_wheel(p, rec)


def test_unsafe_archive(tmp_path):
    p, rec = wheel(tmp_path)
    with zipfile.ZipFile(p, 'a') as z:
        z.writestr('../escape', 'bad')
    rec.update(size=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='UNSAFE_MEMBER'):
        r.verify_wheel(p, rec)


def test_vendored_metadata_is_not_distribution_identity(tmp_path):
    p, rec = wheel(tmp_path)
    with zipfile.ZipFile(p, 'a') as z:
        z.writestr('demo/vendor/other-2.dist-info/METADATA', 'Name: other\nVersion: 2\n')
    rec.update(size=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    assert r.verify_wheel(p, rec)['name'] == 'demo'


def test_legacy_index_version():
    assert r.stable_versions(['0.5.13-hg', '1.0', '2.0rc1', '1.1']) == ['1.1', '1.0']


def test_wrong_wheel_tag(tmp_path):
    p, rec = wheel(tmp_path)
    rec['filename'] = 'demo-1.0-cp312-cp312-manylinux_2_28_x86_64.whl'
    with pytest.raises(ValueError, match='WHEEL_INTERNAL_TAGS'):
        r.verify_wheel(p, rec)


def test_linux_incompatible_tag(tmp_path):
    p, rec = wheel(tmp_path)
    rec['filename'] = 'demo-1.0-cp312-cp312-macosx_11_0_arm64.whl'
    with pytest.raises(ValueError, match='WHEEL_TAG_OR_IDENTITY'):
        r.verify_wheel(p, rec)


@pytest.mark.parametrize('name,version', [('torch-xla', '2.8.0'), ('libtpu', '0.0.20'), ('torch', '2.9.0')])
def test_wrong_profile(name, version):
    versions = dict(r.CORE)
    versions[name] = version
    with pytest.raises(ValueError, match='PROFILE'):
        r.validate_profile(versions)


def test_cuda_dependency():
    with pytest.raises(ValueError, match='FORBIDDEN_DEPENDENCY'):
        r.check_name('nvidia-cublas-cu12')


def test_cpu_fallback():
    pspec = importlib.util.spec_from_file_location('probe', ROOT / 'probe_runtime.py')
    p = importlib.util.module_from_spec(pspec)
    pspec.loader.exec_module(p)
    assert p.require_tpu('TPU', ['TPU:0']) is None
    with pytest.raises(ValueError, match='CPU_FALLBACK'):
        p.require_tpu('CPU', ['CPU:0'])


def test_closure_linux_marker():
    records = [{'name': n, 'version': v, 'requires_dist': []} for n, v in r.CORE.items()]
    records[0]['requires_dist'] = ['absent; sys_platform == "darwin"']
    r.validate_closure(records)
    records[0]['requires_dist'] = ['absent; sys_platform == "linux"']
    with pytest.raises(ValueError, match='DEPENDENCY_UNSATISFIED'):
        r.validate_closure(records)


def test_no_extra_jax_tpu():
    records = [{'name': n, 'version': v, 'requires_dist': []} for n, v in r.CORE.items()]
    next(rec for rec in records if rec['name'] == 'jax')['requires_dist'] = [
        'libtpu==0.0.20; extra == "tpu"']
    r.validate_closure(records)
    next(rec for rec in records if rec['name'] == 'jax')['extras'] = ['tpu']
    with pytest.raises(ValueError, match='DEPENDENCY_UNSATISFIED'):
        r.validate_closure(records)
