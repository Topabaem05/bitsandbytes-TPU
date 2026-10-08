"""Unqualified CPU calling-convention and serialized-evidence controls; no TPU."""
import ast
import copy
import os
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent
NATIVE = Path(os.environ.get('BNB_NATIVE_REPAIR_SOURCE', HERE.parents[1] / 'native')).resolve()
CLOUD = Path(os.environ.get('BNB_NATIVE_REPAIR_CLOUD', HERE.parent)).resolve()
ROOT = next(p for p in HERE.parents if (p / 'packages/bitsandbytes-tpu/pyproject.toml').is_file())
sys.path.insert(0, str(CLOUD))
import native_contract as NC
sys.path.pop(0)
assert NC.sha(NATIVE / 'manifest.json') == NC.MANIFEST_SHA
assert NC.read(NATIVE / 'manifest.json')['sources'] == NC.SOURCES
for name, record in NC.SOURCES.items():
    path = NATIVE / name
    assert not path.is_symlink() and NC.sha(path) == record['sha256'] and path.stat().st_size == record['bytes']
sys.path.insert(0, str(NATIVE))
import device_diagnostic as H
import recorder as E
import probe_backend as B
import probe_precision as P
sys.path.pop(0)
F_PATH = ROOT / 'packages/bitsandbytes-tpu/src/bitsandbytes_tpu/functionalization.py'
assert hashlib.sha256(F_PATH.read_bytes()).hexdigest() == H.S.spec()['plugin_python_files']['functionalization.py']
spec = importlib.util.spec_from_file_location('original_functionalization', F_PATH)
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('functional', [False, True])
@pytest.mark.parametrize('dispatch_included', [False, True])
def test_original_wrapper_boundary(dtype, functional, dispatch_included):
    raw = [torch.tensor([1., 2.], dtype=dtype) for _ in range(3)]
    inputs = [torch._to_functional_tensor(v) for v in raw] if functional else raw
    if functional:
        # The pending update must be synchronized before the backend boundary.
        inputs[0].add_(1)
    expected = [v.clone() for v in raw]
    if functional:
        torch._functionalize_sync(inputs[0])
        expected[0] = torch._from_functional_tensor(inputs[0]).clone()
    seen = []
    def backend(a, packed, shape, scales, blocksize, kind):
        assert shape == (2, 2) and blocksize == 64 and kind == 'nf4'
        assert all(torch._is_functional_tensor(v) for v in (a, packed, scales))
        def native(x, y, z):
            assert not any(torch._is_functional_tensor(v) for v in (x, y, z))
            seen.append((x, y, z))
            return x + y + z
        return H.D.native_boundary(native, torch, a, packed, scales)
    wrapped = F.functionalized_kernel(backend)
    if functional:
        with pytest.raises(RuntimeError, match='inputs must be unwrapped'):
            wrapped(inputs[0], inputs[1], (2, 2), inputs[2], 64, 'nf4')
    with torch._C._IncludeDispatchKeyGuard(torch._C.DispatchKey.Functionalize) if dispatch_included else torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.Functionalize, False):
        result = H.functional_backend_call(wrapped, torch, inputs[0], inputs[1], (2, 2), inputs[2], 64, 'nf4')
    assert len(seen) == 1
    assert torch._is_functional_tensor(result) is functional
    if functional:
        torch._functionalize_sync(result)
        result = torch._from_functional_tensor(result)
    torch.testing.assert_close(result, sum(expected), rtol=0, atol=0)
    for original, current, value in zip(inputs, raw, expected):
        if functional:
            torch._functionalize_sync(original)
            current = torch._from_functional_tensor(original)
        torch.testing.assert_close(current, value, rtol=0, atol=0)


def test_backend_error_is_retained():
    x = torch._to_functional_tensor(torch.ones(2))
    def fail(*args):
        raise ValueError('case remains ERROR')
    with pytest.raises(ValueError, match='case remains ERROR'):
        H.functional_backend_call(F.functionalized_kernel(fail), torch, x, x, (2, 2), x, 64, 'nf4')


def test_non_tensor_native_result_rejected():
    x = torch.ones(2)
    with pytest.raises(ValueError, match='SINGLE_TENSOR_OUTPUT'):
        H.functional_backend_call(lambda *args: None, torch, x, x, (2, 2), x, 64, 'nf4')


class Metrics:
    def __init__(self, value):
        self.value = value
        self.calls = []
    def counter_names(self):
        return list(self.value['counters'])
    def counter_value(self, key):
        return self.value['counters'][key]
    def metric_data(self, key):
        self.calls.append(key)
        value = self.value['execution_metrics'].get(key)
        if value is None:
            return None
        # The pinned API returns a tuple with (TIME, VALUE) sample tuples.
        return (value[0], value[1], [tuple(pair) for pair in value[2]])


def snapshot():
    return {'counters': {'ExecuteComputation': 1}, 'execution_metrics': {'ExecuteTime': [1, 2.0, [[1.0, 2.0]]]}}


def test_captured_real_tuple_snapshot_and_unchanged_verifier():
    value = snapshot()
    shallow = {'counters': value['counters'], 'execution_metrics': {'ExecuteTime': list(Metrics(value).metric_data('ExecuteTime'))}}
    with pytest.raises(ValueError, match='EXECUTION_METRIC_SAMPLES'):
        P.validate_execution(B, shallow)
    metrics = Metrics(value)
    evidence = E.execution_evidence(metrics, B, P)
    assert evidence == value
    assert len(metrics.calls) == len(set(metrics.calls))
    P.validate_execution(B, json.loads(json.dumps(evidence, allow_nan=False)))


@pytest.mark.parametrize('mutation, message', [
    ('fallback', 'CPU_FALLBACK'), ('empty', 'EXECUTION_METRIC_SAMPLES'),
    ('shape', 'EXECUTION_METRIC_SAMPLES'), ('negative', 'EXECUTION_METRIC_SAMPLES'),
    ('boolean', 'EXECUTION_METRIC_SAMPLES'), ('not-observed', 'EXECUTION_NOT_OBSERVED'),
    ('nonfinite', 'Out of range float'),
])
def test_invalid_metrics_remain_invalid(mutation, message):
    value = snapshot()
    data = value['execution_metrics']['ExecuteTime']
    if mutation == 'fallback': value['counters']['aten::clone'] = 1
    if mutation == 'empty': data[2] = []
    if mutation == 'shape': data[2][0].append(4)
    if mutation == 'negative': data[2][0][1] = -1
    if mutation == 'boolean': data[2][0][1] = True
    if mutation == 'not-observed': data[0] = 0
    if mutation == 'nonfinite': data[2][0][1] = float('nan')
    with pytest.raises(ValueError, match=message):
        E.execution_evidence(Metrics(value), B, P)


def test_native_capture_uses_same_canonical_evidence(monkeypatch):
    value = snapshot()
    metrics = Metrics(value)
    metrics.clear_all = lambda: None
    syncs = []
    class Context:
        def build(self, values): pass
        def hlo_text(self): return 'synthetic context'
        def hlo(self): return b'synthetic proto'
        def parameter_id_tensor_mapping(self): return {}
    xlac = SimpleNamespace(lowering=SimpleNamespace(LoweringContext=lambda name: Context()),
        _get_xla_tensors_hlo=lambda values: 'synthetic hlo',
        _get_xla_tensors_hlo_proto=lambda values: b'synthetic printer proto',
        _get_graph_hash=lambda values: b'\x01',
        _xla_sync_multi=lambda *args, **kwargs: syncs.append((args, kwargs)))
    monkeypatch.setattr(E.G, 'bind', lambda *args: {'synthetic': True})
    row = {'status': 'RETURNED', 'tensor_args': (torch.ones(2),) * 3, 'payload': 'synthetic payload'}
    result = E.capture_actual(torch.ones(2), row, SimpleNamespace(_XLAC=xlac),
        SimpleNamespace(wait_device_ops=lambda: None), metrics, B, P, 'synthetic graph')
    assert len(syncs) == 1 and syncs[0][1]['wait'] is True
    assert {key: result[key] for key in value} == value
    P.validate_execution(B, result)


def tail_statements(path):
    tree = ast.parse(path.read_text())
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare) and ast.unparse(node.test) == 'rows == 128'
        and any(isinstance(part, ast.Attribute) and part.attr == 'capture_actual' for part in ast.walk(node)))
    return compile(ast.fix_missing_locations(ast.Module(body=branch.orelse, type_ignores=[])), str(path), 'exec')


def test_actual_tail_branch_uses_serialized_evidence():
    metrics = Metrics(snapshot())
    metrics.clear_all = lambda: None
    namespace = {'metrics': metrics, 'B': B, 'P': P, 'E': E, 'raw': {}, 'actual': torch.ones(2),
        'torch_xla': SimpleNamespace(_XLAC=SimpleNamespace(_xla_sync_multi=lambda *args, **kwargs: None)),
        'xm': SimpleNamespace(wait_device_ops=lambda: None)}
    exec(tail_statements(NATIVE / 'device_diagnostic.py'), namespace)
    assert {key: namespace['raw'][key] for key in snapshot()} == snapshot()
    assert namespace['raw']['output'] == B.tensor_record(namespace['actual'])
    P.validate_execution(B, json.loads(json.dumps(namespace['raw'], allow_nan=False)))


def source_manifest():
    return {'experiment': NC.MODE, 'native_manifest_sha256': NC.MANIFEST_SHA,
        'native_source_variant': NC.VARIANT, 'native_scope': 'BOUNDED_NATIVE_BOUNDARY_ONLY',
        'precision': 'highest', 'm2_dependency_sha256': 'c' * 64,
        'files': {**{'native/' + name: copy.deepcopy(record) for name, record in NC.SOURCES.items()},
            'native/manifest.json': {'sha256': NC.MANIFEST_SHA, 'bytes': (NATIVE / 'manifest.json').stat().st_size},
            'm2-dependency.json': {'sha256': 'c' * 64, 'bytes': 1}}}


def test_current_generation_exact_source_map():
    # Synthetic source-map guard fixture only; not an executable or approved packet.
    NC.verify_manifest(source_manifest())


def test_wrong_manifest_hash_is_rejected():
    value = source_manifest()
    value['native_manifest_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='NATIVE_REVIEWED_SCOPE'):
        NC.verify_manifest(value)


def test_old_kernel_source_map_is_rejected():
    value = source_manifest()
    old = H.S.read(HERE / 'fixtures/native/original-cpu-source-seal.json')
    value['files']['native/kernel.py']['sha256'] = old['sources']['kernel.py']
    assert value['files']['native/kernel.py']['sha256'] != NC.SOURCES['kernel.py']['sha256']
    with pytest.raises(ValueError, match='NATIVE_EXACT_23_SOURCES'):
        NC.verify_manifest(value)


def test_wrong_source_map_is_rejected():
    value = source_manifest()
    value['files']['native/recorder.py']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='NATIVE_EXACT_23_SOURCES'):
        NC.verify_manifest(value)


def test_original_actual_cpu_source_seal_is_rejected(tmp_path):
    source = HERE / 'fixtures/native/original-cpu-source-seal.json'
    metadata = H.S.read(HERE / 'fixtures/native/original-cpu-source-seal-provenance.json')
    assert H.S.sha(source) == metadata['sha256'] and source.stat().st_size == metadata['bytes']
    target = tmp_path / 'old-oracle'
    target.mkdir()
    (target / 'oracle-seal.json').write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match='ORACLE_CANDIDATE_SOURCE'):
        H.S.oracle(target, H.S.sha(source), qualified=True)


def test_harness_does_not_import_jax():
    assert 'jax' not in sys.modules
