"""Transfer controls and saved records. No service calls or device science.

Source integration controls need BNB_TRANSFER_BASE_ARCHIVE set to the retained
immutable Git archive. They check its literal reviewed SHA before use.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import zipfile

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import build_packet as B
import owner
import remote
import transport

ROOT = B.PROJECT
SCIENCE = HERE.parent
PATCH = ROOT / 'patches/params4bit-xla-v1.json'


def write(path, value):
    path.write_text(json.dumps(value))


def source_args():
    paths = {'route_probe': SCIENCE / 'probe_routes.py', 'precision_probe': SCIENCE / 'probe_precision.py',
             'transfer_probe': SCIENCE / 'probe_transfer.py', 'transfer_admission': SCIENCE / 'transfer_admission.py',
             'patch_manifest': PATCH, 'source_controls': ROOT / 'tests/test_transfer_source.py'}
    return dict(experiment='transfer-api42', **paths, **{name + '_sha256': remote.sha(path) for name, path in paths.items()})


@pytest.fixture
def base_archive():
    value = os.environ.get('BNB_TRANSFER_BASE_ARCHIVE')
    if not value:
        pytest.skip('Set BNB_TRANSFER_BASE_ARCHIVE to the retained reviewed Git archive.')
    path = Path(value)
    assert not path.is_symlink() and remote.sha(path) == json.loads(PATCH.read_text())['base_archive_sha256']
    return path


def packet_fixture(base, archive, monkeypatch):
    original_run = B.subprocess.run
    original_read = B.subprocess.check_output
    def run(argv, **kwargs):
        if argv[:1] == ['git'] and 'archive' in argv:
            kwargs['stdout'].write(archive.read_bytes())
            return
        return original_run(argv, **kwargs)
    def read(argv, **kwargs):
        return B.COMMIT + '\n' if argv[:1] == ['git'] and 'rev-parse' in argv else original_read(argv, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(B.subprocess, 'run', run)
        patch.setattr(B.subprocess, 'check_output', read)
        packet = base / 'packet'
        result = B.build(base / 'synthetic-checkout', packet, remote.sha(ROOT / 'packages/bitsandbytes-tpu/source-manifest.json'), **source_args())
    return packet, json.loads((packet / 'manifest.json').read_text()), result['payload_sha256']


def zip_packet(packet, manifest):
    write(packet / 'manifest.json', manifest)
    with zipfile.ZipFile(packet / 'payload.zip', 'w') as archive:
        for name in [*manifest['files'], 'manifest.json']:
            item = zipfile.ZipInfo(name); item.create_system = 3; item.external_attr = 0o100644 << 16
            archive.writestr(item, (packet / name).read_bytes())
    return remote.sha(packet / 'payload.zip')


def controls_fixture():
    rows = []
    for method in ('direct_to_fixture', 'module_to_fixture', 'direct_quantize_meta', 'module_apply_meta'):
        rows.append(dict(method=method, scope='CPU_META_TYPE_MECHANICS_ONLY', identity=True,
                         **{'class': True}, attrs=True, module_alias=True,
                         bias_device='meta' if method == 'module_apply_meta' else 'cpu'))
    rows.extend([{'failure': 'ORIGINAL_INCOMPATIBLE_TYPE', 'original_unchanged': True} for _ in range(3)])
    for negative in ('python_weakref', 'cpp_weakref', 'held_impl', 'requires_grad', 'subclass'):
        rows.append(dict(negative=negative, rejected=True, tensorimpl_unchanged=True,
                         dict_identity_unchanged=True, data_unchanged=True, module_unchanged=True, error='SYNTHETIC rejection'))
    rows.append(dict(compatible_cpu=True, weight_identity=True, custom_attrs=True,
                     state_format=True, weights_only_load=True, keys=['bias', 'weight']))
    for case in ('existing_gradient', 'overwrite_flag', 'swap_flag'):
        flags = ('rejected_before_swap', 'parameter_identity', 'tensorimpl_identity', 'dict_identity',
                 'class_identity', 'values_unchanged', 'quant_state_identity', 'module_alias_unchanged', 'gradient_identity', 'bias_identity')
        rows.append(dict(case=case, scope='CPU_META_TYPE_MECHANICS_ONLY', **{flag: True for flag in flags}))
    rows.append(dict(case='genuine_cpu_forward_backward', forward_exact=True, dx_exact=True, db_exact=True,
                     frozen_base=True, state_plain_tensors=True, forward_values=[[0.0, 0.0]] * 3))
    rows.append(dict(case='fresh_process_cpu_public_restore', forward_exact=True, weights_only=True,
                     original_classes=True, module_state_alias=True, frozen_base=True))
    return {'record_validation': 'PASS', 'qualified_source_controls': True,
            'runtime': {'torch': '2.9.0+cpu', 'python': '3.12.14', 'platform': 'Linux'},
            'patch_manifest_sha256': B.TRANSFER_PATCH_MANIFEST_SHA, 'tpu': 'NOT_RUN', 'xla': 'NOT_RUN',
            'whole_module_transactionality': 'NOT_PROMISED', 'failures': [], 'controls': rows}


@pytest.mark.parametrize('fault', [None, 'truncated', 'duplicate', 'failure', 'false-field', 'unknown',
                                  'python', 'torch', 'qualified', 'tpu', 'original-count'])
def test_complete_source_control_validation(fault):
    record = controls_fixture()
    if fault == 'truncated': record['controls'].pop()
    elif fault == 'duplicate': record['controls'][1] = copy.deepcopy(record['controls'][0])
    elif fault == 'failure': record['failures'] = [{'control': 'SYNTHETIC failed control'}]
    elif fault == 'false-field': record['controls'][0]['identity'] = False
    elif fault == 'unknown': record['controls'][0]['method'] = 'unknown'
    elif fault == 'python': record['runtime']['python'] = '3.12.13'
    elif fault == 'torch': record['runtime']['torch'] = '2.14.1'
    elif fault == 'qualified': record['qualified_source_controls'] = False
    elif fault == 'tpu': record['tpu'] = 'PASS'
    elif fault == 'original-count': record['controls'][4] = copy.deepcopy(record['controls'][0])
    if fault is None:
        assert remote.validate_source_controls(record) is True
    else:
        with pytest.raises(ValueError, match='TRANSFER_SOURCE_CONTROLS'):
            remote.validate_source_controls(record)


@pytest.mark.parametrize('fault', [None, 'changed', 'missing', 'extra', 'bytecode', 'duplicate'])
def test_built_wheel_source_inventory(tmp_path, fault):
    wheel = tmp_path / 'synthetic.whl'
    sources = {'nn/modules.py': b'# reviewed post source\n', '__init__.py': b'# exact initializer\n'}
    expected = {name: remote.hashlib.sha256(value).hexdigest() for name, value in sources.items()}
    if fault == 'changed': sources['nn/modules.py'] = b'# pristine or different source\n'
    elif fault == 'missing': sources.pop('__init__.py')
    elif fault == 'extra': sources['extra.py'] = b'# extra\n'
    elif fault == 'bytecode': sources['nn/modules.pyc'] = b'bytecode'
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name, value in sources.items(): archive.writestr('bitsandbytes/' + name, value)
        if fault == 'duplicate':
            with pytest.warns(UserWarning): archive.writestr('bitsandbytes/nn/modules.py', sources['nn/modules.py'])
    if fault is None:
        assert remote.verify_transfer_wheel(wheel, expected) == expected
    else:
        with pytest.raises(ValueError, match='TRANSFER_WHEEL'):
            remote.verify_transfer_wheel(wheel, expected)


@pytest.mark.parametrize('fault,error', [('manifest', 'TRANSFER_PATCH_MANIFEST_SOURCE'), ('manifest-pin', 'TRANSFER_PATCH_MANIFEST_SOURCE'),
                                        ('precision', 'TRANSFER_PRECISION_SOURCE'), ('source-controls', 'TRANSFER_SOURCE_CONTROLS_SOURCE'),
                                        ('missing-helper', 'TRANSFER_ADMISSION_SOURCE'), ('other-experiment', 'TRANSFER_SOURCE_NOT_REQUESTED')])
def test_transfer_build_admission_before_git(tmp_path, monkeypatch, fault, error):
    args = source_args()
    if fault == 'manifest':
        path = tmp_path / 'unreviewed.json'; path.write_bytes(PATCH.read_bytes() + b' ')
        args.update(patch_manifest=path, patch_manifest_sha256=remote.sha(path))
    elif fault == 'manifest-pin': args['patch_manifest_sha256'] = '0' * 64
    elif fault == 'precision':
        path = tmp_path / 'precision.py'; path.write_text('# Wrong precision source\n')
        args.update(precision_probe=path, precision_probe_sha256=remote.sha(path))
    elif fault == 'source-controls': args['source_controls_sha256'] = '0' * 64
    elif fault == 'missing-helper': args['transfer_admission'] = None
    else: args['experiment'] = 'precision-diagnostic'
    calls = []
    monkeypatch.setattr(B.subprocess, 'check_output', lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match=error): B.build(tmp_path / 'upstream', tmp_path / 'packet', 'fixture', **args)
    assert not calls and not (tmp_path / 'packet').exists()


def test_transfer_build_and_consumer(tmp_path, monkeypatch, base_archive):
    frozen = {name: remote.sha(SCIENCE / name) for name in ('probe_backend.py', 'probe-profile.json', 'probe-inputs.json', 'probe_routes.py', 'probe_precision.py')}
    packet, manifest, expected = packet_fixture(tmp_path, base_archive, monkeypatch)
    assert owner.preflight(packet, expected) == manifest
    patch = json.loads(PATCH.read_text()); admission = json.loads((packet / 'source-admission.json').read_text())
    assert remote.sha(packet / 'upstream.tar') == remote.sha(base_archive) == patch['base_archive_sha256']
    assert manifest['patched_archive_sha256'] != manifest['base_archive_sha256']
    assert admission['bitsandbytes']['commit'] == B.COMMIT
    assert admission['bitsandbytes']['patch_manifest_sha256'] == B.TRANSFER_PATCH_MANIFEST_SHA
    assert admission['bitsandbytes']['files'] == {p: h for p, h in patch['post_patch_package_files'].items() if p.endswith('.py')}
    assert manifest['precision'] == 'highest' and 'diagnostic_only' not in manifest
    assert all(remote.sha(packet / name) == sha == remote.sha(SCIENCE / name) for name, sha in frozen.items())


@pytest.mark.parametrize('fault,error', [('archive', 'PATCH_BASE_ARCHIVE'), ('patch', 'PATCH_BODY_IDENTITY'),
                                        ('preimage', 'PATCH_BASE_PACKAGE_INVENTORY'), ('post', 'PATCH_POST_PACKAGE_INVENTORY')])
def test_patch_rejects_wrong_inputs(tmp_path, base_archive, fault, error):
    manifest = json.loads(PATCH.read_text()); archive = tmp_path / 'base.tar'; shutil.copyfile(base_archive, archive)
    patch = tmp_path / 'known.patch'; shutil.copyfile(ROOT / manifest['patch_path'], patch)
    if fault == 'archive': archive.write_bytes(archive.read_bytes() + b'changed')
    elif fault == 'patch': patch.write_bytes(patch.read_bytes() + b'changed')
    elif fault == 'preimage': manifest['base_package_files']['nn/modules.py'] = '0' * 64
    else:
        manifest['post_patch_package_files']['nn/modules.py'] = '0' * 64
        files = {p: h for p, h in manifest['post_patch_package_files'].items() if p.endswith('.py')}
        manifest['post_patch_python_inventory_sha256'] = remote.hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    with pytest.raises(ValueError, match=error): B.patched_archive(archive, tmp_path / 'post.tar', manifest, patch)


@pytest.mark.parametrize('field', list(remote.TRANSFER_BINDINGS) + ['precision'])
def test_transfer_preflight_binds_every_source(tmp_path, monkeypatch, base_archive, field):
    packet, manifest, _ = packet_fixture(tmp_path, base_archive, monkeypatch)
    manifest[field] = 'wrong'
    expected = zip_packet(packet, manifest)
    with pytest.raises(ValueError): owner.preflight(packet, expected)


def setup_phase(base, archive, monkeypatch):
    packet, manifest, expected = packet_fixture(base, archive, monkeypatch)
    packet.rename(base / 'payload'); payload = base / 'payload'
    shutil.copyfile(payload / 'payload.zip', base / 'payload.zip')
    out = base / 'records'; out.mkdir()
    write(base / 'launch.json', {'oracle_sha256': 'oracle'})
    top = {'status': 'CPU_ORACLE_READY_TPU_NOT_RUN', 'cpu_status': 'PASS', 'tpu_status': 'NOT_RUN',
           'experiment': 'transfer-api42', 'precision': 'highest', 'packet_sha256': expected,
           'manifest_sha256': remote.sha(payload / 'manifest.json'), 'source_admission_sha256': manifest['source_admission_sha256'],
           'runtime_lock_sha256': manifest['runtime_lock_sha256'], 'allocation_epoch': 1000,
           'oracle_sha256': 'oracle', 'science_deadline_epoch': 3000, 'steps': [], 'error': None,
           'built_source_status': 'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS', 'installed_source_status': 'POST_PATCH_PYTHON_SOURCE_PASS',
           'source_controls_status': 'PASS_QUALIFIED_LINUX_SOURCE_CONTROLS',
           **{field: manifest[field] for field in remote.TRANSFER_BINDINGS}}
    write(out / 'receipt.json', top)
    result = {'kind': 'TRANSFER_API42_PROBE', 'status': 'COMPLETE', 'source_pre': manifest['source_admission_sha256'],
              'source_post': manifest['source_admission_sha256'], 'source_admission_sha256': manifest['source_admission_sha256'],
              'oracle_sha256': 'oracle', 'runtime_lock_sha256': manifest['runtime_lock_sha256'], 'pid': 7001, 'pgid': 7001,
              'process_token': '1' * 32, 'precision': {'requested': 'highest', 'readback': 'highest', 'set_calls': 1, 'before_graph': True},
              'backend_probe_sha256': manifest['files']['probe_backend.py']['sha256'],
              **{field: manifest[field] for field in ('route_probe_sha256', 'precision_probe_sha256', 'transfer_probe_sha256', 'transfer_admission_sha256', 'patch_manifest_sha256')}}
    return manifest, expected, out, result


@pytest.mark.parametrize('fault', [None, 'numeric-fail', 'kind', 'partial', 'source', 'patch', 'probe', 'helper', 'backend',
                                  'precision', 'pid', 'pgid', 'group-pgid', 'token', 'cleanup', 'child-error', 'controls-required'])
def test_transfer_phase_receipt_and_deadline(tmp_path, monkeypatch, base_archive, fault):
    manifest, expected, out, result = setup_phase(tmp_path, base_archive, monkeypatch)
    changes = {'kind': ('kind', 'PRECISION_DIAGNOSTIC_ONLY'), 'partial': ('status', 'PARTIAL'),
               'source': ('source_post', 'wrong'), 'patch': ('patch_manifest_sha256', 'wrong'),
               'probe': ('transfer_probe_sha256', 'wrong'), 'helper': ('transfer_admission_sha256', 'wrong'),
               'backend': ('backend_probe_sha256', 'wrong'), 'precision': ('precision', {}), 'pid': ('pid', 7002),
               'pgid': ('pgid', 7002), 'token': ('process_token', '')}
    if fault in changes: result.update([changes[fault]])
    if fault == 'controls-required':
        top = json.loads((out / 'receipt.json').read_text()); top['source_controls_status'] = 'FAIL'; write(out / 'receipt.json', top)
    calls = []
    monkeypatch.setattr(remote.time, 'time', lambda: 1000)
    def child(output, label, argv, deadline, limit, **kwargs):
        calls.append(label)
        assert label == '12-transfer' and deadline == 2500 and limit == 1500 and kwargs['tpu'] is True
        assert argv[2] == str(tmp_path / 'payload/probe_transfer.py')
        for option, path in (('--precision-probe', 'probe_precision.py'), ('--route-probe', 'probe_routes.py'), ('--patch-manifest', 'patches/params4bit-xla-v1.json')):
            assert argv[argv.index(option) + 1] == str(tmp_path / 'payload' / path)
        (out / 'transfer').mkdir(); write(out / 'transfer/receipt.json', result)
        code = 2 if fault == 'numeric-fail' else 1 if fault == 'child-error' else 0
        return {'status': 'CHILD_FAILED' if code else 'PASS', 'exit_code': code,
                'cleanup': {'errors': ['SYNTHETIC'] if fault == 'cleanup' else [],
                            'groups': [{'pid': 7001, 'pgid': 7002 if fault == 'group-pgid' else 7001}]}}
    monkeypatch.setattr(remote, 'run_step', child)
    code = remote.execute(tmp_path, expected, 1000, 'transfer')
    top = json.loads((out / 'receipt.json').read_text())
    if fault in {None, 'numeric-fail'}:
        assert code == 0 and top['status'] == 'TRANSFER_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED'
        assert top['tpu_status'] == 'TRANSFER_RECORDS_COMPLETE'
    else: assert code == 1 and top['status'] == 'BLOCKED'
    assert calls == ([] if fault == 'controls-required' else ['12-transfer'])


@pytest.mark.parametrize('fault', [None, 'truncated', 'reported-failure', 'wrong-runtime'])
def test_transfer_cpu_controls_before_oracle(tmp_path, monkeypatch, base_archive, fault):
    manifest, expected, out, _ = setup_phase(tmp_path, base_archive, monkeypatch)
    top = json.loads((out / 'receipt.json').read_text()); top.update(cpu_status='NOT_RUN', installation_status='PASS')
    write(out / 'receipt.json', top)
    controls = controls_fixture()
    if fault == 'truncated': controls['controls'].pop()
    elif fault == 'reported-failure': controls['failures'] = ['SYNTHETIC failure']
    elif fault == 'wrong-runtime': controls['runtime']['torch'] = '2.14.1'
    calls = []
    monkeypatch.setattr(remote.time, 'time', lambda: 1000)
    def child(output, label, argv, deadline, limit, **kwargs):
        calls.append(label)
        if label == '10-transfer-source-controls':
            assert kwargs.get('tpu', False) is False
            assert argv[argv.index('--patched-source') + 1] == str(tmp_path / 'upstream-patched-controls')
            write(out / 'source-controls.json', controls)
        elif label == '10-runtime-probe': write(out / 'runtime-probe.json', {'status': 'PASS_TPU_RUNTIME_PROBE_ONLY'})
        elif label == '11-cpu-oracle':
            assert argv[2] == str(tmp_path / 'payload/probe_transfer.py')
            assert '--patch-manifest' in argv and '--precision-probe' in argv
            (out / 'cpu-oracle').mkdir(); write(out / 'cpu-oracle/oracle-seal.json', {'scope': 'SYNTHETIC_NO_ORACLE_EXECUTION'})
        return {'status': 'PASS', 'exit_code': 0, 'cleanup': {'errors': [], 'groups': []}}
    monkeypatch.setattr(remote, 'run_step', child)
    code = remote.execute(tmp_path, expected, 1000, 'cpu')
    if fault is None:
        assert code == 0 and calls == ['10-transfer-source-controls', '10-runtime-probe', '11-cpu-oracle']
    else:
        assert code == 1 and calls == ['10-transfer-source-controls']


@pytest.mark.parametrize('phase', ['tpu', 'routes', 'precision'])
def test_transfer_rejects_wrong_scientific_phase(tmp_path, monkeypatch, base_archive, phase):
    _, expected, out, _ = setup_phase(tmp_path, base_archive, monkeypatch)
    calls = []; monkeypatch.setattr(remote, 'run_step', lambda *a, **k: calls.append(a))
    assert remote.execute(tmp_path, expected, 1000, phase) == 1
    assert calls == [] and json.loads((out / 'receipt.json').read_text())['error']['message'] == 'WRONG_SCIENTIFIC_VARIANT'


@pytest.mark.parametrize('field', ['experiment', 'precision', 'patch_manifest_sha256', 'transfer_probe_sha256',
                                  'transfer_admission_sha256', 'source_controls_sha256', 'patched_archive_sha256',
                                  'allocation_epoch', 'source_admission_sha256'])
def test_transfer_top_receipt_rejects_before_child(tmp_path, monkeypatch, base_archive, field):
    _, expected, out, _ = setup_phase(tmp_path, base_archive, monkeypatch)
    top = json.loads((out / 'receipt.json').read_text()); top[field] = 'wrong'; write(out / 'receipt.json', top)
    calls = []; monkeypatch.setattr(remote, 'run_step', lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match='TRANSFER_PHASE'):
        remote.execute(tmp_path, expected, 1000, 'transfer')
    assert calls == []


@pytest.mark.parametrize('name', ['probe_transfer.py', 'transfer_admission.py', 'patches/params4bit-xla-v1.json', 'tests/test_transfer_source.py'])
def test_transfer_missing_dependencies_reject_before_cli(tmp_path, monkeypatch, base_archive, name):
    packet, manifest, _ = packet_fixture(tmp_path, base_archive, monkeypatch)
    manifest['files'].pop(name); (packet / name).unlink(); expected = zip_packet(packet, manifest)
    calls = []; monkeypatch.setattr(owner, 'OfficialCLI', lambda *a: calls.append(a))
    with pytest.raises(ValueError, match='TRANSFER_SOURCE_BINDING'): owner.preflight(packet, expected)
    assert calls == []


@pytest.mark.parametrize('fault', [None, 'matrix-numeric', 'transfer-numeric', 'scalar-agreement', 'source',
                                  'missing-row', 'truncated-controls', 'wheel-source', 'installed-source', 'phase', 'part-bytes', 'stop'])
def test_full_transfer_owner_chain(tmp_path, monkeypatch, base_archive, fault):
    """Fake service, actual source packet, byte recovery, and real retained-array verifier."""
    spec = importlib.util.spec_from_file_location('transfer_owner_fixtures', ROOT / 'tests/test_transfer_probe.py')
    fixtures = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures)
    packet, manifest, expected = packet_fixture(tmp_path, base_archive, monkeypatch)
    data = fixtures.make_fixture(tmp_path / 'fixture', admission=manifest['source_admission_sha256'])
    matrix_path = data.actual / 'matrix/api42-linear-float32-rank2-bias1.json'
    transfer_path = data.actual / 'transfer/module_to_xla-linear-float32-rank2-bias1.json'
    if fault in {'matrix-numeric', 'transfer-numeric'}:
        path = matrix_path if fault == 'matrix-numeric' else transfer_path
        row = data.B.read(path); row['outputs']['y']['values'][0] = 123; data.B.write(path, row)
        fixtures.reseal_fixture(data)
    elif fault == 'scalar-agreement':
        path = data.oracle / 'raw/linear-float32-rank2-bias1.json'
        row = data.B.read(path); row['outputs']['y']['values'][0] = 123; data.B.write(path, row)
        seal = data.B.read(data.oracle / 'oracle-seal.json'); seal['artifacts'] = data.B.inventory(data.oracle, 'oracle-seal.json')
        data.B.write(data.oracle / 'oracle-seal.json', seal); data.oracle_sha256 = data.B.sha(data.oracle / 'oracle-seal.json')
        for path in (matrix_path, transfer_path):
            row = data.B.read(path); row['outputs']['y']['values'][0] = 123; data.B.write(path, row)
        parent = data.B.read(data.actual / 'receipt.json'); parent['oracle_sha256'] = data.oracle_sha256
        data.B.write(data.actual / 'receipt.json', parent); fixtures.reseal_fixture(data)
    elif fault == 'source':
        parent = data.B.read(data.actual / 'receipt.json'); parent['source_post'] = '0' * 64
        data.B.write(data.actual / 'receipt.json', parent); fixtures.reseal_fixture(data)
    elif fault == 'missing-row':
        matrix_path.unlink(); fixtures.reseal_fixture(data)
    records = tmp_path / 'remote-records'; records.mkdir()
    shutil.copytree(data.oracle, records / 'cpu-oracle'); shutil.copytree(data.actual, records / 'transfer')
    original_parent = (records / 'transfer/receipt.json').read_bytes()
    admission_record = json.loads((packet / 'source-admission.json').read_text())
    controls = controls_fixture()
    if fault == 'truncated-controls': controls['controls'].pop()
    write(records / 'source-controls.json', controls)
    installed = {'status': 'POST_PATCH_PYTHON_SOURCE_PASS', 'patch_manifest_sha256': B.TRANSFER_PATCH_MANIFEST_SHA,
                 'source_admission_sha256': manifest['source_admission_sha256'],
                 'installed_python_files': {name: admission_record[name]['files'] for name in ('bitsandbytes', 'bitsandbytes_tpu')}}
    if fault == 'installed-source': installed['installed_python_files']['bitsandbytes']['nn/modules.py'] = 'wrong'
    write(records / 'installed-source.json', installed)
    # Separate maps ensure one negative record does not mutate another fixture.
    admission_record = json.loads((packet / 'source-admission.json').read_text())
    built = {'status': 'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS', 'patch_manifest_sha256': B.TRANSFER_PATCH_MANIFEST_SHA,
             'wheel_name': 'SYNTHETIC-upstream.whl', 'wheel_sha256': 'SYNTHETIC_NO_WHEEL_BODY',
             'python_files': admission_record['bitsandbytes']['files']}
    if fault == 'wheel-source': built['python_files']['nn/modules.py'] = 'wrong'
    write(records / 'built-source.json', built)
    receipt = {'status': 'TRANSFER_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED', 'experiment': 'transfer-api42', 'precision': 'highest',
               'packet_sha256': expected, 'manifest_sha256': remote.sha(packet / 'manifest.json'),
               'source_admission_sha256': manifest['source_admission_sha256'], 'runtime_lock_sha256': manifest['runtime_lock_sha256'],
               'runtime_status': 'PASS_TPU_RUNTIME_PROBE_ONLY', 'cpu_status': 'PASS', 'tpu_status': 'TRANSFER_RECORDS_COMPLETE',
               'built_source_status': 'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS', 'installed_source_status': 'POST_PATCH_PYTHON_SOURCE_PASS',
               'source_controls_status': 'PASS_QUALIFIED_LINUX_SOURCE_CONTROLS', 'tpu_attempted': True, 'oracle_sha256': data.oracle_sha256,
               'built_wheels': [{'name': built['wheel_name'], 'sha256': built['wheel_sha256']}],
               'steps': [{'scope': 'SYNTHETIC_NO_DEVICE_EXECUTION', 'cleanup': {'errors': [], 'groups': []}}],
               **{field: manifest[field] for field in remote.TRANSFER_BINDINGS}}
    write(records / 'receipt.json', receipt)
    out = tmp_path / 'host'; identity = tmp_path / 'identity.json'; identity.write_text('{}')
    gate = {'status': 'ACTUAL_DISPATCH_AUTHORIZED', 'packet_sha256': expected, 'driver_sha256': remote.sha(HERE / 'owner.py'),
            'output': str(out.resolve()), 'plugin_source_manifest_sha256': manifest['plugin_source_manifest_sha256'],
            'runtime_lock_sha256': manifest['runtime_lock_sha256'], 'budget': manifest['budget'], 'one_allocation_only': True,
            'provider_or_solver': 'FORBIDDEN', 'cli_identity_sha256': remote.sha(identity), 'experiment': 'transfer-api42',
            'precision': 'highest', **{field: manifest[field] for field in remote.TRANSFER_BINDINGS}}
    acceptance = tmp_path / 'gate.json'; write(acceptance, gate)
    monkeypatch.setattr(owner, 'verify_cli_identity', lambda *a: None)
    calls, sessions = [], []
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    timer = signal.getitimer(signal.ITIMER_REAL)
    def api(label, argv, timeout):
        calls.append(label)
        if label == '03-one-allocation':
            sessions.append(argv[argv.index('-s') + 1]); receipt['allocation_epoch'] = json.loads((out / 'owner.json').read_text())['allocation_epoch']
            write(records / 'receipt.json', receipt)
        if label in {'04-sessions-after', '90-before-stop'}: return {'status': 'PASS'}, '[' + sessions[0] + ']\nHardware: V6E1'
        if label == '13-tpu':
            assert '"remote.py",\'transfer\'' in Path(argv[-1]).read_text()
            assert json.loads((out / 'launch.json').read_text())['oracle_sha256'] == data.oracle_sha256
        if label == '21-export': write(records / 'receipt.json', {**receipt, **remote.package(records)})
        if label == '24-export-parts':
            transport.split(records / 'evidence.zip', tmp_path / 'remote-parts', expected_sha256=remote.sha(records / 'evidence.zip'), expected_bytes=(records / 'evidence.zip').stat().st_size)
        if argv[0] == 'download':
            destination = Path(argv[-1])
            if label in {'09-install-receipt', '10-cpu-receipt'}:
                phase = {**receipt, 'status': 'INSTALLED_NOT_QUALIFIED' if label.startswith('09-') else 'CPU_ORACLE_READY_TPU_NOT_RUN'}
                if fault == 'phase' and label == '09-install-receipt': phase['patch_manifest_sha256'] = 'wrong'
                write(destination, phase)
            else:
                rel = argv[-2].removeprefix('content/bnb-tpu-first/')
                source = records / rel.removeprefix('records/') if rel.startswith('records/') else tmp_path / 'remote-parts' / rel.removeprefix('result-parts/')
                shutil.copyfile(source, destination)
                if fault == 'part-bytes' and label.startswith('26-part-'): destination.write_bytes(destination.read_bytes() + b'bad')
        if fault == 'stop' and label == '91-stop-exact': return {'status': 'CLI_FAILED'}, 'SYNTHETIC stop refusal'
        return {'status': 'PASS'}, 'Active assignments: 0\nUsage rate: 0.00/hr' if argv == ['usage'] else 'No active sessions found on server.'
    result = owner.drive(packet, out, expected, acceptance, remote.sha(acceptance), Path('fixture-python'), identity, api=api, simulated=True)
    assert result['mode'] == 'SIMULATED_NO_CLOUD' and result['allocation_attempts'] == 1
    assert calls.count('13-tpu') == (0 if fault == 'phase' else 1)
    assert calls[-4:] == ['90-before-stop', '91-stop-exact', '92-after-stop', '93-usage-after']
    assert all(signal.getsignal(sig) == handler for sig, handler in handlers.items()) and signal.getitimer(signal.ITIMER_REAL) == timer
    assert (records / 'transfer/receipt.json').read_bytes() == original_parent
    if fault == 'part-bytes': assert result.get('retrieval_error') and 'local_verifier_exit_code' not in result
    else:
        assert result['retrieval'] == 'COMPLETE_WHOLE_ARCHIVE' and result['whole_archive_verified'] is True
        assert (out / 'recovered/transfer/receipt.json').read_bytes() == original_parent
        assert not result['local_verifier_cleanup']['errors']
        assert all(group['group_absence'] == 'OBSERVED_NO_SUCH_GROUP' for group in result['local_verifier_cleanup']['groups'])
    if fault == 'phase': assert 'local_verifier_exit_code' not in result
    elif fault != 'part-bytes':
        report = json.loads((out / 'verify.json').read_text())
        assert report['m3_status'] == 'NOT_QUALIFIED'
        if fault in {'source', 'missing-row', 'scalar-agreement'}:
            assert report['record_validation'] == 'FAIL' and result['local_verifier_exit_code'] == 1
        else:
            numeric = 'FAIL' if fault in {'matrix-numeric', 'transfer-numeric'} else 'PASS'
            assert report['record_validation'] == 'PASS' and report['numerical_status'] == report['api42_status'] == numeric
            assert len(report['matrix']) == 42 and len(report['transfers']) == 4
            assert result['local_verifier_exit_code'] == (0 if numeric == 'PASS' else 2)
    if fault is None: assert result['status'] == 'PASS_TPU_TRANSFER_API42' and result['api42_status'] == 'PASS'
    elif fault in {'matrix-numeric', 'transfer-numeric'}:
        assert result['status'] == 'FAIL_TPU_TRANSFER_API42' and result['api42_status'] == result['numerical_status'] == 'FAIL'
        assert result['record_validation'] == 'PASS'
    else: assert result['status'] not in {'PASS_TPU_TRANSFER_API42', 'FAIL_TPU_TRANSFER_API42'} and result['api42_status'] == 'NOT_QUALIFIED'
    assert result['m3_status'] == 'NOT_QUALIFIED'
    if fault == 'stop': assert result['status'] == 'BLOCKED_CLEANUP'
