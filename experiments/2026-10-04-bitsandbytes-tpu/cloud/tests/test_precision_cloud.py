"""Precision controls and saved records. No science or service calls."""
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import zipfile

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import build_packet
import owner
import remote

ROUTE = HERE.parent / 'probe_routes.py'
ROUTE_SHA = 'feae751734e57c741b1bdade004ff7ca3c041ee7eb3b7086b531bc6be433038e'


def write(path, data):
    path.write_text(json.dumps(data))


def zip_packet(packet, manifest):
    write(packet / 'manifest.json', manifest)
    with zipfile.ZipFile(packet / 'payload.zip', 'w') as archive:
        for name in [*manifest['files'], 'manifest.json']:
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (packet / name).read_bytes())
    return remote.sha(packet / 'payload.zip')


def packet_fixture(packet):
    packet.mkdir()
    assert remote.sha(ROUTE) == ROUTE_SHA == remote.PRECISION_ROUTE_PROBE_SHA == build_packet.PRECISION_ROUTE_PROBE_SHA
    shutil.copyfile(ROUTE, packet / 'probe_routes.py')
    (packet / 'probe_precision.py').write_text('# Isolated packet admission fixture; no science.\n')
    for path in HERE.rglob('*.py'):
        if 'tests' in path.relative_to(HERE).parts or path.name == 'build_packet.py':
            continue
        dest = packet / 'cloud' / path.relative_to(HERE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    files = {path.relative_to(packet).as_posix(): {'sha256': remote.sha(path), 'bytes': path.stat().st_size}
             for path in packet.rglob('*') if path.is_file()}
    manifest = {'files': files, 'experiment': 'precision-diagnostic', 'diagnostic_only': True,
                'route_probe_sha256': ROUTE_SHA, 'precision_probe_sha256': remote.sha(packet / 'probe_precision.py'),
                'source_admission_sha256': 'source', 'runtime_lock_sha256': 'runtime',
                'plugin_source_manifest_sha256': 'isolated-fixture',
                'budget': {'total': 3600, 'install': 1200, 'science': 1800, 'retrieval': 600, 'cleanup': 60}}
    zip_packet(packet, manifest)
    return manifest


def test_precision_packet_build_and_consumer(tmp_path, monkeypatch):
    """Real packet copy and validation with a synthetic git archive only."""
    precision = tmp_path / 'precision-fixture.py'
    precision.write_text('# Isolated source fixture.\n')
    calls = []
    monkeypatch.setattr(build_packet.subprocess, 'check_output', lambda *a, **k: build_packet.COMMIT + '\n')
    def archive(argv, stdout, check):
        calls.append(argv)
        assert argv[-1] == build_packet.COMMIT and check is True
        raw = b'# Synthetic archived upstream source, not an executed package.\n'
        with tarfile.open(fileobj=stdout, mode='w') as tar:
            member = tarfile.TarInfo('bitsandbytes/__init__.py')
            member.size = len(raw)
            tar.addfile(member, io.BytesIO(raw))
    monkeypatch.setattr(build_packet.subprocess, 'run', archive)
    packet = tmp_path / 'packet'
    plugin = build_packet.PROJECT / 'packages/bitsandbytes-tpu/source-manifest.json'
    result = build_packet.build(tmp_path / 'upstream', packet, remote.sha(plugin),
                                experiment='precision-diagnostic', route_probe=ROUTE, route_probe_sha256=ROUTE_SHA,
                                precision_probe=precision, precision_probe_sha256=remote.sha(precision))
    manifest = owner.preflight(packet, result['payload_sha256'])
    assert manifest['experiment'] == 'precision-diagnostic' and manifest['diagnostic_only'] is True
    assert manifest['route_probe_sha256'] == ROUTE_SHA
    assert manifest['precision_probe_sha256'] == remote.sha(precision)
    assert manifest['files']['probe_routes.py']['sha256'] == ROUTE_SHA
    assert manifest['files']['probe_precision.py']['sha256'] == remote.sha(precision)
    assert manifest['budget'] == {'total': 3600, 'install': 1200, 'science': 1800, 'retrieval': 600, 'cleanup': 60}
    for name in ('probe_backend.py', 'probe-inputs.json', 'probe-profile.json'):
        assert (packet / name).read_bytes() == (HERE.parent / name).read_bytes()
    assert len(calls) == 1


@pytest.mark.parametrize('fault,error', [('missing-route', 'ROUTE_PROBE_SOURCE'),
                                        ('wrong-route', 'PRECISION_ROUTE_PROBE_SOURCE'),
                                        ('missing-precision', 'PRECISION_PROBE_SOURCE'),
                                        ('wrong-precision-hash', 'PRECISION_PROBE_SOURCE'),
                                        ('wrong-experiment', 'EXPERIMENT_VARIANT')])
def test_precision_build_rejects_before_git(tmp_path, monkeypatch, fault, error):
    route = tmp_path / 'route.py'
    shutil.copyfile(ROUTE, route)
    precision = tmp_path / 'precision.py'
    precision.write_text('# Isolated fixture.\n')
    kwargs = dict(experiment='precision-diagnostic', route_probe=route, route_probe_sha256=ROUTE_SHA,
                  precision_probe=precision, precision_probe_sha256=remote.sha(precision))
    if fault == 'missing-route':
        kwargs['route_probe'] = None
    elif fault == 'wrong-route':
        route.write_text('# Wrong dependency fixture.\n')
        kwargs['route_probe_sha256'] = remote.sha(route)
    elif fault == 'missing-precision':
        kwargs['precision_probe'] = None
    elif fault == 'wrong-precision-hash':
        kwargs['precision_probe_sha256'] = '0' * 64
    else:
        kwargs['experiment'] = 'unknown-diagnostic'
    calls = []
    monkeypatch.setattr(build_packet.subprocess, 'check_output', lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match=error):
        build_packet.build(tmp_path / 'upstream', tmp_path / 'packet', 'fixture', **kwargs)
    assert calls == [] and not (tmp_path / 'packet').exists()


@pytest.mark.parametrize('experiment', ['api42', 'device-route-diagnostic'])
def test_other_experiments_cannot_include_precision_source(tmp_path, experiment):
    with pytest.raises(ValueError, match='PRECISION_PROBE_NOT_REQUESTED'):
        build_packet.build(tmp_path / 'upstream', tmp_path / 'packet', 'fixture', experiment=experiment,
                           route_probe=ROUTE if experiment == 'device-route-diagnostic' else None,
                           route_probe_sha256=ROUTE_SHA if experiment == 'device-route-diagnostic' else None,
                           precision_probe=tmp_path / 'precision.py')


@pytest.mark.parametrize('fault,error', [(None, None), ('missing-route', 'ROUTE_PROBE_BINDING'),
                                        ('wrong-route', 'PRECISION_ROUTE_PROBE_BINDING'),
                                        ('missing-precision', 'PRECISION_PROBE_BINDING'),
                                        ('wrong-precision-pin', 'PRECISION_PROBE_BINDING'),
                                        ('scope', 'ROUTE_PROBE_BINDING'),
                                        ('other-experiment', 'PRECISION_PROBE_NOT_REQUESTED'),
                                        ('unknown-experiment', 'EXPERIMENT_VARIANT')])
def test_precision_preflight_binds_dependencies(tmp_path, monkeypatch, fault, error):
    packet = tmp_path / 'packet'
    manifest = packet_fixture(packet)
    if fault in {'missing-route', 'missing-precision'}:
        name = 'probe_routes.py' if fault == 'missing-route' else 'probe_precision.py'
        manifest['files'].pop(name)
        (packet / name).unlink()
    elif fault == 'wrong-route':
        route = packet / 'probe_routes.py'
        route.write_text('# Wrong admitted dependency fixture.\n')
        manifest['route_probe_sha256'] = remote.sha(route)
        manifest['files']['probe_routes.py'] = {'sha256': remote.sha(route), 'bytes': route.stat().st_size}
    elif fault == 'wrong-precision-pin':
        manifest['precision_probe_sha256'] = '0' * 64
    elif fault == 'scope':
        manifest['diagnostic_only'] = False
    elif fault == 'other-experiment':
        manifest['experiment'] = 'device-route-diagnostic'
    elif fault == 'unknown-experiment':
        manifest['experiment'] = 'unknown'
    expected = zip_packet(packet, manifest)
    calls = []
    monkeypatch.setattr(owner, 'OfficialCLI', lambda *a: calls.append(a))
    if fault is None:
        assert owner.preflight(packet, expected) == manifest
    else:
        with pytest.raises(ValueError, match=error):
            owner.preflight(packet, expected)
    assert calls == []


def setup_phase(base):
    payload, out = base / 'payload', base / 'records'
    manifest = packet_fixture(payload)
    shutil.copyfile(payload / 'payload.zip', base / 'payload.zip')
    out.mkdir()
    write(base / 'launch.json', {'oracle_sha256': 'oracle'})
    top = {'status': 'CPU_ORACLE_READY_TPU_NOT_RUN', 'cpu_status': 'PASS', 'tpu_status': 'NOT_RUN',
           'experiment': 'precision-diagnostic', 'diagnostic_only': True,
           'route_probe_sha256': ROUTE_SHA, 'precision_probe_sha256': manifest['precision_probe_sha256'],
           'packet_sha256': remote.sha(base / 'payload.zip'), 'manifest_sha256': remote.sha(payload / 'manifest.json'),
           'source_admission_sha256': manifest['source_admission_sha256'],
           'runtime_lock_sha256': manifest['runtime_lock_sha256'], 'allocation_epoch': 1000,
           'oracle_sha256': 'oracle', 'science_deadline_epoch': 3000, 'steps': [], 'error': None}
    write(out / 'receipt.json', top)
    scientific = {'kind': 'PRECISION_DIAGNOSTIC_ONLY', 'status': 'COMPLETE', 'source_pre': 'source',
                  'source_post': 'source', 'source_admission_sha256': 'source', 'oracle_sha256': 'oracle',
                  'runtime_lock_sha256': 'runtime', 'pid': 7001, 'process_token': 'precision-parent-fixture',
                  'route_probe_sha256': ROUTE_SHA, 'precision_probe_sha256': manifest['precision_probe_sha256'],
                  'numerical_status': 'FAIL'}
    return manifest, scientific, out


@pytest.mark.parametrize('fault', [None, 'kind', 'partial', 'source-pre', 'source-post', 'admission', 'oracle',
                                  'runtime', 'route-pin', 'precision-pin', 'pid', 'process-token',
                                  'cleanup', 'child-exit', 'child-raise'])
def test_precision_remote_receipt_and_parent_identity(tmp_path, monkeypatch, fault):
    manifest, result, out = setup_phase(tmp_path)
    monkeypatch.setattr(remote.time, 'time', lambda: 1000)
    changes = {'kind': ('kind', 'DEVICE_ROUTE_DIAGNOSTIC_ONLY'), 'partial': ('status', 'PARTIAL'),
               'source-pre': ('source_pre', 'wrong'), 'source-post': ('source_post', 'wrong'),
               'admission': ('source_admission_sha256', 'wrong'), 'oracle': ('oracle_sha256', 'wrong'),
               'runtime': ('runtime_lock_sha256', 'wrong'), 'route-pin': ('route_probe_sha256', '0' * 64),
               'precision-pin': ('precision_probe_sha256', '0' * 64), 'pid': ('pid', 7002),
               'process-token': ('process_token', '')}
    if fault in changes:
        key, value = changes[fault]
        result[key] = value
    calls = []
    def child(output, label, argv, deadline, limit, **kwargs):
        calls.append(label)
        running = json.loads((out / 'receipt.json').read_text())
        assert running['status'] == 'PRECISION_CHILD_RUNNING' and running['tpu_attempted'] is True
        assert running['tpu_status'] == 'RUNNING'
        assert label == '12-precision' and limit == 1500 and kwargs['tpu'] is True
        assert argv[2] == str(tmp_path / 'payload/probe_precision.py')
        assert argv[argv.index('--route-probe') + 1] == str(tmp_path / 'payload/probe_routes.py')
        assert argv[argv.index('--backend-probe') + 1] == str(tmp_path / 'payload/probe_backend.py')
        assert argv[argv.index('--oracle-sha256') + 1] == 'oracle'
        assert float(argv[argv.index('--deadline-epoch') + 1]) == deadline
        assert deadline == 2500  # The 1500-second owned child limit is also its supplied deadline.
        if fault == 'child-raise':
            raise RuntimeError('Isolated child failure')
        (out / 'precision').mkdir()
        write(out / 'precision/receipt.json', result)
        return {'status': 'CHILD_FAILED' if fault == 'child-exit' else 'PASS',
                'exit_code': 1 if fault == 'child-exit' else 0,
                'cleanup': {'errors': [{'operation': 'isolated-refusal'}] if fault == 'cleanup' else [],
                            'groups': [{'pid': 7001, 'group_absence': 'OBSERVED_NO_SUCH_GROUP'}]}}
    monkeypatch.setattr(remote, 'run_step', child)
    code = remote.execute(tmp_path, remote.sha(tmp_path / 'payload.zip'), 1000, 'precision')
    receipt = json.loads((out / 'receipt.json').read_text())
    assert calls == ['12-precision'] and receipt['tpu_attempted'] is True
    if fault is None:
        assert code == 0 and receipt['status'] == 'PRECISION_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED'
        assert receipt['tpu_status'] == 'DIAGNOSTIC_RECORDS_COMPLETE'
        assert result['numerical_status'] == 'FAIL'
    else:
        assert code == 1 and receipt['status'] == 'BLOCKED' and receipt['tpu_status'] == 'ATTEMPTED_BLOCKED'


@pytest.mark.parametrize('phase', ['routes', 'tpu'])
def test_precision_rejects_other_science_phases(tmp_path, monkeypatch, phase):
    setup_phase(tmp_path)
    calls = []
    monkeypatch.setattr(remote, 'run_step', lambda *a, **k: calls.append(a))
    assert remote.execute(tmp_path, remote.sha(tmp_path / 'payload.zip'), 1000, phase) == 1
    assert calls == []
    assert json.loads((tmp_path / 'records/receipt.json').read_text())['error']['message'] == 'WRONG_SCIENTIFIC_VARIANT'


@pytest.mark.parametrize('experiment', ['api42', 'device-route-diagnostic'])
def test_precision_phase_rejects_other_experiments(tmp_path, monkeypatch, experiment):
    manifest, _, _ = setup_phase(tmp_path)
    manifest['experiment'] = experiment
    manifest.pop('precision_probe_sha256')
    manifest['files'].pop('probe_precision.py')
    write(tmp_path / 'payload/manifest.json', manifest)
    calls = []
    monkeypatch.setattr(remote, 'run_step', lambda *a, **k: calls.append(a))
    assert remote.execute(tmp_path, remote.sha(tmp_path / 'payload.zip'), 1000, 'precision') == 1
    assert calls == []
    assert json.loads((tmp_path / 'records/receipt.json').read_text())['error']['message'] == 'WRONG_SCIENTIFIC_VARIANT'


@pytest.mark.parametrize('field', ['experiment', 'diagnostic_only', 'route_probe_sha256', 'precision_probe_sha256',
                                  'packet_sha256', 'manifest_sha256', 'source_admission_sha256', 'runtime_lock_sha256',
                                  'allocation_epoch'])
def test_wrong_precision_phase_receipt_rejects_before_child(tmp_path, monkeypatch, field):
    setup_phase(tmp_path)
    receipt = json.loads((tmp_path / 'records/receipt.json').read_text())
    receipt[field] = False if field == 'diagnostic_only' else 'wrong'
    write(tmp_path / 'records/receipt.json', receipt)
    calls = []
    monkeypatch.setattr(remote, 'run_step', lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match='PRECISION_PHASE_RECEIPT_BINDING'):
        remote.execute(tmp_path, remote.sha(tmp_path / 'payload.zip'), 1000, 'precision')
    assert calls == []


@pytest.mark.parametrize('field', ['experiment', 'route_probe_sha256', 'precision_probe_sha256'])
def test_wrong_precision_acceptance_rejects_before_cli(tmp_path, monkeypatch, field):
    packet, out = tmp_path / 'packet', tmp_path / 'out'
    manifest = packet_fixture(packet)
    identity = tmp_path / 'identity.json'
    identity.write_text('{}')
    expected = remote.sha(packet / 'payload.zip')
    gate = {'status': 'ACTUAL_DISPATCH_AUTHORIZED', 'packet_sha256': expected,
            'driver_sha256': remote.sha(HERE / 'owner.py'), 'output': str(out.resolve()),
            'plugin_source_manifest_sha256': manifest['plugin_source_manifest_sha256'],
            'runtime_lock_sha256': manifest['runtime_lock_sha256'], 'budget': manifest['budget'],
            'one_allocation_only': True, 'provider_or_solver': 'FORBIDDEN', 'cli_identity_sha256': remote.sha(identity),
            'experiment': 'precision-diagnostic', 'diagnostic_only': True,
            'route_probe_sha256': ROUTE_SHA, 'precision_probe_sha256': manifest['precision_probe_sha256']}
    gate[field] = 'wrong'
    acceptance = tmp_path / 'acceptance.json'
    write(acceptance, gate)
    calls = []
    monkeypatch.setattr(owner, 'OfficialCLI', lambda *a: calls.append(a))
    with pytest.raises(PermissionError, match='EXACT_ROOT_ACCEPTANCE_REQUIRED'):
        owner.drive(packet, out, expected, acceptance, remote.sha(acceptance), Path('fixture-python'), identity)
    assert calls == [] and not out.exists()


@pytest.mark.parametrize('fault', [None, 'numeric-default', 'numeric-high', 'numeric-highest',
                                  'source', 'phase-binding', 'part-bytes', 'stop'])
def test_full_precision_owner_chain(tmp_path, monkeypatch, fault):
    """Fake service with real recovery, process ownership, and saved-record verifier."""
    import importlib.util
    import signal
    import transport

    fixture_source = build_packet.PROJECT / 'tests/test_precision.py'
    spec = importlib.util.spec_from_file_location('precision_owner_fixtures', fixture_source)
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    packet = tmp_path / 'packet'
    manifest = packet_fixture(packet)
    for name in ('probe_precision.py', 'probe_backend.py', 'probe-profile.json', 'probe-inputs.json'):
        shutil.copyfile(HERE.parent / name, packet / name)
    (packet / 'source-admission.json').write_text('{"scope":"SYNTHETIC_NO_SOURCE_ADMISSION"}\n')
    admission_sha = remote.sha(packet / 'source-admission.json')
    data = fixtures.make_fixture(tmp_path / 'fixture', admission=admission_sha)
    manifest.update(source_admission_sha256=admission_sha,
                    runtime_lock_sha256=data.B.read(packet / 'probe-profile.json')['runtime_lock_sha256'],
                    precision_probe_sha256=remote.sha(packet / 'probe_precision.py'))
    manifest['files'] = {p.relative_to(packet).as_posix(): {'sha256': remote.sha(p), 'bytes': p.stat().st_size}
                         for p in packet.rglob('*') if p.is_file() and p.name not in {'payload.zip', 'manifest.json'}}
    expected = zip_packet(packet, manifest)
    if fault and fault.startswith('numeric-'):
        mode = fault.removeprefix('numeric-')
        path = data.actual / f'modes/{mode}/raw/plain-linear-float32-rank2-bias1.json'
        row = data.B.read(path)
        row['outputs']['y']['values'][0] = 123
        data.B.write(path, row)
        fixtures.reseal_fixture(data)
    elif fault == 'source':
        parent = data.B.read(data.actual / 'receipt.json')
        parent['source_post'] = '0' * 64
        data.B.write(data.actual / 'receipt.json', parent)
        fixtures.reseal_fixture(data)
    records = tmp_path / 'remote-records'
    records.mkdir()
    shutil.copytree(data.oracle, records / 'cpu-oracle')
    shutil.copytree(data.actual, records / 'precision')
    original_parent_bytes = (records / 'precision/receipt.json').read_bytes()
    receipt = {'status': 'PRECISION_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED',
               'packet_sha256': expected, 'manifest_sha256': remote.sha(packet / 'manifest.json'),
               'runtime_status': 'PASS_TPU_RUNTIME_PROBE_ONLY', 'cpu_status': 'PASS',
               'tpu_status': 'DIAGNOSTIC_RECORDS_COMPLETE', 'tpu_attempted': True,
               'oracle_sha256': data.oracle_sha256,
               'experiment': 'precision-diagnostic', 'diagnostic_only': True,
               'route_probe_sha256': ROUTE_SHA, 'precision_probe_sha256': manifest['precision_probe_sha256'],
               'source_admission_sha256': admission_sha, 'runtime_lock_sha256': manifest['runtime_lock_sha256'],
               'steps': [{'scope': 'SYNTHETIC_NO_CHILD_EXECUTION', 'cleanup': {'errors': [], 'groups': []}}]}
    write(records / 'receipt.json', receipt)
    out, identity = tmp_path / 'host', tmp_path / 'identity.json'
    identity.write_text('{}')
    gate = {'status': 'ACTUAL_DISPATCH_AUTHORIZED', 'packet_sha256': expected,
            'driver_sha256': remote.sha(HERE / 'owner.py'), 'output': str(out.resolve()),
            'plugin_source_manifest_sha256': manifest['plugin_source_manifest_sha256'],
            'runtime_lock_sha256': manifest['runtime_lock_sha256'], 'budget': manifest['budget'],
            'one_allocation_only': True, 'provider_or_solver': 'FORBIDDEN',
            'cli_identity_sha256': remote.sha(identity), 'experiment': 'precision-diagnostic',
            'diagnostic_only': True, 'route_probe_sha256': ROUTE_SHA,
            'precision_probe_sha256': manifest['precision_probe_sha256']}
    acceptance = tmp_path / 'gate.json'
    write(acceptance, gate)
    monkeypatch.setattr(owner, 'verify_cli_identity', lambda *args: None)
    calls, session = [], []
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    timer = signal.getitimer(signal.ITIMER_REAL)

    def fake_api(label, argv, timeout):
        calls.append((label, argv, timeout))
        if label == '03-one-allocation':
            session.append(argv[argv.index('-s') + 1])
            receipt['allocation_epoch'] = json.loads((out / 'owner.json').read_text())['allocation_epoch']
            write(records / 'receipt.json', receipt)
        if label in {'04-sessions-after', '90-before-stop'}:
            return {'status': 'PASS'}, '[' + session[0] + ']\nHardware: V6E1'
        if label == '13-tpu':
            code = Path(argv[-1]).read_text()
            assert '"remote.py",\'precision\'' in code
            assert json.loads((out / 'launch.json').read_text())['oracle_sha256'] == data.oracle_sha256
        if label == '21-export':
            write(records / 'receipt.json', {**receipt, **remote.package(records)})
        if label == '24-export-parts':
            transport.split(records / 'evidence.zip', tmp_path / 'remote-parts',
                            expected_sha256=remote.sha(records / 'evidence.zip'),
                            expected_bytes=(records / 'evidence.zip').stat().st_size)
        if argv[0] == 'download':
            source_name, destination = argv[-2], Path(argv[-1])
            if label in {'09-install-receipt', '10-cpu-receipt'}:
                status = 'INSTALLED_NOT_QUALIFIED' if label.startswith('09-') else 'CPU_ORACLE_READY_TPU_NOT_RUN'
                phase = {**receipt, 'status': status}
                if fault == 'phase-binding' and label == '09-install-receipt':
                    phase['allocation_epoch'] = -1
                write(destination, phase)
            else:
                rel = source_name.removeprefix('content/bnb-tpu-first/')
                source = (records / rel.removeprefix('records/') if rel.startswith('records/')
                          else tmp_path / 'remote-parts' / rel.removeprefix('result-parts/'))
                shutil.copyfile(source, destination)
                if fault == 'part-bytes' and label.startswith('26-part-'):
                    destination.write_bytes(destination.read_bytes() + b'bad')
        if fault == 'stop' and label == '91-stop-exact':
            return {'status': 'CLI_FAILED'}, 'SYNTHETIC stop refusal'
        text = 'Active assignments: 0\nUsage rate: 0.00/hr' if argv == ['usage'] else 'No active sessions found on server.'
        return {'status': 'PASS'}, text

    result = owner.drive(packet, out, expected, acceptance, remote.sha(acceptance), Path('fixture-python'),
                         identity, api=fake_api, simulated=True)
    assert result['mode'] == 'SIMULATED_NO_CLOUD' and result['allocation_attempts'] == 1
    assert [label for label, _, _ in calls].count('13-tpu') == (0 if fault == 'phase-binding' else 1)
    assert [label for label, _, _ in calls][-4:] == ['90-before-stop', '91-stop-exact', '92-after-stop', '93-usage-after']
    assert all(signal.getsignal(sig) == handler for sig, handler in handlers.items())
    assert signal.getitimer(signal.ITIMER_REAL) == timer
    assert (records / 'precision/receipt.json').read_bytes() == original_parent_bytes
    if fault == 'part-bytes':
        assert result.get('retrieval_error') and 'local_verifier_exit_code' not in result
    else:
        assert result['retrieval'] == 'COMPLETE_WHOLE_ARCHIVE' and result['whole_archive_verified'] is True
        assert (out / 'recovered/precision/receipt.json').read_bytes() == original_parent_bytes
        assert not result['local_verifier_cleanup']['errors']
        assert all(g['group_absence'] == 'OBSERVED_NO_SUCH_GROUP' for g in result['local_verifier_cleanup']['groups'])
    if fault == 'phase-binding':
        # The owner has no admitted recovered oracle when the install receipt fails.
        assert 'local_verifier_exit_code' not in result and not (out / 'verify.json').exists()
    elif fault != 'part-bytes':
        report = json.loads((out / 'verify.json').read_text())
        assert report == json.loads((out / 'local-verifier.raw').read_text())
        assert report['api42_status'] == report['m3_status'] == 'NOT_QUALIFIED'
        if fault == 'source':
            assert result['local_verifier_exit_code'] == 1 and report['record_validation'] == 'FAIL'
        else:
            assert result['local_verifier_exit_code'] == 0 and report['record_validation'] == 'PASS'
            expected_numeric = 'FAIL' if fault and fault.startswith('numeric-') else 'PASS'
            assert report['numerical_status'] == expected_numeric
            assert [mode['mode'] for mode in report['modes']] == ['default', 'high', 'highest']
            assert all(len(mode['rows']) == 5 for mode in report['modes'])
            if expected_numeric == 'FAIL':
                failed_mode = fault.removeprefix('numeric-')
                assert [mode['mode'] for mode in report['modes'] if mode['numerical_status'] == 'FAIL'] == [failed_mode]
    accepted = fault is None or fault.startswith('numeric-')
    assert (result['status'] == 'PASS_PRECISION_RECORDS') is accepted
    assert result['api42_status'] == result['m3_status'] == 'NOT_QUALIFIED'
    if accepted:
        assert result['record_validation'] == 'PASS'
        assert result['numerical_status'] == ('PASS' if fault is None else 'FAIL')
    if fault == 'stop':
        assert result['status'] == 'BLOCKED_CLEANUP'
    if fault == 'phase-binding':
        assert result['original_error']['message'] == 'PRECISION_PHASE_RECEIPT_BINDING:09-install'
