"""Isolated owner admission and receipt controls. No science or service call."""
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import build_packet
import owner
import remote


def setup_phase(tmp_path):
    payload, out = tmp_path / 'payload', tmp_path / 'records'
    payload.mkdir()
    out.mkdir()
    route = payload / 'probe_routes.py'
    route.write_text('# isolated fixture only\n')
    manifest = {'files': {'probe_routes.py': {'sha256': remote.sha(route), 'bytes': route.stat().st_size}},
                'experiment': 'device-route-diagnostic', 'diagnostic_only': True,
                'route_probe_sha256': remote.sha(route), 'runtime_lock_sha256': 'runtime',
                'source_admission_sha256': 'source'}
    (payload / 'manifest.json').write_text(json.dumps(manifest))
    (tmp_path / 'payload.zip').write_bytes(b'fixture packet')
    (tmp_path / 'launch.json').write_text('{"oracle_sha256":"oracle"}')
    receipt = {'status': 'CPU_ORACLE_READY_TPU_NOT_RUN', 'cpu_status': 'PASS', 'tpu_status': 'NOT_RUN',
               'oracle_sha256': 'oracle', 'science_deadline_epoch': 2000, 'steps': [], 'error': None}
    (out / 'receipt.json').write_text(json.dumps(receipt))
    result = {'kind': 'DEVICE_ROUTE_DIAGNOSTIC_ONLY', 'status': 'COMPLETE', 'source_pre': 'source',
              'source_post': 'source', 'source_admission_sha256': 'source', 'oracle_sha256': 'oracle',
              'runtime_lock_sha256': 'runtime', 'pid': 7001, 'process_token': 'fixture-process',
              'routes': [{'actual_status': 'ERROR', 'expected_error_observed': True}]}
    return manifest, result, out


@pytest.mark.parametrize('failure', [None, 'partial', 'source-pre', 'source-post', 'oracle', 'runtime',
                                      'pid', 'process-token', 'cleanup', 'child-exit', 'child-raise'])
def test_route_child_receipt_and_identity(tmp_path, monkeypatch, failure):
    manifest, result, out = setup_phase(tmp_path)
    changes = {'partial': ('status', 'PARTIAL'), 'source-pre': ('source_pre', 'wrong'),
               'source-post': ('source_post', 'wrong'), 'oracle': ('oracle_sha256', 'wrong'),
               'runtime': ('runtime_lock_sha256', 'wrong'), 'pid': ('pid', 7002),
               'process-token': ('process_token', '')}
    if failure in changes:
        key, value = changes[failure]
        result[key] = value
    def child(output, label, argv, deadline, limit, **kwargs):
        running = json.loads((out / 'receipt.json').read_text())
        assert running['status'] == 'DEVICE_ROUTE_CHILD_RUNNING'
        assert running['tpu_status'] == 'RUNNING' and running['tpu_attempted'] is True
        assert label == '12-device-routes' and limit == 1500 and kwargs['tpu'] is True
        assert argv[argv.index('--backend-probe') + 1] == str(tmp_path / 'payload/probe_backend.py')
        assert argv[argv.index('--oracle-sha256') + 1] == 'oracle'
        assert float(argv[argv.index('--deadline-epoch') + 1]) == deadline
        if failure == 'child-raise':
            raise RuntimeError('isolated fixture refusal')
        target = out / 'device-routes'
        target.mkdir()
        (target / 'receipt.json').write_text(json.dumps(result))
        return {'status': 'CHILD_FAILED' if failure == 'child-exit' else 'PASS',
                'exit_code': 1 if failure == 'child-exit' else 0,
                'cleanup': {'errors': [{'operation': 'fixture-refusal'}] if failure == 'cleanup' else [],
                            'groups': [{'pid': 7001, 'group_absence': 'OBSERVED_NO_SUCH_GROUP'}]}}
    monkeypatch.setattr(remote, 'run_step', child)
    exit_code = remote.execute(tmp_path, remote.sha(tmp_path / 'payload.zip'), 1000, 'routes')
    receipt = json.loads((out / 'receipt.json').read_text())
    assert receipt['tpu_attempted'] is True
    if failure is None:
        assert exit_code == 0
        assert receipt['status'] == 'DEVICE_ROUTE_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED'
        assert receipt['tpu_status'] == 'DIAGNOSTIC_RECORDS_COMPLETE'
        assert result['routes'][0]['actual_status'] == 'ERROR'
        assert result['routes'][0]['expected_error_observed'] is True
    else:
        assert exit_code == 1 and receipt['status'] == 'BLOCKED'
        assert receipt['tpu_status'] == 'ATTEMPTED_BLOCKED'


def test_diagnostic_cannot_run_api42_phase(tmp_path, monkeypatch):
    setup_phase(tmp_path)
    calls = []
    monkeypatch.setattr(remote, 'run_step', lambda *a, **k: calls.append(a))
    assert remote.execute(tmp_path, remote.sha(tmp_path / 'payload.zip'), 1000, 'tpu') == 1
    assert calls == []
    assert json.loads((tmp_path / 'records/receipt.json').read_text())['error']['message'] == 'WRONG_SCIENTIFIC_VARIANT'


def test_wrong_route_source_rejects_before_build(tmp_path, monkeypatch):
    source = tmp_path / 'route.py'
    source.write_text('# fixture only\n')
    calls = []
    monkeypatch.setattr(build_packet.subprocess, 'check_output', lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match='ROUTE_PROBE_SOURCE'):
        build_packet.build(tmp_path / 'upstream', tmp_path / 'packet', 'fixture',
                           experiment='device-route-diagnostic', route_probe=source, route_probe_sha256='0' * 64)
    assert calls == [] and not (tmp_path / 'packet').exists()


def test_api42_cannot_include_diagnostic_source(tmp_path):
    with pytest.raises(ValueError, match='DIAGNOSTIC_PROBE_NOT_REQUESTED'):
        build_packet.build(tmp_path / 'upstream', tmp_path / 'packet', 'fixture',
                           route_probe=tmp_path / 'route.py', route_probe_sha256='fixture')
    assert not (tmp_path / 'packet').exists()


def packet_fixture(target, mutation=None):
    import shutil
    import zipfile
    target.mkdir()
    for p in HERE.rglob('*.py'):
        if 'tests' in p.relative_to(HERE).parts or p.name == 'build_packet.py':
            continue
        dest = target / 'cloud' / p.relative_to(HERE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    (target / 'probe_routes.py').write_text('# isolated source admission fixture\n')
    files = {p.relative_to(target).as_posix(): {'sha256': remote.sha(p), 'bytes': p.stat().st_size}
             for p in target.rglob('*') if p.is_file()}
    manifest = {'files': files, 'experiment': 'device-route-diagnostic', 'diagnostic_only': True,
                'route_probe_sha256': remote.sha(target / 'probe_routes.py'),
                'plugin_source_manifest_sha256': 'fixture', 'runtime_lock_sha256': 'fixture',
                'budget': {'total': 3600, 'install': 1200, 'science': 1800, 'retrieval': 600, 'cleanup': 60}}
    if mutation == 'pin':
        manifest['route_probe_sha256'] = '0' * 64
    elif mutation == 'scope':
        manifest['diagnostic_only'] = False
    (target / 'manifest.json').write_text(json.dumps(manifest))
    with zipfile.ZipFile(target / 'payload.zip', 'w') as z:
        for name in [*files, 'manifest.json']:
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            z.writestr(info, (target / name).read_bytes())
    return manifest


@pytest.mark.parametrize('mutation', [None, 'pin', 'scope'])
def test_real_route_preflight_binding(tmp_path, monkeypatch, mutation):
    packet = tmp_path / 'packet'
    expected = packet_fixture(packet, mutation)
    calls = []
    monkeypatch.setattr(owner, 'OfficialCLI', lambda *a: calls.append(a))
    if mutation is None:
        assert owner.preflight(packet, remote.sha(packet / 'payload.zip')) == expected
    else:
        with pytest.raises(ValueError, match='ROUTE_PROBE_BINDING'):
            owner.preflight(packet, remote.sha(packet / 'payload.zip'))
    assert calls == []


def test_wrong_route_gate_rejects_before_cli(tmp_path, monkeypatch):
    packet = tmp_path / 'packet'
    manifest = packet_fixture(packet)
    out = tmp_path / 'actual'
    identity = tmp_path / 'identity.json'
    identity.write_text('{}')
    gate = {'status': 'ACTUAL_DISPATCH_AUTHORIZED', 'packet_sha256': remote.sha(packet / 'payload.zip'),
            'driver_sha256': remote.sha(HERE / 'owner.py'), 'output': str(out.resolve()),
            'plugin_source_manifest_sha256': 'fixture', 'runtime_lock_sha256': 'fixture',
            'budget': manifest['budget'], 'one_allocation_only': True, 'provider_or_solver': 'FORBIDDEN',
            'cli_identity_sha256': remote.sha(identity), 'experiment': 'device-route-diagnostic',
            'diagnostic_only': True, 'route_probe_sha256': '0' * 64}
    admission = tmp_path / 'admission.json'
    admission.write_text(json.dumps(gate))
    calls = []
    monkeypatch.setattr(owner, 'OfficialCLI', lambda *a: calls.append(a))
    with pytest.raises(PermissionError, match='EXACT_ROOT_ACCEPTANCE_REQUIRED'):
        owner.drive(packet, out, remote.sha(packet / 'payload.zip'), admission, remote.sha(admission),
                    Path('fixture-python'), identity)
    assert calls == [] and not out.exists()


@pytest.mark.parametrize('fault', [None, 'numeric-fail', 'reference', 'packed-roundtrip', 'source', 'part-bytes', 'stop'])
def test_full_diagnostic_owner_chain(tmp_path, monkeypatch, fault):
    """Fake service, real byte recovery and saved-record verifier; no science."""
    import importlib.util
    import shutil
    import signal
    import zipfile
    import transport

    fixture_source = HERE.parent / 'tests/test_routes.py'
    assert remote.sha(fixture_source) == 'e8b55db5266d19c477cd6ebbb7f9d2825b6c0e9d6c7c5af616ce79f4b4e93e85'
    spec = importlib.util.spec_from_file_location('route_owner_fixtures', fixture_source)
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    route_source = HERE.parent / 'probe_routes.py'
    assert remote.sha(route_source) == 'feae751734e57c741b1bdade004ff7ca3c041ee7eb3b7086b531bc6be433038e'
    packet = tmp_path / 'packet'
    manifest = packet_fixture(packet)
    shutil.copyfile(route_source, packet / 'probe_routes.py')
    backend_root = Path(fixtures.BACKEND).parent
    for name in ('probe_backend.py', 'probe-profile.json', 'probe-inputs.json'):
        shutil.copyfile(backend_root / name, packet / name)
    (packet / 'source-admission.json').write_text('{"scope":"SYNTHETIC_NO_SOURCE_ADMISSION"}\n')
    admission_sha = remote.sha(packet / 'source-admission.json')
    manifest.update(source_admission_sha256=admission_sha,
                    runtime_lock_sha256=fixtures.B.read(packet / 'probe-profile.json')['runtime_lock_sha256'],
                    route_probe_sha256=remote.sha(packet / 'probe_routes.py'))
    manifest['files'] = {p.relative_to(packet).as_posix(): {'sha256': remote.sha(p), 'bytes': p.stat().st_size}
                         for p in packet.rglob('*') if p.is_file() and p.name not in {'payload.zip', 'manifest.json'}}
    (packet / 'manifest.json').write_text(json.dumps(manifest))
    with zipfile.ZipFile(packet / 'payload.zip', 'w') as z:
        for name in [*manifest['files'], 'manifest.json']:
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            z.writestr(info, (packet / name).read_bytes())
    expected = remote.sha(packet / 'payload.zip')
    data = fixtures.make_fixture(tmp_path / 'fixture', admission=admission_sha)
    route_path = 'raw/routes-target_constructor-' + fixtures.P.CASE_IDS[0] + '.json'
    if fault == 'reference':
        refs = fixtures.B.read(data.actual / 'references.json')
        refs[fixtures.P.CASE_IDS[0]]['dx']['values'][0] = 123
        fixtures.B.write(data.actual / 'references.json', refs)
        for route in ('target_constructor', 'from_prequantized'):
            path = 'raw/routes-' + route + '-' + fixtures.P.CASE_IDS[0] + '.json'
            fixtures.change(data, path, lambda r: (r['outputs']['dx']['values'].__setitem__(0, 123),
                                                  r['roundtrip_outputs']['dx']['values'].__setitem__(0, 123)))
    elif fault == 'packed-roundtrip':
        fixtures.change(data, route_path, lambda r: r['roundtrip_outputs']['packed']['values'].__setitem__(0, 1))
    elif fault == 'source':
        fixtures.change(data, 'receipt.json', lambda r: r.update(source_post='0' * 64))
    elif fault == 'numeric-fail':
        nonlinear, _ = fixtures.P.selection(fixtures.B)
        fixtures.change(data, 'raw/nonlinear-' + nonlinear[0]['id'] + '.json',
                        lambda r: r['outputs']['packed']['values'].__setitem__(0, 1))
    records = tmp_path / 'remote-records'
    records.mkdir()
    shutil.copytree(data.oracle, records / 'cpu-oracle')
    shutil.copytree(data.actual, records / 'device-routes')
    original_route_bytes = (records / 'device-routes/receipt.json').read_bytes()
    receipt = {'status': 'DEVICE_ROUTE_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED',
               'packet_sha256': expected, 'manifest_sha256': remote.sha(packet / 'manifest.json'),
               'runtime_status': 'PASS_TPU_RUNTIME_PROBE_ONLY', 'cpu_status': 'PASS',
               'tpu_status': 'DIAGNOSTIC_RECORDS_COMPLETE', 'tpu_attempted': True,
               'oracle_sha256': data.oracle_sha256,
               'steps': [{'scope': 'SYNTHETIC_NO_CHILD_EXECUTION', 'cleanup': {'errors': [], 'groups': []}}]}
    (records / 'receipt.json').write_text(json.dumps(receipt))
    out, identity = tmp_path / 'host', tmp_path / 'identity.json'
    identity.write_text('{}')
    gate = {'status': 'ACTUAL_DISPATCH_AUTHORIZED', 'packet_sha256': expected,
            'driver_sha256': remote.sha(HERE / 'owner.py'), 'output': str(out.resolve()),
            'plugin_source_manifest_sha256': manifest['plugin_source_manifest_sha256'],
            'runtime_lock_sha256': manifest['runtime_lock_sha256'], 'budget': manifest['budget'],
            'one_allocation_only': True, 'provider_or_solver': 'FORBIDDEN',
            'cli_identity_sha256': remote.sha(identity), 'experiment': 'device-route-diagnostic',
            'diagnostic_only': True, 'route_probe_sha256': manifest['route_probe_sha256']}
    admission = tmp_path / 'gate.json'
    admission.write_text(json.dumps(gate))
    monkeypatch.setattr(owner, 'verify_cli_identity', lambda *args: None)
    calls, session = [], []
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    timer = signal.getitimer(signal.ITIMER_REAL)

    def fake_api(label, argv, timeout):
        calls.append((label, argv, timeout))
        if label == '03-one-allocation':
            session.append(argv[argv.index('-s') + 1])
        if label == '04-sessions-after' or label == '90-before-stop':
            return {'status': 'PASS'}, '[' + session[0] + ']\nHardware: V6E1'
        if label == '13-tpu':
            code = Path(argv[-1]).read_text()
            assert '"remote.py",\'routes\'' in code
            assert json.loads((out / 'launch.json').read_text())['oracle_sha256'] == data.oracle_sha256
        if label == '21-export':
            exported = {**receipt, **remote.package(records)}
            (records / 'receipt.json').write_text(json.dumps(exported))
        if label == '24-export-parts':
            transport.split(records / 'evidence.zip', tmp_path / 'remote-parts',
                            expected_sha256=remote.sha(records / 'evidence.zip'),
                            expected_bytes=(records / 'evidence.zip').stat().st_size)
        if argv[0] == 'download':
            source_name, destination = argv[-2], Path(argv[-1])
            if label in {'09-install-receipt', '10-cpu-receipt'}:
                status = 'INSTALLED_NOT_QUALIFIED' if label.startswith('09-') else 'CPU_ORACLE_READY_TPU_NOT_RUN'
                destination.write_text(json.dumps({**receipt, 'status': status}))
            else:
                base = 'content/bnb-tpu-first/'
                rel = source_name.removeprefix(base)
                source = records / rel.removeprefix('records/') if rel.startswith('records/') else tmp_path / 'remote-parts' / rel.removeprefix('result-parts/')
                shutil.copyfile(source, destination)
                if fault == 'part-bytes' and label.startswith('26-part-'):
                    destination.write_bytes(destination.read_bytes() + b'bad')
        if fault == 'stop' and label == '91-stop-exact':
            return {'status': 'CLI_FAILED'}, 'SYNTHETIC stop refusal'
        text = 'Active assignments: 0\nUsage rate: 0.00/hr' if argv == ['usage'] else 'No active sessions found on server.'
        return {'status': 'PASS'}, text

    result = owner.drive(packet, out, expected, admission, remote.sha(admission), Path('fixture-python'),
                         identity, api=fake_api, simulated=True)
    assert result['mode'] == 'SIMULATED_NO_CLOUD' and result['allocation_attempts'] == 1
    assert [label for label, _, _ in calls].count('13-tpu') == 1
    assert [label for label, _, _ in calls][-4:] == ['90-before-stop', '91-stop-exact', '92-after-stop', '93-usage-after']
    assert all(signal.getsignal(sig) == handler for sig, handler in handlers.items())
    assert signal.getitimer(signal.ITIMER_REAL) == timer
    assert (records / 'device-routes/receipt.json').read_bytes() == original_route_bytes
    if fault == 'part-bytes':
        assert result.get('retrieval_error') and 'local_verifier_exit_code' not in result
    else:
        assert result['retrieval'] == 'COMPLETE_WHOLE_ARCHIVE' and result['whole_archive_verified'] is True
        assert (out / 'recovered/device-routes/receipt.json').read_bytes() == original_route_bytes
        assert not result['local_verifier_cleanup']['errors']
        assert all(g['group_absence'] == 'OBSERVED_NO_SUCH_GROUP' for g in result['local_verifier_cleanup']['groups'])
        report = json.loads((out / 'local-verifier.raw').read_text())
        if fault in {'reference', 'packed-roundtrip', 'source'}:
            assert result['local_verifier_exit_code'] == 1 and report['record_validation'] == 'FAIL'
        else:
            assert result['local_verifier_exit_code'] == 0 and report['record_validation'] == 'PASS'
            assert report['api42_status'] == report['m3_status'] == 'NOT_QUALIFIED'
            assert report['numerical_status'] == ('FAIL' if fault == 'numeric-fail' else 'INCOMPLETE')
            assert len(report['rows']) == 40
            assert sum(row['expected_error_observed'] for row in report['rows']) == 2
    assert (result['status'] == 'PASS_DEVICE_ROUTE_RECORDS') is (fault in {None, 'numeric-fail'})
    if fault == 'stop':
        assert result['status'] == 'BLOCKED_CLEANUP'
