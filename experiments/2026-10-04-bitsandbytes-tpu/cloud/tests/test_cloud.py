import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import zipfile

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import remote
import owner
import build_packet


def test_budget_clamp_and_exhaustion():
    assert owner.clamp(1000, 1200, 660, clock=lambda: 3800) == 140
    assert remote.allowance(1200, 1800, clock=lambda: 1190) == 10
    with pytest.raises(TimeoutError):
        owner.clamp(1000, 1200, 660, clock=lambda: 4601)


def test_child_environment(monkeypatch):
    for key in ['TPU_LIBRARY_PATH', 'PTXLA_TPU_LIBRARY_PATH', 'PYTHONPATH', 'PYTHONHOME', 'PJRT_DEVICE']:
        monkeypatch.setenv(key, 'fixture')
    before = {key: remote.os.environ.get(key) for key in ['HOME', 'CODEX_HOME']}
    cpu, record = remote.environment(False)
    assert not any(key in cpu for key in record['names_unset'])
    assert {key: cpu.get(key) for key in before} == before
    tpu, record = remote.environment(True)
    assert tpu['PJRT_DEVICE'] == 'TPU'
    assert tpu['TPU_LOG_DIR'] == 'disabled'
    assert tpu['PYTHONNOUSERSITE'] == '1'


def make_zip(tmp_path, unsafe=False):
    packet = tmp_path / 'packet'
    packet.mkdir()
    data = b'fixture'
    (packet / 'fixture.txt').write_bytes(data)
    manifest = {'files': {'fixture.txt': {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}}}
    (packet / 'manifest.json').write_text(json.dumps(manifest))
    with zipfile.ZipFile(packet / 'payload.zip', 'w') as z:
        for name in ['manifest.json', 'fixture.txt']:
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (0o120777 if unsafe and name == 'fixture.txt' else 0o100644) << 16
            z.writestr(info, (packet / name).read_bytes())
    return packet


def test_exact_consumer_archive(tmp_path):
    p = make_zip(tmp_path)
    remote.verify_archive(p / 'payload.zip', remote.sha(p / 'payload.zip'), tmp_path / 'out')
    assert (tmp_path / 'out/fixture.txt').read_bytes() == b'fixture'


def test_consumer_wrong_metadata_before_cli(tmp_path, monkeypatch):
    p = make_zip(tmp_path, unsafe=True)
    calls = []
    monkeypatch.setattr(owner, 'OfficialCLI', lambda *a: calls.append(a))
    with pytest.raises(ValueError, match='PAYLOAD_NONREGULAR'):
        owner.preflight(p, remote.sha(p / 'payload.zip'))
    assert calls == []


def test_loose_manifest_matches_archive(tmp_path, monkeypatch):
    p = make_zip(tmp_path)
    current = tmp_path / 'current-owner'
    current.mkdir()
    monkeypatch.setattr(owner, 'HERE', current)
    manifest = owner.preflight(p, remote.sha(p / 'payload.zip'))
    assert manifest == json.loads((p / 'manifest.json').read_text())


@pytest.mark.parametrize('mutation', ['bytes', 'symlink', 'directory'])
def test_loose_manifest_rejected_before_cli(tmp_path, monkeypatch, mutation):
    p = make_zip(tmp_path)
    manifest = p / 'manifest.json'
    saved = manifest.read_bytes()
    manifest.unlink()
    if mutation == 'bytes':
        manifest.write_bytes(saved + b'\n')
    elif mutation == 'symlink':
        (p / 'manifest-target.json').write_bytes(saved)
        manifest.symlink_to('manifest-target.json')
    else:
        manifest.mkdir()
    calls = []
    monkeypatch.setattr(owner, 'OfficialCLI', lambda *args: calls.append(args))
    expected = 'PACKET_LOCAL_MANIFEST_BYTES' if mutation == 'bytes' else 'PACKET_LOCAL_MANIFEST_NONREGULAR'
    with pytest.raises(ValueError, match=expected):
        owner.drive(p, tmp_path / 'actual', remote.sha(p / 'payload.zip'),
                    tmp_path / 'absent-gate', 'fixture', Path('fixture-python'), tmp_path / 'absent-identity')
    assert calls == []
    assert not (tmp_path / 'actual').exists()


@pytest.mark.parametrize('exit_code,cleanup_errors,child_error,child_status,expected', [
    (0, [], None, 'PASS', 'PASS'),
    (2, [], None, 'CHILD_FAILED', 'FAIL'),
    (1, [], None, 'CHILD_FAILED', 'ATTEMPTED_BLOCKED'),
    (0, [{'operation': 'fixture_cleanup_refusal'}], None, 'BLOCKED_CLEANUP', 'ATTEMPTED_BLOCKED'),
    (2, [{'operation': 'fixture_cleanup_refusal'}], None, 'BLOCKED_CLEANUP', 'ATTEMPTED_BLOCKED'),
    (0, [], {'type': 'RuntimeError', 'message': 'fixture'}, 'PASS', 'ATTEMPTED_BLOCKED'),
    (0, [], None, 'BLOCKED', 'ATTEMPTED_BLOCKED'),
    (None, [], None, 'RAISE_FIXTURE', 'ATTEMPTED_BLOCKED'),
])
def test_tpu_attempt_status(tmp_path, monkeypatch, exit_code, cleanup_errors, child_error, child_status, expected):
    payload, out = tmp_path / 'payload', tmp_path / 'records'
    payload.mkdir()
    out.mkdir()
    (tmp_path / 'payload.zip').write_bytes(b'fixture packet')
    manifest = {'files': {}, 'runtime_lock_sha256': 'runtime', 'source_admission_sha256': 'source'}
    (payload / 'manifest.json').write_text(json.dumps(manifest))
    (tmp_path / 'launch.json').write_text(json.dumps({'oracle_sha256': 'oracle'}))
    receipt_path = out / 'receipt.json'
    receipt_path.write_text(json.dumps({'status': 'CPU_ORACLE_READY_TPU_NOT_RUN', 'cpu_status': 'PASS',
                                      'tpu_status': 'NOT_RUN', 'oracle_sha256': 'oracle',
                                      'science_deadline_epoch': 2000, 'steps': [], 'error': None}))
    def fake_step(*args, **kwargs):
        running = json.loads(receipt_path.read_text())
        assert running['status'] == 'TPU_CHILD_RUNNING'
        assert running['tpu_status'] == 'RUNNING' and running['tpu_attempted'] is True
        if child_status == 'RAISE_FIXTURE':
            raise RuntimeError('fixture_before_child_record')
        return {'status': child_status, 'exit_code': exit_code, 'error': child_error,
                'cleanup': {'errors': cleanup_errors}}
    monkeypatch.setattr(remote, 'run_step', fake_step)
    result = remote.execute(tmp_path, remote.sha(tmp_path / 'payload.zip'), 1000, 'tpu')
    receipt = json.loads(receipt_path.read_text())
    assert receipt['tpu_attempted'] is True and receipt['tpu_status'] == expected
    assert result == (1 if expected == 'ATTEMPTED_BLOCKED' else 0)
    assert receipt['status'] == ('BLOCKED' if expected == 'ATTEMPTED_BLOCKED' else 'TPU_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED')
    if child_status != 'RAISE_FIXTURE':
        assert receipt['steps'][0]['exit_code'] == exit_code
        assert receipt['steps'][0]['cleanup']['errors'] == cleanup_errors
        assert receipt['steps'][0]['error'] == child_error


def test_package_nested_receipt_and_projection(tmp_path):
    out = tmp_path / 'records'
    (out / 'tpu-actual').mkdir(parents=True)
    top = {'status': 'FIXTURE_ONLY'}
    (out / 'receipt.json').write_text(json.dumps(top))
    (out / 'tpu-actual/receipt.json').write_text('{"status":"NUMERICAL_FIXTURE"}')
    exported = {**top, **remote.package(out)}
    (out / 'receipt.json').write_text(json.dumps(exported))
    with zipfile.ZipFile(out / 'evidence.zip') as z:
        assert z.namelist().count('receipt.json') == 1
        assert 'tpu-actual/receipt.json' in z.namelist()
    assert set(owner.verify_result(out, exported)) == {'receipt.json', 'tpu-actual/receipt.json'}


def test_result_symlink_rejected(tmp_path):
    (tmp_path / 'receipt.json').write_text('{}')
    (tmp_path / 'link').symlink_to('receipt.json')
    with pytest.raises(ValueError, match='RESULT_SYMLINK'):
        remote.package(tmp_path)


def test_result_special_rejected(tmp_path):
    (tmp_path / 'receipt.json').write_text('{}')
    remote.os.mkfifo(tmp_path / 'fifo')
    with pytest.raises(ValueError, match='RESULT_NONREGULAR'):
        remote.package(tmp_path)


def fake_admission(tmp_path, monkeypatch):
    packet = tmp_path / 'packet'
    packet.mkdir()
    (packet / 'manifest.json').write_text('{}')
    manifest = {'plugin_source_manifest_sha256': 'plugin', 'runtime_lock_sha256': 'runtime', 'budget': {'total':3600, 'install':1200, 'science':1800, 'retrieval':600, 'cleanup':60}, 'files': {}}
    monkeypatch.setattr(owner, 'preflight', lambda *a: manifest)
    monkeypatch.setattr(owner, 'verify_cli_identity', lambda *a: None)
    ident = tmp_path / 'cli-identity.json'
    ident.write_text('{}')
    out = tmp_path / 'actual'
    gate = {'status':'ACTUAL_DISPATCH_AUTHORIZED', 'packet_sha256':'fixture', 'driver_sha256':owner.sha(HERE/'owner.py'), 'output':str(out.resolve()), 'plugin_source_manifest_sha256':'plugin', 'runtime_lock_sha256':'runtime', 'budget':manifest['budget'], 'one_allocation_only':True, 'provider_or_solver':'FORBIDDEN', 'cli_identity_sha256':owner.sha(ident)}
    admission = tmp_path/'acceptance.json'
    admission.write_text(json.dumps(gate))
    return packet, out, admission, ident


@pytest.mark.parametrize('overdue', [False, True])
def test_allocation_failure_still_stops_and_restores(tmp_path, monkeypatch, overdue):
    packet, out, admission, ident = fake_admission(tmp_path, monkeypatch)
    calls = []
    clock = [1000.0]
    monkeypatch.setattr(owner.time, 'time', lambda: clock[0])
    handlers = {s: signal.getsignal(s) for s in [signal.SIGTERM, signal.SIGINT]}
    timer = signal.getitimer(signal.ITIMER_REAL)
    def api(label, argv, timeout):
        calls.append((label, argv, timeout))
        if label == '03-one-allocation':
            if overdue:
                clock[0] = 5000.0
            return {'status':'CLI_FAILED'}, 'allocation fixture failed'
        text = 'No active sessions found on server.' if argv == ['sessions'] else 'Active assignments: 0\nUsage rate: 0.00/hr'
        return {'status':'PASS'}, text
    result = owner.drive(packet, out, 'fixture', admission, owner.sha(admission), Path('fixture-python'), ident, api=api, simulated=True)
    assert any(a[:2] == ['stop', '-s'] for _, a, _ in calls)
    assert [label for label, _, _ in calls][-4:] == ['90-before-stop','91-stop-exact','92-after-stop','93-usage-after']
    assert sum(t for label, _, t in calls if label.startswith(('90-','91-','92-','93-'))) == 60
    assert result['status'] == ('BLOCKED_CLEANUP' if overdue else 'BLOCKED')
    assert result['original_error']['message'] == 'CLI_STEP_BLOCKED:03-one-allocation'
    assert all(signal.getsignal(s) == handlers[s] for s in handlers)
    assert signal.getitimer(signal.ITIMER_REAL) == timer


def test_cleanup_failure_veto(tmp_path, monkeypatch):
    packet, out, admission, ident = fake_admission(tmp_path, monkeypatch)
    def api(label, argv, timeout):
        if label in ['03-one-allocation','91-stop-exact']:
            return {'status':'CLI_FAILED'}, 'fixture'
        return {'status':'PASS'}, 'No active sessions found on server.\nActive assignments: 0\nUsage rate: 0.00/hr'
    result = owner.drive(packet, out, 'fixture', admission, owner.sha(admission), Path('fixture-python'), ident, api=api, simulated=True)
    assert result['status'] == 'BLOCKED_CLEANUP'
    assert result['cleanup_errors'][0]['operation'] == '91-stop-exact'


def test_wrong_plugin_manifest_rejects_before_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(build_packet.subprocess, 'check_output', lambda *a, **k: build_packet.COMMIT + '\n')
    with pytest.raises(ValueError, match='PLUGIN_MANIFEST'):
        build_packet.build(tmp_path/'upstream', tmp_path/'packet', '0'*64)
    assert not (tmp_path/'packet').exists()


def test_wrong_gate_before_cli(tmp_path, monkeypatch):
    packet, out, admission, ident = fake_admission(tmp_path, monkeypatch)
    gate = json.loads(admission.read_text())
    gate['runtime_lock_sha256'] = 'wrong'
    admission.write_text(json.dumps(gate))
    calls = []
    monkeypatch.setattr(owner, 'OfficialCLI', lambda *args: calls.append(args))
    with pytest.raises(PermissionError, match='EXACT_ROOT_ACCEPTANCE'):
        owner.drive(packet, out, 'fixture', admission, owner.sha(admission), Path('fixture-python'), ident)
    assert calls == []
    assert not out.exists()
