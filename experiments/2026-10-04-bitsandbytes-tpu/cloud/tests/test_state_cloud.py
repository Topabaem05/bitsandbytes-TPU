"""State roundtrip records and real local process ownership. No service calls."""
import importlib.util
import errno
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import build_packet as B
import remote
import owner
import state_coordinator as C
from cleanup_lifecycle import Ownership
import test_transfer_cloud as F

ROOT = B.PROJECT
STATE_PROBE = F.SCIENCE / 'probe_state_roundtrip.py'
base_archive = F.base_archive


def state_args():
    args = F.source_args()
    args.update(experiment='state-roundtrip', state_probe=STATE_PROBE, state_probe_sha256=remote.sha(STATE_PROBE),
                state_helper=F.SCIENCE / 'probe_state.py', state_helper_sha256=B.STATE_HELPER_SHA)
    return args


@pytest.fixture
def state_packet(tmp_path, base_archive, monkeypatch):
    old = F.source_args
    monkeypatch.setattr(F, 'source_args', lambda: dict(old(), experiment='state-roundtrip',
        state_probe=STATE_PROBE, state_probe_sha256=remote.sha(STATE_PROBE),
        state_helper=F.SCIENCE / 'probe_state.py', state_helper_sha256=B.STATE_HELPER_SHA))
    return F.packet_fixture(tmp_path, base_archive, monkeypatch)


@pytest.mark.parametrize('fault', [None, 'missing', 'wrong-helper', 'wrong-pin', 'other-variant', 'scope'])
def test_state_dependency_admission(state_packet, fault):
    packet, manifest, digest = state_packet
    if fault == 'missing': del manifest['files']['probe_state.py']
    elif fault == 'wrong-helper': manifest['state_helper_sha256'] = 'a' * 64
    elif fault == 'wrong-pin': manifest['state_probe_sha256'] = 'a' * 64
    elif fault == 'other-variant': manifest['experiment'] = 'transfer-api42'
    elif fault == 'scope': manifest['state_scope'] = 'ONLY_ONE_CASE'
    if fault is None:
        assert remote.verify_experiment(manifest) == 'state-roundtrip'
        assert owner.preflight(packet, digest)['experiment'] == 'state-roundtrip'
    else:
        with pytest.raises(ValueError, match='STATE_'):
            remote.verify_experiment(manifest)


def launch_script(path, command, deadline_seconds=10):
    code = f'''import json, os, sys, time
from pathlib import Path
sys.path.insert(0, {str(HERE)!r})
import state_coordinator as C
out = Path(sys.argv[1]); out.mkdir()
parent = {{'pid': os.getpid(), 'pgid': os.getpgid(0), 'process_token': 'a'*32, 'deadline_epoch': time.time()+{deadline_seconds}, 'children': []}}
C.run_child({command!r}, output=out, descriptor={{'phase':'save','process_token':'b'*32}}, parent=parent, parent_path=out/'parent.json', deadline=parent['deadline_epoch'])
'''
    path.write_text(code)


@pytest.mark.parametrize('mode', ['success', 'timeout', 'parent-sigkill'])
def test_actual_shared_group_outer_cleanup(tmp_path, mode):
    ready = tmp_path / 'child-ready.json'
    child_code = 'import time;time.sleep(60)' if mode != 'success' else 'pass'
    if mode == 'parent-sigkill':
        child_code = ("import json,os,time;from pathlib import Path;"
                      f"ready=Path({str(ready)!r});temporary=ready.with_suffix('.tmp');"
                      "temporary.write_text(json.dumps({'pid':os.getpid(),'ppid':os.getppid(),'pgid':os.getpgid(0)}));"
                      "os.replace(temporary,ready);time.sleep(60)")
    script = tmp_path / 'coordinator-fixture.py'
    launch_script(script, [sys.executable, '-B', '-c', child_code], 0.15 if mode == 'timeout' else 10)
    out = tmp_path / 'science'
    (tmp_path / 'outer').mkdir()
    outer = Ownership(tmp_path / 'outer')
    with (tmp_path / 'stdout').open('wb') as stdout, (tmp_path / 'stderr').open('wb') as stderr:
        with outer.guard(5):
            process = outer.launch([sys.executable, '-B', str(script), str(out)], record=tmp_path / 'launch.json', stdout=stdout, stderr=stderr)
            until = time.monotonic() + 3
            while not (out / 'parent.json').exists() and time.monotonic() < until:
                time.sleep(0.01)
            parent = json.loads((out / 'parent.json').read_text())
            child = parent['children'][0]
            assert process.pid == parent['pid'] == parent['pgid'] == child['pgid']
            assert process.pid != child['pid']
            if mode == 'parent-sigkill':
                until = time.monotonic() + 3
                while not ready.exists() and time.monotonic() < until:
                    time.sleep(0.005)
                actual = json.loads(ready.read_text())
                assert actual == {'pid': child['pid'], 'ppid': parent['pid'], 'pgid': parent['pgid']}
                os.kill(process.pid, signal.SIGKILL)
            process.wait(timeout=4)
            assert process.returncode == (0 if mode == 'success' else -signal.SIGKILL if mode == 'parent-sigkill' else 1)
    summary = outer.summary()
    assert summary['errors'] == []
    assert summary['groups'][0]['status'] == 'CLEANUP_VERIFIED'
    assert summary['groups'][0]['group_absence'] == 'OBSERVED_NO_SUCH_GROUP'
    if mode != 'parent-sigkill':
        child = json.loads((out / 'parent.json').read_text())['children'][0]
        assert child['reaped'] and child['child_absent'] and child['cleanup_errors'] == []
        assert child['timeout'] == (mode == 'timeout')


def test_owned_group_probe_eperm_remains_unverified(tmp_path):
    """An injected permission denial must never count as observed group absence."""
    class DeniedProbe(Ownership):
        def signal_owned_group(self, process, signum):
            if signum == 0:
                raise PermissionError(errno.EPERM, 'SYNTHETIC_PROBE_PERMISSION_DENIED')
            return super().signal_owned_group(process, signum)
    (tmp_path / 'outer').mkdir()
    outer = DeniedProbe(tmp_path / 'outer')
    with outer.guard(3):
        process = outer.launch([sys.executable, '-B', '-c', 'pass'], record=tmp_path / 'launch.json')
        process.wait(timeout=2)
    summary = outer.summary()
    assert summary['status'] == 'CLEANUP_UNVERIFIED'
    group = summary['groups'][0]
    assert group['status'] == 'CLEANUP_UNVERIFIED' and group['group_absence'] == 'UNVERIFIED'
    assert group['leader_reaped'] is True
    assert any(error['operation'] == 'probe_owned_group' and error['errno'] == errno.EPERM
               for error in summary['errors'])
    # This independent later read cannot rewrite the original guard's failure.
    with pytest.raises(ProcessLookupError): os.killpg(process.pid, 0)


def load_device_fixtures():
    spec = importlib.util.spec_from_file_location('r3_device_fixtures', ROOT / 'tests/state_device_fixture.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('fault', [None, 'numeric', 'missing-row', 'checkpoint', 'launch-pid',
                                  'pgid', 'absence', 'deadline', 'parent-hash'])
def test_real_state_verifier_saved_records(tmp_path, fault):
    fixtures = load_device_fixtures()
    data = fixtures.make_device_fixture(tmp_path)
    D = fixtures.D
    if fault == 'numeric':
        path = data.actual / 'restore/raw' / (D.CASES[0]['id'] + '.json')
        raw = D.read(path); raw['outputs']['y']['values'][0] += 10; D.write(path, raw)
    elif fault == 'missing-row': (data.actual / 'restore/raw' / (D.CASES[0]['id'] + '.json')).unlink()
    elif fault == 'checkpoint':
        name = D.CASES[0]['id']; replacement = b'INVALID CHECKPOINT CONTROL\n'
        for phase in ('save', 'restore'): (data.actual / phase / 'raw' / (name + '.state.pt')).write_bytes(replacement)
        path = data.actual / 'restore/raw' / (name + '.json')
        raw = D.read(path); raw['loaded_checkpoint_sha256'] = D.sha(data.actual / 'save/raw' / (name + '.state.pt')); D.write(path, raw)
    elif fault in {'launch-pid', 'pgid', 'absence', 'deadline'}:
        parent = D.read(data.actual / 'parent.json')
        field, value = {'launch-pid': ('launched_pid', 999), 'pgid': ('pgid', 999),
                        'absence': ('child_absent', False), 'deadline': ('deadline_epoch', 0)}[fault]
        parent['children'][1][field] = value; D.write(data.actual / 'parent.json', parent)
    if fault != 'missing-row': fixtures.reseal_fixture(data)
    # A missing required row can be sealed but cannot pass the row matrix.
    else:
        receipt = D.read(data.actual / 'restore/receipt.json'); receipt['artifacts'] = D.B.inventory(data.actual / 'restore', 'receipt.json'); D.write(data.actual / 'restore/receipt.json', receipt)
        parent = D.read(data.actual / 'parent.json'); parent['children'][1]['receipt_sha256'] = D.sha(data.actual / 'restore/receipt.json'); parent['artifacts'] = D.B.inventory(data.actual, 'parent.json'); D.write(data.actual / 'parent.json', parent)
    if fault in {None, 'numeric'}:
        report = fixtures.verify_fixture(data)
        assert report['record_validation'] == 'PASS' and report['numerical_status'] == ('FAIL' if fault else 'PASS')
        assert len(report['rows']) == 8 and report['m3_status'] == 'NOT_QUALIFIED'
    else:
        with pytest.raises(Exception):
            D.verify_pair(data.actual, False, data.oracle, data.oracle_sha256, data.admission_sha256,
                          '0' * 64 if fault == 'parent-hash' else D.sha(data.actual / 'parent.json'))


@pytest.mark.parametrize('fault', [None, 'kind', 'partial', 'source', 'probe', 'helper', 'patch',
                                  'pid', 'pgid', 'group-pgid', 'token', 'cleanup', 'controls'])
def test_state_remote_receipt_and_deadline(state_packet, tmp_path, monkeypatch, fault):
    packet, manifest, expected = state_packet
    packet.rename(tmp_path / 'payload'); payload = tmp_path / 'payload'
    shutil.copyfile(payload / 'payload.zip', tmp_path / 'payload.zip')
    out = tmp_path / 'records'; out.mkdir()
    F.write(tmp_path / 'launch.json', {'oracle_sha256': 'oracle'})
    top = {'status': 'CPU_ORACLE_READY_TPU_NOT_RUN', 'cpu_status': 'PASS', 'tpu_status': 'NOT_RUN',
           'experiment': 'state-roundtrip', 'precision': 'highest', 'packet_sha256': expected,
           'manifest_sha256': remote.sha(payload / 'manifest.json'), 'source_admission_sha256': manifest['source_admission_sha256'],
           'runtime_lock_sha256': manifest['runtime_lock_sha256'], 'allocation_epoch': 1000,
           'oracle_sha256': 'oracle', 'science_deadline_epoch': 3000, 'steps': [], 'error': None,
           'built_source_status': 'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS', 'installed_source_status': 'POST_PATCH_PYTHON_SOURCE_PASS',
           'source_controls_status': 'FAIL' if fault == 'controls' else 'PASS_QUALIFIED_LINUX_SOURCE_CONTROLS',
           **{field: manifest[field] for field in (*remote.TRANSFER_BINDINGS, *remote.STATE_BINDINGS)}}
    F.write(out / 'receipt.json', top)
    result = {'kind': 'STATE_ROUNDTRIP_PARENT', 'status': 'COMPLETE', 'source_pre': manifest['source_admission_sha256'],
              'source_post': manifest['source_admission_sha256'], 'source_admission_sha256': manifest['source_admission_sha256'],
              'oracle_sha256': 'oracle', 'runtime_lock_sha256': manifest['runtime_lock_sha256'], 'pid': 7001, 'pgid': 7001,
              **{field: manifest[field] for field in (*remote.STATE_BINDINGS, 'patch_manifest_sha256')}}
    changes = {'kind': ('kind', 'TRANSFER_API42_PROBE'), 'partial': ('status', 'PARTIAL'), 'source': ('source_post', 'wrong'),
               'probe': ('state_probe_sha256', 'wrong'), 'helper': ('state_helper_sha256', 'wrong'),
               'patch': ('patch_manifest_sha256', 'wrong'), 'pid': ('pid', 7002), 'pgid': ('pgid', 7002)}
    if fault in changes: result.update([changes[fault]])
    monkeypatch.setattr(remote.time, 'time', lambda: 1000)
    calls = []
    def child(output, label, argv, deadline, limit, **kwargs):
        calls.append(label)
        assert label == '12-state' and deadline == 2500 and limit == 1500 and kwargs['tpu'] is True
        assert argv[2] == str(payload / 'cloud/state_coordinator.py')
        assert argv[argv.index('--state-probe') + 1] == str(payload / 'probe_state_roundtrip.py')
        result.update(process_token='wrong' if fault == 'token' else argv[argv.index('--process-token') + 1], deadline_epoch=deadline)
        (out / 'state').mkdir(); F.write(out / 'state/parent.json', result)
        return {'status': 'PASS', 'exit_code': 0, 'cleanup': {'errors': ['SYNTHETIC'] if fault == 'cleanup' else [],
                'groups': [{'pid': 7001, 'pgid': 7002 if fault == 'group-pgid' else 7001}]}}
    monkeypatch.setattr(remote, 'run_step', child)
    code = remote.execute(tmp_path, expected, 1000, 'state')
    top = json.loads((out / 'receipt.json').read_text())
    assert code == (0 if fault is None else 1)
    assert top['status'] == ('STATE_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED' if fault is None else 'BLOCKED')
    if fault is None: assert top['state_parent_sha256'] == remote.sha(out / 'state/parent.json')
    assert calls == ([] if fault == 'controls' else ['12-state'])


@pytest.mark.parametrize('fault', [None, 'numeric', 'source', 'checkpoint', 'launch-pid', 'truncated-controls', 'phase', 'part-bytes', 'stop'])
def test_full_state_owner_chain(state_packet, tmp_path, monkeypatch, fault):
    """Fake service, actual source packet and real synthetic saved-record verifier."""
    import transport
    write = F.write
    fixtures = load_device_fixtures()
    packet, manifest, expected = state_packet
    data = fixtures.make_device_fixture(tmp_path / 'fixture', admission=manifest['source_admission_sha256'])
    D = fixtures.D
    parent = D.read(data.actual / 'parent.json')
    parent.update(state_probe_sha256=manifest['state_probe_sha256'], state_helper_sha256=manifest['state_helper_sha256'],
                  source_pre=data.admission_sha256, source_post=data.admission_sha256,
                  runtime_lock_sha256=manifest['runtime_lock_sha256'], patch_manifest_sha256=manifest['patch_manifest_sha256'])
    if fault == 'launch-pid': parent['children'][1]['launched_pid'] = 999
    D.write(data.actual / 'parent.json', parent)
    if fault == 'numeric':
        path = data.actual / 'restore/raw' / (D.CASES[0]['id'] + '.json')
        raw = D.read(path); raw['outputs']['y']['values'][0] += 10; D.write(path, raw)
    elif fault == 'source':
        path = data.actual / 'restore/receipt.json'; raw = D.read(path); raw['source_post'] = '0'*64; D.write(path, raw)
    elif fault == 'checkpoint':
        name = D.CASES[0]['id']; replacement = b'INVALID CHECKPOINT CONTROL\n'
        for phase in ('save','restore'): (data.actual / phase / 'raw' / (name + '.state.pt')).write_bytes(replacement)
        path = data.actual / 'restore/raw' / (name + '.json')
        raw = D.read(path); raw['loaded_checkpoint_sha256'] = D.sha(data.actual / 'save/raw' / (name + '.state.pt')); D.write(path, raw)
    fixtures.reseal_fixture(data)
    records = tmp_path / 'remote-records'; records.mkdir()
    shutil.copytree(data.oracle, records / 'cpu-oracle'); shutil.copytree(data.actual, records / 'state')
    original_parent = (records / 'state/parent.json').read_bytes()
    admission_record = json.loads((packet / 'source-admission.json').read_text())
    controls = F.controls_fixture()
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
    receipt = {'status': 'STATE_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED', 'experiment': 'state-roundtrip', 'precision': 'highest',
               'packet_sha256': expected, 'manifest_sha256': remote.sha(packet / 'manifest.json'),
               'source_admission_sha256': manifest['source_admission_sha256'], 'runtime_lock_sha256': manifest['runtime_lock_sha256'],
               'runtime_status': 'PASS_TPU_RUNTIME_PROBE_ONLY', 'cpu_status': 'PASS', 'tpu_status': 'STATE_RECORDS_COMPLETE',
               'built_source_status': 'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS', 'installed_source_status': 'POST_PATCH_PYTHON_SOURCE_PASS',
               'source_controls_status': 'PASS_QUALIFIED_LINUX_SOURCE_CONTROLS', 'tpu_attempted': True, 'oracle_sha256': data.oracle_sha256,
               'built_wheels': [{'name': built['wheel_name'], 'sha256': built['wheel_sha256']}],
               'steps': [{'scope': 'SYNTHETIC_NO_DEVICE_EXECUTION', 'cleanup': {'errors': [], 'groups': []}}],
               **{field: manifest[field] for field in (*remote.TRANSFER_BINDINGS, *remote.STATE_BINDINGS)}}
    receipt['state_parent_sha256'] = remote.sha(records / 'state/parent.json')
    write(records / 'receipt.json', receipt)
    out = tmp_path / 'host'; identity = tmp_path / 'identity.json'; identity.write_text('{}')
    gate = {'status': 'ACTUAL_DISPATCH_AUTHORIZED', 'packet_sha256': expected, 'driver_sha256': remote.sha(HERE / 'owner.py'),
            'output': str(out.resolve()), 'plugin_source_manifest_sha256': manifest['plugin_source_manifest_sha256'],
            'runtime_lock_sha256': manifest['runtime_lock_sha256'], 'budget': manifest['budget'], 'one_allocation_only': True,
            'provider_or_solver': 'FORBIDDEN', 'cli_identity_sha256': remote.sha(identity), 'experiment': 'state-roundtrip',
            'precision': 'highest', **{field: manifest[field] for field in (*remote.TRANSFER_BINDINGS, *remote.STATE_BINDINGS)}}
    gate['state_scope'] = manifest['state_scope']
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
            assert '"remote.py",\'state\'' in Path(argv[-1]).read_text()
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
    assert (records / 'state/parent.json').read_bytes() == original_parent
    if fault == 'part-bytes': assert result.get('retrieval_error') and 'local_verifier_exit_code' not in result
    else:
        assert result['retrieval'] == 'COMPLETE_WHOLE_ARCHIVE' and result['whole_archive_verified'] is True
        assert (out / 'recovered/state/parent.json').read_bytes() == original_parent
        assert not result['local_verifier_cleanup']['errors']
        assert all(group['group_absence'] == 'OBSERVED_NO_SUCH_GROUP' for group in result['local_verifier_cleanup']['groups'])
    if fault == 'phase': assert 'local_verifier_exit_code' not in result
    elif fault != 'part-bytes':
        report = json.loads((out / 'verify.json').read_text())
        assert report['m3_status'] == 'NOT_QUALIFIED'
        if fault in {'source', 'checkpoint', 'launch-pid'}:
            assert report['record_validation'] == 'FAIL' and result['local_verifier_exit_code'] == 1
        else:
            numeric = 'FAIL' if fault == 'numeric' else 'PASS'
            assert report['record_validation'] == 'PASS' and report['numerical_status'] == numeric
            assert len(report['rows']) == 8
            assert result['local_verifier_exit_code'] == (0 if numeric == 'PASS' else 2)
    if fault in {None, 'numeric'}:
        assert result['status'] == ('FAIL' if fault else 'PASS') + '_TPU_STATE_RECORDS'
        assert result['record_validation'] == 'PASS'
    else: assert result['status'] not in {'PASS_TPU_STATE_RECORDS', 'FAIL_TPU_STATE_RECORDS'}
    assert result['api42_status'] == 'NOT_REPEATED' and result['m3_status'] == 'NOT_QUALIFIED'
    if fault == 'stop': assert result['status'] == 'BLOCKED_CLEANUP'
