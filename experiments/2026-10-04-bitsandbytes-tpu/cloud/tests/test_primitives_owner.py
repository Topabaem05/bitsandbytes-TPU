"""Actual pure verifiers and owner/archive path over synthetic Linux-shaped records.

These fixtures make no provider calls and do not establish Linux or TPU qualification.
"""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import zipfile

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import build_packet as B
import owner
import primitive_contract as PC
import remote
import nested_contract as N
import transport
import test_transfer_cloud as F
import test_nested_cloud as NF

SCIENCE = HERE.parent
base_archive = F.base_archive


def tools():
    spec = importlib.util.spec_from_file_location('portable_primitive_saved', HERE / 'tests/primitive_saved_fixture.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def primitive_args():
    args = NF.nested_args()
    args.update(experiment=PC.MODE, primitive_source=SCIENCE,
                primitive_manifest=SCIENCE / 'primitive-manifest.json', primitive_manifest_sha256=PC.MANIFEST_SHA,
                primitive_state_probe=SCIENCE / 'probe_nested_state.py', primitive_state_probe_sha256=PC.STATE_SHA)
    return args


@pytest.fixture
def primitive_packet(tmp_path, base_archive, monkeypatch):
    return NF.build_fixture(tmp_path, base_archive, monkeypatch, args=primitive_args())


def records_fixture(base, packet, manifest, expected):
    T = tools()
    f = T.make_fixture(base / 'scientific-fixture', admission=manifest['source_admission_sha256'],
                       admission_path=packet / 'source-admission.json')
    records = base / 'records'
    records.mkdir()
    shutil.copytree(f.oracle, records / 'cpu-oracle')
    shutil.copytree(f.actual, records / 'primitives')
    admission = PC.read(packet / 'source-admission.json')
    F.write(records / 'source-controls.json', F.controls_fixture())
    installed = {'status': 'POST_PATCH_PYTHON_SOURCE_PASS', 'source_admission_sha256': manifest['source_admission_sha256'],
                 'patch_manifest_sha256': manifest['patch_manifest_sha256'],
                 'installed_python_files': {key: admission[key]['files'] for key in ('bitsandbytes', 'bitsandbytes_tpu')},
                 'source_variant': 'nested-v1', 'nested_source_sha256': N.NESTED_SOURCE_SHA,
                 'plugin_source_manifest_sha256': N.NESTED_PLUGIN_MANIFEST_SHA}
    F.write(records / 'installed-source.json', installed)
    built = {'status': 'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS', 'patch_manifest_sha256': manifest['patch_manifest_sha256'],
             'python_files': admission['bitsandbytes']['files'], 'wheel_name': 'SYNTHETIC-upstream.whl', 'wheel_sha256': '4' * 64}
    plugin = {'status': 'NESTED_PLUGIN_WHEEL_PYTHON_SOURCE_PASS', 'python_files': N.NESTED_FILES,
              'plugin_source_manifest_sha256': N.NESTED_PLUGIN_MANIFEST_SHA, 'nested_source_sha256': N.NESTED_SOURCE_SHA,
              'wheel_name': 'SYNTHETIC-plugin.whl', 'wheel_sha256': '5' * 64}
    F.write(records / 'built-source.json', built)
    F.write(records / 'built-plugin-source.json', plugin)
    locked = {row['name']: row for row in PC.read(packet / 'runtime/requirements.lock.json')['wheels']}
    locked.update({row['name']: row for row in PC.read(packet / 'build-requirements.lock.json')['wheels']})
    F.write(records / 'installed-metadata.json', {'python': '3.12.14', 'executable': '/content/bnb-tpu-first/venv/bin/python',
            'packages': {name: {'version': row['version'], 'metadata_sha256': row['metadata_sha256'], 'wheel_sha256': row['sha256']}
                         for name, row in locked.items()}})
    F.write(records / 'runtime-probe.json', {'status': 'PASS_TPU_RUNTIME_PROBE_ONLY', 'runtime_executed': True,
            'observed_backend': 'TPU', 'error': None, 'python': '3.12.14', 'system': 'Linux', 'machine': 'x86_64',
            'versions': {name: locked[name]['version'] for name in ('torch', 'torch-xla', 'libtpu', 'jax', 'jaxlib')}})
    seal = PC.read(records / 'cpu-oracle/oracle-seal.json')
    payload = Path('/content/bnb-tpu-first/payload')
    argv = ['/content/bnb-tpu-first/venv/bin/python', '-B', str(payload / 'probe_primitives.py'), 'prepare',
            '--admission', str(payload / 'source-admission.json'), '--admission-sha256', manifest['source_admission_sha256'],
            '--output', '/content/bnb-tpu-first/records/cpu-oracle', *[str(x) for x in PC.cli(payload)],
            '--process-token', seal['process_token']]
    cleanup = {'pid': seal['pid'], 'pgid': seal['pgid'], 'status': 'CLEANUP_VERIFIED', 'leader_reaped': True,
               'group_absence': 'OBSERVED_NO_SUCH_GROUP', 'errors': [], 'exit_status': 0}
    cpu_owner = {'pid': seal['pid'], 'pgid': seal['pgid'], 'owner_pid': seal['parent_pid'], 'argv': argv,
                 'cleanup': cleanup, 'exit_status': 0}
    summary = {'errors': [], 'groups': [cleanup], 'restoration': {'handlers_restored': True, 'timer_restored': True}}
    (records / 'steps/11-cpu-oracle').mkdir(parents=True)
    F.write(records / 'steps/11-cpu-oracle/ownership.json', cpu_owner)
    F.write(records / 'steps/11-cpu-oracle/result.json', {'status': 'PASS', 'exit_code': 0, 'argv': argv,
            'cleanup': summary, 'environment': {'set': {}}})
    F.write(records / 'steps/11-cpu-oracle/cleanup.json', summary)
    for name in ('stdout.raw', 'stderr.raw'):
        (records / 'steps/11-cpu-oracle' / name).write_text('SYNTHETIC_NO_LINUX_EXECUTION\n')
    receipt = {'status': 'CPU_ORACLE_READY_TPU_NOT_RUN', 'experiment': PC.MODE, 'precision': 'highest',
               'primitive_variant': PC.VARIANT, 'primitive_scope': PC.SCOPE, 'primitive_diagnostic_only': True,
               'nested_scope': N.NESTED_SCOPE, 'nested_source_variant': 'nested-v1',
               'packet_sha256': expected, 'manifest_sha256': PC.sha(packet / 'manifest.json'),
               'source_admission_sha256': manifest['source_admission_sha256'], 'runtime_lock_sha256': manifest['runtime_lock_sha256'],
               'runtime_status': 'PASS_TPU_RUNTIME_PROBE_ONLY', 'cpu_status': 'PASS', 'tpu_status': 'NOT_RUN', 'tpu_attempted': False,
               'built_source_status': built['status'], 'installed_source_status': installed['status'],
               'built_plugin_source_status': plugin['status'], 'source_controls_status': 'PASS_QUALIFIED_LINUX_SOURCE_CONTROLS',
               'oracle_sha256': f.oracle_sha256, 'built_wheels': [{'name': row['wheel_name'], 'sha256': row['wheel_sha256']}
                   for row in (built, plugin)], 'steps': [{'scope': 'SYNTHETIC', 'cleanup': summary}],
               **{key: manifest[key] for key in (*remote.TRANSFER_BINDINGS, *N.NESTED_BINDINGS, *PC.BINDINGS)}}
    return records, receipt, f, T


@pytest.mark.parametrize('fault', [None, 'numeric', 'native-unsupported', 'cpu-pid', 'cpu-open-group', 'cpu-source',
                                   'cpu-wrong-bits', 'builder-fallback', 'hlo-missing', 'parent-token', 'member-missing',
                                   'member-changed', 'stop'])
def test_full_primitive_owner_host_gate_archive_verifier(primitive_packet, tmp_path, monkeypatch, fault):
    packet, manifest, expected = primitive_packet
    records, cpu_receipt, f, T = records_fixture(tmp_path, packet, manifest, expected)
    row = f.actual / 'builder/raw/float-arithmetic.json'
    if fault == 'numeric':
        raw = T.B.read(row); raw['outputs']['add_zero'] = T.REF.array([0] * 20, [20], 'int32'); T.B.write(row, raw)
    if fault == 'builder-fallback':
        raw = T.B.read(row); raw['counters'] = {'aten::view': 1}; T.B.write(row, raw)
    if fault == 'hlo-missing': (f.actual / 'builder/raw/float-arithmetic.hlo.txt').unlink()
    if fault == 'parent-token':
        parent = T.B.read(f.actual / 'receipt.json'); parent['process_token'] = 'a' * 32; T.B.write(f.actual / 'receipt.json', parent)
    if fault == 'native-unsupported':
        raw_path = f.actual / 'native-view/raw/native-bitcast.json'; raw = T.B.read(raw_path)
        raw.update(status='OBSERVED_UNSUPPORTED', reason='NATIVE_VIEW_EXCEPTION', outputs={}, error_type='RuntimeError',
                   error_message='SYNTHETIC_NATIVE_UNSUPPORTED', traceback='SYNTHETIC_TRACEBACK'); T.B.write(raw_path, raw)
        parent = T.B.read(f.actual / 'receipt.json'); parent['children'][0]['exit_code'] = 2; T.B.write(f.actual / 'receipt.json', parent)
    if fault == 'cpu-source':
        seal = T.B.read(f.oracle / 'oracle-seal.json'); seal['source_post'] = '0' * 64; T.B.write(f.oracle / 'oracle-seal.json', seal)
    if fault == 'cpu-wrong-bits':
        outputs = T.B.read(f.oracle / 'outputs.json'); outputs['host-float-bits']['bits'] = T.REF.array([0] * 20, [20], 'int32')
        T.B.write(f.oracle / 'outputs.json', outputs)
    T.reseal(f, oracle=fault in {'cpu-source', 'cpu-wrong-bits'})
    shutil.rmtree(records / 'primitives'); shutil.copytree(f.actual, records / 'primitives')
    shutil.rmtree(records / 'cpu-oracle'); shutil.copytree(f.oracle, records / 'cpu-oracle')
    cpu_receipt['oracle_sha256'] = f.oracle_sha256
    if fault in {'cpu-pid', 'cpu-open-group'}:
        p = records / 'steps/11-cpu-oracle/ownership.json'; value = PC.read(p)
        if fault == 'cpu-pid': value['pid'] = value['pgid'] = 99999
        else: value['cleanup']['group_absence'] = 'UNVERIFIED'
        F.write(p, value)
    out = tmp_path / 'host'; identity = tmp_path / 'identity.json'; F.write(identity, {})
    gate = {'status': 'ACTUAL_DISPATCH_AUTHORIZED', 'packet_sha256': expected, 'driver_sha256': PC.sha(HERE / 'owner.py'),
            'output': str(out.resolve()), 'plugin_source_manifest_sha256': manifest['plugin_source_manifest_sha256'],
            'runtime_lock_sha256': manifest['runtime_lock_sha256'], 'budget': manifest['budget'], 'one_allocation_only': True,
            'provider_or_solver': 'FORBIDDEN', 'cli_identity_sha256': PC.sha(identity), 'experiment': PC.MODE, 'precision': 'highest',
            'nested_scope': N.NESTED_SCOPE, 'nested_source_variant': 'nested-v1', 'primitive_scope': PC.SCOPE,
            'primitive_variant': PC.VARIANT, 'primitive_diagnostic_only': True,
            **{key: manifest[key] for key in (*remote.TRANSFER_BINDINGS, *N.NESTED_BINDINGS, *PC.BINDINGS)}}
    acceptance = tmp_path / 'gate.json'; F.write(acceptance, gate)
    monkeypatch.setattr(owner, 'verify_cli_identity', lambda *args: None)
    calls = []; sessions = []; final_receipt = copy.deepcopy(cpu_receipt)
    def api(label, argv, timeout):
        calls.append(label)
        if label == '03-one-allocation':
            sessions.append(argv[argv.index('-s') + 1]); epoch = PC.read(out / 'owner.json')['allocation_epoch']
            cpu_receipt['allocation_epoch'] = final_receipt['allocation_epoch'] = epoch
            F.write(records / 'receipt.json', cpu_receipt); cpu_receipt.update(PC.cpu_archive(records))
            transport.split(records / 'primitive-cpu-evidence.zip', tmp_path / 'primitive-cpu-parts',
                            expected_sha256=cpu_receipt['primitive_cpu_evidence_sha256'], expected_bytes=cpu_receipt['primitive_cpu_evidence_bytes'])
            final_receipt.update(cpu_receipt)
        if label in ('04-sessions-after', '90-before-stop'):
            return {'status': 'PASS'}, '[' + sessions[0] + ']\nHardware: V6E1'
        if label == '13-tpu':
            assert "'primitives'" in Path(argv[-1]).read_text()
            final_receipt.update(status='PRIMITIVE_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED', tpu_status='PRIMITIVE_RECORDS_COMPLETE',
                                 tpu_attempted=True, primitive_root_gate_sha256=PC.sha(out / 'launch.json'),
                                 primitive_receipt_sha256=PC.sha(records / 'primitives/receipt.json'))
            F.write(records / 'receipt.json', final_receipt)
        if label == '21-export':
            F.write(records / 'receipt.json', final_receipt); exported = remote.package(records)
            if fault in {'member-missing', 'member-changed'}:
                archive = records / 'evidence.zip'; replacement = records / 'changed.zip'; target = 'primitives/builder/raw/float-arithmetic.json'
                with zipfile.ZipFile(archive) as original, zipfile.ZipFile(replacement, 'w') as changed:
                    for info in original.infolist():
                        if info.filename == target and fault == 'member-missing': continue
                        body = original.read(info)
                        if info.filename == target and fault == 'member-changed': body += b'SYNTHETIC_CORRUPTION'
                        changed.writestr(info, body)
                replacement.replace(archive); exported.update(evidence_sha256=PC.sha(archive), evidence_bytes=archive.stat().st_size)
            F.write(records / 'receipt.json', {**final_receipt, **exported})
        if label == '24-export-parts':
            transport.split(records / 'evidence.zip', tmp_path / 'result-parts',
                            expected_sha256=PC.sha(records / 'evidence.zip'), expected_bytes=(records / 'evidence.zip').stat().st_size)
        if argv[0] == 'download':
            destination = Path(argv[-1])
            if label in {'09-install-receipt', '10-cpu-receipt'}:
                phase = copy.deepcopy(cpu_receipt)
                if label == '09-install-receipt': phase['status'] = 'INSTALLED_NOT_QUALIFIED'
                F.write(destination, phase)
            else:
                rel = argv[-2].removeprefix('content/bnb-tpu-first/')
                source = records / rel.removeprefix('records/') if rel.startswith('records/') else tmp_path / rel
                shutil.copyfile(source, destination)
        if fault == 'stop' and label == '91-stop-exact': return {'status': 'CLI_FAILED'}, 'SYNTHETIC_STOP_REFUSAL'
        return {'status': 'PASS'}, 'Active assignments: 0\nUsage rate: 0.00/hr' if argv == ['usage'] else 'No active sessions found on server.'
    result = owner.drive(packet, out, expected, acceptance, PC.sha(acceptance), Path('SYNTHETIC_PYTHON'), identity, api=api, simulated=True)
    assert result['mode'] == 'SIMULATED_NO_CLOUD'
    assert result['allocation_attempts'] == 1
    assert calls[-4:] == ['90-before-stop', '91-stop-exact', '92-after-stop', '93-usage-after']
    if fault in {'cpu-pid', 'cpu-open-group', 'cpu-source', 'cpu-wrong-bits'}:
        assert '13-tpu' not in calls
        assert result['status'] != 'PASS_PRIMITIVE_RECORDS'
        assert result['local_verifier_status'] == 'NOT_RUN_INCOMPLETE_PRIMITIVE_RECORDS'
        assert 'local_readback_error' not in result and 'local_verifier_exit_code' not in result
        assert result.get('original_error')
    elif fault in {None, 'numeric', 'native-unsupported'}:
        assert result['status'] == 'PASS_PRIMITIVE_RECORDS'
        report = PC.read(out / 'verify.json')
        assert report['record_validation'] == 'PASS' and report['m4_status'] == 'NOT_QUALIFIED'
        assert report['numerical_status'] == ('FAIL' if fault == 'numeric' else 'PASS')
        assert result['whole_archive_verified'] is True
    else:
        assert result['status'] != 'PASS_PRIMITIVE_RECORDS'
    if fault == 'stop': assert result['status'] == 'BLOCKED_CLEANUP'


@pytest.mark.parametrize('fault', [None, 'parent-pid', 'group', 'source', 'helper', 'token', 'deadline',
                                   'oracle', 'cleanup', 'gate', 'wrong-phase'])
def test_remote_primitive_exact_parent_launch_and_phase(primitive_packet, tmp_path, monkeypatch, fault):
    """Run the actual phase coordinator with an isolated owned-child record boundary."""
    packet, manifest, expected = primitive_packet
    records, receipt, fixture, T = records_fixture(tmp_path, packet, manifest, expected)
    packet.rename(tmp_path / 'payload'); payload = tmp_path / 'payload'
    shutil.copyfile(payload / 'payload.zip', tmp_path / 'payload.zip')
    receipt.update(allocation_epoch=1000, science_deadline_epoch=3000, error=None,
                   primitive_cpu_inventory_sha256='a' * 64, primitive_cpu_evidence_sha256='b' * 64,
                   primitive_cpu_evidence_bytes=123)
    gate = {'kind': 'ROOT_PRIMITIVE_CPU_ORACLE_GATE', 'status': 'QUALIFIED_LINUX_CPU_ORACLE_VERIFIED',
            'oracle_sha256': receipt['oracle_sha256'], 'source_admission_sha256': manifest['source_admission_sha256'],
            'runtime_lock_sha256': manifest['runtime_lock_sha256'], 'source_variant': 'nested-v1',
            'primitive_variant': PC.VARIANT, 'packet_sha256': expected, 'manifest_sha256': PC.sha(payload / 'manifest.json'),
            'cpu_observations_sha256': 'c' * 64,
            **{key: manifest[key] for key in PC.BINDINGS},
            **{key: receipt[key] for key in ('primitive_cpu_inventory_sha256', 'primitive_cpu_evidence_sha256', 'primitive_cpu_evidence_bytes')},
            **{key: 'NOT_QUALIFIED' for key in ('api42_status', 'm3_status', 'm4_status', 'm5_status')}}
    if fault == 'gate': gate['cpu_observations_sha256'] = 'NOT-A-HASH'
    F.write(tmp_path / 'launch.json', gate); F.write(records / 'receipt.json', receipt)
    monkeypatch.setattr(remote.time, 'time', lambda: 1000)
    calls = []
    def child(output, label, argv, deadline, limit, **kwargs):
        calls.append(label)
        assert label == '12-primitives' and limit == 1500 and deadline == 2500 and kwargs['tpu'] is True
        assert argv[2:4] == [str(payload / 'probe_primitives.py'), 'execute']
        token = argv[argv.index('--process-token') + 1]
        assert len(token) == 32 and argv[argv.index('--deadline-epoch') + 1] == '2500'
        M, args, helpers = PC.science(payload, manifest['source_admission_sha256'])
        B, R, P, A, nested, S = helpers
        result = {**M.binding(B, nested, args), 'status': 'COMPLETE', 'pid': 7001, 'pgid': 7001, 'parent_pid': 999,
                  'process_token': token, 'deadline_epoch': deadline, 'oracle_sha256': gate['oracle_sha256'],
                  'source_pre': manifest['source_admission_sha256'], 'source_post': manifest['source_admission_sha256']}
        changes = {'parent-pid': ('parent_pid', 998), 'group': ('pgid', 7002), 'source': ('source_post', '0' * 64),
                   'helper': ('helper_sha256', {}), 'token': ('process_token', '0' * 32),
                   'deadline': ('deadline_epoch', 3000), 'oracle': ('oracle_sha256', '0' * 64)}
        if fault in changes: result.update([changes[fault]])
        F.write(records / 'primitives/receipt.json', result)
        (records / 'steps/12-primitives').mkdir(parents=True)
        F.write(records / 'steps/12-primitives/ownership.json', {'owner_pid': 999})
        return {'status': 'PASS', 'exit_code': 0,
                'cleanup': {'errors': ['SYNTHETIC_CLOSE_FAILURE'] if fault == 'cleanup' else [],
                            'groups': [{'pid': 7001, 'pgid': 7001}]}}
    monkeypatch.setattr(remote, 'run_step', child)
    code = remote.execute(tmp_path, expected, 1000, 'nested' if fault == 'wrong-phase' else 'primitives')
    rec = PC.read(records / 'receipt.json')
    assert code == (0 if fault is None else 1)
    assert rec['status'] == ('PRIMITIVE_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED' if fault is None else 'BLOCKED')
    assert calls == ([] if fault in {'gate', 'wrong-phase'} else ['12-primitives'])


@pytest.mark.parametrize('fault', [None, 'pin', 'missing', 'source', 'variant', 'scope', 'wrong-experiment'])
def test_primitive_source_protocol_is_explicit_and_immutable(primitive_packet, fault):
    packet, manifest, expected = primitive_packet
    if fault == 'pin': manifest['primitive_reference_sha256'] = '0' * 64
    if fault == 'missing': manifest['files'].pop('primitive_kernels.py')
    if fault == 'variant': manifest['primitive_variant'] = 'unreviewed-v2'
    if fault == 'scope': manifest['primitive_diagnostic_only'] = False
    if fault == 'wrong-experiment': manifest['experiment'] = 'nested-79'
    if fault == 'source': (packet / 'primitive-inputs.json').write_bytes(b'UNREVIEWED_INPUTS')
    if fault is None:
        assert owner.preflight(packet, expected)['experiment'] == PC.MODE
        PC.payload(packet, manifest)
    else:
        with pytest.raises(ValueError, match='PRIMITIVE_'):
            PC.payload(packet, manifest)


def test_exact_primitive_packet_build_is_deterministic(tmp_path, base_archive, monkeypatch):
    left = tmp_path / 'left'; right = tmp_path / 'right'; left.mkdir(); right.mkdir()
    a = NF.build_fixture(left, base_archive, monkeypatch, args=primitive_args())
    b = NF.build_fixture(right, base_archive, monkeypatch, args=primitive_args())
    assert a[2] == b[2]
    assert (a[0] / 'payload.zip').read_bytes() == (b[0] / 'payload.zip').read_bytes()
    assert (a[0] / 'manifest.json').read_bytes() == (b[0] / 'manifest.json').read_bytes()


@pytest.mark.parametrize('mode', ['old', 'fixed', 'cached-transport'])
def test_primitive_cpu_archive_split_real_fresh_runpy(primitive_packet, tmp_path, monkeypatch, mode):
    from cleanup_lifecycle import Ownership
    packet, manifest, expected = primitive_packet
    records, receipt, fixture, T = records_fixture(tmp_path, packet, manifest, expected)
    if mode == 'old':
        source = packet / 'cloud/remote.py'; text = source.read_text()
        text = text.replace("                _transport.split(out/'primitive-cpu-evidence.zip'", "                import transport\n                transport.split(out/'primitive-cpu-evidence.zip'")
        source.write_text(text); manifest['files']['cloud/remote.py'] = {'sha256': PC.sha(source), 'bytes': source.stat().st_size}
        expected = F.zip_packet(packet, manifest)
    packet.rename(tmp_path / 'payload'); payload = tmp_path / 'payload'
    shutil.copytree(payload / 'cloud', tmp_path / 'control')
    shutil.copyfile(payload / 'payload.zip', tmp_path / 'payload.zip')
    receipt.update(status='INSTALLED_NOT_QUALIFIED', installation_status='PASS', cpu_status='NOT_RUN',
                   allocation_epoch=__import__('time').time(), steps=[], error=None, manifest_sha256=PC.sha(payload / 'manifest.json'), packet_sha256=expected)
    F.write(records / 'receipt.json', receipt)
    neutral = tmp_path / 'neutral'; neutral.mkdir()
    (tmp_path / 'runpy-process').mkdir()
    guard = Ownership(tmp_path / 'runpy-process')
    streams = [(tmp_path / name).open('wb') for name in ('runpy.stdout', 'runpy.stderr')]
    try:
        with guard.guard(20):
            argv = [sys.executable, '-I', '-B', str(HERE / 'tests/primitive_cpu_runpy_child.py'), '--base', str(tmp_path)]
            if mode == 'cached-transport': argv.append('--cached-transport')
            child = guard.launch(argv, cwd=neutral, stdout=streams[0], stderr=streams[1], record=tmp_path / 'runpy-process/ownership.json')
            child.wait()
    finally:
        for stream in streams: stream.close()
    assert child.returncode == 0, (tmp_path / 'runpy.stderr').read_text()
    assert not guard.summary()['errors']
    result = PC.read(tmp_path / 'fresh-runpy-result.json')
    assert result['pid'] == result['pgid'] == child.pid and result['provider_calls'] == 0
    if mode == 'old':
        assert result['remote_exit_code'] == 1 and result['receipt_status'] == 'BLOCKED'
        assert result['receipt_error']['type'] == 'ModuleNotFoundError' and "No module named 'transport'" in result['receipt_error']['message']
        assert result['parts_present'] is False
    else:
        assert result['remote_exit_code'] == 0 and result['receipt_status'] == 'CPU_ORACLE_READY_TPU_NOT_RUN'
        assert result['parts_present'] is True
