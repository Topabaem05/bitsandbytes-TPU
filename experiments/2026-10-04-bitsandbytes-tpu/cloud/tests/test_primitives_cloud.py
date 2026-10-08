"""Bounded primitive CPU recovery controls. All records are isolated and synthetic."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import primitive_contract as PC


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


def cpu_records(base):
    output = base / 'records'
    output.mkdir()
    write(output / 'receipt.json', {'scope': 'SYNTHETIC_ARCHIVE_ONLY', 'status': 'CPU_ORACLE_READY_TPU_NOT_RUN'})
    for name in PC.REQUIRED_PROOFS:
        write(output / name, {'scope': 'SYNTHETIC_ARCHIVE_ONLY', 'name': name})
    for name in ('oracle-seal.json', 'inputs.json', 'outputs.json', 'cpu-evidence.json'):
        write(output / 'cpu-oracle' / name, {'scope': 'SYNTHETIC_ARCHIVE_ONLY', 'name': name})
    for name in ('ownership.json', 'result.json', 'cleanup.json'):
        write(output / 'steps/11-cpu-oracle' / name, {'scope': 'SYNTHETIC_ARCHIVE_ONLY'})
    for name in ('stdout.raw', 'stderr.raw'):
        (output / 'steps/11-cpu-oracle' / name).write_text('SYNTHETIC_ARCHIVE_ONLY\n')
    receipt = PC.cpu_archive(output)
    recovered = base / 'transport'
    recovered.mkdir()
    (recovered / 'evidence.zip').write_bytes((output / 'primitive-cpu-evidence.zip').read_bytes())
    (recovered / 'inventory.json').write_bytes((output / 'primitive-cpu-inventory.json').read_bytes())
    return output, recovered, receipt


def zip_entries(root, entries):
    with zipfile.ZipFile(root / 'evidence.zip', 'w') as archive:
        for name, data, attributes in entries:
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = attributes
            archive.writestr(info, data)


@pytest.mark.parametrize('fault', [None, 'missing', 'extra', 'duplicate', 'bytes', 'type', 'traversal',
                                   'archive-sha', 'archive-size', 'inventory-sha', 'bad-size', 'bad-sha'])
def test_cpu_archive_complete_regular_member_recovery(tmp_path, fault):
    output, root, receipt = cpu_records(tmp_path)
    with zipfile.ZipFile(root / 'evidence.zip') as archive:
        entries = [(row.filename, archive.read(row), row.external_attr) for row in archive.infolist()]
    inventory = PC.read(root / 'inventory.json')
    first = entries[0][0]
    if fault == 'missing':
        entries = entries[1:]
    elif fault == 'extra':
        entries.append(('UNREVIEWED.json', b'{}', 0o100644 << 16))
    elif fault == 'duplicate':
        entries.append(entries[0])
    elif fault == 'bytes':
        entries[0] = (first, b'CHANGED', entries[0][2])
    elif fault == 'type':
        entries[0] = (first, entries[0][1], 0o120777 << 16)
    elif fault == 'traversal':
        name = '../outside.json'
        entries[0] = (name, entries[0][1], entries[0][2])
        inventory[name] = inventory.pop(first)
    elif fault == 'bad-size':
        inventory[first]['bytes'] = True
    elif fault == 'bad-sha':
        inventory[first]['sha256'] = 'NOT-A-HASH'
    if fault in {'missing', 'extra', 'duplicate', 'bytes', 'type', 'traversal'}:
        zip_entries(root, entries)
    PC.write(root / 'inventory.json', inventory)
    receipt.update(primitive_cpu_inventory_sha256=PC.sha(root / 'inventory.json'),
                   primitive_cpu_evidence_sha256=PC.sha(root / 'evidence.zip'),
                   primitive_cpu_evidence_bytes=(root / 'evidence.zip').stat().st_size)
    if fault == 'archive-sha':
        receipt['primitive_cpu_evidence_sha256'] = 'f' * 64
    elif fault == 'archive-size':
        receipt['primitive_cpu_evidence_bytes'] += 1
    elif fault == 'inventory-sha':
        receipt['primitive_cpu_inventory_sha256'] = 'f' * 64
    if fault is None:
        recovered = PC.recover_cpu(root, receipt)
        assert {path.relative_to(recovered).as_posix(): PC.sha(path) for path in recovered.rglob('*') if path.is_file()} == {
            name: row['sha256'] for name, row in inventory.items()}
        assert PC.read(recovered / 'primitive-cpu-receipt.json')['scope'] == 'SYNTHETIC_ARCHIVE_ONLY'
    else:
        with pytest.raises(ValueError, match='PRIMITIVE_CPU_'):
            PC.recover_cpu(root, receipt)
    assert not (tmp_path / 'outside.json').exists()


def test_cpu_archive_refuses_missing_launch_and_symlink(tmp_path):
    output, root, receipt = cpu_records(tmp_path)
    (output / 'steps/11-cpu-oracle/ownership.json').unlink()
    with pytest.raises(ValueError, match='PRIMITIVE_CPU_LAUNCH_RECORDS'):
        PC.cpu_archive(output)
    (output / 'steps/11-cpu-oracle/ownership.json').symlink_to(output / 'cpu-oracle/oracle-seal.json')
    with pytest.raises(ValueError, match='PRIMITIVE_CPU_SYMLINK'):
        PC.cpu_archive(output)


def cpu_report():
    return {'record_validation': 'PASS', 'oracle_status': 'VERIFIED', 'oracle_sha256': 'a' * 64,
            'row_count': 32, 'mean_case_count': 28, 'output_count': 206,
            **{key: 'NOT_QUALIFIED' for key in ('api42_status', 'm3_status', 'm4_status', 'm5_status')}}


@pytest.mark.parametrize('fault', [None, 'truncated', 'qualified', 'wrong-oracle', 'boolean-count'])
def test_cpu_report_cannot_promote_labels_or_incomplete_inventory(fault):
    report = cpu_report()
    if fault == 'truncated': report['output_count'] -= 1
    if fault == 'qualified': report['m4_status'] = 'PASS'
    if fault == 'wrong-oracle': report['oracle_sha256'] = 'b' * 64
    if fault == 'boolean-count': report['row_count'] = True
    if fault is None:
        assert PC.validate_cpu_report(report, 'a' * 64)
    else:
        with pytest.raises(ValueError, match='PRIMITIVE_CPU_INDEPENDENT_AUDIT'):
            PC.validate_cpu_report(report, 'a' * 64)


@pytest.mark.parametrize('fault', [None, 'extra', 'oracle', 'archive', 'qualified', 'cpu-audit'])
def test_root_gate_is_exact_and_requires_independent_cpu_audit_digest(fault):
    receipt = {'oracle_sha256': 'a' * 64, 'packet_sha256': 'b' * 64, 'manifest_sha256': 'c' * 64,
               'primitive_cpu_inventory_sha256': 'd' * 64, 'primitive_cpu_evidence_sha256': 'e' * 64,
               'primitive_cpu_evidence_bytes': 123}
    manifest = {'source_admission_sha256': 'f' * 64, 'runtime_lock_sha256': '1' * 64,
                **{key: '2' * 64 for key in PC.BINDINGS}}
    gate = {'kind': 'ROOT_PRIMITIVE_CPU_ORACLE_GATE', 'status': 'QUALIFIED_LINUX_CPU_ORACLE_VERIFIED',
            **receipt, **manifest, 'source_variant': 'nested-v1', 'primitive_variant': PC.VARIANT,
            'cpu_observations_sha256': '3' * 64,
            **{key: 'NOT_QUALIFIED' for key in ('api42_status', 'm3_status', 'm4_status', 'm5_status')}}
    if fault == 'extra': gate['manual_override'] = True
    if fault == 'oracle': gate['oracle_sha256'] = '0' * 64
    if fault == 'archive': gate['primitive_cpu_evidence_bytes'] += 1
    if fault == 'qualified': gate['m4_status'] = 'PASS'
    if fault == 'cpu-audit': del gate['cpu_observations_sha256']
    if fault is None:
        assert PC.verify_root_gate(gate, receipt, manifest) == gate
    else:
        with pytest.raises(ValueError, match='PRIMITIVE_ROOT_CPU_GATE'):
            PC.verify_root_gate(gate, receipt, manifest)


@pytest.mark.parametrize('fault', [None, 'numeric-fail', 'native-unsupported', 'qualified', 'truncated',
                                   'duplicate', 'missing-gate', 'forged-aggregate', 'forged-gate'])
def test_observation_status_and_complete_inventory_are_distinct(tmp_path, monkeypatch, fault):
    """Isolated contract test, not scientific or Linux execution evidence."""
    from types import SimpleNamespace
    data = {'mean_cases': [{'case_id': str(i)} for i in range(28)]}
    ids = [('native-view', 'native-bitcast'), *[('builder', name) for name in
        ('host-float-bits', 'device-int-float', 'float-arithmetic', *['mean-' + str(i) for i in range(28)])]]
    expected = {name: {'float': {}} if name == 'device-int-float' else {'bits': {}} for _, name in ids}
    ref = SimpleNamespace(expected=lambda data: expected)
    module = SimpleNamespace(spec=lambda B, N: data, pure=lambda: (ref, None),
                             source_inputs=lambda Ref, data, name, materialized: {'input': {}})
    monkeypatch.setattr(PC, 'science', lambda packet, admission: (module, None, (None,) * 6))
    write(tmp_path / 'manifest.json', {'source_admission_sha256': 'a' * 64})
    gate = {'status': 'PASS', 'actual_bytes_sha256': 'b' * 64, 'expected_bytes_sha256': 'b' * 64}
    rows = [{'phase': phase, 'case_id': name, 'status': 'PASS',
             'gates': {next(iter(expected[name])): copy.deepcopy(gate), 'parameter_input': copy.deepcopy(gate)}}
            for phase, name in ids]
    report = {'record_validation': 'PASS', 'phase_ids': ['native-view', 'builder'], 'row_count': 32,
              'rows': rows, 'gate_count': 64, 'numerical_status': 'PASS', 'builder_bit_status': 'PASS',
              'native_view_status': 'SUPPORTED', 'cpu_oracle': cpu_report(),
              **{key: 'NOT_QUALIFIED' for key in ('api42_status', 'm3_status', 'm4_status', 'm5_status')}}
    if fault == 'numeric-fail':
        rows[1]['gates']['bits'].update(status='FAIL', actual_bytes_sha256='c' * 64)
        rows[1]['status'] = 'FAIL'
        report.update(numerical_status='FAIL', builder_bit_status='FAIL')
    if fault == 'native-unsupported':
        rows[0].update(status='UNSUPPORTED', gates={})
        report.update(native_view_status='UNSUPPORTED', gate_count=62)
    if fault == 'qualified': report['m4_status'] = 'PASS'
    if fault == 'truncated': rows.pop()
    if fault == 'duplicate': rows[-1] = copy.deepcopy(rows[-2])
    if fault == 'missing-gate': del rows[1]['gates']['parameter_input']
    if fault == 'forged-aggregate': report['numerical_status'] = 'FAIL'
    if fault == 'forged-gate': rows[1]['gates']['bits']['status'] = 'FAIL'
    if fault in {None, 'numeric-fail', 'native-unsupported'}:
        assert PC.validate_report(report, tmp_path)
        assert report['m4_status'] == 'NOT_QUALIFIED'
    else:
        with pytest.raises(ValueError, match='PRIMITIVE_'):
            PC.validate_report(report, tmp_path)


@pytest.mark.parametrize('fault', [None, 'field', 'payload', 'scope'])
def test_older_variant_rejects_primitive_admission(fault):
    manifest = {'experiment': 'state-roundtrip', 'files': {'probe_nested_state.py': {'sha256': 'a' * 64}}}
    if fault == 'field': manifest['primitive_probe_sha256'] = 'b' * 64
    if fault == 'payload': manifest['files']['primitive_kernels.py'] = {'sha256': 'c' * 64}
    if fault == 'scope': manifest['primitive_scope'] = PC.SCOPE
    if fault is None:
        PC.verify_manifest(manifest)
    else:
        with pytest.raises(ValueError, match='PRIMITIVE_NOT_REQUESTED'):
            PC.verify_manifest(manifest)
