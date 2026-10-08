"""Explicit arithmetic diagnostic source and independent recovered CPU gate."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
from types import SimpleNamespace
import zipfile

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('_primitive_admitted_nested', HERE / 'nested_contract.py')
N = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(N)
MODE = 'arithmetic-primitives'
VARIANT = 'arithmetic-v1'
SCOPE = 'BOUNDED_FP32_BITS_MEAN_FTZ_DIAGNOSTIC'
# Exact independently frozen scientific sources; preparation does not qualify a milestone.
PROBE_SHA = '80ba23466e22f3695501aac64cc71affbfbccd458e4e8804b8b1f5bf275c2865'
KERNEL_SHA = 'a532a70b86001f8d3cd755ddd95ed03ba78b2c5570e2e6b278e92577ec52f81a'
INPUT_SHA = '7cb1be0bf806f3da98f9617f90ba99e042331ecbc07bf79799d779088919e5b5'
REFERENCE_SHA = '7e0bc0f7e6cfbf049609001613157ba7fd3ec123b5e29b8ed104087d642b5ee8'
HLO_SHA = '60f8ac1ea1b0a7ebfce27036da02470dae6ad1bb824dc894d462d6e54ce1a39b'
SOURCES = {'probe_primitives.py': {'sha256': '80ba23466e22f3695501aac64cc71affbfbccd458e4e8804b8b1f5bf275c2865', 'bytes': 46342}, 'primitive_kernels.py': {'sha256': 'a532a70b86001f8d3cd755ddd95ed03ba78b2c5570e2e6b278e92577ec52f81a', 'bytes': 3343}, 'primitive-inputs.json': {'sha256': '7cb1be0bf806f3da98f9617f90ba99e042331ecbc07bf79799d779088919e5b5', 'bytes': 87001}, 'primitive_reference.py': {'sha256': '7e0bc0f7e6cfbf049609001613157ba7fd3ec123b5e29b8ed104087d642b5ee8', 'bytes': 3922}, 'hlo_bitcast.py': {'sha256': '60f8ac1ea1b0a7ebfce27036da02470dae6ad1bb824dc894d462d6e54ce1a39b', 'bytes': 9783}}
STATE_SHA = '1233803f173b9e0d6e864e1fa3081f66408b357ffe17d925b0a69459514fb1f1'
MANIFEST_SHA = 'e377614105391502a379b0a6c5b7a0d441e2a5e83b126c7c759140820a0e954c'
BINDINGS = {'primitive_probe_sha256': 'probe_primitives.py',
            'primitive_kernels_sha256': 'primitive_kernels.py',
            'primitive_inputs_sha256': 'primitive-inputs.json',
            'primitive_reference_sha256': 'primitive_reference.py',
            'primitive_hlo_sha256': 'hlo_bitcast.py',
            'primitive_state_probe_sha256': 'probe_nested_state.py',
            'primitive_manifest_sha256': 'primitive-manifest.json'}
REQUIRED_PROOFS = ('source-controls.json', 'installed-source.json', 'built-source.json',
                   'built-plugin-source.json', 'runtime-probe.json', 'installed-metadata.json')
MAX_BYTES = 100 * 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def regular(path):
    path = Path(path)
    require(not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode) and path.lstat().st_nlink == 1,
            'PRIMITIVE_NONREGULAR')
    return path


def verify_manifest(manifest):
    if manifest.get('experiment') != MODE:
        require(not any(key in manifest for key in (*BINDINGS, 'primitive_variant', 'primitive_scope', 'primitive_diagnostic_only'))
                and not any(name in manifest['files'] for name in BINDINGS.values() if name != 'probe_nested_state.py'),
                'PRIMITIVE_NOT_REQUESTED')
        return
    N.verify_nested_manifest(manifest)
    for (field, name), pin in zip(BINDINGS.items(), (PROBE_SHA, KERNEL_SHA, INPUT_SHA, REFERENCE_SHA, HLO_SHA, STATE_SHA, MANIFEST_SHA)):
        require(re.fullmatch('[0-9a-f]{64}', pin) and manifest.get(field) == pin == manifest['files'].get(name, {}).get('sha256'),
                'PRIMITIVE_SOURCE_BINDING:' + field)
    require(manifest.get('primitive_variant') == VARIANT and manifest.get('primitive_scope') == SCOPE
            and manifest.get('primitive_diagnostic_only') is True and manifest.get('precision') == 'highest',
            'PRIMITIVE_REVIEWED_SCOPE')


def payload(root, manifest):
    root = Path(root)
    verify_manifest(manifest)
    N.verify_nested_payload(root, manifest)
    for field, name in BINDINGS.items():
        require(sha(regular(root / name)) == manifest[field], 'PRIMITIVE_PAYLOAD_SOURCE:' + name)
    protocol = read(root / 'primitive-manifest.json')
    require(protocol.get('format') == 'bnb-tpu.arithmetic-diagnostic.v1'
            and protocol.get('source_variant') == 'nested-v1' and protocol.get('primitive_variant') == VARIANT
            and protocol.get('scope') == SCOPE
            and protocol.get('sources') == SOURCES
            and all(manifest['files'].get(name) == row for name, row in SOURCES.items()),
            'PRIMITIVE_SOURCE_PROTOCOL')


def cli(root):
    root = Path(root)
    return [*N.nested_cli(root), '--nested-probe', root / 'probe_nested.py',
            '--nested-state-probe', root / 'probe_nested_state.py']


def cpu_archive(output):
    """Seal complete CPU data and actual launch/source proof before device science."""
    output = Path(output)
    write(output / 'primitive-cpu-receipt.json', read(output / 'receipt.json'))
    names = ['primitive-cpu-receipt.json', *REQUIRED_PROOFS]
    for relative in ('cpu-oracle', 'steps/11-cpu-oracle'):
        source = output / relative
        require(source.is_dir() and not source.is_symlink(), 'PRIMITIVE_CPU_DIRECTORY')
        for path in sorted(source.rglob('*')):
            require(not path.is_symlink(), 'PRIMITIVE_CPU_SYMLINK')
            if path.is_file():
                names.append(path.relative_to(output).as_posix())
            else:
                require(path.is_dir(), 'PRIMITIVE_CPU_NONREGULAR')
    require({'steps/11-cpu-oracle/' + name for name in ('ownership.json', 'result.json', 'cleanup.json', 'stdout.raw', 'stderr.raw')}
            <= set(names), 'PRIMITIVE_CPU_LAUNCH_RECORDS')
    require(len(names) == len(set(names)) and len(names) <= 2000, 'PRIMITIVE_CPU_MEMBER_COUNT')
    inventory = {name: {'sha256': sha(regular(output / name)), 'bytes': (output / name).stat().st_size}
                 for name in sorted(names)}
    require(sum(row['bytes'] for row in inventory.values()) <= MAX_BYTES, 'PRIMITIVE_CPU_ARCHIVE_SIZE')
    write(output / 'primitive-cpu-inventory.json', inventory)
    with zipfile.ZipFile(output / 'primitive-cpu-evidence.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(inventory):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 8, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (output / name).read_bytes())
    return {'primitive_cpu_inventory_sha256': sha(output / 'primitive-cpu-inventory.json'),
            'primitive_cpu_evidence_sha256': sha(output / 'primitive-cpu-evidence.zip'),
            'primitive_cpu_evidence_bytes': (output / 'primitive-cpu-evidence.zip').stat().st_size}


def recover_cpu(directory, receipt):
    root = Path(directory)
    inventory_file = regular(root / 'inventory.json')
    archive_file = regular(root / 'evidence.zip')
    require(sha(inventory_file) == receipt['primitive_cpu_inventory_sha256']
            and sha(archive_file) == receipt['primitive_cpu_evidence_sha256']
            and archive_file.stat().st_size == receipt['primitive_cpu_evidence_bytes'], 'PRIMITIVE_CPU_ARCHIVE_SEAL')
    inventory = read(inventory_file)
    require(type(inventory) is dict and 1 <= len(inventory) <= 2000, 'PRIMITIVE_CPU_INVENTORY')
    require(all(type(row) is dict and set(row) == {'bytes', 'sha256'} and type(row['bytes']) is int
                and 0 <= row['bytes'] <= MAX_BYTES and isinstance(row['sha256'], str)
                and re.fullmatch('[0-9a-f]{64}', row['sha256']) for row in inventory.values())
            and sum(row['bytes'] for row in inventory.values()) <= MAX_BYTES, 'PRIMITIVE_CPU_INVENTORY_RECORD')
    target = root / 'recovered'
    require(not target.exists() and not target.is_symlink(), 'PRIMITIVE_CPU_FRESH_RECOVERY')
    with zipfile.ZipFile(archive_file) as archive:
        require(len(archive.infolist()) == len(inventory) and set(archive.namelist()) == set(inventory),
                'PRIMITIVE_CPU_MEMBER_SET')
        for info in archive.infolist():
            name = info.filename
            path = PurePosixPath(name)
            require(not path.is_absolute() and '..' not in path.parts and '\\' not in name
                    and name == path.as_posix() and info.create_system == 3
                    and info.external_attr == 0o100644 << 16 and info.file_size == inventory[name]['bytes'],
                    'PRIMITIVE_CPU_MEMBER_TYPE')
        target.mkdir()
        for info in archive.infolist():
            data = archive.read(info)
            row = inventory[info.filename]
            require(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'],
                    'PRIMITIVE_CPU_MEMBER_BYTES')
            path = target / info.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    return target


def science(packet, admission_sha):
    packet = Path(packet)
    spec = importlib.util.spec_from_file_location('_admitted_primitive_science', packet / 'probe_primitives.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(backend_probe=packet / 'probe_backend.py', route_probe=packet / 'probe_routes.py',
                           precision_probe=packet / 'probe_precision.py', transfer_admission=packet / 'transfer_admission.py',
                           nested_probe=packet / 'probe_nested.py', nested_state_probe=packet / 'probe_nested_state.py',
                           patch_manifest=packet / 'patches/params4bit-xla-v1.json', admission_sha256=admission_sha)
    return module, args, module.helpers(args)


def cpu_gate(packet, recovered, receipt):
    """Independently audit recovered CPU data and actual process/source proofs."""
    packet, recovered = Path(packet), Path(recovered)
    manifest = read(packet / 'manifest.json')
    payload(packet, manifest)
    archive_fields = {'primitive_cpu_inventory_sha256', 'primitive_cpu_evidence_sha256', 'primitive_cpu_evidence_bytes'}
    snapshot = read(recovered / 'primitive-cpu-receipt.json')
    require(snapshot == {key: value for key, value in receipt.items() if key not in archive_fields},
            'PRIMITIVE_CPU_RECEIPT_SNAPSHOT')
    expected = {'status': 'CPU_ORACLE_READY_TPU_NOT_RUN', 'cpu_status': 'PASS', 'tpu_status': 'NOT_RUN',
                'runtime_status': 'PASS_TPU_RUNTIME_PROBE_ONLY', 'experiment': MODE,
                'precision': 'highest', 'primitive_variant': VARIANT, 'primitive_scope': SCOPE,
                'primitive_diagnostic_only': True,
                'built_source_status': 'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS',
                'installed_source_status': 'POST_PATCH_PYTHON_SOURCE_PASS',
                'built_plugin_source_status': 'NESTED_PLUGIN_WHEEL_PYTHON_SOURCE_PASS',
                'source_controls_status': 'PASS_QUALIFIED_LINUX_SOURCE_CONTROLS',
                'manifest_sha256': sha(packet / 'manifest.json'),
                'packet_sha256': sha(packet / 'payload.zip'),
                'source_admission_sha256': manifest['source_admission_sha256'],
                'runtime_lock_sha256': manifest['runtime_lock_sha256'],
                **{key: manifest[key] for key in (*BINDINGS, *N.NESTED_BINDINGS)}}
    require(all(receipt.get(key) == value for key, value in expected.items()), 'PRIMITIVE_CPU_PHASE_PROOFS')
    module, args, helpers = science(packet, manifest['source_admission_sha256'])
    B, R, P, A, nested, S = helpers
    report = module.verify_cpu_oracle(B, R, P, A, nested, S, args,
                                     root=recovered / 'cpu-oracle', expected_sha=receipt['oracle_sha256'])
    validate_cpu_report(report, receipt['oracle_sha256'])
    seal = read(recovered / 'cpu-oracle/oracle-seal.json')
    ownership = read(recovered / 'steps/11-cpu-oracle/ownership.json')
    result = read(recovered / 'steps/11-cpu-oracle/result.json')
    require(type(seal.get('pid')) is int and seal['pid'] > 0 and
            seal['pid'] == seal.get('pgid') == ownership.get('pid') == ownership.get('pgid')
            and seal.get('parent_pid') == ownership.get('owner_pid')
            and type(seal.get('parent_pid')) is int and seal['parent_pid'] > 0,
            'PRIMITIVE_CPU_ACTUAL_IDENTITY')
    token = seal.get('process_token')
    require(isinstance(token, str) and re.fullmatch('[0-9a-f]{32}', token), 'PRIMITIVE_CPU_TOKEN')
    base = Path('/content/bnb-tpu-first')
    payload_root = base / 'payload'
    expected_argv = [base / 'venv/bin/python', '-B', payload_root / 'probe_primitives.py', 'prepare',
                     '--admission', payload_root / 'source-admission.json',
                     '--admission-sha256', manifest['source_admission_sha256'],
                     '--output', base / 'records/cpu-oracle', *cli(payload_root), '--process-token', token]
    require(ownership.get('argv') == result.get('argv') == [str(x) for x in expected_argv],
            'PRIMITIVE_CPU_ARGV')
    cleanup = ownership.get('cleanup', {})
    require(cleanup.get('pid') == cleanup.get('pgid') == seal['pid'] and cleanup.get('status') == 'CLEANUP_VERIFIED'
            and cleanup.get('leader_reaped') is True and cleanup.get('group_absence') == 'OBSERVED_NO_SUCH_GROUP'
            and cleanup.get('errors') == [] and cleanup.get('exit_status') == ownership.get('exit_status') == 0,
            'PRIMITIVE_CPU_GROUP_CLOSED')
    group_summary = result.get('cleanup', {})
    require(result.get('status') == 'PASS' and result.get('exit_code') == 0 and not result.get('error')
            and group_summary.get('errors') == [] and group_summary.get('groups') == [cleanup]
            and group_summary.get('restoration', {}).get('handlers_restored') is True
            and group_summary.get('restoration', {}).get('timer_restored') is True,
            'PRIMITIVE_CPU_STEP_PASS')
    require('PJRT_DEVICE' not in result.get('environment', {}).get('set', {}), 'PRIMITIVE_CPU_ENVIRONMENT')
    from remote import validate_source_controls
    validate_source_controls(read(recovered / 'source-controls.json'))
    admission = read(packet / 'source-admission.json')
    installed = read(recovered / 'installed-source.json')
    built = read(recovered / 'built-source.json')
    plugin = read(recovered / 'built-plugin-source.json')
    require(installed.get('source_admission_sha256') == manifest['source_admission_sha256']
            and installed.get('patch_manifest_sha256') == manifest['patch_manifest_sha256']
            and installed.get('status') == 'POST_PATCH_PYTHON_SOURCE_PASS'
            and installed.get('installed_python_files') == {key: admission[key]['files'] for key in ('bitsandbytes', 'bitsandbytes_tpu')}
            and installed.get('source_variant') == 'nested-v1' and installed.get('nested_source_sha256') == N.NESTED_SOURCE_SHA
            and installed.get('plugin_source_manifest_sha256') == N.NESTED_PLUGIN_MANIFEST_SHA,
            'PRIMITIVE_CPU_INSTALLED_SOURCE')
    require(built.get('status') == 'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS'
            and built.get('patch_manifest_sha256') == manifest['patch_manifest_sha256']
            and built.get('python_files') == admission['bitsandbytes']['files']
            and any(row['name'] == built.get('wheel_name') and row['sha256'] == built.get('wheel_sha256')
                    for row in receipt['built_wheels']), 'PRIMITIVE_CPU_BUILT_SOURCE')
    require(plugin.get('status') == 'NESTED_PLUGIN_WHEEL_PYTHON_SOURCE_PASS'
            and plugin.get('python_files') == N.NESTED_FILES
            and plugin.get('plugin_source_manifest_sha256') == N.NESTED_PLUGIN_MANIFEST_SHA
            and plugin.get('nested_source_sha256') == N.NESTED_SOURCE_SHA
            and any(row['name'] == plugin.get('wheel_name') and row['sha256'] == plugin.get('wheel_sha256')
                    for row in receipt['built_wheels']), 'PRIMITIVE_CPU_BUILT_PLUGIN')
    installed_metadata = read(recovered / 'installed-metadata.json')
    locked = {row['name']: row for row in read(packet / 'runtime/requirements.lock.json')['wheels']}
    locked.update({row['name']: row for row in read(packet / 'build-requirements.lock.json')['wheels']})
    require(installed_metadata.get('python') == '3.12.14'
            and installed_metadata.get('executable') == str(base / 'venv/bin/python')
            and installed_metadata.get('packages') == {name: {'version': row['version'],
                'metadata_sha256': row['metadata_sha256'], 'wheel_sha256': row['sha256']}
                for name, row in locked.items()}, 'PRIMITIVE_CPU_INSTALLED_METADATA')
    runtime = read(recovered / 'runtime-probe.json')
    require(runtime.get('status') == 'PASS_TPU_RUNTIME_PROBE_ONLY' and runtime.get('runtime_executed') is True
            and runtime.get('observed_backend') == 'TPU' and runtime.get('error') is None
            and runtime.get('python') == '3.12.14' and runtime.get('system') == 'Linux'
            and runtime.get('machine') == 'x86_64'
            and runtime.get('versions') == {name: locked[name]['version']
                for name in ('torch', 'torch-xla', 'libtpu', 'jax', 'jaxlib')},
            'PRIMITIVE_CPU_RUNTIME_PROOF')
    gate = {'kind': 'ROOT_PRIMITIVE_CPU_ORACLE_GATE', 'status': 'QUALIFIED_LINUX_CPU_ORACLE_VERIFIED',
            'oracle_sha256': receipt['oracle_sha256'], 'source_admission_sha256': manifest['source_admission_sha256'],
            'runtime_lock_sha256': manifest['runtime_lock_sha256'], 'source_variant': 'nested-v1',
            'primitive_variant': VARIANT, 'packet_sha256': sha(packet / 'payload.zip'),
            'manifest_sha256': sha(packet / 'manifest.json'),
            'cpu_observations_sha256': hashlib.sha256(json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
            **{key: manifest[key] for key in BINDINGS},
            **{key: receipt[key] for key in archive_fields},
            **{key: 'NOT_QUALIFIED' for key in ('api42_status', 'm3_status', 'm4_status', 'm5_status')}}
    return gate, report


def verify_root_gate(gate, receipt, manifest):
    expected = {'kind': 'ROOT_PRIMITIVE_CPU_ORACLE_GATE', 'status': 'QUALIFIED_LINUX_CPU_ORACLE_VERIFIED',
                'oracle_sha256': receipt['oracle_sha256'], 'source_admission_sha256': manifest['source_admission_sha256'],
                'runtime_lock_sha256': manifest['runtime_lock_sha256'], 'source_variant': 'nested-v1',
                'primitive_variant': VARIANT, 'packet_sha256': receipt['packet_sha256'],
                'manifest_sha256': receipt['manifest_sha256'],
                **{key: manifest[key] for key in BINDINGS},
                **{key: receipt[key] for key in ('primitive_cpu_inventory_sha256', 'primitive_cpu_evidence_sha256', 'primitive_cpu_evidence_bytes')},
                **{key: 'NOT_QUALIFIED' for key in ('api42_status', 'm3_status', 'm4_status', 'm5_status')}}
    require(type(gate) is dict and set(gate) == set(expected) | {'cpu_observations_sha256'}
            and all(gate.get(key) == value for key, value in expected.items())
            and isinstance(gate.get('cpu_observations_sha256'), str)
            and re.fullmatch('[0-9a-f]{64}', gate['cpu_observations_sha256']), 'PRIMITIVE_ROOT_CPU_GATE')
    return gate


def validate_cpu_report(report, oracle_sha256):
    require(type(report) is dict and report.get('record_validation') == 'PASS'
            and report.get('oracle_status') == 'VERIFIED' and report.get('oracle_sha256') == oracle_sha256
            and isinstance(oracle_sha256, str) and re.fullmatch('[0-9a-f]{64}', oracle_sha256)
            and type(report.get('row_count')) is int and report['row_count'] == 32
            and type(report.get('mean_case_count')) is int and report['mean_case_count'] == 28
            and type(report.get('output_count')) is int and report['output_count'] == 206
            and all(report.get(key) == 'NOT_QUALIFIED' for key in ('api42_status', 'm3_status', 'm4_status', 'm5_status')),
            'PRIMITIVE_CPU_INDEPENDENT_AUDIT')
    return True


def validate_report(report, packet):
    """Check the exact audited observation matrix without turning differences into qualification."""
    manifest = read(Path(packet) / 'manifest.json')
    module, args, helpers = science(packet, manifest['source_admission_sha256'])
    B, R, P, A, nested, S = helpers
    data = module.spec(B, nested)
    Ref, _ = module.pure()
    expected = Ref.expected(data)
    ids = [('native-view', 'native-bitcast'), *[('builder', name) for name in
        ('host-float-bits', 'device-int-float', 'float-arithmetic', *['mean-' + row['case_id'] for row in data['mean_cases']])]]
    require(type(report) is dict and report.get('record_validation') == 'PASS'
            and report.get('phase_ids') == ['native-view', 'builder']
            and type(report.get('row_count')) is int and report['row_count'] == 32
            and all(report.get(key) == 'NOT_QUALIFIED' for key in ('api42_status', 'm3_status', 'm4_status', 'm5_status')),
            'PRIMITIVE_DIAGNOSTIC_REPORT')
    rows = report.get('rows')
    require(type(rows) is list and [(row.get('phase'), row.get('case_id')) for row in rows] == ids,
            'PRIMITIVE_DIAGNOSTIC_MATRIX')
    for row in rows:
        name = row['case_id']
        gates = row.get('gates')
        if row.get('status') == 'UNSUPPORTED':
            require(row['phase'] == 'native-view' and gates == {}, 'PRIMITIVE_UNSUPPORTED_SCOPE')
            continue
        parameters = module.source_inputs(Ref, data, name, expected['device-int-float']['float'])
        require(type(gates) is dict and set(gates) == set(expected[name]) | {'parameter_' + key for key in parameters} | {'auxiliary_' + key for key in module.auxiliary_sources(Ref, name)},
                'PRIMITIVE_GATE_MATRIX')
        for gate in gates.values():
            require(type(gate) is dict and set(gate) == {'status', 'actual_bytes_sha256', 'expected_bytes_sha256'}
                    and all(isinstance(gate[key], str) and re.fullmatch('[0-9a-f]{64}', gate[key])
                            for key in ('actual_bytes_sha256', 'expected_bytes_sha256'))
                    and gate['status'] == ('PASS' if gate['actual_bytes_sha256'] == gate['expected_bytes_sha256'] else 'FAIL'),
                    'PRIMITIVE_GATE_STATUS')
        for role, value in module.auxiliary_sources(Ref, name).items():
            require(gates['auxiliary_' + role]['expected_bytes_sha256'] == value['bytes_sha256'],
                    'PRIMITIVE_AUXILIARY_SOURCE_GATE')
        require(row.get('status') == ('PASS' if all(g['status'] == 'PASS' for g in gates.values()) else 'FAIL'),
                'PRIMITIVE_ROW_STATUS')
    require(report.get('native_view_status') == ('UNSUPPORTED' if rows[0]['status'] == 'UNSUPPORTED' else 'SUPPORTED')
            and report.get('builder_bit_status') == ('FAIL' if any(row['status'] == 'FAIL' for row in rows[1:]) else 'PASS')
            and report.get('numerical_status') == ('FAIL' if any(row['status'] == 'FAIL' for row in rows) else 'PASS')
            and type(report.get('gate_count')) is int and report['gate_count'] == sum(len(row['gates']) for row in rows),
            'PRIMITIVE_AGGREGATE_STATUS')
    cpu = report.get('cpu_oracle')
    require(type(cpu) is dict, 'PRIMITIVE_CPU_REPORT_MISSING')
    validate_cpu_report(cpu, cpu.get('oracle_sha256'))
    return True


class ReadbackUnavailable(Exception):
    """Complete archive recovery can precede an admitted diagnostic result."""


def readback_missing(owner, receipt, output, gate):
    missing = []
    if not isinstance(gate, dict) or gate.get('status') != 'QUALIFIED_LINUX_CPU_ORACLE_VERIFIED':
        missing.append('independent_root_cpu_gate')
    if 'recovered_oracle_sha256' not in owner:
        missing.append('recovered_cpu_oracle_seal')
    if receipt.get('status') != 'PRIMITIVE_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED' or receipt.get('tpu_status') != 'PRIMITIVE_RECORDS_COMPLETE':
        missing.append('terminal_primitive_receipt')
    if 'primitive_receipt_sha256' not in receipt:
        missing.append('primitive_receipt_sha256')
    if not (Path(output) / 'recovered/primitives/receipt.json').is_file():
        missing.append('primitives/receipt.json')
    return missing
