"""Install fixed files and run separate runtime, CPU, and TPU child processes."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import secrets
import stat
import sys
import tarfile
import time
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'ownership'))
from cleanup_lifecycle import Ownership
from lifecycle import durable_json

PRECISION_ROUTE_PROBE_SHA = 'feae751734e57c741b1bdade004ff7ca3c041ee7eb3b7086b531bc6be433038e'
TRANSFER_PATCH_MANIFEST_SHA = 'e745fbf21aac10ed9118167a131d6505bf6dbe03e1fab0a669c663b26c5a5732'
TRANSFER_PRECISION_PROBE_SHA = 'e0534955507e5f91e67ecddfffc738a367ad752e69a4eea7ae8052c6d76a8852'
TRANSFER_SOURCE_CONTROLS_SHA = '5f5c8b6c3d04a0d8658dae1f4d70d7a909dfe63977b953936aecfb69bf9a16fe'
from nested_contract import (NESTED_BINDINGS,NESTED_SCOPE,NESTED_SOURCE_SHA,NESTED_PLUGIN_MANIFEST_SHA,
    NESTED_FILES,verify_nested_manifest,verify_nested_payload,verify_nested_wheel,nested_cli)

TRANSFER_BINDINGS = {'route_probe_sha256': 'probe_routes.py', 'precision_probe_sha256': 'probe_precision.py',
                     'transfer_probe_sha256': 'probe_transfer.py', 'transfer_admission_sha256': 'transfer_admission.py',
                     'source_controls_sha256': 'tests/test_transfer_source.py',
                     'patch_manifest_sha256': 'patches/params4bit-xla-v1.json',
                     'patch_sha256': 'patches/params4bit-xla-v1.patch',
                     'base_archive_sha256': 'upstream.tar', 'patched_archive_sha256': 'patched-upstream.tar'}


STATE_HELPER_SHA = '006604d18d10c202583c5d42dfa361a83da441ea99d625241101b58d5948da45'
STATE_BINDINGS = {'state_probe_sha256': 'probe_state_roundtrip.py', 'state_helper_sha256': 'probe_state.py'}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def verify_experiment(manifest):
    experiment = manifest.get('experiment', 'api42')
    require(experiment in {'api42', 'device-route-diagnostic', 'precision-diagnostic', 'transfer-api42', 'state-roundtrip', 'nested-79'}, 'EXPERIMENT_VARIANT')
    if experiment in {'device-route-diagnostic', 'precision-diagnostic'}:
        pin = manifest.get('route_probe_sha256')
        require(manifest.get('diagnostic_only') is True and isinstance(pin, str) and len(pin) == 64 and
                pin == manifest['files'].get('probe_routes.py', {}).get('sha256'), 'ROUTE_PROBE_BINDING')
    if experiment in {'precision-diagnostic', 'transfer-api42', 'state-roundtrip', 'nested-79'}:
        require(manifest.get('route_probe_sha256') == PRECISION_ROUTE_PROBE_SHA, 'PRECISION_ROUTE_PROBE_BINDING')
        pin = manifest.get('precision_probe_sha256')
        require(isinstance(pin, str) and len(pin) == 64 and
                pin == manifest['files'].get('probe_precision.py', {}).get('sha256'), 'PRECISION_PROBE_BINDING')
    else:
        require('precision_probe_sha256' not in manifest and 'probe_precision.py' not in manifest['files'],
                'PRECISION_PROBE_NOT_REQUESTED')
    if experiment in {'transfer-api42', 'state-roundtrip', 'nested-79'}:
        for field, name in TRANSFER_BINDINGS.items():
            pin = manifest.get(field)
            require(isinstance(pin, str) and len(pin) == 64 and pin == manifest['files'].get(name, {}).get('sha256'),
                    'TRANSFER_SOURCE_BINDING:' + field)
        require(manifest['patch_manifest_sha256'] == TRANSFER_PATCH_MANIFEST_SHA and
                manifest['precision_probe_sha256'] == TRANSFER_PRECISION_PROBE_SHA and
                manifest['source_controls_sha256'] == TRANSFER_SOURCE_CONTROLS_SHA and
                manifest.get('precision') == 'highest' and not manifest.get('diagnostic_only'), 'TRANSFER_REVIEWED_SCOPE')
    else:
        transfer_only = set(TRANSFER_BINDINGS) - {'route_probe_sha256', 'precision_probe_sha256'}
        require(not any(field in manifest or (field != 'base_archive_sha256' and TRANSFER_BINDINGS[field] in manifest['files']) for field in transfer_only),
                'TRANSFER_SOURCE_NOT_REQUESTED')
    if experiment == 'state-roundtrip':
        for field, name in STATE_BINDINGS.items():
            pin = manifest.get(field)
            require(isinstance(pin, str) and len(pin) == 64 and pin == manifest['files'].get(name, {}).get('sha256'), 'STATE_SOURCE_BINDING:' + field)
        require(manifest['state_helper_sha256'] == STATE_HELPER_SHA and manifest.get('state_scope') == 'FRESH_SAVE_AND_RESTORE_PROCESSES_ALL8_LINEAR', 'STATE_REVIEWED_SCOPE')
    else:
        require(not any(field in manifest or name in manifest['files'] for field, name in STATE_BINDINGS.items()) and 'state_scope' not in manifest, 'STATE_SOURCE_NOT_REQUESTED')
    if experiment == 'nested-79':
        verify_nested_manifest(manifest)
    else:
        require(manifest.get('plugin_source_manifest_sha256')!=NESTED_PLUGIN_MANIFEST_SHA,'NESTED_PLUGIN_NOT_REQUESTED')
        require(not any(field in manifest or name in manifest['files'] for field,name in NESTED_BINDINGS.items()) and
                'nested_scope' not in manifest and 'nested_source_variant' not in manifest,'NESTED_SOURCE_NOT_REQUESTED')
    return experiment


def verify_transfer_payload(payload, manifest):
    patch_path = payload / 'patches/params4bit-xla-v1.json'
    require(sha(patch_path) == TRANSFER_PATCH_MANIFEST_SHA, 'TRANSFER_PATCH_MANIFEST_SOURCE')
    patch = json.loads(patch_path.read_text())
    require(patch['base_archive_sha256'] == manifest['base_archive_sha256'] and
            patch['patch_sha256'] == manifest['patch_sha256'] and
            patch['runtime_lock_sha256'] == manifest['runtime_lock_sha256'], 'TRANSFER_PATCH_PROVENANCE')
    admission = json.loads((payload / 'source-admission.json').read_text())
    post = {p: h for p, h in patch['post_patch_package_files'].items() if p.endswith('.py')}
    require(sha(payload / 'source-admission.json') == manifest['source_admission_sha256'] and
            admission.get('format') == 'bnb-tpu.probe-source-admission.v1' and
            admission.get('runtime_lock_sha256') == manifest['runtime_lock_sha256'] and
            admission['bitsandbytes'].get('commit') == patch['base_revision'] and
            admission['bitsandbytes'].get('patch_manifest_sha256') == TRANSFER_PATCH_MANIFEST_SHA and
            admission['bitsandbytes'].get('files') == post, 'TRANSFER_ADMISSION_OVERLAY')


def validate_source_controls(report):
    """Require the complete reviewed Linux source-control record."""
    require(report.get('record_validation') == 'PASS' and report.get('qualified_source_controls') is True and
            report.get('runtime') == {'torch': '2.9.0+cpu', 'python': '3.12.14', 'platform': 'Linux'} and
            report.get('patch_manifest_sha256') == TRANSFER_PATCH_MANIFEST_SHA and
            report.get('tpu') == report.get('xla') == 'NOT_RUN' and report.get('failures') == [] and
            report.get('whole_module_transactionality') == 'NOT_PROMISED', 'TRANSFER_SOURCE_CONTROLS_RUNTIME')
    rows = report.get('controls')
    require(isinstance(rows, list) and len(rows) == 18 and all(isinstance(row, dict) for row in rows), 'TRANSFER_SOURCE_CONTROLS_MATRIX')
    methods, negatives, cases, originals, compatible = [], [], [], [], []
    for row in rows:
        if 'method' in row:
            methods.append(row['method'])
            flags = ('identity', 'class', 'attrs', 'module_alias')
            keys = set(flags) | {'method', 'scope', 'bias_device'}
            require(row.get('scope') == 'CPU_META_TYPE_MECHANICS_ONLY' and
                    row.get('bias_device') == ('meta' if row['method'] == 'module_apply_meta' else 'cpu'), 'TRANSFER_SOURCE_CONTROLS_ROW')
        elif 'negative' in row:
            negatives.append(row['negative'])
            flags = ('rejected', 'tensorimpl_unchanged', 'dict_identity_unchanged', 'data_unchanged', 'module_unchanged')
            keys = set(flags) | {'negative', 'error'}
            require(isinstance(row.get('error'), str) and bool(row['error']), 'TRANSFER_SOURCE_CONTROLS_ROW')
        elif 'failure' in row:
            originals.append(row['failure']); flags = ('original_unchanged',); keys = set(flags) | {'failure'}
        elif 'compatible_cpu' in row:
            compatible.append(row); flags = ('compatible_cpu', 'weight_identity', 'custom_attrs', 'state_format', 'weights_only_load')
            keys = set(flags) | {'keys'}
            require(isinstance(row.get('keys'), list) and len(set(row['keys'])) == len(row['keys']) and
                    {'weight', 'bias'}.issubset(row['keys']), 'TRANSFER_SOURCE_CONTROLS_ROW')
        elif row.get('case') in {'existing_gradient', 'overwrite_flag', 'swap_flag'}:
            cases.append(row['case'])
            flags = ('rejected_before_swap', 'parameter_identity', 'tensorimpl_identity', 'dict_identity', 'class_identity',
                     'values_unchanged', 'quant_state_identity', 'module_alias_unchanged', 'gradient_identity', 'bias_identity')
            keys = set(flags) | {'case', 'scope'}
            require(row.get('scope') == 'CPU_META_TYPE_MECHANICS_ONLY', 'TRANSFER_SOURCE_CONTROLS_ROW')
        elif row.get('case') == 'genuine_cpu_forward_backward':
            cases.append(row['case']); flags = ('forward_exact', 'dx_exact', 'db_exact', 'frozen_base', 'state_plain_tensors')
            keys = set(flags) | {'case', 'forward_values'}
            values = row.get('forward_values')
            require(isinstance(values, list) and len(values) == 3 and all(isinstance(r, list) and len(r) == 2 and
                    all(type(v) in (int, float) and math.isfinite(v) for v in r) for r in values), 'TRANSFER_SOURCE_CONTROLS_ROW')
        elif row.get('case') == 'fresh_process_cpu_public_restore':
            cases.append(row['case']); flags = ('forward_exact', 'weights_only', 'original_classes', 'module_state_alias', 'frozen_base')
            keys = set(flags) | {'case'}
        else:
            raise ValueError('TRANSFER_SOURCE_CONTROLS_ROW')
        require(set(row) == keys and all(row.get(flag) is True for flag in flags), 'TRANSFER_SOURCE_CONTROLS_ROW')
    require(sorted(methods) == sorted(('direct_to_fixture', 'module_to_fixture', 'direct_quantize_meta', 'module_apply_meta')) and
            sorted(negatives) == sorted(('python_weakref', 'cpp_weakref', 'held_impl', 'requires_grad', 'subclass')) and
            sorted(cases) == sorted(('existing_gradient', 'overwrite_flag', 'swap_flag', 'genuine_cpu_forward_backward', 'fresh_process_cpu_public_restore')) and
            originals == ['ORIGINAL_INCOMPATIBLE_TYPE'] * 3 and len(compatible) == 1, 'TRANSFER_SOURCE_CONTROLS_IDENTITIES')
    return True


def verify_transfer_wheel(wheel, expected):
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'TRANSFER_WHEEL_DUPLICATE')
        observed = {name.removeprefix('bitsandbytes/'): hashlib.sha256(archive.read(name)).hexdigest()
                    for name in names if name.startswith('bitsandbytes/') and name.endswith('.py')}
        require(not any(name.startswith('bitsandbytes/') and name.endswith('.pyc') for name in names) and
                observed == expected, 'TRANSFER_WHEEL_PYTHON_SOURCE')
    return observed


def verify_archive(archive, expected_sha, target):
    require(sha(archive) == expected_sha, 'PAYLOAD_SHA')
    require(not target.exists() and not target.is_symlink(), 'FRESH_PAYLOAD_TARGET')
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        require(len(names) == len(set(names)) and len(names) <= 1000, 'PAYLOAD_MEMBER_COUNT')
        require(sum(i.file_size for i in infos) <= 50 * 1024 * 1024, 'PAYLOAD_SIZE')
        for i in infos:
            p = PurePosixPath(i.filename)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in i.filename and
                    i.create_system == 3 and i.external_attr == (0o100644 << 16), 'PAYLOAD_NONREGULAR_OR_PATH')
        manifest = json.loads(z.read('manifest.json'))
        require(set(names) == set(manifest['files']) | {'manifest.json'}, 'PAYLOAD_INVENTORY')
        for name, rec in manifest['files'].items():
            raw = z.read(name)
            require(len(raw) == rec['bytes'] and hashlib.sha256(raw).hexdigest() == rec['sha256'], 'PAYLOAD_MEMBER_SHA')
        target.mkdir(mode=0o700)
        for name in names:
            p = target / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(z.read(name))
        return manifest


def environment(tpu=False):
    env = dict(os.environ)
    removed = ['PYTHONPATH', 'PYTHONHOME', 'TPU_LIBRARY_PATH', 'PTXLA_TPU_LIBRARY_PATH', 'PJRT_DEVICE']
    for key in removed:
        env.pop(key, None)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', TPU_LOG_DIR='disabled')
    if tpu:
        env['PJRT_DEVICE'] = 'TPU'
    return env, {'names_unset': removed, 'set': {k: env[k] for k in ['PYTHONDONTWRITEBYTECODE', 'PYTHONNOUSERSITE', 'TPU_LOG_DIR', 'PJRT_DEVICE'] if k in env},
                 'HOME_and_CODEX_HOME': 'INHERITED_WITHOUT_CHANGE'}


def allowance(deadline, requested, clock=time.time):
    value = min(float(requested), float(deadline) - clock())
    if not math.isfinite(value) or value <= 0:
        raise TimeoutError('WORK_OR_STAGE_BUDGET_EXHAUSTED')
    return value


def run_step(out, label, argv, deadline, limit, *, cwd, tpu=False, extra_env=None):
    directory = out / 'steps' / label
    directory.mkdir(parents=True)
    owner = Ownership(directory)
    env, env_record = environment(tpu)
    if extra_env:
        env.update(extra_env)
        env_record['set'].update(extra_env)
    streams = []
    record = {'argv': argv, 'status': 'BLOCKED', 'exit_code': None, 'environment': env_record}
    try:
        for name in ['stdout.raw', 'stderr.raw']:
            streams.append((directory / name).open('wb', buffering=0))
        budget = allowance(deadline, limit)
        record['timeout_seconds'] = budget
        with owner.guard(budget):
            p = owner.launch(argv, record=directory / 'ownership.json', cwd=cwd, env=env,
                             stdout=streams[0], stderr=streams[1])
            p.wait()
            record.update(exit_code=p.returncode, status='PASS' if p.returncode == 0 else 'CHILD_FAILED')
    except BaseException as error:
        record['error'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        record['original_outcome'] = {'status': record['status'], 'exit_code': record['exit_code'], 'error': record.get('error')}
        for stream in streams:
            try:
                os.fsync(stream.fileno())
            except Exception as error:
                owner.note_error('fsync_child_log', error)
            finally:
                try:
                    stream.close()
                except Exception as error:
                    owner.note_error('close_child_log', error)
        record['cleanup'] = owner.summary()
        if record['cleanup']['errors']:
            record['status'] = 'BLOCKED_CLEANUP'
        durable_json(directory / 'result.json', record)
    return record


def unpack_source(archive, target):
    require(not target.exists(), 'FRESH_SOURCE_TARGET')
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        require(len(members) <= 5000 and sum(m.size for m in members) <= 100 * 1024 * 1024, 'SOURCE_SIZE')
        names = set()
        for m in members:
            p = PurePosixPath(m.name)
            require(m.name not in names and not p.is_absolute() and '..' not in p.parts and '\\' not in m.name and
                    (m.isfile() or m.isdir()), 'SOURCE_UNSAFE_MEMBER')
            names.add(m.name)
        target.mkdir(mode=0o700)
        for m in members:
            p = target / m.name
            if m.isdir():
                p.mkdir(parents=True, exist_ok=True)
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(m) as src, p.open('xb') as dst:
                    shutil.copyfileobj(src, dst)


def download_wheels(records, target):
    target.mkdir(exist_ok=True)
    for rec in records:
        require(Path(rec['filename']).name == rec['filename'] and rec['url'].startswith(('https://files.pythonhosted.org/', 'https://download.pytorch.org/')), 'WHEEL_URL_OR_NAME')
        p = target / rec['filename']
        if not p.exists():
            with urllib.request.urlopen(rec['url'], timeout=60) as src, p.open('xb') as dst:
                size = 0
                while chunk := src.read(1024 * 1024):
                    size += len(chunk)
                    require(size <= rec['size'], 'WHEEL_DOWNLOAD_SIZE')
                    dst.write(chunk)
        require(not p.is_symlink() and sha(p) == rec['sha256'] and p.stat().st_size == rec['size'], 'WHEEL_DOWNLOAD_SHA')


def package(out):
    # Only test output enters the archive. Interpreters, source trees, wheels and scratch are outside out.
    members = {}
    for p in sorted(out.rglob('*')):
        require(not p.is_symlink(), 'RESULT_SYMLINK')
        require(stat.S_ISREG(p.lstat().st_mode) or stat.S_ISDIR(p.lstat().st_mode), 'RESULT_NONREGULAR')
        if p.is_file() and p.relative_to(out).as_posix() not in {'evidence.zip', 'archive-members.json', 'receipt.json'}:
            require(stat.S_ISREG(p.lstat().st_mode) and p.lstat().st_nlink == 1, 'RESULT_NONREGULAR')
            members[p.relative_to(out).as_posix()] = {'bytes': p.stat().st_size, 'sha256': sha(p)}
    require(len(members) <= 2000 and sum(r['bytes'] for r in members.values()) <= 100 * 1024 * 1024, 'RESULT_SIZE')
    # The archive receipt precedes self-referential export fields.
    members['receipt.json'] = {'bytes': (out / 'receipt.json').stat().st_size, 'sha256': sha(out / 'receipt.json')}
    durable_json(out / 'archive-members.json', members)
    with zipfile.ZipFile(out / 'evidence.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 4, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, (out / name).read_bytes())
    return {'evidence_sha256': sha(out / 'evidence.zip'), 'evidence_bytes': (out / 'evidence.zip').stat().st_size,
            'archive_members_sha256': sha(out / 'archive-members.json')}


def execute(base, packet_sha, allocation_epoch, phase):
    out, payload = base / 'records', base / 'payload'
    manifest = json.loads((payload / 'manifest.json').read_text())
    require(sha(base / 'payload.zip') == packet_sha, 'PACKET_SHA')
    for name, rec in manifest['files'].items():
        require(sha(payload / name) == rec['sha256'], 'POST_UPLOAD_SOURCE_SHA')
    experiment = verify_experiment(manifest)
    diagnostic = experiment == 'device-route-diagnostic'
    precision = experiment == 'precision-diagnostic'
    nested = experiment == 'nested-79'
    state = experiment == 'state-roundtrip'
    transfer = experiment in {'transfer-api42', 'state-roundtrip', 'nested-79'}
    if transfer:
        verify_transfer_payload(payload, manifest)
    out.mkdir(exist_ok=True)
    receipt_path = out / 'receipt.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {
        'status': 'RUNNING', 'runtime_status': 'NOT_RUN', 'cpu_status': 'NOT_RUN', 'tpu_status': 'NOT_RUN',
        'packet_sha256': packet_sha, 'manifest_sha256': sha(payload / 'manifest.json'),
        'runtime_lock_sha256': manifest['runtime_lock_sha256'], 'source_admission_sha256': manifest['source_admission_sha256'],
        'allocation_epoch': allocation_epoch, 'steps': [], 'error': None}
    if precision or transfer:
        if receipt_path.exists():
            require(receipt.get('experiment') == experiment and
                    (receipt.get('diagnostic_only') is True if precision else receipt.get('precision') == 'highest') and
                    receipt.get('route_probe_sha256') == manifest['route_probe_sha256'] and
                    receipt.get('precision_probe_sha256') == manifest['precision_probe_sha256'] and
                    receipt.get('packet_sha256') == packet_sha and
                    receipt.get('manifest_sha256') == sha(payload / 'manifest.json') and
                    receipt.get('source_admission_sha256') == manifest['source_admission_sha256'] and
                    receipt.get('runtime_lock_sha256') == manifest['runtime_lock_sha256'] and
                    receipt.get('allocation_epoch') == allocation_epoch,
                    'TRANSFER_PHASE_RECEIPT_BINDING' if transfer else 'PRECISION_PHASE_RECEIPT_BINDING')
            if transfer:
                require(all(receipt.get(field) == manifest[field] for field in TRANSFER_BINDINGS), 'TRANSFER_PHASE_SOURCE_BINDING')
                if state:
                    require(all(receipt.get(field) == manifest[field] for field in STATE_BINDINGS), 'STATE_PHASE_SOURCE_BINDING')
                if nested:
                    require(all(receipt.get(field)==manifest[field] for field in NESTED_BINDINGS),'NESTED_PHASE_SOURCE_BINDING')
        else:
            receipt.update(experiment=experiment,
                           route_probe_sha256=manifest['route_probe_sha256'],
                           precision_probe_sha256=manifest['precision_probe_sha256'])
            if precision:
                receipt['diagnostic_only'] = True
            else:
                receipt.update({field: manifest[field] for field in TRANSFER_BINDINGS})
                receipt.update(precision='highest', m3_status='NOT_QUALIFIED')
                if state:
                    receipt.update({field: manifest[field] for field in STATE_BINDINGS})
                    receipt['state_scope'] = manifest['state_scope']
                if nested:
                    receipt.update({field:manifest[field] for field in NESTED_BINDINGS})
                    receipt.update(nested_scope=NESTED_SCOPE,nested_source_variant='nested-v1')
    work_deadline = allocation_epoch + 3600 - 600 - 60
    if nested:
        verify_nested_payload(payload,manifest)
    installed = base / 'venv/bin/python'

    def step(label, argv, deadline, limit, **kwargs):
        rec = run_step(out, label, [str(x) for x in argv], min(work_deadline, deadline), limit,
                       cwd=payload, **kwargs)
        receipt['steps'].append({'label': label, **rec})
        durable_json(receipt_path, receipt)
        require(rec['status'] == 'PASS', 'CHILD_STEP_BLOCKED:' + label)
        return rec

    try:
        if phase == 'install':
            require(receipt['steps'] == [], 'NO_INSTALL_RESUBMISSION')
            deadline = min(work_deadline, time.time() + 1200)
            receipt['kernel_metadata'] = {'python': platform.python_version(), 'libc': list(platform.libc_ver()),
                                          'system': platform.system(), 'machine': platform.machine(),
                                          'uid': os.getuid(), 'gid': os.getgid(), 'executable': sys.executable}
            durable_json(receipt_path, receipt)
            require(platform.system() == 'Linux' and platform.machine() == 'x86_64', 'LINUX_X86_64_REQUIRED')
            step('01-python-bootstrap', [sys.executable, '-B', payload / 'cloud/python_bootstrap.py', '--output', base / 'bootstrap'], deadline, 180)
            python = base / 'bootstrap/interpreter/python/bin/python3.12'
            step('02-create-venv', [python, '-B', '-m', 'venv', base / 'venv'], deadline, 90)
            runtime = json.loads((payload / 'runtime/requirements.lock.json').read_text())
            build = json.loads((payload / 'build-requirements.lock.json').read_text())
            require(runtime['marker_environment'] == build['marker_environment'], 'BUILD_MARKER_ENVIRONMENT')
            all_records = {r['name']: r for r in runtime['wheels']}
            for rec in build['wheels']:
                if rec['name'] in all_records:
                    require(all_records[rec['name']]['sha256'] == rec['sha256'], 'BUILD_RUNTIME_CONFLICT')
                all_records[rec['name']] = rec
            # The download worker has the same owned deadline as installation.
            download = payload / 'download-config.json'
            durable_json(download, {'wheels': list(all_records.values())})
            step('03-download-wheels', [python, '-B', payload / 'cloud/remote.py', 'download', '--config', download, '--base', base], deadline, 600)
            requirements = base / 'all-requirements.txt'
            requirements.write_text(''.join(f"{r['name']} @ {(base / 'wheels' / r['filename']).as_uri()} --hash=sha256:{r['sha256']}\n" for r in all_records.values()))
            step('04-install-wheels', [installed, '-B', '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-compile', '--require-hashes', '-r', requirements], deadline, 240)
            step('05-verify-runtime-wheels', [installed, '-B', payload / 'runtime/resolve.py', '--verify-only', '--cache', base / 'wheels', '--lock', payload / 'runtime/requirements.lock.json'], deadline, 90)
            if transfer:
                unpack_source(payload / 'upstream.tar', base / 'upstream-base')
                unpack_source(payload / 'patched-upstream.tar', base / 'upstream-patched-controls')
                unpack_source(payload / 'patched-upstream.tar', base / 'upstream')
            else:
                unpack_source(payload / 'upstream.tar', base / 'upstream')
            step('06-build-upstream', [installed, '-B', '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation', '--no-cache-dir', '--wheel-dir', base / 'built-upstream', base / 'upstream'], deadline, 180, extra_env={'BNB_SKIP_CMAKE': '1'})
            step('07-build-plugin', [installed, '-B', '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation', '--no-cache-dir', '--wheel-dir', base / 'built-plugin', payload / 'plugin'], deadline, 120)
            wheels = [*sorted((base / 'built-upstream').glob('*.whl')), *sorted((base / 'built-plugin').glob('*.whl'))]
            require(len(wheels) == 2, 'BUILT_WHEEL_COUNT')
            receipt['built_wheels'] = [{'name': p.name, 'sha256': sha(p), 'bytes': p.stat().st_size} for p in wheels]
            if transfer:
                upstream_wheels = sorted((base / 'built-upstream').glob('*.whl'))
                require(len(upstream_wheels) == 1, 'TRANSFER_UPSTREAM_WHEEL_COUNT')
                admission = json.loads((payload / 'source-admission.json').read_text())
                observed = verify_transfer_wheel(upstream_wheels[0], admission['bitsandbytes']['files'])
                durable_json(out / 'built-source.json', {'status': 'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS',
                             'wheel_name': upstream_wheels[0].name, 'wheel_sha256': sha(upstream_wheels[0]),
                             'patch_manifest_sha256': TRANSFER_PATCH_MANIFEST_SHA, 'python_files': observed})
                receipt['built_source_status'] = 'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS'
                if nested:
                    plugin_wheels=sorted((base/'built-plugin').glob('*.whl'));require(len(plugin_wheels)==1,'NESTED_PLUGIN_WHEEL_COUNT')
                    observed_plugin=verify_nested_wheel(plugin_wheels[0])
                    durable_json(out/'built-plugin-source.json',{'status':'NESTED_PLUGIN_WHEEL_PYTHON_SOURCE_PASS',
                        'wheel_name':plugin_wheels[0].name,'wheel_sha256':sha(plugin_wheels[0]),'python_files':observed_plugin,
                        'plugin_source_manifest_sha256':NESTED_PLUGIN_MANIFEST_SHA,'nested_source_sha256':NESTED_SOURCE_SHA})
                    receipt['built_plugin_source_status']='NESTED_PLUGIN_WHEEL_PYTHON_SOURCE_PASS'
            step('08-install-source-wheels', [installed, '-B', '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-compile', *wheels], deadline, 120)
            # Query installed distribution records without native imports.
            config = payload / 'installed-proof-config.json'
            durable_json(config, {'wheels': list(all_records.values())})
            step('09-installed-metadata', [installed, '-B', payload / 'cloud/remote.py', 'metadata', '--config', config, '--base', base], deadline, 30)
            if transfer:
                step('09-transfer-installed-source', [installed, '-B', payload / 'cloud/remote.py', 'transfer-source', '--base', base], deadline, 30)
                proof = json.loads((out / 'installed-source.json').read_text())
                require(proof.get('status') == 'POST_PATCH_PYTHON_SOURCE_PASS' and
                        proof.get('source_admission_sha256') == manifest['source_admission_sha256'] and
                        proof.get('patch_manifest_sha256') == TRANSFER_PATCH_MANIFEST_SHA and
                        proof.get('installed_python_files') == {name: admission[name]['files'] for name in ('bitsandbytes', 'bitsandbytes_tpu')}, 'TRANSFER_INSTALLED_SOURCE')
                if nested:
                    require(proof.get('source_variant')=='nested-v1' and proof.get('nested_source_sha256')==NESTED_SOURCE_SHA and
                            proof.get('plugin_source_manifest_sha256')==NESTED_PLUGIN_MANIFEST_SHA,'NESTED_INSTALLED_SOURCE')
                receipt['installed_source_status'] = 'POST_PATCH_PYTHON_SOURCE_PASS'
            receipt.update(status='INSTALLED_NOT_QUALIFIED', installation_status='PASS')
        elif phase == 'cpu':
            if nested:
                require(receipt.get('built_plugin_source_status')=='NESTED_PLUGIN_WHEEL_PYTHON_SOURCE_PASS',
                        'NESTED_PLUGIN_WHEEL_REQUIRED_BEFORE_CPU_ORACLE')
            require(receipt.get('installation_status') == 'PASS' and receipt['cpu_status'] == 'NOT_RUN', 'CPU_PHASE_ORDER')
            receipt['science_start_epoch'] = time.time()
            receipt['science_deadline_epoch'] = min(work_deadline, time.time() + 1800)
            deadline = receipt['science_deadline_epoch']
            if transfer:
                require(receipt.get('built_source_status') == 'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS' and
                        receipt.get('installed_source_status') == 'POST_PATCH_PYTHON_SOURCE_PASS', 'TRANSFER_INSTALLED_SOURCE_REQUIRED')
                step('10-transfer-source-controls', [installed, '-B', payload / 'tests/test_transfer_source.py',
                     '--base-source', base / 'upstream-base', '--patched-source', base / 'upstream-patched-controls',
                     '--output', out / 'source-controls.json'], deadline, 300)
                controls = json.loads((out / 'source-controls.json').read_text())
                validate_source_controls(controls)
                receipt['source_controls_status'] = 'PASS_QUALIFIED_LINUX_SOURCE_CONTROLS'
            step('10-runtime-probe', [installed, '-B', payload / 'runtime/probe_runtime.py', '--out', out / 'runtime-probe.json'], deadline, 120, tpu=True)
            require(json.loads((out / 'runtime-probe.json').read_text())['status'] == 'PASS_TPU_RUNTIME_PROBE_ONLY', 'RUNTIME_PROBE_BLOCKED')
            receipt['runtime_status'] = 'PASS_TPU_RUNTIME_PROBE_ONLY'
            argv = [installed, '-B', payload / ('probe_nested.py' if nested else 'probe_transfer.py' if transfer else 'probe_backend.py'), 'prepare',
                    '--admission', payload / 'source-admission.json', '--admission-sha256', manifest['source_admission_sha256'],
                    '--output', out / 'cpu-oracle']
            if nested:
                argv += nested_cli(payload)
            elif transfer:
                argv += ['--backend-probe', payload / 'probe_backend.py', '--route-probe', payload / 'probe_routes.py',
                         '--precision-probe', payload / 'probe_precision.py', '--patch-manifest', payload / 'patches/params4bit-xla-v1.json']
            step('11-cpu-oracle', argv, deadline, 300)
            receipt.update(status='CPU_ORACLE_READY_TPU_NOT_RUN', cpu_status='PASS', oracle_sha256=sha(out / 'cpu-oracle/oracle-seal.json'))
        elif phase == 'nested':
            require(nested,'WRONG_SCIENTIFIC_VARIANT')
            require(receipt.get('installed_source_status')=='POST_PATCH_PYTHON_SOURCE_PASS' and
                    receipt.get('built_source_status')=='POST_PATCH_WHEEL_PYTHON_SOURCE_PASS' and
                    receipt.get('built_plugin_source_status')=='NESTED_PLUGIN_WHEEL_PYTHON_SOURCE_PASS' and
                    receipt.get('source_controls_status')=='PASS_QUALIFIED_LINUX_SOURCE_CONTROLS','NESTED_CPU_CONTROLS_REQUIRED')
            launch=json.loads((base/'launch.json').read_text())
            require(receipt['cpu_status']=='PASS' and receipt['tpu_status']=='NOT_RUN' and
                    launch['oracle_sha256']==receipt['oracle_sha256'],'EXPLICIT_RECOVERED_ORACLE_HASH_REQUIRED')
            deadline=min(work_deadline,receipt['science_deadline_epoch'],time.time()+1500)
            receipt.update(status='NESTED_CHILD_RUNNING',tpu_status='RUNNING',tpu_attempted=True)
            durable_json(receipt_path,receipt)
            argv=[installed,'-B',payload/'probe_nested.py','execute',*nested_cli(payload),
                  '--admission',payload/'source-admission.json','--admission-sha256',manifest['source_admission_sha256'],
                  '--oracle',out/'cpu-oracle','--oracle-sha256',launch['oracle_sha256'],
                  '--deadline-epoch',deadline,'--output',out/'nested']
            rec=run_step(out,'12-nested',[str(x) for x in argv],deadline,1500,cwd=payload,tpu=True)
            receipt['steps'].append({'label':'12-nested',**rec})
            require(not rec['cleanup']['errors'] and not rec.get('error') and
                    (rec['status'],rec['exit_code']) in {('PASS',0),('CHILD_FAILED',2)},'NESTED_CHILD_UNQUALIFIED_OR_CLEANUP')
            result=json.loads((out/'nested/receipt.json').read_text())
            expected={'kind':'PRIVATE_NESTED_PROBE','status':'COMPLETE','probe_sha256':manifest['nested_probe_sha256'],
                      'inputs_sha256':manifest['nested_inputs_sha256'],'nested_source_sha256':manifest['nested_source_sha256'],
                      'schema_sha256':manifest['nested_schemas_sha256'],'source_variant':'nested-v1',
                      'patch_manifest_sha256':manifest['patch_manifest_sha256'],'source_admission_sha256':manifest['source_admission_sha256'],
                      'source_pre':manifest['source_admission_sha256'],'source_post':manifest['source_admission_sha256'],
                      'oracle_sha256':launch['oracle_sha256'],'runtime_lock_sha256':manifest['runtime_lock_sha256']}
            require(all(result.get(field)==value for field,value in expected.items()),'NESTED_RECEIPT_BINDING')
            require(type(result.get('pid')) is int and result.get('pid')==result.get('pgid') and
                    any(group.get('pid')==group.get('pgid')==result['pid'] for group in rec['cleanup']['groups']),
                    'NESTED_PROCESS_GROUP_IDENTITY')
            receipt.update(status='NESTED_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED',tpu_status='NESTED_RECORDS_COMPLETE',
                           nested_receipt_sha256=sha(out/'nested/receipt.json'),m4_status='NOT_QUALIFIED')
        elif phase == 'state':
            require(state, 'WRONG_SCIENTIFIC_VARIANT')
            require(receipt.get('installed_source_status') == 'POST_PATCH_PYTHON_SOURCE_PASS' and
                    receipt.get('built_source_status') == 'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS' and
                    receipt.get('source_controls_status') == 'PASS_QUALIFIED_LINUX_SOURCE_CONTROLS', 'STATE_CPU_CONTROLS_REQUIRED')
            launch = json.loads((base / 'launch.json').read_text())
            require(receipt['cpu_status'] == 'PASS' and receipt['tpu_status'] == 'NOT_RUN' and
                    launch['oracle_sha256'] == receipt['oracle_sha256'], 'EXPLICIT_RECOVERED_ORACLE_HASH_REQUIRED')
            receipt.update(status='STATE_CHILD_RUNNING', tpu_status='RUNNING', tpu_attempted=True)
            durable_json(receipt_path, receipt)
            deadline = min(work_deadline, receipt['science_deadline_epoch'], time.time() + 1500)
            parent_token = secrets.token_hex(16)
            argv = [installed, '-B', payload / 'cloud/state_coordinator.py', 'execute',
                    '--state-probe', payload / 'probe_state_roundtrip.py', '--manifest', payload / 'manifest.json',
                    '--manifest-sha256', sha(payload / 'manifest.json'), '--admission', payload / 'source-admission.json',
                    '--admission-sha256', manifest['source_admission_sha256'], '--patch-manifest', payload / 'patches/params4bit-xla-v1.json',
                    '--oracle', out / 'cpu-oracle', '--oracle-sha256', launch['oracle_sha256'],
                    '--deadline-epoch', deadline, '--output', out / 'state', '--process-token', parent_token]
            rec = run_step(out, '12-state', [str(value) for value in argv], deadline, 1500, cwd=payload, tpu=True)
            receipt['steps'].append({'label': '12-state', **rec})
            require(not rec['cleanup']['errors'] and not rec.get('error') and (rec['status'], rec['exit_code']) == ('PASS', 0), 'STATE_CHILD_UNQUALIFIED_OR_CLEANUP')
            result = json.loads((out / 'state/parent.json').read_text())
            require(result.get('kind') == 'STATE_ROUNDTRIP_PARENT' and result.get('status') == 'COMPLETE', 'STATE_RECEIPT_INCOMPLETE')
            require(result.get('source_pre') == result.get('source_post') == result.get('source_admission_sha256') == manifest['source_admission_sha256'] and
                    result.get('oracle_sha256') == launch['oracle_sha256'] and result.get('runtime_lock_sha256') == manifest['runtime_lock_sha256'] and
                    all(result.get(field) == manifest[field] for field in (*STATE_BINDINGS, 'patch_manifest_sha256')), 'STATE_RECEIPT_BINDING')
            require(result.get('process_token') == parent_token and result.get('deadline_epoch') == deadline, 'STATE_PARENT_LAUNCH_BINDING')
            require(type(result.get('pid')) is int and result.get('pid') == result.get('pgid') and
                    any(group.get('pid') == group.get('pgid') == result['pid'] for group in rec['cleanup']['groups']), 'STATE_PROCESS_GROUP_IDENTITY')
            receipt.update(status='STATE_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED', tpu_status='STATE_RECORDS_COMPLETE',
                           state_parent_sha256=sha(out / 'state/parent.json'), m3_status='NOT_QUALIFIED')
        elif phase in ('routes', 'precision', 'transfer'):
            require({'routes': diagnostic, 'precision': precision, 'transfer': transfer and not state and not nested}[phase], 'WRONG_SCIENTIFIC_VARIANT')
            label = '12-transfer' if transfer else '12-precision' if precision else '12-device-routes'
            directory = 'transfer' if transfer else 'precision' if precision else 'device-routes'
            status_prefix = 'TRANSFER' if transfer else 'PRECISION' if precision else 'DEVICE_ROUTE'
            if transfer:
                require(receipt.get('installed_source_status') == 'POST_PATCH_PYTHON_SOURCE_PASS' and
                        receipt.get('built_source_status') == 'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS' and
                        receipt.get('source_controls_status') == 'PASS_QUALIFIED_LINUX_SOURCE_CONTROLS', 'TRANSFER_CPU_CONTROLS_REQUIRED')
            launch = json.loads((base / 'launch.json').read_text())
            require(receipt['cpu_status'] == 'PASS' and receipt['tpu_status'] == 'NOT_RUN' and launch['oracle_sha256'] == receipt['oracle_sha256'], 'EXPLICIT_RECOVERED_ORACLE_HASH_REQUIRED')
            receipt.update(status=status_prefix + '_CHILD_RUNNING', tpu_status='RUNNING', tpu_attempted=True)
            durable_json(receipt_path, receipt)
            deadline = min(work_deadline, receipt['science_deadline_epoch'])
            if precision or transfer:
                deadline = min(deadline, time.time() + 1500)
            argv = [installed, '-B', payload / ('probe_transfer.py' if transfer else 'probe_precision.py' if precision else 'probe_routes.py'), 'execute',
                           '--backend-probe', payload / 'probe_backend.py', '--admission', payload / 'source-admission.json',
                           '--admission-sha256', manifest['source_admission_sha256'], '--oracle', out / 'cpu-oracle',
                           '--oracle-sha256', launch['oracle_sha256'], '--deadline-epoch', deadline,
                           '--output', out / directory]
            if precision or transfer:
                argv += ['--route-probe', payload / 'probe_routes.py']
            if transfer:
                argv += ['--precision-probe', payload / 'probe_precision.py', '--patch-manifest', payload / 'patches/params4bit-xla-v1.json']
            rec = run_step(out, label, [str(x) for x in argv], deadline, 1500, cwd=payload, tpu=True)
            receipt['steps'].append({'label': label, **rec})
            require(not rec['cleanup']['errors'] and not rec.get('error') and
                    (rec['status'], rec['exit_code']) in ({('PASS', 0), ('CHILD_FAILED', 2)} if transfer else {('PASS', 0)}),
                    ('TRANSFER' if transfer else 'PRECISION' if precision else 'ROUTE') + '_CHILD_UNQUALIFIED_OR_CLEANUP')
            result = json.loads((out / directory / 'receipt.json').read_text())
            error_prefix = 'TRANSFER' if transfer else 'PRECISION' if precision else 'ROUTE'
            require(result.get('kind') == ('TRANSFER_API42_PROBE' if transfer else 'PRECISION_DIAGNOSTIC_ONLY' if precision else 'DEVICE_ROUTE_DIAGNOSTIC_ONLY') and
                    result.get('status') == 'COMPLETE', error_prefix + '_RECEIPT_INCOMPLETE')
            require(result.get('source_pre') == manifest['source_admission_sha256'] and
                    result.get('source_post') == manifest['source_admission_sha256'] and
                    result.get('source_admission_sha256') == manifest['source_admission_sha256'] and
                    result.get('oracle_sha256') == launch['oracle_sha256'] and
                    result.get('runtime_lock_sha256') == manifest['runtime_lock_sha256'], error_prefix + '_RECEIPT_BINDING')
            require(type(result.get('pid')) is int and any(g.get('pid') == result['pid'] for g in rec['cleanup']['groups']) and
                    isinstance(result.get('process_token'), str) and bool(result['process_token']), error_prefix + '_PROCESS_IDENTITY')
            if precision or transfer:
                require(result.get('route_probe_sha256') == manifest['route_probe_sha256'] and
                        result.get('precision_probe_sha256') == manifest['precision_probe_sha256'], error_prefix + '_RECEIPT_PROBE_BINDING')
            if transfer:
                require(type(result.get('pgid')) is int and result['pgid'] == result['pid'] and
                        any(group.get('pid') == group.get('pgid') == result['pid'] for group in rec['cleanup']['groups']),
                        'TRANSFER_PROCESS_GROUP_IDENTITY')
                require(all(result.get(field) == manifest[field] for field in
                            ('transfer_probe_sha256', 'transfer_admission_sha256', 'patch_manifest_sha256')) and
                        result.get('backend_probe_sha256') == manifest['files']['probe_backend.py']['sha256'] and
                        result.get('precision') == {'requested': 'highest', 'readback': 'highest', 'set_calls': 1, 'before_graph': True},
                        'TRANSFER_RECEIPT_SOURCE_PRECISION_BINDING')
            receipt.update(status=status_prefix + '_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED',
                           tpu_status='TRANSFER_RECORDS_COMPLETE' if transfer else 'DIAGNOSTIC_RECORDS_COMPLETE')
            if not transfer:
                receipt['diagnostic_only'] = True
        elif phase == 'tpu':
            require(not diagnostic and not precision and not transfer, 'WRONG_SCIENTIFIC_VARIANT')
            launch = json.loads((base / 'launch.json').read_text())
            require(receipt['cpu_status'] == 'PASS' and receipt['tpu_status'] == 'NOT_RUN' and launch['oracle_sha256'] == receipt['oracle_sha256'], 'EXPLICIT_RECOVERED_ORACLE_HASH_REQUIRED')
            deadline = receipt['science_deadline_epoch']
            receipt.update(status='TPU_CHILD_RUNNING', tpu_status='RUNNING', tpu_attempted=True)
            durable_json(receipt_path, receipt)
            rec = run_step(out, '12-tpu-probe', [str(x) for x in [installed, '-B', payload / 'probe_backend.py', 'execute', '--admission', payload / 'source-admission.json', '--admission-sha256', manifest['source_admission_sha256'], '--oracle', out / 'cpu-oracle', '--oracle-sha256', launch['oracle_sha256'], '--output', out / 'tpu-actual']], min(work_deadline, deadline), 1500, cwd=payload, tpu=True)
            receipt['steps'].append({'label': '12-tpu-probe', **rec})
            require(not rec['cleanup']['errors'] and not rec.get('error') and
                    (rec['status'], rec['exit_code']) in {('PASS', 0), ('CHILD_FAILED', 2)},
                    'TPU_CHILD_UNQUALIFIED_OR_CLEANUP')
            receipt.update(status='TPU_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED', tpu_status='PASS' if rec['exit_code'] == 0 else 'FAIL')
        else:
            raise ValueError('UNKNOWN_PHASE')
    except BaseException as error:
        if phase in ('tpu', 'routes', 'precision', 'transfer', 'state', 'nested') and receipt.get('tpu_status') == 'RUNNING':
            receipt['tpu_status'] = 'ATTEMPTED_BLOCKED'
        receipt.update(status='BLOCKED', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        durable_json(receipt_path, receipt)
    print(json.dumps({'status': receipt['status'], 'phase': phase, 'error': receipt['error']}))
    return 0 if receipt['status'] != 'BLOCKED' else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=['unpack', 'install', 'cpu', 'tpu', 'routes', 'precision', 'transfer', 'state', 'nested', 'transfer-source', 'export', 'download', 'metadata'])
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--packet-sha256')
    p.add_argument('--allocation-epoch', type=float)
    p.add_argument('--config', type=Path)
    a = p.parse_args()
    if a.operation == 'unpack':
        manifest = verify_archive(a.base / 'payload.zip', a.packet_sha256, a.base / 'payload')
        print(json.dumps({'status': 'PAYLOAD_VERIFIED', 'manifest_sha256': sha(a.base / 'payload/manifest.json')}))
        return 0
    if a.operation == 'download':
        download_wheels(json.loads(a.config.read_text())['wheels'], a.base / 'wheels')
        return 0
    if a.operation == 'transfer-source':
        payload = a.base / 'payload'
        manifest = json.loads((payload / 'manifest.json').read_text())
        require(verify_experiment(manifest) in {'transfer-api42', 'state-roundtrip', 'nested-79'}, 'TRANSFER_SOURCE_VARIANT')
        for name, rec in manifest['files'].items():
            require(not (payload / name).is_symlink() and sha(payload / name) == rec['sha256'], 'TRANSFER_SOURCE_FILES')
        verify_transfer_payload(payload, manifest)
        spec = importlib.util.spec_from_file_location('transfer_backend_admission', payload / 'probe_backend.py')
        B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
        spec = importlib.util.spec_from_file_location('transfer_source_admission', payload / 'transfer_admission.py')
        A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
        admission, roots = A.admit(B, payload / 'source-admission.json', manifest['source_admission_sha256'], payload / 'patches/params4bit-xla-v1.json')
        nested_proof={}
        if manifest.get('experiment')=='nested-79':
            verify_nested_payload(payload,manifest)
            spec=importlib.util.spec_from_file_location('nested_source_admission',payload/'probe_nested.py')
            N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)
            N.validate_admission(B,A,admission,A.load_manifest(B,payload/'patches/params4bit-xla-v1.json'))
            nested_proof={'source_variant':'nested-v1','nested_source_sha256':NESTED_SOURCE_SHA,
                          'plugin_source_manifest_sha256':NESTED_PLUGIN_MANIFEST_SHA}
        durable_json(a.base / 'records/installed-source.json', {'status': 'POST_PATCH_PYTHON_SOURCE_PASS',
                     'source_admission_sha256': manifest['source_admission_sha256'], 'patch_manifest_sha256': TRANSFER_PATCH_MANIFEST_SHA,
                     'installed_python_files': {name: admission[name]['files'] for name in roots},**nested_proof})
        return 0
    if a.operation == 'metadata':
        import importlib.metadata as md
        records = json.loads(a.config.read_text())['wheels']
        proof = {}
        for rec in records:
            dist = md.distribution(rec['name'])
            m = next(f for f in dist.files if str(f).endswith('.dist-info/METADATA') and len(PurePosixPath(str(f)).parts) == 2)
            require(dist.version == rec['version'] and sha(dist.locate_file(m)) == rec['metadata_sha256'], 'INSTALLED_METADATA')
            proof[rec['name']] = {'version': dist.version, 'metadata_sha256': rec['metadata_sha256'], 'wheel_sha256': rec['sha256']}
        durable_json(a.base / 'records/installed-metadata.json', {'python': platform.python_version(), 'executable': sys.executable, 'packages': proof})
        return 0
    if a.operation == 'export':
        out = a.base / 'records'
        receipt = json.loads((out / 'receipt.json').read_text())
        for key in ['evidence_sha256', 'evidence_bytes', 'archive_members_sha256']:
            receipt.pop(key, None)
        durable_json(out / 'receipt.json', receipt)
        try:
            receipt.update(package(out))
        except BaseException as error:
            receipt.update(status='BLOCKED_EXPORT', export_error={'type': type(error).__name__, 'message': str(error)})
        durable_json(out / 'receipt.json', receipt)
        print(json.dumps(receipt))
        return 0 if 'evidence_sha256' in receipt else 1
    return execute(a.base, a.packet_sha256, a.allocation_epoch, a.operation)


if __name__ == '__main__':
    raise SystemExit(main())
