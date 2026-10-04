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


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


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
    out.mkdir(exist_ok=True)
    receipt_path = out / 'receipt.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {
        'status': 'RUNNING', 'runtime_status': 'NOT_RUN', 'cpu_status': 'NOT_RUN', 'tpu_status': 'NOT_RUN',
        'packet_sha256': packet_sha, 'manifest_sha256': sha(payload / 'manifest.json'),
        'runtime_lock_sha256': manifest['runtime_lock_sha256'], 'source_admission_sha256': manifest['source_admission_sha256'],
        'allocation_epoch': allocation_epoch, 'steps': [], 'error': None}
    work_deadline = allocation_epoch + 3600 - 600 - 60
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
            unpack_source(payload / 'upstream.tar', base / 'upstream')
            step('06-build-upstream', [installed, '-B', '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation', '--no-cache-dir', '--wheel-dir', base / 'built-upstream', base / 'upstream'], deadline, 180, extra_env={'BNB_SKIP_CMAKE': '1'})
            step('07-build-plugin', [installed, '-B', '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation', '--no-cache-dir', '--wheel-dir', base / 'built-plugin', payload / 'plugin'], deadline, 120)
            wheels = [*sorted((base / 'built-upstream').glob('*.whl')), *sorted((base / 'built-plugin').glob('*.whl'))]
            require(len(wheels) == 2, 'BUILT_WHEEL_COUNT')
            receipt['built_wheels'] = [{'name': p.name, 'sha256': sha(p), 'bytes': p.stat().st_size} for p in wheels]
            step('08-install-source-wheels', [installed, '-B', '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-compile', *wheels], deadline, 120)
            # Query installed distribution records without native imports.
            config = payload / 'installed-proof-config.json'
            durable_json(config, {'wheels': list(all_records.values())})
            step('09-installed-metadata', [installed, '-B', payload / 'cloud/remote.py', 'metadata', '--config', config, '--base', base], deadline, 30)
            receipt.update(status='INSTALLED_NOT_QUALIFIED', installation_status='PASS')
        elif phase == 'cpu':
            require(receipt.get('installation_status') == 'PASS' and receipt['cpu_status'] == 'NOT_RUN', 'CPU_PHASE_ORDER')
            receipt['science_start_epoch'] = time.time()
            receipt['science_deadline_epoch'] = min(work_deadline, time.time() + 1800)
            deadline = receipt['science_deadline_epoch']
            step('10-runtime-probe', [installed, '-B', payload / 'runtime/probe_runtime.py', '--out', out / 'runtime-probe.json'], deadline, 120, tpu=True)
            require(json.loads((out / 'runtime-probe.json').read_text())['status'] == 'PASS_TPU_RUNTIME_PROBE_ONLY', 'RUNTIME_PROBE_BLOCKED')
            receipt['runtime_status'] = 'PASS_TPU_RUNTIME_PROBE_ONLY'
            step('11-cpu-oracle', [installed, '-B', payload / 'probe_backend.py', 'prepare', '--admission', payload / 'source-admission.json', '--admission-sha256', manifest['source_admission_sha256'], '--output', out / 'cpu-oracle'], deadline, 300)
            receipt.update(status='CPU_ORACLE_READY_TPU_NOT_RUN', cpu_status='PASS', oracle_sha256=sha(out / 'cpu-oracle/oracle-seal.json'))
        elif phase == 'tpu':
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
        if phase == 'tpu' and receipt.get('tpu_status') == 'RUNNING':
            receipt['tpu_status'] = 'ATTEMPTED_BLOCKED'
        receipt.update(status='BLOCKED', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        durable_json(receipt_path, receipt)
    print(json.dumps({'status': receipt['status'], 'phase': phase, 'error': receipt['error']}))
    return 0 if receipt['status'] != 'BLOCKED' else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=['unpack', 'install', 'cpu', 'tpu', 'export', 'download', 'metadata'])
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
