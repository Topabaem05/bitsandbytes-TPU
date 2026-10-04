"""One admitted Colab session. Retrieve test records and stop the exact session."""
import argparse
import email
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'ownership'))
from cleanup_lifecycle import Ownership
from lifecycle import durable_json
from remote import sha, verify_archive


def clamp(started, limit, reserve, clock=None):
    value = min(limit, started + 3600 - (clock or time.time)() - reserve)
    if not math.isfinite(value) or value <= 0:
        raise TimeoutError('TOTAL_LIFECYCLE_RESERVE')
    return value


def preflight(packet, expected):
    if sha(packet / 'payload.zip') != expected:
        raise ValueError('PACKET_SHA')
    # Use the same consumer and canonical regular-file guard before any CLI instance.
    with tempfile.TemporaryDirectory(prefix='bnb-preflight-') as temporary:
        manifest = verify_archive(packet / 'payload.zip', expected, Path(temporary) / 'payload')
    for name, rec in manifest['files'].items():
        p = packet / name
        if p.is_symlink() or sha(p) != rec['sha256'] or p.stat().st_size != rec['bytes']:
            raise ValueError('PACKET_LOCAL_FILE')
    for p in HERE.rglob('*.py'):
        if 'tests' in p.relative_to(HERE).parts or p.name == 'build_packet.py':
            continue
        key = 'cloud/' + p.relative_to(HERE).as_posix()
        if manifest['files'].get(key, {}).get('sha256') != sha(p):
            raise ValueError('CURRENT_OWNER_SOURCE')
    return manifest


def verify_cli_identity(python, identity):
    """Read the admitted installed CLI files. Do not import auth or execute a command."""
    if identity.get('distribution') != 'google-colab-cli' or identity.get('version') != '0.7.4':
        raise ValueError('CLI_PROFILE')
    site = python.absolute().parent.parent / 'lib/python3.12/site-packages'
    metadata_path = site / 'google_colab_cli-0.7.4.dist-info/METADATA'
    record_path = site / 'google_colab_cli-0.7.4.dist-info/RECORD'
    cfg = python.absolute().parent.parent / 'pyvenv.cfg'
    if sha(python) != identity.get('python_sha256') or sha(cfg) != identity.get('venv_config_sha256') or sha(metadata_path) != identity.get('metadata_sha256') or sha(record_path) != identity.get('record_sha256'):
        raise ValueError('CLI_INTERPRETER_OR_DISTRIBUTION_SHA')
    meta = email.message_from_bytes(metadata_path.read_bytes())
    if meta['Name'] != 'google-colab-cli' or meta['Version'] != '0.7.4':
        raise ValueError('CLI_INSTALLED_VERSION')
    files = identity.get('files', {})
    actual = {p.relative_to(site).as_posix() for p in (site / 'colab_cli').rglob('*.py')}
    if not files or actual != set(files):
        raise ValueError('CLI_SOURCE_INVENTORY')
    for name, digest in files.items():
        p = site / name
        if not name.startswith('colab_cli/') or '..' in Path(name).parts or p.is_symlink() or sha(p) != digest:
            raise ValueError('CLI_SOURCE_SHA')


class OfficialCLI:
    def __init__(self, python, out):
        self.prefix = [str(python), '-B', str(HERE / 'official_cli_no_oauth.py')]
        self.out = out

    def __call__(self, label, argv, timeout):
        directory = self.out / 'cli' / label
        directory.mkdir(parents=True)
        owner = Ownership(directory)
        record = {'status': 'BLOCKED', 'argv': self.prefix + argv, 'timeout_seconds': timeout}
        stream = None
        try:
            stream = (directory / 'stdout-stderr.raw').open('wb', buffering=0)
            with owner.guard(timeout):
                p = owner.launch(record['argv'], record=directory / 'ownership.json', cwd=HERE,
                                 env={**{k: v for k, v in os.environ.items() if k not in {'PYTHONPATH', 'PYTHONHOME'}}, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1'}, stdout=stream, stderr=stream)
                p.wait()
                record.update(status='PASS' if p.returncode == 0 else 'CLI_FAILED', exit_code=p.returncode)
        except BaseException as error:
            record['error'] = {'type': type(error).__name__, 'message': str(error)}
        finally:
            record['original_outcome'] = {'status': record['status'], 'exit_code': record.get('exit_code'), 'error': record.get('error')}
            if stream is not None:
                try:
                    os.fsync(stream.fileno())
                except Exception as error:
                    owner.note_error('fsync_cli_log', error)
                finally:
                    try:
                        stream.close()
                    except Exception as error:
                        owner.note_error('close_cli_log', error)
            record['cleanup'] = owner.summary()
            if record['cleanup']['errors']:
                record['status'] = 'BLOCKED_CLEANUP'
            durable_json(directory / 'result.json', record)
        text = (directory / 'stdout-stderr.raw').read_text(errors='replace') if (directory / 'stdout-stderr.raw').exists() else ''
        return record, text


def verify_result(out, receipt):
    inventory = json.loads((out / 'archive-members.json').read_text())
    if sha(out / 'archive-members.json') != receipt['archive_members_sha256'] or sha(out / 'evidence.zip') != receipt['evidence_sha256'] or (out / 'evidence.zip').stat().st_size != receipt['evidence_bytes']:
        raise ValueError('RESULT_WHOLE_OR_INVENTORY')
    with zipfile.ZipFile(out / 'evidence.zip') as z:
        if set(z.namelist()) != set(inventory) or len(z.infolist()) != len(inventory):
            raise ValueError('RESULT_MEMBER_SET')
        for info in z.infolist():
            p = Path(info.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in info.filename or info.create_system != 3 or info.external_attr != (0o100644 << 16):
                raise ValueError('RESULT_NONREGULAR_OR_PATH')
            raw = z.read(info)
            import hashlib
            if len(raw) != inventory[info.filename]['bytes'] or hashlib.sha256(raw).hexdigest() != inventory[info.filename]['sha256']:
                raise ValueError('RESULT_MEMBER_SHA')
        archived = json.loads(z.read('receipt.json'))
        if archived != {k: v for k, v in receipt.items() if k not in {'evidence_sha256', 'evidence_bytes', 'archive_members_sha256'}}:
            raise ValueError('RESULT_RECEIPT_PROJECTION')
        target = out / 'recovered'
        if target.exists():
            raise ValueError('FRESH_RECOVERY_TARGET')
        target.mkdir()
        for name in z.namelist():
            p = target / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(z.read(name))
    return inventory


def drive(packet, output, expected, acceptance, acceptance_sha, cli_python, cli_identity, *, api=None, simulated=False):
    manifest = preflight(packet, expected)
    if sha(acceptance) != acceptance_sha:
        raise PermissionError('ROOT_ACCEPTANCE_SHA')
    gate = json.loads(acceptance.read_text())
    fields = {'status': 'ACTUAL_DISPATCH_AUTHORIZED', 'packet_sha256': expected,
              'driver_sha256': sha(HERE / 'owner.py'), 'output': str(output.resolve()),
              'plugin_source_manifest_sha256': manifest['plugin_source_manifest_sha256'],
              'runtime_lock_sha256': manifest['runtime_lock_sha256'],
              'budget': manifest['budget'], 'one_allocation_only': True, 'provider_or_solver': 'FORBIDDEN'}
    if any(gate.get(k) != v for k, v in fields.items()):
        raise PermissionError('EXACT_ROOT_ACCEPTANCE_REQUIRED')
    if gate.get('cli_identity_sha256') != sha(cli_identity):
        raise PermissionError('CLI_IDENTITY_ADMISSION')
    verify_cli_identity(cli_python, json.loads(cli_identity.read_text()))
    if output.exists() or output.is_symlink():
        raise FileExistsError('FRESH_OUTPUT_REQUIRED')
    if simulated and api is None:
        raise ValueError('SIMULATION_REQUIRES_FAKE_API')
    output.mkdir(parents=True)
    durable_json(output / 'root-acceptance.json', gate)
    api = api or OfficialCLI(cli_python, output)
    session = 'bnb-tpu-first-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    base = '/content/bnb-tpu-first'
    owner = {'status': 'RUNNING', 'mode': 'SIMULATED_NO_CLOUD' if simulated else 'ACTUAL_COLAB',
             'session': session, 'allocation_attempts': 0, 'commands': [], 'cleanup_errors': [],
             'packet_sha256': expected, 'budget': manifest['budget'], 'remote_base': base,
             'provider_or_solver': 'NOT_RUN', 'started_utc': datetime.now(timezone.utc).isoformat()}
    started = None
    prior = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def interrupted(signum, frame):
        for sig in prior:
            signal.signal(sig, signal.SIG_IGN)
        raise KeyboardInterrupt('owned experiment interrupted')

    def call(label, argv, limit=30, reserve=660, cleanup=False):
        # Cleanup always has an independent bounded allowance, including an overdue run.
        timeout = limit if cleanup or started is None else clamp(started, limit, reserve)
        record, text = api(label, argv, timeout)
        owner['commands'].append({'label': label, **record})
        durable_json(output / 'owner.json', owner)
        if record['status'] != 'PASS':
            raise RuntimeError('CLI_STEP_BLOCKED:' + label)
        return text

    def upload(label, source, remote, reserve=660):
        return call(label, ['upload', '-s', session, str(source), remote.lstrip('/')], reserve=reserve)

    def download(label, remote, local, reserve=60):
        local.parent.mkdir(parents=True, exist_ok=True)
        return call(label, ['download', '-s', session, remote.lstrip('/'), str(local)], reserve=reserve)

    def kernel(label, code, limit, reserve=660):
        path = output / (label + '.py')
        path.write_text(code)
        timeout = math.floor(clamp(started, limit, reserve))
        return call(label, ['exec', '-s', session, '--timeout', str(timeout), '-f', str(path)], timeout, reserve)

    def operation(label, phase, limit, reserve=660):
        # Bind the initial control source before importing its ownership helpers.
        identity = {name.removeprefix('cloud/'): rec['sha256'] for name, rec in manifest['files'].items() if name.startswith('cloud/') and name.endswith('.py')}
        code = ('import hashlib,json,runpy,sys\nfrom pathlib import Path\n'
                f'b=Path({base!r}); expected={identity!r}\n'
                'for n,h in expected.items():\n'
                ' p=b/"control"/n\n'
                ' if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=h: raise ValueError("CONTROL_SOURCE_HASH")\n'
                f'sys.argv=["remote.py",{phase!r},"--base",str(b),"--packet-sha256",{expected!r},"--allocation-epoch",{str(started)!r}]\n'
                'runpy.run_path(str(b/"control/remote.py"),run_name="__main__")\n')
        return kernel(label, code, limit, reserve)

    def phase_receipt(label, expected_status):
        path = output / (label + '-receipt.json')
        download(label + '-receipt', base + '/records/receipt.json', path, reserve=660)
        r = json.loads(path.read_text())
        if r.get('packet_sha256') != expected or r.get('manifest_sha256') != sha(packet / 'manifest.json') or r.get('status') != expected_status:
            raise ValueError('REMOTE_PHASE_BLOCKED:' + label)
        return r

    receipt = None
    try:
        for sig in prior:
            signal.signal(sig, interrupted)
        if 'No active sessions found on server.' not in call('01-sessions-before', ['sessions'], 60):
            raise RuntimeError('ACTIVE_SESSION_NO_ALLOCATION')
        call('02-usage-before', ['usage'], 60)
        started = time.time()
        owner.update(allocation_attempts=1, allocation_epoch=started, deadline_utc=datetime.fromtimestamp(started + 3600, timezone.utc).isoformat())
        durable_json(output / 'owner.json', owner)
        call('03-one-allocation', ['new', '-s', session, '--tpu', 'v6e1'], 180)
        text = call('04-sessions-after', ['sessions'], 60)
        if '[' + session + ']' not in text or 'Hardware: V6E1' not in text:
            raise RuntimeError('OWNED_V6E1_NOT_OBSERVED')
        readiness = ('from pathlib import Path\nimport json,platform,sys,os\n'
                     f'b=Path({base!r})\n'
                     'if b.exists() or b.is_symlink(): raise ValueError("FRESH_REMOTE_ROOT_REQUIRED")\n'
                     'b.mkdir(mode=0o700); (b/"control/ownership").mkdir(parents=True)\n'
                     'print(json.dumps({"python":platform.python_version(),"libc":platform.libc_ver(),"executable":sys.executable,"uid":os.getuid(),"gid":os.getgid(),"system":platform.system(),"machine":platform.machine(),"parents":["control","control/ownership"]}))\n')
        owner['kernel_readiness_stdout'] = kernel('05-readiness', readiness, 30)
        for name in sorted(manifest['files']):
            if name.startswith('cloud/') and name.endswith('.py'):
                upload('06-' + name.replace('/', '-'), packet / name, base + '/control/' + name.removeprefix('cloud/'))
        upload('07-payload', packet / 'payload.zip', base + '/payload.zip')
        operation('08-unpack', 'unpack', 60)
        operation('09-install', 'install', 1200)
        phase_receipt('09-install', 'INSTALLED_NOT_QUALIFIED')
        operation('10-cpu', 'cpu', 1800)
        phase_receipt('10-cpu', 'CPU_ORACLE_READY_TPU_NOT_RUN')
        download('11-cpu-seal', base + '/records/cpu-oracle/oracle-seal.json', output / 'cpu-oracle-seal.json', reserve=660)
        seal = json.loads((output / 'cpu-oracle-seal.json').read_text())
        if seal.get('runtime_lock_sha256') != manifest['runtime_lock_sha256'] or seal.get('source_admission_sha256') != manifest['source_admission_sha256']:
            raise ValueError('RECOVERED_ORACLE_BINDING')
        owner['recovered_oracle_sha256'] = sha(output / 'cpu-oracle-seal.json')
        launch = output / 'launch.json'
        durable_json(launch, {'oracle_sha256': owner['recovered_oracle_sha256']})
        upload('12-oracle-admission', launch, base + '/launch.json')
        operation('13-tpu', 'tpu', 1800)
        phase_receipt('13-tpu', 'TPU_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED')
        owner['status'] = 'CHILD_TERMINAL_RETRIEVAL_REQUIRED'
    except BaseException as error:
        owner.update(status='BLOCKED', original_error={'type': type(error).__name__, 'message': str(error)})
    finally:
        for sig in prior:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if started is not None:
                # Retrieval is independent of the child result. Never repeat a science phase.
                try:
                    download('20-critical-receipt', base + '/records/receipt.json', output / 'receipt-before-export.json')
                    operation('21-export', 'export', 90, 60)
                    download('22-terminal-receipt', base + '/records/receipt.json', output / 'receipt.json')
                    receipt = json.loads((output / 'receipt.json').read_text())
                    download('23-inventory', base + '/records/archive-members.json', output / 'archive-members.json')
                    # The full result may exceed the official per-file transport size.
                    code = ('import sys\nfrom pathlib import Path\n'
                            f'sys.path.insert(0,{(base + "/control")!r})\n'
                            'import transport\n'
                            f'b=Path({base!r});p=b/"records/evidence.zip"\n'
                            'transport.split(p,b/"result-parts",expected_sha256=transport.digest(p),expected_bytes=p.stat().st_size)\n')
                    kernel('24-export-parts', code, 60, 60)
                    download('25-parts-manifest', base + '/result-parts/manifest.json', output / 'parts-manifest.json')
                    import transport
                    parts = transport.validate_manifest(json.loads((output / 'parts-manifest.json').read_text()), receipt['evidence_sha256'], receipt['evidence_bytes'])
                    target = output / 'parts'
                    target.mkdir()
                    for index, part in enumerate(parts['parts']):
                        download('26-part-' + str(index), base + '/result-parts/' + part['name'], target / part['name'])
                    transport.assemble(output / 'parts-manifest.json', target, output / 'evidence.zip', expected_manifest_sha256=sha(output / 'parts-manifest.json'), expected_sha256=receipt['evidence_sha256'], expected_bytes=receipt['evidence_bytes'])
                    owner['retrieval'] = 'COMPLETE_WHOLE_ARCHIVE'
                except BaseException as error:
                    owner['retrieval_error'] = {'type': type(error).__name__, 'message': str(error)}
                for label, argv, limit, field in [
                    ('90-before-stop', ['sessions'], 15, 'sessions_before_stop'),
                    ('91-stop-exact', ['stop', '-s', session], 25, 'stop_exact'),
                    ('92-after-stop', ['sessions'], 10, 'sessions_after_stop'),
                    ('93-usage-after', ['usage'], 10, 'usage_after_stop')]:
                    try:
                        owner[field] = call(label, argv, limit, 0, cleanup=True)
                    except BaseException as error:
                        owner['cleanup_errors'].append({'operation': label, 'type': type(error).__name__, 'message': str(error)})
                owner['server_empty_observed'] = 'No active sessions found on server.' in owner.get('sessions_after_stop', '')
                owner['usage_zero_observed'] = 'Active assignments: 0' in owner.get('usage_after_stop', '') and 'Usage rate: 0.00/hr' in owner.get('usage_after_stop', '')
                owner['elapsed_lifecycle_seconds'] = time.time() - started
                if not owner['server_empty_observed'] or not owner['usage_zero_observed'] or owner['cleanup_errors'] or owner['elapsed_lifecycle_seconds'] > 3600:
                    owner['status'] = 'BLOCKED_CLEANUP'
            else:
                owner['allocation'] = 'NOT_ATTEMPTED'
        finally:
            owner['handler_restoration_errors'] = []
            for sig, handler in prior.items():
                try:
                    signal.signal(sig, handler)
                except Exception as error:
                    owner['handler_restoration_errors'].append(str(error))
            if owner['handler_restoration_errors']:
                owner['status'] = 'BLOCKED_CLEANUP'
            durable_json(output / 'owner.json', owner)
    if owner.get('retrieval') == 'COMPLETE_WHOLE_ARCHIVE':
        try:
            inventory = verify_result(output, receipt)
            owner['whole_archive_verified'] = True
            # The existing probe verifier uses retained arrays; no TPU or CPU science is repeated.
            verify = Ownership(output)
            log = (output / 'local-verifier.raw').open('wb', buffering=0)
            try:
                with verify.guard(clamp(started, 120, 0)):
                    proc = verify.launch([sys.executable, '-B', str(packet / 'probe_backend.py'), 'verify', '--admission-sha256', manifest['source_admission_sha256'], '--oracle', str(output / 'recovered/cpu-oracle'), '--oracle-sha256', owner['recovered_oracle_sha256'], '--actual', str(output / 'recovered/tpu-actual')], record=output / 'local-verifier-ownership.json', env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), stdout=log, stderr=log)
                    proc.wait()
                    owner['local_verifier_exit_code'] = proc.returncode
            finally:
                try:
                    os.fsync(log.fileno())
                except Exception as error:
                    verify.note_error('fsync_local_verifier', error)
                finally:
                    try:
                        log.close()
                    except Exception as error:
                        verify.note_error('close_local_verifier', error)
                owner['local_verifier_cleanup'] = verify.summary()
            if owner['status'] == 'CHILD_TERMINAL_RETRIEVAL_REQUIRED' and not owner['local_verifier_cleanup']['errors'] and owner.get('local_verifier_exit_code') in (0, 2) and receipt['runtime_status'] == 'PASS_TPU_RUNTIME_PROBE_ONLY' and receipt['cpu_status'] == 'PASS' and receipt['tpu_status'] in ('PASS', 'FAIL') and all(not step['cleanup']['errors'] for step in receipt['steps']):
                owner['status'] = 'PASS_TPU_API_PROBE' if owner['local_verifier_exit_code'] == 0 else 'FAIL_TPU_API_PROBE'
        except BaseException as error:
            owner['local_readback_error'] = {'type': type(error).__name__, 'message': str(error)}
    owner['finished_utc'] = datetime.now(timezone.utc).isoformat()
    durable_json(output / 'owner.json', owner)
    return owner


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--packet-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--root-acceptance', type=Path, required=True)
    p.add_argument('--root-acceptance-sha256', required=True)
    p.add_argument('--cli-python', type=Path, required=True)
    p.add_argument('--cli-identity', type=Path, required=True)
    p.add_argument('--preflight-only', action='store_true')
    a = p.parse_args()
    if a.preflight_only:
        m = preflight(a.packet, a.packet_sha256)
        verify_cli_identity(a.cli_python, json.loads(a.cli_identity.read_text()))
        print(json.dumps({'status': 'LOCAL_PREFLIGHT_NO_CLI', 'files': len(m['files']), 'CLI_file_identity': 'PASS_NO_API'}))
    else:
        r = drive(a.packet.resolve(), a.output.resolve(), a.packet_sha256, a.root_acceptance, a.root_acceptance_sha256, a.cli_python, a.cli_identity)
        print(json.dumps(r))
        raise SystemExit(0 if r['status'] == 'PASS_TPU_API_PROBE' else 2)
