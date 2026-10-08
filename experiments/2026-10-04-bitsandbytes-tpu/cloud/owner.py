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
import stat
import sys
import tempfile
import time
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'ownership'))
from cleanup_lifecycle import Ownership
from lifecycle import durable_json
import native_contract as NC
import primitive_contract as PC
from remote import sha, verify_archive, verify_experiment, verify_transfer_payload, validate_source_controls, TRANSFER_BINDINGS, TRANSFER_PATCH_MANIFEST_SHA, STATE_BINDINGS
from nested_contract import (NESTED_BINDINGS,NESTED_SCOPE,NESTED_SOURCE_SHA,NESTED_PLUGIN_MANIFEST_SHA,
    NESTED_FILES,verify_nested_payload,nested_cli)

from browser_adoption import MODE as BROWSER_MODE, record as browser_record

from nested_state_contract import (NESTED_STATE_BINDINGS,NESTED_STATE_SCOPE,verify_state_payload,state_cli)


class _NativeReadbackUnavailable(Exception):
    """The archive is recoverable, but no terminal native proof can be verified."""


def native_readback_missing(owner, receipt, output, native_gate):
    missing=[]
    if not isinstance(native_gate,dict) or native_gate.get('status')!='QUALIFIED_LINUX_CPU_ORACLE_VERIFIED':missing.append('root_cpu_oracle_gate')
    if 'recovered_oracle_sha256' not in owner:missing.append('recovered_cpu_oracle_seal')
    if receipt.get('status')!='NATIVE_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED' or receipt.get('tpu_status')!='NATIVE_RECORDS_TERMINAL':missing.append('terminal_native_receipt')
    for field in ('native_expected_sha256','native_outer_owner_sha256','native_parent_sha256'):
        if field not in receipt:missing.append(field)
    for name in ('native/expected.json','native/parent.json','steps/12-native-parent/ownership.json'):
        if not (output/'recovered'/name).is_file():missing.append(name)
    return missing


def clamp(started, limit, reserve, clock=None):
    value = min(limit, started + 3600 - (clock or time.time)() - reserve)
    if not math.isfinite(value) or value <= 0:
        raise TimeoutError('TOTAL_LIFECYCLE_RESERVE')
    return value


def preflight(packet, expected):
    loose_manifest = packet / 'manifest.json'
    if not stat.S_ISREG(loose_manifest.lstat().st_mode):
        raise ValueError('PACKET_LOCAL_MANIFEST_NONREGULAR')
    if sha(packet / 'payload.zip') != expected:
        raise ValueError('PACKET_SHA')
    # Use the same consumer and canonical regular-file guard before any CLI instance.
    with tempfile.TemporaryDirectory(prefix='bnb-preflight-') as temporary:
        target = Path(temporary) / 'payload'
        manifest = verify_archive(packet / 'payload.zip', expected, target)
        if loose_manifest.read_bytes() != (target / 'manifest.json').read_bytes():
            raise ValueError('PACKET_LOCAL_MANIFEST_BYTES')
    verify_experiment(manifest)
    for name, rec in manifest['files'].items():
        p = packet / name
        if p.is_symlink() or sha(p) != rec['sha256'] or p.stat().st_size != rec['bytes']:
            raise ValueError('PACKET_LOCAL_FILE')
    if manifest.get('experiment') in {'transfer-api42', 'state-roundtrip', 'nested-79', 'nested-state-8', 'm6-native-boundary', PC.MODE}:
        verify_transfer_payload(packet, manifest)
    if manifest.get('experiment') in {'nested-79','nested-state-8',PC.MODE}:
        verify_nested_payload(packet,manifest)
    if manifest.get('experiment')=='nested-state-8':
        verify_state_payload(packet,manifest)
    if manifest.get('experiment')==NC.MODE: NC.payload(packet,manifest)
    if manifest.get('experiment')==PC.MODE:PC.payload(packet,manifest)
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


def drive(packet, output, expected, acceptance, acceptance_sha, cli_python, cli_identity, *, api=None, simulated=False, browser_adoption=None, browser_adoption_sha256=None):
    manifest = preflight(packet, expected)
    if sha(acceptance) != acceptance_sha:
        raise PermissionError('ROOT_ACCEPTANCE_SHA')
    gate = json.loads(acceptance.read_text())
    fields = {'status': 'ACTUAL_DISPATCH_AUTHORIZED', 'packet_sha256': expected,
              'driver_sha256': sha(HERE / 'owner.py'), 'output': str(output.resolve()),
              'plugin_source_manifest_sha256': manifest['plugin_source_manifest_sha256'],
              'runtime_lock_sha256': manifest['runtime_lock_sha256'],
              'budget': manifest['budget'], 'one_allocation_only': True, 'provider_or_solver': 'FORBIDDEN'}
    experiment = manifest.get('experiment', 'api42')
    primitive=experiment==PC.MODE
    diagnostic = experiment in {'device-route-diagnostic', 'precision-diagnostic'}
    precision = experiment == 'precision-diagnostic'
    nested_state = experiment=='nested-state-8'
    nested = experiment in {'nested-79','nested-state-8',PC.MODE}
    state = experiment == 'state-roundtrip'
    native = experiment == 'm6-native-boundary'
    transfer = experiment in {'transfer-api42', 'state-roundtrip', 'nested-79', 'nested-state-8', 'm6-native-boundary', PC.MODE}
    if diagnostic:
        fields.update(experiment=experiment, diagnostic_only=True, route_probe_sha256=manifest['route_probe_sha256'])
    if precision:
        fields['precision_probe_sha256'] = manifest['precision_probe_sha256']
    if transfer:
        fields.update({field: manifest[field] for field in TRANSFER_BINDINGS})
        fields.update(experiment=experiment, precision='highest')
        if nested:
            fields.update({field:manifest[field] for field in NESTED_BINDINGS})
            fields.update(nested_scope=NESTED_SCOPE,nested_source_variant='nested-v1')
        if nested_state:
            fields.update({field:manifest[field] for field in NESTED_STATE_BINDINGS})
            fields.update(nested_state_scope=NESTED_STATE_SCOPE,nested_state_variant='nested-state-v1')
        if state:
            fields.update({field: manifest[field] for field in STATE_BINDINGS})
            fields['state_scope'] = manifest['state_scope']
    if native:
        fields.update(native_manifest_sha256=NC.MANIFEST_SHA,m2_dependency_sha256=manifest['m2_dependency_sha256'],native_source_variant=NC.VARIANT,native_scope=manifest['native_scope'])
    if primitive:
        fields.update(**{field:manifest[field] for field in PC.BINDINGS},primitive_variant=PC.VARIANT,
                      primitive_scope=PC.SCOPE,primitive_diagnostic_only=True)
    adopting=browser_adoption is not None or browser_adoption_sha256 is not None
    adoption=None
    if adopting:
        if browser_adoption is None or browser_adoption_sha256 is None:raise PermissionError('BROWSER_ADOPTION_ARGUMENTS')
        adoption=browser_record(browser_adoption,browser_adoption_sha256)
        if adoption['packet_sha256']!=expected or adoption['driver_sha256']!=sha(HERE/'owner.py') or adoption['cli_identity_sha256']!=sha(cli_identity):raise PermissionError('BROWSER_ADOPTION_SOURCE_BINDING')
        fields.update(runtime_mode=BROWSER_MODE,browser_adoption_sha256=browser_adoption_sha256,
            adopted_endpoint=adoption['endpoint'],adopted_session=adoption['session'],adopted_hardware=adoption['hardware'],
            original_allocation_epoch=adoption['allocation_epoch'],marker_path=adoption['marker_path'],marker_sha256=adoption['marker_sha256'])
    elif any(k in gate for k in ('runtime_mode','browser_adoption_sha256','adopted_endpoint','adopted_session','adopted_hardware','original_allocation_epoch','marker_path','marker_sha256')):
        raise PermissionError('BROWSER_ADOPTION_NOT_REQUESTED')
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
    session = adoption['session'] if adopting else 'bnb-tpu-first-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    base = '/content/bnb-tpu-first'
    owner = {'status': 'RUNNING', 'mode': 'SIMULATED_NO_CLOUD' if simulated else 'ACTUAL_COLAB',
             'session': session, 'allocation_attempts': 0, 'commands': [], 'cleanup_errors': [],
             'packet_sha256': expected, 'budget': manifest['budget'], 'remote_base': base,
             'provider_or_solver': 'NOT_RUN', 'started_utc': datetime.now(timezone.utc).isoformat()}
    if diagnostic:
        owner.update(experiment=experiment, diagnostic_only=True, api42_status='NOT_QUALIFIED')
    if precision:
        owner['m3_status'] = 'NOT_QUALIFIED'
    if transfer:
        owner.update(experiment=experiment, precision='highest', api42_status='NOT_QUALIFIED', m3_status='NOT_QUALIFIED')
        if state:
            owner['api42_status'] = 'NOT_REPEATED'
    if native: owner.update(m6_status='NOT_QUALIFIED',m4_source_compatibility='NOT_QUALIFIED',native_source_variant=NC.VARIANT,optimized_executable_link='UNKNOWN',compiler_body_memory_allocator='NOT_QUALIFIED')
    adoption_verified=False
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
        if (precision or transfer) and (r.get('experiment') != experiment or
                          (r.get('diagnostic_only') is not True if precision else r.get('precision') != 'highest') or
                          r.get('route_probe_sha256') != manifest['route_probe_sha256'] or
                          r.get('precision_probe_sha256') != manifest['precision_probe_sha256'] or
                          r.get('source_admission_sha256') != manifest['source_admission_sha256'] or
                          r.get('runtime_lock_sha256') != manifest['runtime_lock_sha256'] or
                          r.get('allocation_epoch') != started):
            raise ValueError(('TRANSFER' if transfer else 'PRECISION') + '_PHASE_RECEIPT_BINDING:' + label)
        if transfer and any(r.get(field) != manifest[field] for field in TRANSFER_BINDINGS):
            raise ValueError('TRANSFER_PHASE_SOURCE_BINDING:' + label)
        if nested and any(r.get(field)!=manifest[field] for field in NESTED_BINDINGS):
            raise ValueError('NESTED_REMOTE_PHASE_BINDING:'+label)
        if nested_state and any(r.get(field)!=manifest[field] for field in NESTED_STATE_BINDINGS):
            raise ValueError('NESTED_STATE_REMOTE_PHASE_BINDING:'+label)
        if state and any(r.get(field) != manifest[field] for field in STATE_BINDINGS):
            raise ValueError('STATE_PHASE_SOURCE_BINDING:' + label)
        if primitive and (any(r.get(field)!=manifest[field] for field in PC.BINDINGS) or
                r.get('primitive_variant')!=PC.VARIANT or r.get('primitive_scope')!=PC.SCOPE or r.get('primitive_diagnostic_only') is not True):
            raise ValueError('PRIMITIVE_REMOTE_PHASE_BINDING:'+label)
        return r

    receipt = None
    native_gate = None
    primitive_gate = None
    try:
        for sig in prior:
            signal.signal(sig, interrupted)
        before=call('01-sessions-before', ['sessions'], 60)
        if not adopting and 'No active sessions found on server.' not in before:
            raise RuntimeError('ACTIVE_SESSION_NO_ALLOCATION')
        call('02-usage-before', ['usage'], 60)
        started=adoption['allocation_epoch'] if adopting else time.time()
        owner.update(allocation_attempts=0 if adopting else 1,allocation_epoch=started,deadline_utc=datetime.fromtimestamp(started+3600,timezone.utc).isoformat())
        if adopting:
            owner.update(runtime_mode=BROWSER_MODE,browser_adoption_sha256=browser_adoption_sha256,adoption_verified=False,allocation='NOT_ATTEMPTED_BROWSER_RUNTIME')
        durable_json(output / 'owner.json', owner)
        if adopting:
            call('03-browser-registration',['--adopt-root-browser-v1',str(browser_adoption),browser_adoption_sha256],60)
        else:
            call('03-one-allocation', ['--allocation-transport-v1', 'new', '-s', session, '--tpu', 'v6e1'], 360)
        text=call('04-sessions-after',['sessions'],60)
        if adopting:
            assignment_lines=[line for line in text.splitlines() if 'Hardware:' in line]
            segments=assignment_lines[0].split(' | ') if len(assignment_lines)==1 else []
            if not segments or segments[0]!='['+session+'] '+adoption['endpoint'] or segments.count('Hardware: V6E1')!=1 or segments.count('Variant: TPU')!=1:raise RuntimeError('ADOPTED_SESSION_NOT_OBSERVED')
            marker_code=('import hashlib,json\nfrom pathlib import Path\n'+f'p=Path({adoption["marker_path"]!r})\n'+
                f'assert p.is_file() and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()=={adoption["marker_sha256"]!r}, "ROOT_BROWSER_MARKER_MISMATCH"\n'+
                f'print(json.dumps({{"status":"ROOT_BROWSER_MARKER_MATCH","marker_sha256":{adoption["marker_sha256"]!r}}},sort_keys=True))\n')
            proof=kernel('04b-browser-marker',marker_code,30)
            expected_proof={'status':'ROOT_BROWSER_MARKER_MATCH','marker_sha256':adoption['marker_sha256']}
            if not any(line.strip().startswith('{') and json.loads(line)==expected_proof for line in proof.splitlines() if line.strip().startswith('{')):raise RuntimeError('ROOT_BROWSER_MARKER_NOT_PROVED')
            adoption_verified=True;owner.update(adoption_verified=True,marker_readback=expected_proof,adopted_endpoint_sha256=__import__('hashlib').sha256(adoption['endpoint'].encode()).hexdigest())
            durable_json(output/'owner.json',owner)
        elif '['+session+']' not in text or 'Hardware: V6E1' not in text:
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
        cpu_receipt=phase_receipt('10-cpu', 'CPU_ORACLE_READY_TPU_NOT_RUN')
        download('11-cpu-seal', base + '/records/cpu-oracle/oracle-seal.json', output / 'cpu-oracle-seal.json', reserve=660)
        seal = json.loads((output / 'cpu-oracle-seal.json').read_text())
        if primitive:
            primitive_cpu=output/'primitive-cpu';primitive_cpu.mkdir()
            download('11a-primitive-inventory',base+'/records/primitive-cpu-inventory.json',primitive_cpu/'inventory.json',reserve=660)
            download('11b-primitive-parts',base+'/primitive-cpu-parts/manifest.json',primitive_cpu/'parts-manifest.json',reserve=660)
            import transport
            parts=transport.validate_manifest(json.loads((primitive_cpu/'parts-manifest.json').read_text()),cpu_receipt['primitive_cpu_evidence_sha256'],cpu_receipt['primitive_cpu_evidence_bytes'])
            (primitive_cpu/'parts').mkdir()
            for i,part in enumerate(parts['parts']):
                download('11c-primitive-part-'+str(i),base+'/primitive-cpu-parts/'+part['name'],primitive_cpu/'parts'/part['name'],reserve=660)
            transport.assemble(primitive_cpu/'parts-manifest.json',primitive_cpu/'parts',primitive_cpu/'evidence.zip',
                               expected_manifest_sha256=sha(primitive_cpu/'parts-manifest.json'),
                               expected_sha256=cpu_receipt['primitive_cpu_evidence_sha256'],expected_bytes=cpu_receipt['primitive_cpu_evidence_bytes'])
            recovered_cpu=PC.recover_cpu(primitive_cpu,cpu_receipt)
            primitive_gate,cpu_observations=PC.cpu_gate(packet,recovered_cpu,cpu_receipt)
            PC.verify_root_gate(primitive_gate,cpu_receipt,manifest)
            durable_json(output/'primitive-cpu-validation.json',cpu_observations)
            owner['primitive_root_cpu_gate']=primitive_gate
            if sha(output/'cpu-oracle-seal.json')!=primitive_gate['oracle_sha256']:raise ValueError('PRIMITIVE_STANDALONE_ARCHIVED_ORACLE_SEAL')
        elif native:
            cpu_receipt=phase_receipt('11-native-cpu','CPU_ORACLE_READY_TPU_NOT_RUN')
            native_cpu=output/'native-cpu';native_cpu.mkdir()
            download('11a-native-inventory',base+'/records/native-cpu-inventory.json',native_cpu/'inventory.json',reserve=660)
            download('11b-native-parts',base+'/native-cpu-parts/manifest.json',native_cpu/'parts-manifest.json',reserve=660)
            import transport
            parts=transport.validate_manifest(json.loads((native_cpu/'parts-manifest.json').read_text()),cpu_receipt['native_cpu_evidence_sha256'],cpu_receipt['native_cpu_evidence_bytes']);(native_cpu/'parts').mkdir()
            for i,part in enumerate(parts['parts']): download('11c-native-part-'+str(i),base+'/native-cpu-parts/'+part['name'],native_cpu/'parts'/part['name'],reserve=660)
            transport.assemble(native_cpu/'parts-manifest.json',native_cpu/'parts',native_cpu/'evidence.zip',expected_manifest_sha256=sha(native_cpu/'parts-manifest.json'),expected_sha256=cpu_receipt['native_cpu_evidence_sha256'],expected_bytes=cpu_receipt['native_cpu_evidence_bytes'])
            recovered_cpu=NC.recover_cpu(native_cpu,cpu_receipt);native_gate=NC.cpu_gate(packet,recovered_cpu,cpu_receipt);owner['native_root_oracle_gate']=native_gate
            if sha(output/'cpu-oracle-seal.json')!=native_gate['oracle_sha256']:raise ValueError('NATIVE_STANDALONE_ARCHIVED_ORACLE_SEAL')
        elif seal.get('runtime_lock_sha256') != manifest['runtime_lock_sha256'] or seal.get('source_admission_sha256') != manifest['source_admission_sha256']:
            raise ValueError('RECOVERED_ORACLE_BINDING')
        owner['recovered_oracle_sha256'] = sha(output / 'cpu-oracle-seal.json')
        if nested_state:
            owner['recovered_nested_oracle_sha256']=owner['recovered_oracle_sha256']
            nested_launch=output/'nested-launch.json'
            durable_json(nested_launch,{'nested_oracle_sha256':owner['recovered_nested_oracle_sha256']})
            upload('11a-nested-oracle-admission',nested_launch,base+'/nested-launch.json')
            operation('11b-state-cpu','nested-state-cpu',300)
            phase_receipt('11b-state-cpu','NESTED_STATE_CPU_ORACLE_READY_TPU_NOT_RUN')
            download('11c-state-cpu-seal',base+'/records/state-cpu-oracle/oracle-seal.json',output/'state-cpu-oracle-seal.json',reserve=660)
            state_seal=json.loads((output/'state-cpu-oracle-seal.json').read_text())
            if (state_seal.get('runtime_lock_sha256')!=manifest['runtime_lock_sha256'] or
                    state_seal.get('source_admission_sha256')!=manifest['source_admission_sha256'] or
                    state_seal.get('nested_oracle_sha256')!=owner['recovered_nested_oracle_sha256'] or
                    state_seal.get('probe_sha256')!=manifest['nested_state_probe_sha256']):
                raise ValueError('RECOVERED_NESTED_STATE_ORACLE_BINDING')
            owner['recovered_oracle_sha256']=sha(output/'state-cpu-oracle-seal.json')
        launch = output / 'launch.json'
        durable_json(launch,primitive_gate if primitive else native_gate if native else {'oracle_sha256': owner['recovered_oracle_sha256'],**({'nested_oracle_sha256':owner['recovered_nested_oracle_sha256']} if nested_state else {})})
        upload('12-oracle-admission', launch, base + '/launch.json')
        operation('13-tpu', 'primitives' if primitive else 'native' if native else 'nested-state' if nested_state else 'nested' if nested else 'state' if state else 'transfer' if transfer else 'precision' if precision else 'routes' if diagnostic else 'tpu', 1800)
        phase_receipt('13-tpu', 'PRIMITIVE_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED' if primitive else 'NATIVE_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED' if native else 'NESTED_STATE_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED' if nested_state else 'NESTED_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED' if nested else 'STATE_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED' if state else 'TRANSFER_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED' if transfer else
                      'PRECISION_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED' if precision else
                      'DEVICE_ROUTE_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED' if diagnostic else 'TPU_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED')
        owner['status'] = 'CHILD_TERMINAL_RETRIEVAL_REQUIRED'
    except BaseException as error:
        owner.update(status='BLOCKED', original_error={'type': type(error).__name__, 'message': str(error)})
    finally:
        for sig in prior:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if started is not None and (not adopting or adoption_verified):
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
                owner['allocation']='NOT_ATTEMPTED'
                if adopting:
                    owner['remote_termination']='NOT_RUN_UNPROVEN_BROWSER_IDENTITY'
                    try:call('89-remove-provisional',['--remove-browser-provisional-v1',str(browser_adoption),browser_adoption_sha256],15,0,cleanup=True)
                    except BaseException as error:
                        owner['cleanup_errors'].append({'operation':'89-remove-provisional','type':type(error).__name__,'message':str(error)})
                        owner['status']='BLOCKED_CLEANUP'
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
            if native:
                missing=native_readback_missing(owner,receipt,output,native_gate)
                if missing:raise _NativeReadbackUnavailable(','.join(missing))
            if primitive:
                missing=PC.readback_missing(owner,receipt,output,primitive_gate)
                if missing:raise PC.ReadbackUnavailable(','.join(missing))
            # The existing probe verifier uses retained arrays; no TPU or CPU science is repeated.
            verify = Ownership(output)
            log = (output / 'local-verifier.raw').open('wb', buffering=0)
            error_log = log
            try:
                if precision or transfer:
                    error_log = (output / 'local-verifier.stderr.raw').open('wb', buffering=0)
                with verify.guard(clamp(started, 120, 0)):
                    probe = 'probe_primitives.py' if primitive else 'probe_nested_state.py' if nested_state else 'probe_nested.py' if nested else 'probe_state_roundtrip.py' if state else 'probe_transfer.py' if transfer else 'probe_precision.py' if precision else 'probe_routes.py' if diagnostic else 'probe_backend.py'
                    actual = 'recovered/primitives' if primitive else 'recovered/nested-state' if nested_state else 'recovered/nested' if nested else 'recovered/state' if state else 'recovered/transfer' if transfer else 'recovered/precision' if precision else 'recovered/device-routes' if diagnostic else 'recovered/tpu-actual'
                    argv = [sys.executable, '-B', str(packet / probe), 'verify', '--admission-sha256', manifest['source_admission_sha256'], '--oracle', str(output / ('recovered/state-cpu-oracle' if nested_state else 'recovered/cpu-oracle')), '--oracle-sha256', owner['recovered_oracle_sha256'], '--actual', str(output / actual)]
                    if (diagnostic or transfer) and not state and not nested:
                        argv += ['--backend-probe', str(packet / 'probe_backend.py')]
                    if (precision or transfer) and not state and not nested:
                        argv += ['--route-probe', str(packet / 'probe_routes.py')]
                    if transfer and not state and not nested:
                        argv += ['--precision-probe', str(packet / 'probe_precision.py'), '--patch-manifest', str(packet / 'patches/params4bit-xla-v1.json')]
                    if primitive:
                        argv += [str(x) for x in PC.cli(packet)]
                    elif nested_state:
                        argv += [str(x) for x in state_cli(packet,output/'recovered/cpu-oracle',owner['recovered_nested_oracle_sha256'])]
                    elif nested:
                        argv += [str(x) for x in nested_cli(packet)]
                    if state:
                        argv += ['--patch-manifest', str(packet / 'patches/params4bit-xla-v1.json'), '--parent-receipt-sha256', receipt['state_parent_sha256']]
                    if native:
                        if receipt.get('native_expected_sha256')!=sha(output/'recovered/native/expected.json') or receipt.get('native_outer_owner_sha256')!=sha(output/'recovered/steps/12-native-parent/ownership.json') or receipt.get('oracle_sha256')!=native_gate['oracle_sha256']:raise ValueError('NATIVE_RECOVERED_BOUNDARY_SEALS')
                        argv=[sys.executable,'-B',str(packet/'native/verify_run.py'),'--actual',str(output/'recovered/native'),'--oracle',str(output/'native-cpu/recovered/cpu-oracle'),'--oracle-sha256',native_gate['oracle_sha256'],'--admission-sha256',manifest['source_admission_sha256'],'--expected-sha256',receipt['native_expected_sha256'],'--outer-ownership',str(output/'recovered/steps/12-native-parent/ownership.json'),'--outer-ownership-sha256',receipt['native_outer_owner_sha256'],'--output',str(output/'verify.json')]
                    proc = verify.launch(argv, record=output / 'local-verifier-ownership.json', env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), stdout=log, stderr=error_log)
                    proc.wait()
                    owner['local_verifier_exit_code'] = proc.returncode
            finally:
                for stream in [log, error_log] if error_log is not log else [log]:
                    try:
                        os.fsync(stream.fileno())
                    except Exception as error:
                        verify.note_error('fsync_local_verifier', error)
                    finally:
                        try:
                            stream.close()
                        except Exception as error:
                            verify.note_error('close_local_verifier', error)
                owner['local_verifier_cleanup'] = verify.summary()
            if native:
                report=json.loads((output/'verify.json').read_text())
                bound=(receipt.get('experiment')==NC.MODE and receipt.get('native_manifest_sha256')==NC.MANIFEST_SHA and receipt.get('native_source_variant')==NC.VARIANT and receipt.get('native_oracle_gate_sha256')==sha(output/'launch.json') and receipt.get('source_admission_sha256')==manifest['source_admission_sha256'] and receipt.get('allocation_epoch')==started and receipt.get('native_parent_sha256')==sha(output/'recovered/native/parent.json'))
                if owner['status']=='CHILD_TERMINAL_RETRIEVAL_REQUIRED' and owner.get('local_verifier_exit_code')==0 and not owner['local_verifier_cleanup']['errors'] and bound and report.get('status')=='BOUNDED_NATIVE_DIAGNOSTIC_PASS' and report.get('actual_device') is True and report.get('m6')=='NOT_QUALIFIED' and receipt.get('tpu_status')=='NATIVE_RECORDS_TERMINAL' and receipt.get('status')=='NATIVE_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED' and all(not s['cleanup']['errors'] for s in receipt['steps']):owner.update(status='PASS_BOUNDED_NATIVE_DIAGNOSTIC',record_validation='PASS',m6_status='NOT_QUALIFIED',m4_source_compatibility='NOT_QUALIFIED',optimized_executable_link='UNKNOWN',compiler_body_memory_allocator='NOT_QUALIFIED')
            elif transfer:
                report = json.loads((output / 'local-verifier.raw').read_text())
                durable_json(output / 'verify.json', report)
                valid = (report.get('record_validation') == 'PASS' and report.get('m3_status') == 'NOT_QUALIFIED' and
                         report.get('numerical_status') in {'PASS', 'FAIL'} and
                         (state or report.get('api42_status') == report['numerical_status']))
                if nested and not nested_state and not primitive:
                    expected_ids=[case['id'] for case in json.loads((packet/'nested-inputs.json').read_text())['cases']]
                    valid=(report.get('record_validation')=='PASS' and report.get('m4_status')=='NOT_QUALIFIED' and
                           report.get('m3_status')=='DEPENDENCY_NOT_ACCEPTED_BY_THIS_PROBE' and
                           report.get('numerical_status') in {'PASS','FAIL','ERROR'} and
                           [row.get('case_id') for row in report.get('rows',[])]==expected_ids)
                if nested_state:
                    expected_ids=[case['id'] for case in json.loads((packet/'nested-inputs.json').read_text())['cases'] if case['kind']=='module']
                    valid=(report.get('record_validation')=='PASS' and report.get('m4_status')=='NOT_QUALIFIED' and
                           report.get('numerical_status') in {'PASS','FAIL','ERROR'} and
                           [row.get('case_id') for row in report.get('rows',[])]==expected_ids and
                           report.get('native_cpu_golden')==report.get('cuda_golden')=='NOT_RUN')
                if primitive:
                    valid=PC.validate_report(report, packet)
                bound = (receipt.get('experiment') == experiment and receipt.get('precision') == 'highest' and
                         all(receipt.get(field) == manifest[field] for field in TRANSFER_BINDINGS) and
                         (not state or all(receipt.get(field) == manifest[field] for field in STATE_BINDINGS)) and
                         receipt.get('packet_sha256') == expected and receipt.get('manifest_sha256') == sha(packet / 'manifest.json') and
                         receipt.get('source_admission_sha256') == manifest['source_admission_sha256'] and
                         receipt.get('runtime_lock_sha256') == manifest['runtime_lock_sha256'] and receipt.get('allocation_epoch') == started)
                if nested:
                    bound=(bound and all(receipt.get(field)==manifest[field] for field in NESTED_BINDINGS))
                    if primitive:
                        bound=(bound and all(receipt.get(field)==manifest[field] for field in PC.BINDINGS) and
                            receipt.get('primitive_root_gate_sha256')==sha(output/'launch.json') and
                            receipt.get('primitive_receipt_sha256')==sha(output/'recovered/primitives/receipt.json') and
                            receipt.get('oracle_sha256')==primitive_gate['oracle_sha256']==owner['recovered_oracle_sha256'])
                    elif nested_state:
                        bound=(bound and all(receipt.get(field)==manifest[field] for field in NESTED_STATE_BINDINGS) and
                            receipt.get('nested_state_receipt_sha256')==sha(output/'recovered/nested-state/receipt.json') and
                            receipt.get('nested_oracle_sha256')==owner['recovered_nested_oracle_sha256']==sha(output/'recovered/cpu-oracle/oracle-seal.json') and
                            receipt.get('oracle_sha256')==owner['recovered_oracle_sha256']==sha(output/'recovered/state-cpu-oracle/oracle-seal.json') and
                            receipt.get('nested_state_cpu_status')=='PASS')
                    else:bound=(bound and receipt.get('nested_receipt_sha256')==sha(output/'recovered/nested/receipt.json'))
                controls = json.loads((output / 'recovered/source-controls.json').read_text())
                validate_source_controls(controls)
                proof = json.loads((output / 'recovered/installed-source.json').read_text())
                built = json.loads((output / 'recovered/built-source.json').read_text())
                admission_record = json.loads((packet / 'source-admission.json').read_text())
                controls_valid = (proof.get('status') == 'POST_PATCH_PYTHON_SOURCE_PASS' and
                                  proof.get('patch_manifest_sha256') == TRANSFER_PATCH_MANIFEST_SHA and
                                  proof.get('source_admission_sha256') == manifest['source_admission_sha256'] and
                                  proof.get('installed_python_files') == {name: admission_record[name]['files'] for name in ('bitsandbytes', 'bitsandbytes_tpu')} and
                                  built.get('status') == 'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS' and
                                  built.get('patch_manifest_sha256') == TRANSFER_PATCH_MANIFEST_SHA and
                                  built.get('python_files') == admission_record['bitsandbytes']['files'] and
                                  any(row.get('name') == built.get('wheel_name') and row.get('sha256') == built.get('wheel_sha256') for row in receipt.get('built_wheels', [])))
                if nested:
                    plugin_built=json.loads((output/'recovered/built-plugin-source.json').read_text())
                    controls_valid=(controls_valid and proof.get('source_variant')=='nested-v1' and
                        proof.get('nested_source_sha256')==NESTED_SOURCE_SHA and
                        proof.get('plugin_source_manifest_sha256')==NESTED_PLUGIN_MANIFEST_SHA and
                        plugin_built.get('status')=='NESTED_PLUGIN_WHEEL_PYTHON_SOURCE_PASS' and
                        plugin_built.get('python_files')==NESTED_FILES and
                        plugin_built.get('plugin_source_manifest_sha256')==NESTED_PLUGIN_MANIFEST_SHA and
                        plugin_built.get('nested_source_sha256')==NESTED_SOURCE_SHA and
                        receipt.get('built_plugin_source_status')=='NESTED_PLUGIN_WHEEL_PYTHON_SOURCE_PASS' and
                        any(row.get('name')==plugin_built.get('wheel_name') and row.get('sha256')==plugin_built.get('wheel_sha256') for row in receipt.get('built_wheels',[])))
                if owner['status'] == 'CHILD_TERMINAL_RETRIEVAL_REQUIRED' and not owner['local_verifier_cleanup']['errors'] and owner.get('local_verifier_exit_code') == (0 if primitive or report.get('numerical_status') == 'PASS' else 2) and valid and bound and controls_valid and receipt.get('built_source_status') == 'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS' and receipt.get('installed_source_status') == 'POST_PATCH_PYTHON_SOURCE_PASS' and receipt.get('source_controls_status') == 'PASS_QUALIFIED_LINUX_SOURCE_CONTROLS' and receipt.get('runtime_status') == 'PASS_TPU_RUNTIME_PROBE_ONLY' and receipt.get('cpu_status') == 'PASS' and receipt.get('tpu_status') == ('PRIMITIVE_RECORDS_COMPLETE' if primitive else 'NESTED_STATE_RECORDS_COMPLETE' if nested_state else 'NESTED_RECORDS_COMPLETE' if nested else 'STATE_RECORDS_COMPLETE' if state else 'TRANSFER_RECORDS_COMPLETE') and receipt.get('status') == ('PRIMITIVE_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED' if primitive else 'NESTED_STATE_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED' if nested_state else 'NESTED_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED' if nested else 'STATE_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED' if state else 'TRANSFER_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED') and all(not step['cleanup']['errors'] for step in receipt['steps']):
                    owner.update(status=('PASS' if report['numerical_status'] == 'PASS' else 'FAIL') + ('_TPU_NESTED_STATE_RECORDS' if nested_state else '_TPU_NESTED_RECORDS' if nested else '_TPU_STATE_RECORDS' if state else '_TPU_TRANSFER_API42'),
                                 record_validation='PASS', api42_status='NOT_REPEATED' if state or nested else report['api42_status'], numerical_status=report['numerical_status'], m3_status='NOT_QUALIFIED')
                    if primitive:
                        owner.update(status='PASS_PRIMITIVE_RECORDS',primitive_diagnostic_only=True,
                                     numerical_status=report['numerical_status'],builder_bit_status=report['builder_bit_status'],
                                     native_view_status=report['native_view_status'],
                                     **{key:'NOT_QUALIFIED' for key in ('api42_status','m3_status','m4_status','m5_status')})
                    elif nested:owner.update(m3_status='DEPENDENCY_NOT_ACCEPTED_BY_THIS_PROBE',m4_status='NOT_QUALIFIED')
            elif precision:
                report = json.loads((output / 'local-verifier.raw').read_text())
                durable_json(output / 'verify.json', report)
                complete = (report.get('record_validation') == 'PASS' and
                            report.get('api42_status') == report.get('m3_status') == 'NOT_QUALIFIED')
                bound = (receipt.get('experiment') == experiment and receipt.get('diagnostic_only') is True and
                         receipt.get('route_probe_sha256') == manifest['route_probe_sha256'] and
                         receipt.get('precision_probe_sha256') == manifest['precision_probe_sha256'] and
                         receipt.get('packet_sha256') == expected and receipt.get('manifest_sha256') == sha(packet / 'manifest.json') and
                         receipt.get('source_admission_sha256') == manifest['source_admission_sha256'] and
                         receipt.get('runtime_lock_sha256') == manifest['runtime_lock_sha256'] and
                         receipt.get('allocation_epoch') == started)
                if owner['status'] == 'CHILD_TERMINAL_RETRIEVAL_REQUIRED' and not owner['local_verifier_cleanup']['errors'] and owner.get('local_verifier_exit_code') == 0 and complete and bound and receipt['runtime_status'] == 'PASS_TPU_RUNTIME_PROBE_ONLY' and receipt['cpu_status'] == 'PASS' and receipt['tpu_status'] == 'DIAGNOSTIC_RECORDS_COMPLETE' and receipt['status'] == 'PRECISION_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED' and all(not step['cleanup']['errors'] for step in receipt['steps']):
                    owner.update(status='PASS_PRECISION_RECORDS', record_validation='PASS', api42_status='NOT_QUALIFIED',
                                 m3_status='NOT_QUALIFIED', numerical_status=report.get('numerical_status'))
            elif diagnostic:
                if owner['status'] == 'CHILD_TERMINAL_RETRIEVAL_REQUIRED' and not owner['local_verifier_cleanup']['errors'] and owner.get('local_verifier_exit_code') == 0 and receipt['runtime_status'] == 'PASS_TPU_RUNTIME_PROBE_ONLY' and receipt['cpu_status'] == 'PASS' and receipt['tpu_status'] == 'DIAGNOSTIC_RECORDS_COMPLETE' and receipt['status'] == 'DEVICE_ROUTE_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED' and all(not step['cleanup']['errors'] for step in receipt['steps']):
                    owner.update(status='PASS_DEVICE_ROUTE_RECORDS', record_validation='PASS', api42_status='NOT_QUALIFIED')
            elif owner['status'] == 'CHILD_TERMINAL_RETRIEVAL_REQUIRED' and not owner['local_verifier_cleanup']['errors'] and owner.get('local_verifier_exit_code') in (0, 2) and receipt['runtime_status'] == 'PASS_TPU_RUNTIME_PROBE_ONLY' and receipt['cpu_status'] == 'PASS' and receipt['tpu_status'] in ('PASS', 'FAIL') and all(not step['cleanup']['errors'] for step in receipt['steps']):
                owner['status'] = 'PASS_TPU_API_PROBE' if owner['local_verifier_exit_code'] == 0 else 'FAIL_TPU_API_PROBE'
        except PC.ReadbackUnavailable as error:
            owner.update(local_verifier_status='NOT_RUN_INCOMPLETE_PRIMITIVE_RECORDS',primitive_record_validation='NOT_RUN',local_verifier_unavailable_reason=str(error))
            if owner['status']=='CHILD_TERMINAL_RETRIEVAL_REQUIRED':owner['status']='BLOCKED'
        except _NativeReadbackUnavailable as error:
            owner.update(local_verifier_status='NOT_RUN_INCOMPLETE_NATIVE_RECORDS',native_record_validation='NOT_RUN',local_verifier_unavailable_reason=str(error))
            if owner['status']=='CHILD_TERMINAL_RETRIEVAL_REQUIRED':owner['status']='BLOCKED'
        except BaseException as error:
            owner['local_readback_error'] = {'type': type(error).__name__, 'message': str(error)}
    owner['finished_utc'] = datetime.now(timezone.utc).isoformat()
    durable_json(output / 'owner.json', owner)
    return owner


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--packet-sha256', required=True)
    p.add_argument('--browser-adoption',type=Path)
    p.add_argument('--browser-adoption-sha256')
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
        r = drive(a.packet.resolve(), a.output.resolve(), a.packet_sha256, a.root_acceptance, a.root_acceptance_sha256, a.cli_python, a.cli_identity,**({'browser_adoption':a.browser_adoption,'browser_adoption_sha256':a.browser_adoption_sha256} if a.browser_adoption is not None or a.browser_adoption_sha256 is not None else {}))
        print(json.dumps(r))
        raise SystemExit(0 if r['status'] in ('PASS_TPU_API_PROBE', 'PASS_DEVICE_ROUTE_RECORDS', 'PASS_PRECISION_RECORDS', 'PASS_TPU_TRANSFER_API42', 'PASS_TPU_STATE_RECORDS', 'PASS_TPU_NESTED_RECORDS', 'PASS_TPU_NESTED_STATE_RECORDS','PASS_BOUNDED_NATIVE_DIAGNOSTIC', 'PASS_PRIMITIVE_RECORDS') else 2)
