"""Private R3 coordinator. An outer Ownership guard owns this whole process group."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import secrets
import re
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'ownership'))
from lifecycle import durable_json
from remote import sha, require, verify_experiment, verify_transfer_payload

STATE_PROBE_SHA = '06a6d702bd74cc4602955249d31f1199b5aeaaf7bbecc5043a5ac25c349a9f67'


def run_child(argv, *, output, descriptor, parent, parent_path, deadline, env=None, interrupts=None):
    """Launch one direct child in the inherited group; persist identity before waiting."""
    require(math.isfinite(deadline) and deadline <= parent['deadline_epoch'] and time.time() < deadline,
            'STATE_CHILD_DEADLINE')
    streams = []
    process = None
    descriptor.update(argv=list(argv), deadline_epoch=deadline, parent_pid=parent['pid'],
                      parent_process_token=parent['process_token'], started_epoch=time.time(),
                      started_monotonic=time.monotonic(), reaped=False, child_absent=False,
                      cleanup_errors=[], timeout=False, exit_code=None)
    try:
        for filename in ('stdout.raw', 'stderr.raw'):
            streams.append((output / filename).open('wb', buffering=0))
        # A new PID is sufficient. The outer owner must retain the same PGID even
        # if this coordinator receives SIGKILL and cannot execute its finally.
        if interrupts is not None:
            interrupts['launching'] = True
        try:
            process = subprocess.Popen(argv, stdout=streams[0], stderr=streams[1], env=env)
            descriptor.update(pid=process.pid, launched_pid=process.pid, pgid=os.getpgid(process.pid))
            parent['children'].append(descriptor)
            durable_json(parent_path, parent)
        finally:
            if interrupts is not None:
                interrupts['launching'] = False
        if interrupts is not None and interrupts['pending'] is not None:
            raise InterruptedError('STATE_PARENT_SIGNAL:' + str(interrupts['pending']))
        require(process.pid != parent['pid'] and descriptor['pgid'] == parent['pgid'], 'STATE_INHERITED_GROUP')
        try:
            process.wait(timeout=max(0.001, deadline - time.time()))
        except subprocess.TimeoutExpired:
            descriptor['timeout'] = True
            raise TimeoutError('STATE_CHILD_DEADLINE_EXPIRED')
    finally:
        if process is not None:
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)
                else:
                    process.wait(timeout=2)
                descriptor.update(exit_code=process.returncode, reaped=True)
                try:
                    os.kill(process.pid, 0)
                except ProcessLookupError:
                    descriptor['child_absent'] = True
                require(descriptor['child_absent'], 'STATE_CHILD_STILL_PRESENT')
            except BaseException as error:
                descriptor['cleanup_errors'].append({'type': type(error).__name__, 'message': str(error)})
            descriptor.update(finished_epoch=time.time(), finished_monotonic=time.monotonic())
        for stream in streams:
            try:
                os.fsync(stream.fileno())
                stream.close()
            except Exception as error:
                descriptor['cleanup_errors'].append({'type': type(error).__name__, 'message': str(error)})
        if process is not None:
            durable_json(parent_path, parent)
    require(descriptor['cleanup_errors'] == [] and descriptor['reaped'] and descriptor['child_absent'] and
            not descriptor['timeout'] and descriptor['exit_code'] in (0, 2), 'STATE_CHILD_TERMINAL')
    return descriptor


def execute(args):
    output = Path(args.output)
    require(not output.exists() and os.getpid() == os.getpgid(0), 'STATE_FRESH_OWNED_PARENT')
    require(re.fullmatch('[0-9a-f]{32}', args.process_token), 'STATE_PARENT_LAUNCH_TOKEN')
    require(math.isfinite(args.deadline_epoch) and time.time() + 15 < args.deadline_epoch, 'STATE_PARENT_DEADLINE')
    manifest_path = Path(args.manifest)
    payload = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    require(sha(manifest_path) == args.manifest_sha256 and verify_experiment(manifest) == 'state-roundtrip', 'STATE_MANIFEST')
    for name, record in manifest['files'].items():
        require(not (payload / name).is_symlink() and sha(payload / name) == record['sha256'], 'STATE_PAYLOAD_SOURCE')
    require(sha(args.state_probe) == manifest['state_probe_sha256'] == STATE_PROBE_SHA, 'STATE_PROBE_PIN')
    verify_transfer_payload(payload, manifest)
    spec = importlib.util.spec_from_file_location('state_candidate', args.state_probe)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    B, P, A = candidate.B, candidate.P, candidate.A
    P.validate_environment(B, P.precision_environment())
    candidate.T.verify_oracle(B, A, args)
    admission, roots = A.admit(B, args.admission, args.admission_sha256, args.patch_manifest)
    output.mkdir(parents=True)
    parent_path = output / 'parent.json'
    parent = {'kind': 'STATE_ROUNDTRIP_PARENT', 'status': 'PARTIAL', 'pid': os.getpid(), 'pgid': os.getpgid(0),
              'process_token': args.process_token, 'protocol_sha256': candidate.PROTOCOL_SHA,
              'probe_sha256': sha(args.state_probe), 'helper_pins': candidate.PINS,
              'state_probe_sha256': manifest['state_probe_sha256'], 'state_helper_sha256': manifest['state_helper_sha256'],
              'source_pre': args.admission_sha256, 'source_post': None, 'source_admission_sha256': args.admission_sha256,
              'oracle_sha256': args.oracle_sha256, 'runtime_lock_sha256': manifest['runtime_lock_sha256'],
              'patch_manifest_sha256': manifest['patch_manifest_sha256'], 'precision_environment': P.precision_environment(),
              'deadline_epoch': args.deadline_epoch, 'children': [], 'm3_status': 'NOT_QUALIFIED',
              'oracle_provenance': str(Path(args.oracle).resolve())}
    durable_json(parent_path, parent)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    interrupts = {'launching': False, 'pending': None}
    def interrupted(signum, frame):
        interrupts['pending'] = signum
        if interrupts['launching']:
            return
        raise InterruptedError('STATE_PARENT_SIGNAL:' + str(signum))
    for sig in previous:
        signal.signal(sig, interrupted)
    try:
        for index, phase in enumerate(('save', 'restore')):
            # Reserve cleanup time inside the externally enforced science deadline.
            remaining = args.deadline_epoch - time.time() - 10
            require(remaining > 0, 'STATE_PHASE_BUDGET')
            deadline = time.time() + remaining / (2 - index)
            token = secrets.token_hex(16)
            argv = [sys.executable, '-B', str(args.state_probe), 'device-child', '--phase', phase,
                    '--output', str(output / phase), '--admission', str(args.admission),
                    '--admission-sha256', args.admission_sha256, '--patch-manifest', str(args.patch_manifest),
                    '--oracle', str(args.oracle), '--oracle-sha256', args.oracle_sha256,
                    '--parent-pid', str(parent['pid']), '--parent-process-token', parent['process_token'],
                    '--process-token', token, '--deadline-epoch', str(deadline)]
            if phase == 'restore':
                argv += ['--saved', str(output / 'save'), '--saved-sha256', sha(output / 'save/receipt.json')]
            logs = output / (phase + '-logs')
            logs.mkdir()
            descriptor = run_child(argv, output=logs, descriptor={'phase': phase, 'process_token': token},
                                   parent=parent, parent_path=parent_path, deadline=deadline, interrupts=interrupts)
            descriptor.update(receipt=phase + '/receipt.json', receipt_sha256=sha(output / phase / 'receipt.json'))
            durable_json(parent_path, parent)
        A.verify_installed(B, admission, roots)
        parent.update(source_post=args.admission_sha256, status='COMPLETE', artifacts=B.inventory(output, 'parent.json'))
        durable_json(parent_path, parent)
        return 0
    except BaseException as error:
        parent.update(status='BLOCKED', error={'type': type(error).__name__, 'message': str(error)})
        parent['artifacts'] = B.inventory(output, 'parent.json')
        durable_json(parent_path, parent)
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['execute'])
    for name in ('state-probe', 'manifest', 'manifest-sha256', 'admission', 'admission-sha256',
                 'patch-manifest', 'oracle', 'oracle-sha256', 'output', 'process-token'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    args = parser.parse_args()
    return execute(args)


if __name__ == '__main__':
    raise SystemExit(main())
