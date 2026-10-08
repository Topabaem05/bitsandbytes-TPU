"""Local direct-parent fixture using exact production child-owner and exec wrapper."""
import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent / 'compiler-native'; sys.path.insert(0, str(NATIVE)); sys.path.insert(0, str(NATIVE / 'ownership'))
import compiler_contract as C
from lifecycle import durable_json
from state_child_owner import run_child

p = argparse.ArgumentParser()
p.add_argument('--output', required=True); p.add_argument('--mode', required=True)
p.add_argument('--process-token', required=True); p.add_argument('--deadline-epoch', type=float, required=True)
p.add_argument('--compiler-dumps', required=True); p.add_argument('--compiler-policy-sha256', required=True)
a = p.parse_args(); output = Path(a.output); output.mkdir()
parent = {'kind': 'LOCAL_OS_TOPOLOGY_PARENT_ONLY', 'pid': os.getpid(), 'pgid': os.getpgrp(),
          'process_token': a.process_token, 'deadline_epoch': a.deadline_epoch, 'children': [],
          'manifest_sha256': C.sha(NATIVE/'manifest.json'), 'oracle_sha256': 'b' * 64, 'source_admission_sha256': 'c' * 64,
          'compiler_source_pre': C.verify_manifest(NATIVE/'manifest.json',C.sha(NATIVE/'manifest.json')),
          'compiler_source_post': C.verify_manifest(NATIVE/'manifest.json',C.sha(NATIVE/'manifest.json')),
          'expected_sha256': 'd' * 64, 'compiler_policy_sha256': a.compiler_policy_sha256, 'status': 'ERROR'}
token = secrets.token_hex(16); deadline = a.deadline_epoch - .3
launch = {'parent_pid': parent['pid'], 'parent_pgid': parent['pgid'], 'parent_process_token': a.process_token,
          'process_token': token, 'deadline_epoch': deadline}
durable_json(output / 'expected.json', {'launch': launch})
argv = [sys.executable, '-B', str(HERE / 'topology_child.py'), '--output', str(output / 'child.json'),
        '--mode', a.mode]
for key, value in [('parent-pid', parent['pid']), ('parent-process-token', a.process_token),
                   ('process-token', token), ('deadline-epoch', deadline), ('oracle-sha256', 'b'*64),
                   ('expected-sha256', 'd'*64), ('admission-sha256', 'c'*64)]:
    argv += ['--' + key, str(value)]
cfg = {'entries': 8, 'file_bytes': 8192, 'total_bytes': 12288}
policy = {'limits': cfg, 'minimum_free_bytes': 268435456, 'poll_seconds': .01, 'shutdown_grace': .05}
root = output.with_name(output.name + '.compiler-private')
previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
interrupts = {'launching': False, 'pending': None}
def interrupted(sig, frame):
    interrupts['pending'] = sig
    if not interrupts['launching']:
        raise InterruptedError('LOCAL_PARENT_SIGNAL')
for sig in previous:
    signal.signal(sig, interrupted)
try:
    expected = C.seal(policy, 'e'*64, root, parent, argv, token, deadline, parent['expected_sha256'])
    env = dict(os.environ, BNB_COMPILER_DUMP_DIRECTORY=str(root / 'raw-private'),
               XLA_FLAGS=' '.join(['--xla_dump_to=' + str(root / 'raw-private')] + C.DUMP_FLAGS))
    expected['xla_flags'] = env['XLA_FLAGS']
    durable_json(root / 'expected.json', expected); sealed = C.sha(root / 'expected.json')
    parent['compiler_expected_sha256'] = sealed
    parent['compiler_dump_request'] = {'flags': env['XLA_FLAGS'], 'mode': 'request', 'policy_sha256': 'e'*64, 'sibling_evidence': str(root)}
    durable_json(output / 'parent.json', parent)
    wrapped = [sys.executable, '-B', str(NATIVE / 'compiler_exec.py'), '--contract', str(root / 'expected.json'),
               '--contract-sha256', sealed, '--'] + argv
    logs = output / 'logs'; logs.mkdir()
    result = run_child(wrapped, output=logs, descriptor={'phase': 'native-boundary', 'process_token': token},
                       parent=parent, parent_path=output / 'parent.json', deadline=deadline, env=env,
                       interrupts=interrupts, compiler=(expected, root / 'expected.json', sealed))
    parent['status'] = 'COMPLETE' if result['exit_code'] == 0 else 'ERROR'
except BaseException as e:
    parent['error'] = {'type': type(e).__name__, 'message': str(e)}
finally:
    for sig, handler in previous.items():
        signal.signal(sig, handler)
    durable_json(output / 'parent.json', parent)
raise SystemExit(0 if parent['status'] == 'COMPLETE' else 2)
