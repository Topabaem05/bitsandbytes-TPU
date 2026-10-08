"""Direct child resource-limit wrapper. exec preserves PID, PPID and PGID."""
import argparse
import os
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'ownership'))
from lifecycle import durable_json
import compiler_contract as C

def execute(path, expected_sha, argv):
    C.require(C.sha(path) == expected_sha, 'EXEC_PRELAUNCH_HASH')
    expected = C.read(path)
    C.require(expected['kind'] == 'R6_COMPILER_PRELAUNCH_V1' and expected['sources'] == C.sources(),
              'EXEC_SOURCE_SEAL')
    C.require(argv == expected['argv'] and argv[:2] == [sys.executable, '-B'] and
              C.sha(argv[2]) == expected['target_source_sha256'], 'EXEC_EXACT_ARGV_SOURCE')
    C.require(os.getpid() != expected['parent_pid'] and os.getppid() == expected['parent_pid'] ==
              expected['parent_pgid'] == os.getpgrp(), 'EXEC_DIRECT_PARENT_GROUP')
    C.require(C.finite(expected['deadline_epoch']) and time.time() < expected['deadline_epoch'],
              'EXEC_ORIGINAL_DEADLINE')
    C.require(os.environ.get('XLA_FLAGS') == expected['xla_flags'], 'EXEC_EXACT_DUMP_FLAGS')
    limit = expected['limits']['file_bytes']
    C.require(type(limit) is int and 0 < limit <= 16777216, 'EXEC_FILE_LIMIT_SCOPE')
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
    actual = resource.getrlimit(resource.RLIMIT_FSIZE)
    C.require(actual == (limit, limit), 'EXEC_FILE_LIMIT_READBACK')
    record = {'kind': 'R6_SAME_PID_EXEC_PRELUDE', 'pid': os.getpid(), 'pgid': os.getpgrp(),
              'parent_pid': os.getppid(), 'process_token': expected['process_token'],
              'parent_process_token': expected['parent_process_token'],
              'deadline_epoch': expected['deadline_epoch'], 'prelaunch_sha256': expected_sha,
              'wrapper_source_sha256': C.sha(__file__), 'target_source_sha256': C.sha(argv[2]),
              'argv': argv, 'xla_flags': os.environ.get('XLA_FLAGS'), 'file_limit_soft_hard': list(actual),
              'started_epoch': time.time(), 'started_monotonic': time.monotonic(),
              'claim': 'BEFORE_EXEC_ONLY_REQUIRE_INDEPENDENT_POST_EXEC_CHILD_RECEIPT'}
    durable_json(Path(expected['evidence_directory']) / 'exec.json', record)
    os.execv(argv[0], argv)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--contract', required=True)
    p.add_argument('--contract-sha256', required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    try:
        execute(a.contract, a.contract_sha256, command)
    except BaseException as e:
        print(type(e).__name__ + ': ' + str(e), file=sys.stderr)
        raise SystemExit(2)
