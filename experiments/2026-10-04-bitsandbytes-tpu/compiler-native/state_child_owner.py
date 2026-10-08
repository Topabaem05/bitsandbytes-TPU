"""Explicit compiler variant: direct inherited child and monitored bounded wait."""
import os
import math
import subprocess
import time
from pathlib import Path
from compiler_contract import require
from lifecycle import durable_json
import compiler_contract as C
from dump_observation import observe
import dump_inventory as D

def run_child(argv, *, output, descriptor, parent, parent_path, deadline, env=None, interrupts=None, compiler=None):
    """Launch one direct child in the inherited group; persist identity before waiting."""
    require(math.isfinite(deadline) and deadline <= parent['deadline_epoch'] and time.time() < deadline,
            'STATE_CHILD_DEADLINE')
    require(compiler is not None, 'EXPLICIT_COMPILER_MONITOR_REQUIRED')
    expected, expected_path, expected_sha = compiler
    require(C.sha(expected_path) == expected_sha and expected['sources'] == C.sources(), 'MONITOR_SOURCE_SEAL')
    evidence = Path(expected['evidence_directory'])
    monitor = {'kind': 'R6_DIRECT_CHILD_DUMP_MONITOR', 'status': 'PARTIAL',
               'pid': os.getpid(), 'pgid': os.getpgrp(), 'prelaunch_sha256': expected_sha,
               'source_pre': C.sources(), 'source_post': None,
               'limits': expected['limits'], 'deadline_epoch': deadline,
               'hard_aggregate_quota': 'UNAVAILABLE',
               'polling_overshoot': 'POSSIBLE_UNBOUNDED_BY_WRITER_RATE',
               'observations': 0, 'overflow_observation': None, 'failure_reason': None,
               'closure': 'REQUIRES_EXTERNAL_OWNER_GROUP_CLEANUP',
               'started_epoch': time.time(), 'started_monotonic': time.monotonic()}
    durable_json(evidence / 'monitor.json', monitor)
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
        monitor['child_pid'] = process.pid
        durable_json(evidence / 'monitor.json', monitor)
        stop_monotonic = time.monotonic() + max(0, deadline - time.time())
        while True:
            snapshot = observe(expected['dump_directory'], tuple(expected['dump_identity']), expected['limits'])
            monitor['observations'] += 1
            monitor['last_observation'] = snapshot
            monitor['last_observed_epoch'] = time.time()
            monitor['last_observed_monotonic'] = time.monotonic()
            if snapshot['status'] != 'WITHIN_OBSERVED_LIMITS':
                monitor['overflow_observation'] = snapshot
                monitor['failure_reason'] = snapshot['status']
                raise ValueError(snapshot['status'])
            if time.time() >= deadline or time.monotonic() >= stop_monotonic:
                descriptor['timeout'] = True
                monitor['failure_reason'] = 'STATE_CHILD_DEADLINE_EXPIRED'
                raise TimeoutError('STATE_CHILD_DEADLINE_EXPIRED')
            if process.poll() is not None:
                cfg = expected['limits']
                monitor['terminal_snapshot'] = D.capture(expected['dump_directory'], max_entries=cfg['entries'],
                    max_file_bytes=cfg['file_bytes'], max_total_bytes=cfg['total_bytes'])
                if time.time() >= deadline or time.monotonic() >= stop_monotonic:
                    descriptor['timeout'] = True
                    monitor['failure_reason'] = 'STATE_CHILD_DEADLINE_EXPIRED'
                    raise TimeoutError('STATE_CHILD_DEADLINE_EXPIRED')
                monitor['status'] = 'COMPLETE_BOUNDED_CAPTURE'
                break
            time.sleep(min(expected['poll_seconds'], max(0, stop_monotonic - time.monotonic())))
    except BaseException as error:
        monitor.update(status='BLOCKED', failure_reason=monitor['failure_reason'] or str(error),
                       error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        if process is not None:
            if process.poll() is None:
                monitor.update(receipt_before_term=True, shutdown_requested_epoch=time.time(),
                               shutdown_requested_monotonic=time.monotonic())
                durable_json(evidence / 'monitor.json', monitor)

            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=expected['shutdown_grace'])
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
        monitor.update(finished_epoch=time.time(), finished_monotonic=time.monotonic(),
                       source_post=C.sources(), child_terminal=dict(descriptor))
        if descriptor.get('exit_code') not in (0, 2) or descriptor.get('cleanup_errors'):
            monitor.update(status='BLOCKED', failure_reason=monitor['failure_reason'] or 'CHILD_FAILED_OR_FILE_LIMIT')
        durable_json(evidence / 'monitor.json', monitor)
    require(descriptor['cleanup_errors'] == [] and descriptor['reaped'] and descriptor['child_absent'] and
            not descriptor['timeout'] and descriptor['exit_code'] in (0, 2), 'STATE_CHILD_TERMINAL')
    return descriptor
