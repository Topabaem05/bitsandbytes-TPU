"""Exact reviewed R3 run_child function, with local pure imports."""
import os
import math
import subprocess
import time
from protocol import require
from lifecycle import durable_json

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
