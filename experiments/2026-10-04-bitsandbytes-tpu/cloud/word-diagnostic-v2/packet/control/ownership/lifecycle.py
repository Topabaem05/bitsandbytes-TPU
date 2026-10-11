"""Finite per-run catchable-signal ownership, durable metadata, no daemon."""
import contextlib
import json
import os
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from bootstrap import ProviderError, stop_group


class Interrupted(ProviderError):
    pass


class DeadlineExceeded(ProviderError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def durable_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as file:
        file.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class RawLog:
    def __init__(self, path):
        self.file = Path(path).open('wb', buffering=0)

    def write(self, chunk):
        self.file.write(chunk)
        os.fsync(self.file.fileno())

    def close(self):
        self.file.close()


class Ownership:
    def __init__(self, output):
        self.output = Path(output)
        self.procs = {}
        self.signal_received = None
        self.deadline_received = False
        self.launching = False
        self.pending = None
        self.cleaning = False
        self.events = self.output / 'lifecycle.jsonl'

    def event(self, value):
        with self.events.open('ab', buffering=0) as file:
            file.write((json.dumps(dict(value, recorded_at=utc()), sort_keys=True) + '\n').encode())
            os.fsync(file.fileno())

    def handle(self, signum, frame):
        if self.cleaning:
            return  # Cleanup must not be recursively interrupted.
        self.pending = signum
        if signum == signal.SIGALRM:
            self.deadline_received = True
        else:
            self.signal_received = signum
        if self.launching:
            return  # Defer until Popen has returned and its group is registered.
        self.event({'kind': 'signal', 'signal': signum})
        if signum == signal.SIGALRM:
            raise DeadlineExceeded('total work deadline expired')
        raise Interrupted(f'owner received signal {signum}')

    def launch(self, argv, *, record, **kwargs):
        self.launching = True
        try:
            proc = subprocess.Popen(argv, start_new_session=True, **kwargs)
            metadata = {'argv': list(argv), 'pid': proc.pid, 'pgid': proc.pid,
                        'started_at': utc(), 'owner_pid': os.getpid(),
                        'ownership': 'Popen child in dedicated start_new_session group'}
            self.procs[proc.pid] = (proc, Path(record), metadata)
            durable_json(record, metadata)
            self.event({'kind': 'child_started', 'pid': proc.pid, 'pgid': proc.pid})
        finally:
            self.launching = False
        if self.pending is not None:
            self.handle(self.pending, None)
        return proc

    def stop(self, proc):
        if proc.pid not in self.procs:
            raise ValueError('refusing cleanup of unregistered process')
        old = self.cleaning
        self.cleaning = True
        try:
            _, path, metadata = self.procs[proc.pid]
            stop_group(proc)
            metadata.update(ended_at=utc(), exit_status=proc.returncode,
                            cleanup='registered group TERM, bounded wait, KILL; leader reaped')
            durable_json(path, metadata)
            self.event({'kind': 'child_stopped', 'pid': proc.pid, 'exit_status': proc.returncode})
            del self.procs[proc.pid]
        finally:
            self.cleaning = old

    @contextlib.contextmanager
    def guard(self, deadline_seconds=None):
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        for sig in previous:
            signal.signal(sig, self.handle)
        if deadline_seconds is not None:
            signal.setitimer(signal.ITIMER_REAL, deadline_seconds)
        try:
            yield self
        finally:
            self.cleaning = True
            signal.setitimer(signal.ITIMER_REAL, 0)
            for proc, _, _ in list(self.procs.values()):
                self.stop(proc)
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            if previous_timer[0]:
                signal.setitimer(signal.ITIMER_REAL, *previous_timer)
