"""V5-owned cleanup/guard correction; immutable V2 launch/signal ownership reused."""
import contextlib
import os
import signal
import subprocess
import time
from lifecycle import Ownership as ParentOwnership, durable_json, utc


def describe_error(operation, error):
    return {'operation': operation, 'type': type(error).__name__, 'message': str(error),
            'errno': getattr(error, 'errno', None)}


class Ownership(ParentOwnership):
    def __init__(self, output):
        super().__init__(output)
        self.cleanup_records = []
        self.guard_errors = []
        self.attempted = {}
        self.restoration = None
        self.guard_outcome = None

    def signal_owned_group(self, proc, signum):
        # Test injection belongs to this new adapter, never to a frozen module.
        if proc.pid not in self.procs or self.procs[proc.pid][0] is not proc:
            raise ValueError('refusing unregistered process group')
        os.killpg(proc.pid, signum)

    def note_error(self, operation, error):
        self.guard_errors.append(describe_error(operation, error))

    def summary(self):
        errors = [error for record in self.cleanup_records for error in record['errors']]
        errors += self.guard_errors
        return {'status': 'CLEANUP_UNVERIFIED' if errors else 'NO_CLEANUP_ERRORS',
                'groups': self.cleanup_records, 'errors': errors,
                'restoration': self.restoration, 'guard_original_outcome': self.guard_outcome}

    def persist_summary(self):
        try:
            durable_json(self.output / 'cleanup.json', self.summary())
        except Exception as error:
            self.note_error('persist_cleanup_summary', error)

    def stop(self, proc):
        if proc.pid in self.attempted:
            registered = self.attempted[proc.pid]
            if registered[0] is not proc:
                raise ValueError('refusing reused process identity')
            return registered[1]  # Do not silently repeat an unverified signal attempt.
        if proc.pid not in self.procs or self.procs[proc.pid][0] is not proc:
            raise ValueError('refusing cleanup of unregistered process')
        previous_cleaning = self.cleaning
        self.cleaning = True
        _, path, metadata = self.procs[proc.pid]
        record = {'pid': proc.pid, 'pgid': metadata['pgid'], 'started_at': utc(),
                  'status': 'CLEANUP_UNVERIFIED', 'errors': [], 'signals': [],
                  'group_absence': 'UNVERIFIED', 'leader_reaped': False}
        self.cleanup_records.append(record)
        self.attempted[proc.pid] = (proc, record)

        def failed(operation, error):
            record['errors'].append(describe_error(operation, error))

        def send(signum):
            try:
                self.signal_owned_group(proc, signum)
                record['signals'].append({'signal': signum, 'result': 'SENT'})
            except ProcessLookupError:
                record['signals'].append({'signal': signum, 'result': 'NO_SUCH_GROUP'})
            except Exception as error:
                failed('signal_' + str(signum), error)

        def reap():
            try:
                proc.wait(timeout=0.5)
                record['leader_reaped'] = True
            except subprocess.TimeoutExpired:
                record['leader_wait_timeout'] = True
            except Exception as error:
                failed('reap_leader', error)

        try:
            # Poll reaps an already-exited immediate leader before group operations.
            try:
                proc.poll()
            except Exception as error:
                failed('poll_leader', error)
            send(signal.SIGTERM)
            reap()
            # Still signal the registered group when its leader has exited: owned
            # descendants can remain. No scan, descendant PID guessing or unowned kill.
            send(signal.SIGKILL)
            reap()
            if not record['errors']:
                until = time.monotonic() + 0.3
                while True:
                    try:
                        self.signal_owned_group(proc, 0)
                    except ProcessLookupError:
                        record['group_absence'] = 'OBSERVED_NO_SUCH_GROUP'
                        break
                    except Exception as error:
                        failed('probe_owned_group', error)
                        break
                    if time.monotonic() >= until:
                        failed('probe_owned_group', RuntimeError('owned group still present after bounded cleanup'))
                        break
                    time.sleep(0.01)
            if not record['leader_reaped']:
                failed('reap_leader', RuntimeError('leader not reaped after bounded cleanup'))
        except Exception as error:
            failed('cleanup_unexpected', error)
        finally:
            try:
                record.update(ended_at=utc(), exit_status=proc.returncode)
                if not record['errors'] and record['group_absence'] == 'OBSERVED_NO_SUCH_GROUP':
                    record['status'] = 'CLEANUP_VERIFIED'
                try:
                    self.event({'kind': 'child_cleanup_checks', 'pid': proc.pid,
                                'errors': record['errors']})
                except Exception as error:
                    failed('persist_cleanup_event', error)
                if record['errors']:
                    record['status'] = 'CLEANUP_UNVERIFIED'
                    record['group_absence'] = 'UNVERIFIED'
                metadata.update(ended_at=record['ended_at'], exit_status=proc.returncode,
                                cleanup=record)
                try:
                    durable_json(path, metadata)
                except Exception as error:
                    failed('persist_ownership', error)
                if record['errors']:
                    record['status'] = 'CLEANUP_UNVERIFIED'
                    record['group_absence'] = 'UNVERIFIED'
                elif record['status'] == 'CLEANUP_VERIFIED':
                    del self.procs[proc.pid]
                self.persist_summary()
            finally:
                self.cleaning = previous_cleaning
        return record

    @contextlib.contextmanager
    def guard(self, deadline_seconds=None):
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        previous_cleaning = self.cleaning
        self.restoration = {'handlers_restored': None, 'timer_restored': None, 'state': 'GUARD_ACTIVE',
                            'previous_timer': list(previous_timer),
                            'timer_policy': 'PAUSE_AND_RESTORE_CAPTURED_TIMER'}
        self.guard_outcome = {'status': 'ENTERING', 'error': None}
        try:
            for sig in previous:
                signal.signal(sig, self.handle)
            signal.setitimer(signal.ITIMER_REAL, deadline_seconds or 0)
            yield self
            self.guard_outcome = {'status': 'COMPLETED', 'error': None}
        except BaseException as exc:
            self.guard_outcome = {'status': 'BODY_EXCEPTION', 'error': describe_error('guard_body', exc)}
            raise
        finally:
            self.cleaning = True
            try:
                try:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                except Exception as error:
                    self.note_error('disable_guard_timer', error)
                finally:
                    for proc, _, _ in list(self.procs.values()):
                        try:
                            self.stop(proc)
                        except Exception as error:
                            self.note_error('cleanup_registered_group_' + str(proc.pid), error)
            finally:
                try:
                    handler_errors = []
                    for sig, handler in previous.items():
                        try:
                            signal.signal(sig, handler)
                        except Exception as error:
                            handler_errors.append(describe_error('restore_handler_' + str(sig), error))
                    self.guard_errors.extend(handler_errors)
                    self.restoration['handlers_restored'] = not handler_errors
                finally:
                    try:
                        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
                        self.restoration['timer_restored'] = True
                    except Exception as error:
                        self.note_error('restore_guard_timer', error)
                    finally:
                        self.cleaning = previous_cleaning
                        self.restoration['state'] = 'RESTORATION_ATTEMPTED'
                        self.persist_summary()


def apply_cleanup(record, owner, *, original_status=None, original_error=None):
    if 'original_outcome' not in record:
        record['original_outcome'] = {'status': original_status or record['status'],
                                      'error': original_error if original_error is not None else record.get('error')}
    record['cleanup'] = owner.summary()
    record['cleanup_errors'] = record['cleanup']['errors']
    if record['cleanup_errors']:
        record['status'] = 'BLOCKED_CLEANUP'
    return record
