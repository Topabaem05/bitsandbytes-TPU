"""Run the reviewed local owner within a bounded, owned process group."""
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

ROOT = Path('/Users/topamini/code/bitsandbytes-TPU')
CLOUD = Path(__file__).resolve().parent / 'cloud'
sys.path.insert(0, str(CLOUD / 'ownership'))
from cleanup_lifecycle import Ownership
from lifecycle import durable_json


def guard_seconds(config,now=None):
    if 'absolute_deadline_epoch' not in config:return 3660
    deadline=config['absolute_deadline_epoch'];now=time.time() if now is None else now
    if type(deadline) not in (int,float) or not math.isfinite(deadline) or deadline<=now:raise ValueError('HOST_ABSOLUTE_DEADLINE_EXPIRED')
    return min(3660,deadline+60-now)


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    seconds=guard_seconds(config)
    out = Path(config['host_output'])
    out.mkdir(parents=True, exist_ok=False)
    owner = Ownership(out)
    stream = (out / 'driver.stdout-stderr.raw').open('wb', buffering=0)
    started = time.monotonic()
    rec = {'argv': config['argv'], 'status': 'STARTING',
           'started_utc': datetime.now(timezone.utc).isoformat(),
           'owner_budget_seconds': 3600, 'outer_guard_seconds': seconds, 'absolute_deadline_epoch':config.get('absolute_deadline_epoch'),
           'emergency_closure_allowance_seconds': 60}
    proc = None
    try:
        with owner.guard(guard_seconds(config)):
            proc = owner.launch(config['argv'], record=out / 'driver-ownership.json', cwd=ROOT,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), stdout=stream, stderr=stream)
            rec.update(status='RUNNING', driver_pid=proc.pid, driver_pgid=proc.pid)
            durable_json(out / 'host-process.json', rec)
            proc.wait()
            rec.update(status='DRIVER_EXITED', exit_code=proc.returncode)
    except BaseException as error:
        rec.update(status='DRIVER_ERROR', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        rec['original_outcome'] = {'status': rec['status'], 'exit_code': rec.get('exit_code'), 'error': rec.get('error')}
        try:
            os.fsync(stream.fileno())
        except Exception as error:
            owner.note_error('fsync_driver', error)
        finally:
            stream.close()
        rec.update(cleanup=owner.summary(), elapsed_seconds=time.monotonic() - started,
                   finished_utc=datetime.now(timezone.utc).isoformat())
        if rec['cleanup']['errors']:
            rec['status'] = 'BLOCKED_CLEANUP'
        durable_json(out / 'host-finalization.json', rec)
    print(json.dumps({k: rec.get(k) for k in ['status', 'exit_code', 'driver_pid', 'elapsed_seconds']}))
    return 0 if rec.get('exit_code') == 0 and not rec['cleanup']['errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
