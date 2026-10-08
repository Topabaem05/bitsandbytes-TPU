"""Local process fixture only. No compiler, client or tensor imports."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--output', required=True); p.add_argument('--mode', required=True)
for key in ('parent-pid', 'parent-process-token', 'process-token', 'deadline-epoch',
            'oracle-sha256', 'expected-sha256', 'admission-sha256'):
    p.add_argument('--' + key, required=True)
a = p.parse_args(); output = Path(a.output)
record = {'scope': 'LOCAL_OS_TOPOLOGY_FIXTURE_ONLY', 'pid': os.getpid(), 'pgid': os.getpgrp(),
          'parent_pid': os.getppid(), 'process_token': a.process_token,
          'parent_process_token': a.parent_process_token, 'deadline_epoch': float(a.deadline_epoch),
          'file_limit_soft_hard': list(resource.getrlimit(resource.RLIMIT_FSIZE))}
output.write_text(json.dumps(record) + '\n')
dump = Path(os.environ['BNB_COMPILER_DUMP_DIRECTORY'])
if a.mode == 'correct':
    (dump / 'one.hlo.pb').write_bytes(b'LOCAL_FIXTURE_NOT_HLO')
elif a.mode == 'total-overflow':
    for i in range(4):
        (dump / str(i)).write_bytes(b'x' * 4096)
    time.sleep(10)
elif a.mode == 'count-overflow':
    for i in range(9):
        (dump / str(i)).write_bytes(b'x')
    time.sleep(10)
elif a.mode == 'file-overflow':
    with (dump / 'large').open('wb') as f:
        f.write(b'x' * 65536)
elif a.mode == 'timeout':
    time.sleep(10)
elif a.mode == 'ignored-term-descendant':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready = output.with_name('descendant.json')
    code = ('import os,signal,time,json;from pathlib import Path;'
            'signal.signal(signal.SIGTERM,signal.SIG_IGN);'
            'Path(' + repr(str(ready)) + ').write_text(json.dumps({"pid":os.getpid(),"pgid":os.getpgrp()}));'
            'time.sleep(30)')
    child = subprocess.Popen([sys.executable, '-B', '-c', code])
    until = time.monotonic() + 1
    while not ready.exists() and time.monotonic() < until:
        time.sleep(.005)
    if not ready.exists():
        child.kill(); child.wait(); raise RuntimeError('DESCENDANT_NOT_READY')
    time.sleep(30)
else:
    raise ValueError('UNKNOWN_FIXTURE_MODE')
