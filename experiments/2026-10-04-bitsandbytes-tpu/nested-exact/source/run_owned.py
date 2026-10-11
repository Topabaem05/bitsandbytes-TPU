"""Bounded offline controls through unchanged canonical Ownership."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser()
for key in ('repo', 'generation', 'oracle', 'readback-lock', 'mul-reference', 'old-failed-receipt', 'output'):
    p.add_argument('--' + key, type=Path, required=True)
a = p.parse_args(); repo = a.repo.resolve(); output = a.output.resolve(); output.mkdir(parents=True, exist_ok=False)
sys.path[:0] = [str(repo / 'experiments/2026-10-04-bitsandbytes-tpu/cloud/ownership'),
                str(repo / 'experiments/2026-10-04-bitsandbytes-tpu/cloud')]
from cleanup_lifecycle import Ownership
argv = [sys.executable, '-B', str(Path(__file__).with_name('controls.py'))]
for name in ('repo', 'generation', 'oracle', 'readback-lock', 'mul-reference', 'old-failed-receipt'):
    argv += ['--' + name, str(getattr(a, name.replace('-', '_')).resolve())]
argv += ['--output', str(output / 'controls')]
files = sorted(Path(__file__).parent.glob('*.py')) + sorted(Path(__file__).parent.glob('*.json'))
pre = {str(f.resolve()): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
owner = Ownership(output); started = time.monotonic(); code = None
with owner.guard(deadline_seconds=240):
    with (output / 'stdout.raw').open('wb') as out, (output / 'stderr.raw').open('wb') as err:
        child = owner.launch(argv, record=output / 'ownership.json', stdout=out, stderr=err, cwd=repo)
        code = child.wait(timeout=220)
post = {str(f.resolve()): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
report = {'exit_code': code, 'source_pre': pre, 'source_post': post, 'source_unchanged': pre == post,
          'elapsed_seconds': time.monotonic() - started, 'cleanup': owner.summary(), 'provider_calls': 0, 'TPU': 'NOT_RUN'}
(output / 'result.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
print(json.dumps(report)); raise SystemExit(0 if code == 0 and pre == post and not report['cleanup']['errors'] else 1)
