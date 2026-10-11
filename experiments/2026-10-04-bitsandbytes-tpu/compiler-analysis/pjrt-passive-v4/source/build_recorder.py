"""Build the shim from one exact upstream header. Keep no header body."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import urllib.request

import sys
sys.dont_write_bytecode = True

from run_controls import HEADER_SHA, HEADER_URL

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--header', type=Path)
options = parser.parse_args()
assert not options.output.exists()
source = Path(__file__).resolve().parent
compiler = shutil.which('clang++') or shutil.which('c++')
assert compiler
with tempfile.TemporaryDirectory(prefix='m6-pjrt-build-') as td:
    body = options.header.read_bytes() if options.header else urllib.request.urlopen(HEADER_URL, timeout=30).read()
    assert len(body) == 102174 and hashlib.sha256(body).hexdigest() == HEADER_SHA
    header = Path(td) / 'pjrt_c_api.h'
    header.write_bytes(body)
    argv = [compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-pthread',
            '-dynamiclib' if platform.system() == 'Darwin' else '-shared',
            '-fPIC', '-I', td, str(source / 'recorder.cpp'), '-o', str(options.output)]
    if platform.system() != 'Darwin':
        argv.append('-ldl')
    subprocess.run(argv, check=True, timeout=60)
result = {'library': str(options.output.absolute()),
          'bytes': options.output.stat().st_size,
          'sha256': hashlib.sha256(options.output.read_bytes()).hexdigest(),
          'header_sha256': HEADER_SHA,
          'source_sha256': hashlib.sha256((source / 'recorder.cpp').read_bytes()).hexdigest(),
          'retention': 'Temporary header deleted'}
print(json.dumps(result, indent=2, sort_keys=True))
