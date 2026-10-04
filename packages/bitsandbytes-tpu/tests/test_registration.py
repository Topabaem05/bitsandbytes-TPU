"""Registration controls. No XLA tensor or TPU client is created."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

PACKAGE = Path(__file__).resolve().parents[1]


def child(code, extra_path=None):
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONPATH'] = str(extra_path or PACKAGE / 'src')
    p = subprocess.run([sys.executable, '-B', '-c', code], env=env,
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
    return json.loads(p.stdout.splitlines()[-1])


def test_register_preserves_classes_schemas_cpu_and_is_idempotent():
    assert importlib.util.find_spec('bitsandbytes_tpu.registration') is not None, 'Registration is not implemented'
    got = child('''
import sys, json, torch, bitsandbytes as b
from bitsandbytes_tpu import register
ops = ['quantize_4bit', 'dequantize_4bit', 'dequantize_4bit.out', 'gemm_4bit']
classes = (b.nn.Linear4bit, b.nn.Params4bit, b.functional.QuantState)
pre = {n: torch._C._dispatch_dump_table('bitsandbytes::'+n) for n in ops}
schemas = {n: str(torch._C._dispatch_find_schema_or_throw('bitsandbytes::'+n.split('.')[0], n.split('.')[1] if '.' in n else '').schema()) for n in ops}
register(); register()
assert classes == (b.nn.Linear4bit, b.nn.Params4bit, b.functional.QuantState)
assert all(torch._C._dispatch_has_kernel_for_dispatch_key('bitsandbytes::'+n, 'XLA') for n in ops)
for n in ops:
 post = torch._C._dispatch_dump_table('bitsandbytes::'+n)
 for key in ('CPU:', 'CUDA:', 'CompositeExplicitAutograd:'):
  assert [x for x in pre[n].splitlines() if x.startswith(key)] == [x for x in post.splitlines() if x.startswith(key)]
 assert schemas[n] == str(torch._C._dispatch_find_schema_or_throw('bitsandbytes::'+n.split('.')[0], n.split('.')[1] if '.' in n else '').schema())
assert 'xla' in b.supported_torch_devices
assert not any(n == 'jax' or n.startswith('jax.') or n == 'torch_xla' or n.startswith('torch_xla.') for n in sys.modules)
print(json.dumps({'registered': len(ops), 'classes_unchanged': True, 'cpu_unchanged': True, 'runtime_imports': False}))
''')
    assert got['registered'] == 4 and got['classes_unchanged'] and not got['runtime_imports']


def test_plugin_import_does_not_import_torch_or_runtime():
    got = child('''
import sys,json,bitsandbytes_tpu
assert 'torch' not in sys.modules and 'jax' not in sys.modules and 'torch_xla' not in sys.modules
print(json.dumps({'version':bitsandbytes_tpu.__version__,'lazy':True}))
''')
    assert got == {'version': '0.1.0', 'lazy': True}


def test_existing_xla_kernel_is_rejected_without_partial_registration():
    got = child('''
import json,torch,bitsandbytes
lib = torch.library.Library('bitsandbytes','IMPL','XLA')
lib.impl('gemm_4bit', lambda *a,**k: None)
from bitsandbytes_tpu import register
try: register()
except RuntimeError as e: assert 'existing XLA' in str(e)
else: raise AssertionError('Existing implementation was overwritten')
assert not torch._C._dispatch_has_kernel_for_dispatch_key('bitsandbytes::quantize_4bit','XLA')
assert torch._C._dispatch_has_kernel_for_dispatch_key('bitsandbytes::gemm_4bit','XLA')
assert 'xla' not in bitsandbytes.supported_torch_devices
print(json.dumps({'rejected':True,'partial':False}))
''')
    assert got == {'rejected': True, 'partial': False}


def test_source_change_is_rejected_before_any_registration(tmp_path):
    got = child('''
import json,torch,bitsandbytes as b, pathlib, shutil, tempfile
from bitsandbytes_tpu import register
with tempfile.TemporaryDirectory() as d:
 root=pathlib.Path(d)/'bitsandbytes'; shutil.copytree(pathlib.Path(b.__file__).parent, root, ignore=shutil.ignore_patterns('__pycache__'))
 with (root/'functional.py').open('a') as f: f.write('\\n# Unreviewed change\\n')
 old=b.__file__; b.__file__=str(root/'__init__.py')
 try:
  try: register()
  except RuntimeError as e: assert 'source pin' in str(e)
  else: raise AssertionError('Unreviewed source admitted')
 finally: b.__file__=old
assert not torch._C._dispatch_has_kernel_for_dispatch_key('bitsandbytes::quantize_4bit','XLA')
print(json.dumps({'rejected':True}))
''')
    assert got['rejected']


def test_wrong_schema_is_rejected_before_registration():
    got = child('''
import json,torch,bitsandbytes as b
from bitsandbytes_tpu import register
from bitsandbytes_tpu import compatibility as c
c.SCHEMAS['quantize_4bit'] = '(Tensor A, int blocksize, str quant_type, ScalarType quant_storage) -> Tensor'
try: register()
except RuntimeError as e: assert 'schema' in str(e)
else: raise AssertionError('Wrong ABI admitted')
assert not torch._C._dispatch_has_kernel_for_dispatch_key('bitsandbytes::quantize_4bit','XLA')
print(json.dumps({'rejected':True}))
''')
    assert got['rejected']


def test_gate_does_not_require_version_during_backend_autoload():
    got = child('''
import json,torch,bitsandbytes as b
from bitsandbytes_tpu import register
old=b.__version__; del b.__version__
try: register()
finally: b.__version__=old
assert torch._C._dispatch_has_kernel_for_dispatch_key('bitsandbytes::gemm_4bit','XLA')
print(json.dumps({'version_unset_accepted':True}))
''')
    assert got['version_unset_accepted']


def test_failed_registration_removes_partial_kernels_and_can_retry():
    got = child('''
import json,torch,bitsandbytes as b
from bitsandbytes_tpu import register
original=torch.library.register_kernel
count=0
def fail_second(*a,**kw):
 global count
 count+=1
 if count==2: raise RuntimeError('Injected registration failure')
 return original(*a,**kw)
torch.library.register_kernel=fail_second
try:
 try: register()
 except RuntimeError as e: assert 'Injected' in str(e)
 else: raise AssertionError('Failure was hidden')
finally: torch.library.register_kernel=original
assert not torch._C._dispatch_has_kernel_for_dispatch_key('bitsandbytes::quantize_4bit','XLA')
assert 'xla' not in b.supported_torch_devices
register()
assert torch._C._dispatch_has_kernel_for_dispatch_key('bitsandbytes::gemm_4bit','XLA')
print(json.dumps({'rolled_back':True,'retry_registered':True}))
''')
    assert got == {'rolled_back': True, 'retry_registered': True}


def test_real_wheel_target_install_autoloads_without_runtime_import(tmp_path):
    import shutil
    source = tmp_path / 'source'
    shutil.copytree(PACKAGE, source, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache', 'build', '*.egg-info'))
    wheels = tmp_path / 'wheels'
    wheels.mkdir()
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    build = subprocess.run([sys.executable, '-B', '-c',
                            'from setuptools.build_meta import build_wheel; import sys; build_wheel(sys.argv[1])', str(wheels)],
                           cwd=source, capture_output=True, text=True, env=env, timeout=60)
    assert build.returncode == 0, build.stdout + build.stderr
    wheel, = wheels.glob('*.whl')
    import zipfile
    with zipfile.ZipFile(wheel) as archive:
        prefix = 'bitsandbytes_tpu-0.1.0.dist-info/'
        assert 'License-Expression: Apache-2.0 AND MIT' in archive.read(prefix + 'METADATA').decode()
        assert archive.read(prefix + 'licenses/LICENSE') == (PACKAGE / 'LICENSE').read_bytes()
        assert archive.read(prefix + 'licenses/THIRD_PARTY_NOTICES.md') == (PACKAGE / 'THIRD_PARTY_NOTICES.md').read_bytes()
    target = tmp_path / 'install'
    install = subprocess.run(['uv', 'pip', 'install', '--python', sys.executable,
                              '--target', str(target), '--no-deps', str(wheel)],
                             capture_output=True, text=True, env=env, timeout=60)
    assert install.returncode == 0, install.stdout + install.stderr
    # Keep actual builder and installer output. This is not hand-written dist-info.
    (tmp_path / 'wheel-build.stdout').write_text(build.stdout)
    (tmp_path / 'wheel-build.stderr').write_text(build.stderr)
    (tmp_path / 'wheel-install.stdout').write_text(install.stdout)
    (tmp_path / 'wheel-install.stderr').write_text(install.stderr)
    got = child('''
import sys,json,importlib.metadata as m
import bitsandbytes as b,torch,bitsandbytes_tpu
entries=[e for e in m.entry_points(group='bitsandbytes.backends') if e.name=='tpu']
assert len(entries)==1 and entries[0].value=='bitsandbytes_tpu:register'
assert m.version('bitsandbytes-tpu')=='0.1.0'
assert torch._C._dispatch_has_kernel_for_dispatch_key('bitsandbytes::gemm_4bit','XLA')
assert b.__version__=='0.50.3.dev0'
assert not any(n=='jax' or n.startswith('jax.') or n=='torch_xla' or n.startswith('torch_xla.') for n in sys.modules)
entries[0].load()()
print(json.dumps({'distribution':m.version('bitsandbytes-tpu'),'entrypoint_autoload':True,'runtime_imports':False,'plugin':bitsandbytes_tpu.__file__}))
''', target)
    assert Path(got['plugin']).is_relative_to(target)
    assert got['entrypoint_autoload'] and not got['runtime_imports']
    (tmp_path / 'autoload-result.json').write_text(json.dumps(got, indent=2) + '\n')
