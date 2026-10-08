"""Create a small, source-bound packet. Do not install or allocate resources."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
COMMIT = '833649043474794b8fe7a4136e0c40faf077b2e0'
RUNTIME_SHA = '323371ff61c5fbcc4f79fd6a358cf2ba17cb72b907382a5ceaac07f91dc66ed6'
PRECISION_ROUTE_PROBE_SHA = 'feae751734e57c741b1bdade004ff7ca3c041ee7eb3b7086b531bc6be433038e'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def python_tree(root):
    files = {}
    for p in sorted(root.rglob('*.py')):
        if p.is_symlink() or not p.is_file():
            raise ValueError('SOURCE_NONREGULAR')
        files[p.relative_to(root).as_posix()] = sha(p)
    if not files:
        raise ValueError('SOURCE_EMPTY')
    return files


def build(upstream, out, plugin_manifest_sha256, *, experiment='api42', route_probe=None, route_probe_sha256=None,
          precision_probe=None, precision_probe_sha256=None):
    if experiment not in {'api42', 'device-route-diagnostic', 'precision-diagnostic'}:
        raise ValueError('EXPERIMENT_VARIANT')
    if experiment in {'device-route-diagnostic', 'precision-diagnostic'}:
        if route_probe is None or route_probe.is_symlink() or not route_probe.is_file() or sha(route_probe) != route_probe_sha256:
            raise ValueError('ROUTE_PROBE_SOURCE')
    elif route_probe is not None or route_probe_sha256 is not None:
        raise ValueError('DIAGNOSTIC_PROBE_NOT_REQUESTED')
    if experiment == 'precision-diagnostic':
        if route_probe_sha256 != PRECISION_ROUTE_PROBE_SHA:
            raise ValueError('PRECISION_ROUTE_PROBE_SOURCE')
        if precision_probe is None or precision_probe.is_symlink() or not precision_probe.is_file() or sha(precision_probe) != precision_probe_sha256:
            raise ValueError('PRECISION_PROBE_SOURCE')
    elif precision_probe is not None or precision_probe_sha256 is not None:
        raise ValueError('PRECISION_PROBE_NOT_REQUESTED')
    if out.exists() or out.is_symlink():
        raise FileExistsError('FRESH_PACKET_REQUIRED')
    if subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != COMMIT:
        raise ValueError('UPSTREAM_COMMIT')
    scientific = PROJECT / 'experiments/2026-10-04-bitsandbytes-tpu'
    runtime = scientific / 'runtime'
    if sha(runtime / 'requirements.lock.json') != RUNTIME_SHA:
        raise ValueError('RUNTIME_LOCK')
    plugin = PROJECT / 'packages/bitsandbytes-tpu'
    if sha(plugin / 'source-manifest.json') != plugin_manifest_sha256:
        raise ValueError('PLUGIN_MANIFEST')
    plugin_manifest = json.loads((plugin / 'source-manifest.json').read_text())
    if python_tree(plugin / 'src/bitsandbytes_tpu') != plugin_manifest['installed_python_files']:
        raise ValueError('PLUGIN_SOURCE_BYTES')
    for rec in plugin_manifest['files']:
        p = plugin / rec['path']
        if p.is_symlink() or sha(p) != rec['sha256'] or p.stat().st_size != rec['bytes']:
            raise ValueError('PLUGIN_BOUND_FILE')
    out.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix='bnb-packet-') as temporary:
        tree = Path(temporary)
        archive = tree / 'upstream.tar'
        with archive.open('wb') as f:
            subprocess.run(['git', '-C', str(upstream), 'archive', '--format=tar', COMMIT], stdout=f, check=True)
        # Read exact archive package source; the mutable checkout is not the input.
        with tarfile.open(archive) as tar:
            bnb = {m.name.removeprefix('bitsandbytes/'): hashlib.sha256(tar.extractfile(m).read()).hexdigest()
                   for m in tar.getmembers() if m.isfile() and m.name.startswith('bitsandbytes/') and m.name.endswith('.py')}
        shutil.copyfile(archive, out / 'upstream.tar')
    admission = {'format': 'bnb-tpu.probe-source-admission.v1', 'runtime_lock_sha256': RUNTIME_SHA,
                 'bitsandbytes': {'commit': COMMIT, 'files': bnb},
                 'bitsandbytes_tpu': {'files': plugin_manifest['installed_python_files']}}
    write(out / 'source-admission.json', admission)
    for p in sorted((plugin / 'src').rglob('*.py')):
        relative = p.relative_to(plugin)
        dest = out / 'plugin' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    for name in ('pyproject.toml', 'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'source-manifest.json'):
        shutil.copyfile(plugin / name, out / 'plugin' / name)
    if python_tree(out / 'plugin/src/bitsandbytes_tpu') != plugin_manifest['installed_python_files']:
        raise ValueError('PLUGIN_COPIED_BYTES')
    for name in ('probe_backend.py', 'probe-profile.json', 'probe-inputs.json'):
        shutil.copyfile(scientific / name, out / name)
    if experiment in {'device-route-diagnostic', 'precision-diagnostic'}:
        shutil.copyfile(route_probe, out / 'probe_routes.py')
    if experiment == 'precision-diagnostic':
        shutil.copyfile(precision_probe, out / 'probe_precision.py')
        if sha(out / 'probe_routes.py') != PRECISION_ROUTE_PROBE_SHA or sha(out / 'probe_precision.py') != precision_probe_sha256:
            raise ValueError('PRECISION_PROBE_COPIED_BYTES')
    for name in ('resolve.py', 'probe_runtime.py', 'requirements.lock.json'):
        dest = out / 'runtime' / name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(runtime / name, dest)
    for p in sorted(HERE.rglob('*.py')):
        if 'tests' in p.relative_to(HERE).parts or p.name == 'build_packet.py':
            continue
        dest = out / 'cloud' / p.relative_to(HERE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    shutil.copyfile(HERE / 'build-requirements.lock.json', out / 'build-requirements.lock.json')
    members = {p.relative_to(out).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size}
               for p in sorted(out.rglob('*')) if p.is_file()}
    manifest = {'format': 'bnb-tpu.first-packet.v1', 'runtime_lock_sha256': RUNTIME_SHA,
                'source_admission_sha256': sha(out / 'source-admission.json'), 'files': members,
                'plugin_source_manifest_sha256': plugin_manifest_sha256,
                'upstream_commit': COMMIT, 'upstream_source_url': f'https://github.com/bitsandbytes-foundation/bitsandbytes/tree/{COMMIT}',
                'budget': {'total': 3600, 'install': 1200, 'science': 1800, 'retrieval': 600, 'cleanup': 60}}
    if experiment in {'device-route-diagnostic', 'precision-diagnostic'}:
        manifest.update(experiment=experiment, diagnostic_only=True, route_probe_sha256=sha(out / 'probe_routes.py'))
    if experiment == 'precision-diagnostic':
        manifest['precision_probe_sha256'] = sha(out / 'probe_precision.py')
    write(out / 'manifest.json', manifest)
    with zipfile.ZipFile(out / 'payload.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in sorted([*members, 'manifest.json']):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 4, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, (out / name).read_bytes())
    return {'manifest_sha256': sha(out / 'manifest.json'), 'payload_sha256': sha(out / 'payload.zip'),
            'payload_bytes': (out / 'payload.zip').stat().st_size, 'members': len(members) + 1}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--plugin-manifest-sha256', required=True)
    p.add_argument('--experiment', choices=['api42', 'device-route-diagnostic', 'precision-diagnostic'], default='api42')
    p.add_argument('--route-probe', type=Path)
    p.add_argument('--route-probe-sha256')
    p.add_argument('--precision-probe', type=Path)
    p.add_argument('--precision-probe-sha256')
    a = p.parse_args()
    print(json.dumps(build(a.upstream, a.out, a.plugin_manifest_sha256, experiment=a.experiment,
                          route_probe=a.route_probe, route_probe_sha256=a.route_probe_sha256,
                          precision_probe=a.precision_probe, precision_probe_sha256=a.precision_probe_sha256), sort_keys=True))
