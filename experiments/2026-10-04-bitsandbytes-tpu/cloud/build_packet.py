"""Create a small, source-bound packet. Do not install or allocate resources."""
import argparse
import hashlib
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from nested_contract import (NESTED_BINDINGS, NESTED_PROBE_SHA, NESTED_INPUT_SHA, NESTED_SOURCE_SHA,
    NESTED_SCHEMA_SHA, NESTED_PLUGIN_MANIFEST_SHA, NESTED_SCOPE, NESTED_FILES, NESTED_SCHEMAS)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
COMMIT = '833649043474794b8fe7a4136e0c40faf077b2e0'
RUNTIME_SHA = '323371ff61c5fbcc4f79fd6a358cf2ba17cb72b907382a5ceaac07f91dc66ed6'
PRECISION_ROUTE_PROBE_SHA = 'feae751734e57c741b1bdade004ff7ca3c041ee7eb3b7086b531bc6be433038e'
TRANSFER_PATCH_MANIFEST_SHA = 'e745fbf21aac10ed9118167a131d6505bf6dbe03e1fab0a669c663b26c5a5732'
TRANSFER_PRECISION_PROBE_SHA = 'e0534955507e5f91e67ecddfffc738a367ad752e69a4eea7ae8052c6d76a8852'
TRANSFER_SOURCE_CONTROLS_SHA = '5f5c8b6c3d04a0d8658dae1f4d70d7a909dfe63977b953936aecfb69bf9a16fe'
STATE_HELPER_SHA = '006604d18d10c202583c5d42dfa361a83da441ea99d625241101b58d5948da45'


def package_tree(root):
    files = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink() or (not p.is_file() and not p.is_dir()):
            raise ValueError('PATCH_SOURCE_NONREGULAR')
        if p.is_file():
            files[p.relative_to(root).as_posix()] = sha(p)
    return files


def patched_archive(archive, output, manifest, patch):
    """Apply the one inspected patch to exact archived bytes in a fresh tree."""
    from remote import unpack_source
    if sha(archive) != manifest['base_archive_sha256']:
        raise ValueError('PATCH_BASE_ARCHIVE')
    if patch.is_symlink() or sha(patch) != manifest['patch_sha256']:
        raise ValueError('PATCH_BODY_IDENTITY')
    with tempfile.TemporaryDirectory(prefix='bnb-patched-source-') as temporary:
        tree = Path(temporary) / 'source'
        unpack_source(archive, tree)
        if package_tree(tree / 'bitsandbytes') != manifest['base_package_files']:
            raise ValueError('PATCH_BASE_PACKAGE_INVENTORY')
        for prefix in ('base', 'post_patch'):
            files = {p: h for p, h in manifest[prefix + '_package_files'].items() if p.endswith('.py')}
            if hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != manifest[prefix + '_python_inventory_sha256']:
                raise ValueError('PATCH_PYTHON_INVENTORY_HASH')
        base = package_tree(tree)
        subprocess.run(['git', 'apply', '--check', str(patch.resolve())], cwd=tree, check=True, capture_output=True)
        subprocess.run(['git', 'apply', str(patch.resolve())], cwd=tree, check=True, capture_output=True)
        if package_tree(tree / 'bitsandbytes') != manifest['post_patch_package_files']:
            raise ValueError('PATCH_POST_PACKAGE_INVENTORY')
        post = package_tree(tree)
        changed = {name for name in base if base[name] != post.get(name)}
        if set(base) != set(post) or changed != {'bitsandbytes/nn/modules.py'}:
            raise ValueError('PATCH_CHANGED_FILE_SCOPE')
        with tarfile.open(archive) as original, tarfile.open(output, 'w') as target:
            for member in original.getmembers():
                current = copy.copy(member)
                if member.isfile():
                    path = tree / member.name
                    current.size = path.stat().st_size
                    with path.open('rb') as stream:
                        target.addfile(current, stream)
                else:
                    target.addfile(current)
    return {name: digest for name, digest in manifest['post_patch_package_files'].items() if name.endswith('.py')}


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
          precision_probe=None, precision_probe_sha256=None, patch_manifest=None, patch_manifest_sha256=None,
          transfer_probe=None, transfer_probe_sha256=None, transfer_admission=None, transfer_admission_sha256=None,
          source_controls=None, source_controls_sha256=None, state_probe=None, state_probe_sha256=None,
          state_helper=None, state_helper_sha256=None, nested_probe=None, nested_probe_sha256=None,
          nested_inputs=None, nested_inputs_sha256=None, nested_source=None, nested_source_sha256=None,
          nested_schemas=None, nested_schemas_sha256=None, nested_plugin=None):
    nested = experiment == 'nested-79'
    state = experiment == 'state-roundtrip'
    transfer = experiment in {'transfer-api42', 'state-roundtrip', 'nested-79'}
    if experiment not in {'api42', 'device-route-diagnostic', 'precision-diagnostic', 'transfer-api42', 'state-roundtrip', 'nested-79'}:
        raise ValueError('EXPERIMENT_VARIANT')
    if experiment in {'device-route-diagnostic', 'precision-diagnostic', 'transfer-api42', 'state-roundtrip', 'nested-79'}:
        if route_probe is None or route_probe.is_symlink() or not route_probe.is_file() or sha(route_probe) != route_probe_sha256:
            raise ValueError('ROUTE_PROBE_SOURCE')
    elif route_probe is not None or route_probe_sha256 is not None:
        raise ValueError('DIAGNOSTIC_PROBE_NOT_REQUESTED')
    if experiment in {'precision-diagnostic', 'transfer-api42', 'state-roundtrip', 'nested-79'}:
        if route_probe_sha256 != PRECISION_ROUTE_PROBE_SHA:
            raise ValueError('PRECISION_ROUTE_PROBE_SOURCE')
        if precision_probe is None or precision_probe.is_symlink() or not precision_probe.is_file() or sha(precision_probe) != precision_probe_sha256:
            raise ValueError('PRECISION_PROBE_SOURCE')
    elif precision_probe is not None or precision_probe_sha256 is not None:
        raise ValueError('PRECISION_PROBE_NOT_REQUESTED')
    patch_record = None
    transfer_fields = ((transfer_probe, transfer_probe_sha256, 'TRANSFER_PROBE_SOURCE'),
                       (transfer_admission, transfer_admission_sha256, 'TRANSFER_ADMISSION_SOURCE'),
                       (source_controls, source_controls_sha256, 'TRANSFER_SOURCE_CONTROLS_SOURCE'))
    if transfer:
        if (patch_manifest is None or patch_manifest.is_symlink() or not patch_manifest.is_file() or
                patch_manifest_sha256 != TRANSFER_PATCH_MANIFEST_SHA or sha(patch_manifest) != TRANSFER_PATCH_MANIFEST_SHA):
            raise ValueError('TRANSFER_PATCH_MANIFEST_SOURCE')
        patch_record = json.loads(patch_manifest.read_text())
        patch_path = PROJECT / 'patches/params4bit-xla-v1.patch'
        if (patch_record['format'] != 'bnb-tpu.upstream-patch.v1' or patch_record['id'] != 'params4bit-xla-v1' or
                patch_record['base_revision'] != COMMIT or patch_record['runtime_lock_sha256'] != RUNTIME_SHA or
                patch_record['patch_path'] != 'patches/params4bit-xla-v1.patch' or patch_path.is_symlink() or
                sha(patch_path) != patch_record['patch_sha256']):
            raise ValueError('TRANSFER_PATCH_SOURCE')
        if precision_probe_sha256 != TRANSFER_PRECISION_PROBE_SHA:
            raise ValueError('TRANSFER_PRECISION_SOURCE')
        if source_controls_sha256 != TRANSFER_SOURCE_CONTROLS_SHA:
            raise ValueError('TRANSFER_SOURCE_CONTROLS_SOURCE')
        for path, pin, error in transfer_fields:
            if path is None or path.is_symlink() or not path.is_file() or sha(path) != pin:
                raise ValueError(error)
    elif (patch_manifest is not None or patch_manifest_sha256 is not None or
          any(path is not None or pin is not None for path, pin, _ in transfer_fields)):
        raise ValueError('TRANSFER_SOURCE_NOT_REQUESTED')
    if state:
        if state_helper_sha256 != STATE_HELPER_SHA:
            raise ValueError('STATE_HELPER_IDENTITY')
        for path, pin, error in ((state_probe, state_probe_sha256, 'STATE_PROBE_SOURCE'),
                                 (state_helper, state_helper_sha256, 'STATE_HELPER_SOURCE')):
            if path is None or path.is_symlink() or not path.is_file() or sha(path) != pin:
                raise ValueError(error)
    elif any(value is not None for value in (state_probe, state_probe_sha256, state_helper, state_helper_sha256)):
        raise ValueError('STATE_SOURCE_NOT_REQUESTED')
    nested_fields=((nested_probe,nested_probe_sha256,NESTED_PROBE_SHA,'probe_nested.py'),
                   (nested_inputs,nested_inputs_sha256,NESTED_INPUT_SHA,'nested-inputs.json'),
                   (nested_source,nested_source_sha256,NESTED_SOURCE_SHA,'nested-source.json'),
                   (nested_schemas,nested_schemas_sha256,NESTED_SCHEMA_SHA,'nested-schemas.json'))
    if nested:
        for path,pin,expected,name in nested_fields:
            if path is None or path.is_symlink() or not path.is_file() or pin!=expected or sha(path)!=expected:
                raise ValueError('NESTED_REVIEWED_SOURCE:'+name)
        if nested_plugin is None or nested_plugin.is_symlink() or not nested_plugin.is_dir() or plugin_manifest_sha256!=NESTED_PLUGIN_MANIFEST_SHA:
            raise ValueError('NESTED_PLUGIN_VARIANT')
    elif nested_plugin is not None or any(path is not None or pin is not None for path,pin,_,_ in nested_fields):
        raise ValueError('NESTED_SOURCE_NOT_REQUESTED')
    if out.exists() or out.is_symlink():
        raise FileExistsError('FRESH_PACKET_REQUIRED')
    if subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != COMMIT:
        raise ValueError('UPSTREAM_COMMIT')
    scientific = PROJECT / 'experiments/2026-10-04-bitsandbytes-tpu'
    runtime = scientific / 'runtime'
    if sha(runtime / 'requirements.lock.json') != RUNTIME_SHA:
        raise ValueError('RUNTIME_LOCK')
    plugin = nested_plugin if nested else PROJECT / 'packages/bitsandbytes-tpu'
    if sha(plugin / 'source-manifest.json') != plugin_manifest_sha256:
        raise ValueError('PLUGIN_MANIFEST')
    plugin_manifest = json.loads((plugin / 'source-manifest.json').read_text())
    if not nested and (plugin_manifest.get('source_variant')=='nested-v1' or plugin_manifest.get('installed_python_files')==NESTED_FILES):
        raise ValueError('NESTED_PLUGIN_NOT_REQUESTED')
    if nested and (plugin_manifest.get('source_variant')!='nested-v1' or plugin_manifest.get('installed_python_files')!=NESTED_FILES or plugin_manifest.get('operator_schemas')!=NESTED_SCHEMAS):
        raise ValueError('NESTED_PLUGIN_EXACT_MAP')
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
        if transfer:
            bnb = patched_archive(archive, out / 'patched-upstream.tar', patch_record, patch_path)
    admission = {'format': 'bnb-tpu.probe-source-admission.v1', 'runtime_lock_sha256': RUNTIME_SHA,
                 'bitsandbytes': {'commit': COMMIT, 'files': bnb},
                 'bitsandbytes_tpu': {'files': plugin_manifest['installed_python_files']}}
    if transfer:
        admission['bitsandbytes']['patch_manifest_sha256'] = TRANSFER_PATCH_MANIFEST_SHA
    if nested:
        admission['bitsandbytes_tpu'].update(source_variant='nested-v1',nested_source_sha256=NESTED_SOURCE_SHA)
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
    if nested:
        for path,pin,_,name in nested_fields:
            shutil.copyfile(path,out/name)
            if sha(out/name)!=pin: raise ValueError('NESTED_COPIED_SOURCE')
    for name in ('probe_backend.py', 'probe-profile.json', 'probe-inputs.json'):
        shutil.copyfile(scientific / name, out / name)
    if experiment in {'device-route-diagnostic', 'precision-diagnostic', 'transfer-api42', 'state-roundtrip', 'nested-79'}:
        shutil.copyfile(route_probe, out / 'probe_routes.py')
    if experiment in {'precision-diagnostic', 'transfer-api42', 'state-roundtrip', 'nested-79'}:
        shutil.copyfile(precision_probe, out / 'probe_precision.py')
        if sha(out / 'probe_routes.py') != PRECISION_ROUTE_PROBE_SHA or sha(out / 'probe_precision.py') != precision_probe_sha256:
            raise ValueError('PRECISION_PROBE_COPIED_BYTES')
    if transfer:
        for source, relative in ((patch_manifest, 'patches/params4bit-xla-v1.json'),
                                 (patch_path, 'patches/params4bit-xla-v1.patch'),
                                 (transfer_probe, 'probe_transfer.py'), (transfer_admission, 'transfer_admission.py'),
                                 (source_controls, 'tests/test_transfer_source.py')):
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha(target) != sha(source):
                raise ValueError('TRANSFER_COPIED_BYTES')
    if state:
        for source, name, expected in ((state_probe, 'probe_state_roundtrip.py', state_probe_sha256),
                                       (state_helper, 'probe_state.py', STATE_HELPER_SHA)):
            shutil.copyfile(source, out / name)
            if sha(out / name) != expected:
                raise ValueError('STATE_COPIED_BYTES')
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
    if transfer:
        manifest.update(experiment=experiment, route_probe_sha256=PRECISION_ROUTE_PROBE_SHA,
                        precision_probe_sha256=TRANSFER_PRECISION_PROBE_SHA,
                        transfer_probe_sha256=transfer_probe_sha256, transfer_admission_sha256=transfer_admission_sha256,
                        source_controls_sha256=source_controls_sha256, patch_manifest_sha256=TRANSFER_PATCH_MANIFEST_SHA,
                        patch_sha256=patch_record['patch_sha256'], base_archive_sha256=patch_record['base_archive_sha256'],
                        patched_archive_sha256=sha(out / 'patched-upstream.tar'), precision='highest')
    if state:
        manifest.update(state_probe_sha256=state_probe_sha256, state_helper_sha256=STATE_HELPER_SHA,
                        state_scope='FRESH_SAVE_AND_RESTORE_PROCESSES_ALL8_LINEAR')
    if nested:
        manifest.update(**{field:sha(out/name) for field,name in NESTED_BINDINGS.items()}, nested_scope=NESTED_SCOPE, nested_source_variant='nested-v1')
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
    p.add_argument('--experiment', choices=['api42', 'device-route-diagnostic', 'precision-diagnostic', 'transfer-api42', 'state-roundtrip', 'nested-79'], default='api42')
    p.add_argument('--route-probe', type=Path)
    p.add_argument('--route-probe-sha256')
    p.add_argument('--precision-probe', type=Path)
    p.add_argument('--precision-probe-sha256')
    for name in ('patch-manifest', 'transfer-probe', 'transfer-admission', 'source-controls', 'state-probe', 'state-helper', 'nested-probe', 'nested-inputs', 'nested-source', 'nested-schemas'):
        p.add_argument('--' + name, type=Path)
        p.add_argument('--' + name + '-sha256')
    p.add_argument('--nested-plugin',type=Path)
    a = p.parse_args()
    print(json.dumps(build(a.upstream, a.out, a.plugin_manifest_sha256, experiment=a.experiment,
                          route_probe=a.route_probe, route_probe_sha256=a.route_probe_sha256,
                          precision_probe=a.precision_probe, precision_probe_sha256=a.precision_probe_sha256,
                          patch_manifest=a.patch_manifest, patch_manifest_sha256=a.patch_manifest_sha256,
                          transfer_probe=a.transfer_probe, transfer_probe_sha256=a.transfer_probe_sha256,
                          transfer_admission=a.transfer_admission, transfer_admission_sha256=a.transfer_admission_sha256,
                          source_controls=a.source_controls, source_controls_sha256=a.source_controls_sha256,
                          state_probe=a.state_probe, state_probe_sha256=a.state_probe_sha256,
                          state_helper=a.state_helper, state_helper_sha256=a.state_helper_sha256,
                          nested_probe=a.nested_probe,nested_probe_sha256=a.nested_probe_sha256,
                          nested_inputs=a.nested_inputs,nested_inputs_sha256=a.nested_inputs_sha256,
                          nested_source=a.nested_source,nested_source_sha256=a.nested_source_sha256,
                          nested_schemas=a.nested_schemas,nested_schemas_sha256=a.nested_schemas_sha256,nested_plugin=a.nested_plugin), sort_keys=True))
