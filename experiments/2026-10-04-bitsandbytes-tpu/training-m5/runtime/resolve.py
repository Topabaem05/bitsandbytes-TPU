"""Resolve and verify the Linux CPython 3.12 TPU wheel profile. Do not install."""
import argparse
import email
import hashlib
import json
import stat
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import compatible_tags, cpython_tags, parse_tag
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

CORE = {'torch': '2.9.0+cpu', 'torch-xla': '2.9.0', 'libtpu': '0.0.21',
        'jax': '0.7.1', 'jaxlib': '0.7.1'}
ENV = {'implementation_name': 'cpython', 'implementation_version': '3.12.14',
       'os_name': 'posix', 'platform_machine': 'x86_64', 'platform_release': '',
       'platform_system': 'Linux', 'platform_version': '', 'python_full_version': '3.12.14',
       'platform_python_implementation': 'CPython', 'python_version': '3.12',
       'sys_platform': 'linux'}
PLATFORMS = [f'manylinux_2_{n}_x86_64' for n in range(31, 4, -1)] + [
    'manylinux2014_x86_64', 'manylinux2010_x86_64', 'manylinux1_x86_64']
TAGS = set(cpython_tags((3, 12), ['cp312'], PLATFORMS)) | set(
    compatible_tags((3, 12), 'cp312', PLATFORMS))
CPU = {'name': 'torch', 'version': '2.9.0+cpu',
       'filename': 'torch-2.9.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl',
       'url': 'https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl',
       'sha256': '28f6eb31b08180a5c5e98d5bc14eef6909c9f5a1dbff9632c3e02a8773449349',
       'size': 184388168}
SOURCES = {
 'setup.py': '2769e5ba7953a8a93bdc086a2ea2d85c59ea2e63f782b813c04a4c528bc999aa',
 'torch_xla/experimental/custom_kernel.py': 'c347f8fcb4844fa8849109680ad81b94e221a2c8b011462553b48ec9f67e6338',
 'torch_xla/runtime.py': '26e6f341ba5507e3cb8c437d121aa2a50055e9ba7acacc72a2274a6a54dadd83',
 'torch_xla/_internal/jax_workarounds.py': 'fd8c53eb24fb373fba35c938e00b1ef36caadf3eaf8a74d266528ec6602ad42f',
}
SOURCE_REVISION = '5fab7053df86c8d503b98d9e7202ca8b8d4978c7'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check_name(name):
    n = canonicalize_name(name)
    if n.startswith(('nvidia-', 'jax-cuda', 'jax-rocm')) or n in {'triton', 'torchvision', 'torchaudio'}:
        raise ValueError('FORBIDDEN_DEPENDENCY:' + n)


def stable_versions(values):
    result = []
    for value in values:
        try:
            version = Version(value)
        except InvalidVersion:
            continue  # Legacy index entries cannot meet PEP 440 requirements.
        if not version.is_prerelease and not version.is_devrelease:
            result.append(value)
    return sorted(result, key=Version, reverse=True)


def active(text, extras=()):
    req = Requirement(text)
    return req.marker is None or any(req.marker.evaluate({**ENV, 'extra': e}) for e in ['', *extras])


def validate_profile(versions):
    for name, version in CORE.items():
        if versions.get(name) != version:
            raise ValueError('PROFILE:' + name)
    for name in versions:
        check_name(name)


def metadata(raw):
    msg = email.message_from_bytes(raw)
    if not msg.get('Name') or not msg.get('Version'):
        raise ValueError('METADATA_IDENTITY_MISSING')
    return {'name': canonicalize_name(msg['Name']), 'version': msg['Version'],
            'requires_python': msg.get('Requires-Python', ''),
            'requires_dist': msg.get_all('Requires-Dist', [])}


def verify_wheel(path, rec):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('WHEEL_NOT_REGULAR')
    if sha(path) != rec['sha256']:
        raise ValueError('WHEEL_SHA')
    if path.stat().st_size != rec['size']:
        raise ValueError('WHEEL_SIZE')
    name, version, _, tags = parse_wheel_filename(rec['filename'])
    if canonicalize_name(name) != rec['name'] or str(version) != rec['version'] or not tags & TAGS:
        raise ValueError('WHEEL_TAG_OR_IDENTITY')
    with zipfile.ZipFile(path) as z:
        seen = set()
        for info in z.infolist():
            p = PurePosixPath(info.filename)
            mode = info.external_attr >> 16
            if (info.filename in seen or p.is_absolute() or '..' in p.parts or
                '\\' in info.filename or '\x00' in info.filename or
                (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR))):
                raise ValueError('UNSAFE_MEMBER:' + info.filename)
            seen.add(info.filename)
        if z.testzip() is not None:
            raise ValueError('ZIP_CRC')
        metas = [n for n in seen if n.endswith('.dist-info/METADATA') and len(PurePosixPath(n).parts) == 2]
        if len(metas) != 1:
            raise ValueError('METADATA_COUNT')
        wheel_file = str(PurePosixPath(metas[0]).parent / 'WHEEL')
        wheel_metadata = email.message_from_bytes(z.read(wheel_file))
        internal_tags = set()
        for text in wheel_metadata.get_all('Tag', []):
            internal_tags.update(parse_tag(text))
        if wheel_metadata.get('Wheel-Version') != '1.0' or internal_tags != set(tags):
            raise ValueError('WHEEL_INTERNAL_TAGS')
        raw = z.read(metas[0])
        result = metadata(raw)
        if result['name'] != rec['name'] or result['version'] != rec['version']:
            raise ValueError('WHEEL_IDENTITY')
        if result['requires_python'] and not SpecifierSet(result['requires_python']).contains(ENV['python_full_version']):
            raise ValueError('PYTHON_REQUIREMENT')
        result.update(metadata_sha256=hashlib.sha256(raw).hexdigest(), zip_members=len(seen))
        return result


def validate_closure(records):
    by_name = {rec['name']: rec for rec in records}
    if len(by_name) != len(records):
        raise ValueError('DUPLICATE_PACKAGE')
    validate_profile({n: r['version'] for n, r in by_name.items()})
    for rec in records:
        for text in rec['requires_dist']:
            if not active(text, rec.get('extras', [])):
                continue
            req = Requirement(text)
            name = canonicalize_name(req.name)
            check_name(name)
            if req.url or name not in by_name or not req.specifier.contains(by_name[name]['version']):
                raise ValueError('DEPENDENCY_UNSATISFIED:' + text)
            if not set(req.extras) <= set(by_name[name].get('extras', [])):
                raise ValueError('DEPENDENCY_EXTRAS_UNSATISFIED:' + text)


def read_url(url):
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != 'https' or parts.hostname not in {'pypi.org', 'files.pythonhosted.org', 'download.pytorch.org', 'raw.githubusercontent.com'}:
        raise ValueError('SOURCE_URL')
    with urllib.request.urlopen(url, timeout=60) as f:
        return f.read()


class Resolver:
    def __init__(self):
        self.indexes = {}
        self.releases = {}
        self.cpu_meta = metadata(read_url(CPU['url'] + '.metadata'))

    def index(self, name):
        if name not in self.indexes:
            self.indexes[name] = json.loads(read_url(f'https://pypi.org/pypi/{name}/json'))
        return self.indexes[name]

    def release(self, name, version):
        key = (name, version)
        if key not in self.releases:
            self.releases[key] = json.loads(read_url(f'https://pypi.org/pypi/{name}/{version}/json'))
        return self.releases[key]

    def choose(self, name, requirements):
        check_name(name)
        if name == 'torch':
            rec = {**CPU, **self.cpu_meta}
            if not all(Requirement(r).specifier.contains(rec['version']) for r in requirements):
                raise ValueError('CORE_CONFLICT:torch')
            return rec
        versions = [CORE[name]] if name in CORE else stable_versions(self.index(name)['releases'])
        for version in versions:
            if not all(Requirement(r).specifier.contains(version) for r in requirements):
                continue
            release = self.release(name, version)
            wheels = []
            for f in release['urls']:
                if f['packagetype'] != 'bdist_wheel' or f.get('yanked'):
                    continue
                if f.get('requires_python') and not SpecifierSet(f['requires_python']).contains(ENV['python_full_version']):
                    continue
                _, _, _, tags = parse_wheel_filename(f['filename'])
                if tags & TAGS:
                    wheels.append(f)
            if wheels:
                f = sorted(wheels, key=lambda f: f['filename'])[0]
                return {'name': name, 'version': version, 'filename': f['filename'], 'url': f['url'],
                        'sha256': f['digests']['sha256'], 'size': f['size'],
                        'requires_dist': release['info']['requires_dist'] or []}
        raise ValueError('NO_COMPATIBLE_WHEEL:' + name)

    def resolve(self):
        current = {}
        for _ in range(20):
            requirements = {n: [f'{n}=={v}'] for n, v in CORE.items()}
            extras = {'torch-xla': {'tpu', 'pallas'}}
            queue = list(requirements)
            selected = {}
            while queue:
                name = queue.pop(0)
                rec = self.choose(name, requirements[name])
                selected[name] = rec
                rec['extras'] = sorted(extras.get(name, set()))
                # Use previous full selection on the next pass to propagate constraints.
                source = current.get(name, rec)
                for text in source['requires_dist']:
                    if not active(text, rec['extras']):
                        continue
                    req = Requirement(text)
                    dep = canonicalize_name(req.name)
                    check_name(dep)
                    if req.url:
                        raise ValueError('UNPINNED_DIRECT_DEPENDENCY')
                    if dep not in requirements:
                        requirements[dep] = []
                        queue.append(dep)
                    if text not in requirements[dep]:
                        requirements[dep].append(text)
                    extras.setdefault(dep, set()).update(req.extras)
            if json.dumps(selected, sort_keys=True) == json.dumps(current, sort_keys=True):
                result = [selected[n] for n in sorted(selected)]
                validate_closure(result)
                return result
            current = selected
        raise ValueError('RESOLUTION_DID_NOT_CONVERGE')


def fetch_wheel(rec, cache):
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / rec['filename']
    if not path.exists():
        tmp = path.with_suffix('.partial')
        with urllib.request.urlopen(rec['url'], timeout=60) as src, tmp.open('xb') as dst:
            while chunk := src.read(1024 * 1024):
                dst.write(chunk)
        if sha(tmp) != rec['sha256'] or tmp.stat().st_size != rec['size']:
            raise ValueError('DOWNLOAD_SHA_OR_SIZE:' + rec['name'])
        tmp.rename(path)
    actual = verify_wheel(path, rec)
    rec.update(actual)
    return path


def verify_sources(cache, xla_path):
    receipts = []
    for relative, expected in SOURCES.items():
        url = f'https://raw.githubusercontent.com/pytorch/xla/{SOURCE_REVISION}/{relative}'
        raw = read_url(url)
        got = hashlib.sha256(raw).hexdigest()
        if got != expected:
            raise ValueError('TAG_SOURCE_SHA:' + relative)
        parity = 'BUILD_METADATA_ONLY'
        if relative.startswith('torch_xla/'):
            with zipfile.ZipFile(xla_path) as z:
                if z.read(relative) != raw:
                    raise ValueError('WHEEL_TAG_SOURCE_MISMATCH:' + relative)
            parity = 'WHEEL_BYTES_IDENTICAL'
        receipts.append({'path': relative, 'url': url, 'sha256': got, 'comparison': parity})
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--from-lock', action='store_true',
                      help='Download only the exact files in the existing lock.')
    parser.add_argument('--requirements-out', type=Path,
                        help='Write the verified closed set as pip --require-hashes input.')
    args = parser.parse_args()
    if args.verify_only or args.from_lock:
        lock = json.loads(args.lock.read_text())
        if lock['marker_environment'] != ENV:
            raise ValueError('MARKER_ENVIRONMENT')
        records = lock['wheels']
        validate_closure(records)
        for rec in records:
            if args.from_lock:
                # Do not mutate the admitted record with newly read metadata.
                fetch_wheel(dict(rec), args.cache)
            actual = verify_wheel(args.cache / rec['filename'], rec)
            for key in ('name', 'version', 'requires_dist', 'metadata_sha256', 'zip_members'):
                if actual[key] != rec[key]:
                    raise ValueError('LOCK_METADATA:' + rec['name'])
        validate_closure(records)
        sources = verify_sources(args.cache, args.cache / next(r['filename'] for r in records if r['name'] == 'torch-xla'))
        if sources != lock['xla_sources']:
            raise ValueError('LOCK_SOURCES')
    else:
        records = Resolver().resolve()
        for rec in records:
            print('VERIFY', rec['name'], rec['version'], rec['size'], flush=True)
            fetch_wheel(rec, args.cache)
        validate_closure(records)
        sources = verify_sources(args.cache, args.cache / next(r['filename'] for r in records if r['name'] == 'torch-xla'))
        lock = {'schema': 1, 'status': 'ARTIFACT_VERIFIED_RUNTIME_NOT_RUN',
                'resolved_at_utc': datetime.now(timezone.utc).isoformat(),
                'target': 'CPython 3.12.14 / Linux x86_64 / glibc >= 2.31',
                'marker_environment': ENV, 'xla_extras': ['tpu', 'pallas'],
                'wheel_count': len(records), 'total_bytes': sum(r['size'] for r in records),
                'wheels': records, 'xla_source_revision': SOURCE_REVISION, 'xla_sources': sources}
        args.lock.write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
    if args.requirements_out:
        args.requirements_out.write_text(''.join(
            f"{r['name']} @ {r['url']} --hash=sha256:{r['sha256']}\n" for r in records))
    print(json.dumps({'status': 'ARTIFACT_VERIFIED_RUNTIME_NOT_RUN', 'wheel_count': len(records),
                      'total_bytes': sum(r['size'] for r in records)}, sort_keys=True))


if __name__ == '__main__':
    main()
