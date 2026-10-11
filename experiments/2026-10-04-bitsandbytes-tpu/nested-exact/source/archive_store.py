"""Lossless logical files with exact-content reuse and bounded, selected recovery."""
import contextlib
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
import archive_objects as O

KIND = 'NESTED_EXACT_LOSSLESS_LOGICAL_FILES_V1'
OUTER_BYTES = 100 * O.MiB
MAX_MEMBERS = 2000
INDEX_BYTES = 4 * O.MiB

def require(value, reason):
    if not value:
        raise ValueError(reason)

def valid_name(name):
    require(type(name) is str and 0 < len(name) <= 512 and '\\' not in name
            and not Path(name).is_absolute() and all(p not in ('', '.', '..') for p in name.split('/'))
            and not name.startswith('objects/') and name != 'logical-files.json', 'LOGICAL_FILE_PATH')
    return name

def physical(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), 'ARCHIVE_ROOT')
    result = {}
    for path in root.rglob('*'):
        require(not path.is_symlink(), 'ARCHIVE_SYMLINK')
        if path.is_file():
            result[path.relative_to(root).as_posix()] = {'bytes': path.stat().st_size, 'sha256': O.digest(path.read_bytes())}
    require(len(result) <= MAX_MEMBERS and sum(v['bytes'] for v in result.values()) <= OUTER_BYTES, 'ARCHIVE_OUTER_QUOTA')
    return result

class Store:
    def __init__(self, root, create=False):
        self.root = Path(root); self.path = self.root / 'logical-files.json'
        if create:
            require(not self.path.exists(), 'LOGICAL_STORE_EXISTS')
            self.index = {'kind': KIND, 'profile': O.PROFILE, 'files': {}}
            self.flush()
        else:
            require(self.path.is_file() and not self.path.is_symlink() and self.path.stat().st_size <= INDEX_BYTES, 'LOGICAL_INDEX_SIZE')
            self.index = json.loads(self.path.read_bytes())
            require(set(self.index) == {'kind', 'profile', 'files'} and self.index['kind'] == KIND
                    and self.index['profile'] == O.PROFILE and type(self.index['files']) is dict, 'LOGICAL_INDEX_SCHEMA')
            for name in self.index['files']:
                valid_name(name)
            O.inventory(list(self.index['files'].values()))
    @property
    def files(self):
        return self.index['files']
    def flush(self):
        encoded = (json.dumps(self.index, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        require(len(encoded) <= INDEX_BYTES, 'LOGICAL_INDEX_SIZE')
        temp = self.path.with_suffix('.tmp'); temp.write_bytes(encoded); temp.replace(self.path)
    def put(self, name, raw, kind):
        valid_name(name); compressed, record = O.encode(raw, kind)
        require(name not in self.files or self.files[name] == record, 'LOGICAL_FILE_REWRITE')
        prospective = dict(self.files, **{name: record}); O.inventory(list(prospective.values()))
        path = self.root / record['path']; path.parent.mkdir(exist_ok=True)
        if path.exists():
            require(not path.is_symlink() and path.read_bytes() == compressed, 'OBJECT_CONTENT_CONFLICT')
        else:
            path.write_bytes(compressed)
        self.files[name] = record; self.flush()
        return record
    def pack(self, path):
        path = Path(path); name = path.relative_to(self.root).as_posix(); valid_name(name)
        require(path.is_file() and not path.is_symlink(), 'LOGICAL_SOURCE_FILE')
        kind = 'json' if path.suffix == '.json' else 'graph'
        raw = path.read_bytes()
        try:
            record = self.put(name, raw, kind)
        except ValueError as error:
            diagnostic = {'status': 'EVIDENCE_LIMIT', 'logical_file': name, 'raw_bytes': len(raw),
                          'raw_sha256': O.digest(raw), 'reason': str(error), 'required_evidence': 'MISSING_NOT_ACCEPTED'}
            failure = self.root / 'evidence-limit.json'
            existing = json.loads(failure.read_text()) if failure.exists() else []
            require(len(existing) < 32, 'EVIDENCE_LIMIT_DIAGNOSTIC_QUOTA')
            existing.append(diagnostic); failure.write_text(json.dumps(existing, sort_keys=True, indent=2)+'\n')
            path.unlink()
            raise ValueError('EVIDENCE_LIMIT:' + str(error)) from error
        path.unlink()
        return record
    def read(self, name):
        valid_name(name); require(name in self.files, 'MISSING_LOGICAL_FILE')
        record = self.files[name]; path = self.root / record['path']
        require(path.is_file() and not path.is_symlink(), 'MISSING_OBJECT')
        require(path.stat().st_size == record['gzip_bytes'], 'OBJECT_FILE_SIZE')
        return O.decode(path.read_bytes(), record)
    def audit(self):
        inventory = physical(self.root)
        declared = {r['path'] for r in self.files.values()}
        require({p for p in inventory if p.startswith('objects/')} == declared, 'EXACT_OBJECT_INVENTORY')
        # Audit each unique body once. Logical names remain separate even if bytes coincide.
        for name in dict((r['path'], n) for n, r in self.files.items()).values():
            self.read(name)
        return {'logical_files': len(self.files), **O.inventory(list(self.files.values())), 'physical_bytes': sum(r['bytes'] for r in inventory.values())}
    @contextlib.contextmanager
    def selected(self, names, destination):
        names = list(names); require(len(names) == len(set(names)), 'SELECTED_DUPLICATE')
        require(sum(self.files[n]['raw_bytes'] for n in names) <= 132 * O.MiB, 'SELECTED_RAW_QUOTA')
        destination = Path(destination); written = []
        try:
            for name in names:
                path = destination / valid_name(name)
                require(not path.exists(), 'RECOVERY_TARGET_EXISTS')
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(self.read(name)); written.append(path)
            yield destination
        finally:
            for path in written:
                path.unlink(missing_ok=True)


def pack_directory(root, directory):
    store = Store(root)
    failure = None
    for path in sorted(Path(directory).rglob('*')):
        if path.is_file():
            try:
                store.pack(path)
            except ValueError as error:
                # Remove all unsealed raw bodies from the affected call after recording each failure.
                failure = failure or error
                path.unlink(missing_ok=True)
    if failure is not None:
        raise failure


def seal(root):
    store = Store(root); report = store.audit(); require(report['physical_bytes'] <= 96 * O.MiB, 'ARCHIVE_INNER_QUOTA')
    return report


def write_tar(root, output):
    """Finite physical envelope. Never package raw expanded evidence or partial ZIPs."""
    root, output = Path(root), Path(output)
    require(not output.exists() and not output.resolve().is_relative_to(root.resolve()), 'ARCHIVE_DESTINATION')
    members = physical(root)
    require(sum(v['bytes'] for v in members.values()) <= 96 * O.MiB, 'ARCHIVE_INNER_QUOTA')
    try:
        with tarfile.open(output, 'w:gz') as archive:
            for name in sorted(members):
                archive.add(root / name, arcname=name, recursive=False)
        require(output.stat().st_size <= OUTER_BYTES, 'ARCHIVE_OUTER_QUOTA')
        return {'bytes': output.stat().st_size, 'sha256': O.digest(output.read_bytes()), 'members': len(members)}
    except BaseException:
        output.unlink(missing_ok=True)
        raise
