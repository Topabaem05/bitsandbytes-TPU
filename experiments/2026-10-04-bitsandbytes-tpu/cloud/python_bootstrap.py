"""One pinned public CPython download/extraction into a fresh owned subtree."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import stat
import tarfile
import urllib.request

URL = 'https://github.com/astral-sh/python-build-standalone/releases/download/20260929/cpython-3.12.14%2B20260929-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz'
SHA = 'ef605200f8174e87ecfc308e52a88127543f85dd5c940dc5e92cab244b98a003'
SIZE = 34277962
MAX_BYTES = 128 * 1024 * 1024
PYTHON_SHA = '5ebcd7993f0e40cb4b16e15d2c4c3c343ecad419966299b8e8f6f9e6a36bf779'


def digest(path): return sha256(path.read_bytes()).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def inspect(archive, expected_sha):
    if digest(archive) != expected_sha: raise ValueError('PYTHON_ARCHIVE_HASH_MISMATCH')
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        names = {member.name for member in members}
        if len(members) > 6000 or len(members) != len(names) or sum(member.size for member in members) > MAX_BYTES:
            raise ValueError('PYTHON_ARCHIVE_MEMBER_OR_SIZE_LIMIT')
        links = {}
        rows = []
        for member in members:
            name = member.name
            if not name or '\\' in name or '\x00' in name or PurePosixPath(name).is_absolute() or any(p in ('', '.', '..') for p in name.split('/')) or PurePosixPath(name).parts[0] != 'python':
                raise ValueError('UNSAFE_PYTHON_ARCHIVE_PATH')
            if not (member.isfile() or member.isdir() or member.issym()) or member.mode & 0o7000:
                raise ValueError('UNSAFE_PYTHON_ARCHIVE_TYPE_OR_MODE')
            if member.issym():
                if not member.linkname or PurePosixPath(member.linkname).is_absolute() or '\\' in member.linkname:
                    raise ValueError('UNSAFE_PYTHON_ARCHIVE_LINK')
                target = posixpath.normpath(posixpath.join(posixpath.dirname(name), member.linkname))
                if not target.startswith('python/') or target not in names:
                    raise ValueError('UNSAFE_PYTHON_ARCHIVE_LINK')
                links[name] = target
            rows.append({'path': name, 'kind': 'file' if member.isfile() else 'directory' if member.isdir() else 'symlink',
                         'bytes': member.size, 'mode': oct(member.mode),
                         **({'sha256': sha256(tar.extractfile(member).read()).hexdigest()} if member.isfile() else {}),
                         **({'link': member.linkname, 'resolved_archive_target': links[name]} if member.issym() else {})})
        for name in names:
            if any(str(parent) in links for parent in PurePosixPath(name).parents):
                raise ValueError('SYMLINK_ARCHIVE_PARENT')
        for name in links:
            target = name; seen = set()
            while target in links:
                if target in seen: raise ValueError('CYCLIC_PYTHON_ARCHIVE_LINK')
                seen.add(target); target = links[target]
        folded = {}
        collisions = []
        for name in sorted(names):
            if name.casefold() in folded: collisions.append([folded[name.casefold()], name])
            else: folded[name.casefold()] = name
        return {'members': len(rows), 'total_file_bytes': sum(member.size for member in members),
                'casefold_collisions': collisions, 'files': rows}


def extract(archive, target, expected_sha):
    inventory = inspect(archive, expected_sha)
    if target.exists() or target.is_symlink(): raise ValueError('FRESH_PYTHON_TARGET_REQUIRED')
    target.mkdir(mode=0o700)
    if inventory['casefold_collisions']:
        probe = target / '.python-case-probe-a'
        probe.write_bytes(b'owned filesystem case check')
        try:
            if (target / '.python-case-probe-A').exists(): raise ValueError('CASE_SENSITIVE_PYTHON_TARGET_REQUIRED')
        finally: probe.unlink()
    # Create directories, then regular files, then confined links; no extractall.
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        directories = {member.name for member in members if member.isdir()}
        directories.update(str(parent) for member in members for parent in PurePosixPath(member.name).parents if str(parent) != '.')
        for name in sorted(directories, key=lambda name: (len(PurePosixPath(name).parts), name)):
            (target / name).mkdir(mode=0o700)
        for member in members:
            if member.isfile():
                path = target / member.name
                if path.parent.is_symlink() or not path.parent.is_dir(): raise ValueError('MISSING_PYTHON_ARCHIVE_PARENT')
                with tar.extractfile(member) as source, path.open('xb') as dest:
                    while chunk := source.read(1024 * 1024): dest.write(chunk)
                path.chmod(0o555 if member.mode & 0o111 else 0o444)
        for member in members:
            if member.issym(): (target / member.name).symlink_to(member.linkname)
        for name in directories: (target / name).chmod(0o555)
    for row in inventory['files']:
        path = target / row['path']
        if not path.resolve().is_relative_to(target.resolve()): raise ValueError('RELOCATED_PYTHON_PATH_ESCAPE')
        if row['kind'] == 'file' and digest(path) != row['sha256']: raise ValueError('RELOCATED_PYTHON_FILE_HASH_MISMATCH')
    return inventory


def main(output):
    if output.exists() or output.is_symlink(): raise ValueError('FRESH_PYTHON_BOOTSTRAP_REQUIRED')
    output.mkdir(mode=0o711); output.chmod(0o711)
    record = {'status': 'RUNNING', 'publisher': 'astral-sh/python-build-standalone',
              'url': URL, 'expected_sha256': SHA, 'expected_bytes': SIZE, 'attempts': 1,
              'interpreter_execution': 'NOT_RUN_BY_BOOTSTRAP', 'global_paths_modified': False}
    try:
        archive = output / 'python.tar.gz'
        count = 0
        with urllib.request.urlopen(urllib.request.Request(URL, headers={'User-Agent': 'Port2TPU-pinned-runtime-bootstrap'}), timeout=15) as source, archive.open('xb') as dest:
            while chunk := source.read(1024 * 1024):
                count += len(chunk)
                if count > SIZE: raise ValueError('PYTHON_DOWNLOAD_SIZE_EXCEEDED')
                dest.write(chunk)
        if count != SIZE: raise ValueError('PYTHON_DOWNLOAD_SIZE_MISMATCH')
        target = output / 'interpreter'
        inventory = extract(archive, target, SHA)
        python = target / 'python/bin/python3.12'
        if python.is_symlink() or digest(python) != PYTHON_SHA: raise ValueError('PINNED_PYTHON_BINARY_MISMATCH')
        target.chmod(0o711)
        write(output / 'archive-inventory.json', inventory)
        record.update(status='PINNED_ARCHIVE_EXTRACTED_NOT_EXECUTED', archive_sha256=digest(archive),
                      archive_bytes=count, python=str(python), python_sha256=digest(python),
                      relocation='ALL_CONFINED_PATHS_AND_BYTES_VERIFIED_RUNTIME_NOT_YET_TESTED',
                      inventory_sha256=digest(output / 'archive-inventory.json'))
    except BaseException as error:
        record.update(status='BLOCKED', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally: write(output / 'bootstrap-result.json', record)
    print(json.dumps(record, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); main(args.output)
