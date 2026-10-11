"""Operational byte transport only. Stdlib; never imports JAX or edits campaign bytes."""
from pathlib import Path
from hashlib import sha256
import json
import os
import re
import shutil
import stat

FORMAT = 'port2tpu.byte-transport.v1'
PART_BYTES = 2 * 1024 * 1024
MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024
MAX_PARTS = MAX_FILE_BYTES // PART_BYTES


def regular(path):
    path = Path(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('regular non-symlink file required: ' + str(path))
    return path


def digest(path):
    h = sha256()
    with regular(path).open('rb') as stream:
        for block in iter(lambda: stream.read(PART_BYTES), b''):
            h.update(block)
    return h.hexdigest()


def require_digest(value):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError('explicit SHA256 required')


def validate_manifest(data, expected_sha256, expected_bytes):
    require_digest(expected_sha256)
    if type(expected_bytes) is not int or not 1 <= expected_bytes <= MAX_FILE_BYTES:
        raise ValueError('explicit supported source size required')
    if set(data) != {'format', 'source_name', 'source_sha256', 'source_bytes', 'part_bytes', 'parts'}:
        raise ValueError('unexpected transport manifest fields')
    if data['format'] != FORMAT or data['source_sha256'] != expected_sha256 or type(data['source_bytes']) is not int or data['source_bytes'] != expected_bytes:
        raise ValueError('transport manifest differs from independently expected source')
    if not isinstance(data['source_name'], str) or re.fullmatch(r'[A-Za-z0-9_.-]+', data['source_name']) is None:
        raise ValueError('plain source filename required')
    if data['part_bytes'] != PART_BYTES or not isinstance(data['parts'], list):
        raise ValueError('fixed bounded parts required')
    expected_count = (expected_bytes + PART_BYTES - 1) // PART_BYTES
    if len(data['parts']) != expected_count or not 1 <= expected_count <= MAX_PARTS:
        raise ValueError('complete part coverage required')
    for i, part in enumerate(data['parts']):
        if set(part) != {'name', 'bytes', 'sha256'} or part['name'] != f'part-{i+1:06d}.bin':
            raise ValueError('unique complete ordered part names required')
        require_digest(part['sha256'])
        expected_size = PART_BYTES if i+1 < expected_count else expected_bytes - i * PART_BYTES
        if type(part['bytes']) is not int or part['bytes'] != expected_size:
            raise ValueError('part size differs from complete source partition')
    return data


def split(source, out, *, expected_sha256, expected_bytes):
    source, out = regular(source), Path(out)
    require_digest(expected_sha256)
    if type(expected_bytes) is not int or not 1 <= expected_bytes <= MAX_FILE_BYTES or source.stat().st_size != expected_bytes:
        raise ValueError('source size differs or exceeds transport ceiling')
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    out.mkdir()
    h, parts = sha256(), []
    try:
        with source.open('rb') as stream:
            for i, block in enumerate(iter(lambda: stream.read(PART_BYTES), b''), 1):
                if i > MAX_PARTS:
                    raise ValueError('too many transport parts')
                name = f'part-{i:06d}.bin'
                with (out / name).open('xb') as target:
                    target.write(block)
                h.update(block)
                parts.append({'name':name,'bytes':len(block),'sha256':sha256(block).hexdigest()})
        if h.hexdigest() != expected_sha256 or sum(p['bytes'] for p in parts) != expected_bytes:
            raise ValueError('source SHA256/size differs during split')
        manifest = validate_manifest({'format':FORMAT,'source_name':source.name,'source_sha256':h.hexdigest(),'source_bytes':expected_bytes,'part_bytes':PART_BYTES,'parts':parts}, expected_sha256, expected_bytes)
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        return manifest
    except BaseException:
        shutil.rmtree(out)  # Only this freshly created operation-owned directory.
        raise


def assemble(manifest_path, parts_dir, destination, *, expected_manifest_sha256, expected_sha256, expected_bytes):
    manifest_path, parts_dir, destination = regular(manifest_path), Path(parts_dir), Path(destination)
    require_digest(expected_manifest_sha256)
    if digest(manifest_path) != expected_manifest_sha256:
        raise ValueError('transport manifest hash mismatch')
    manifest = validate_manifest(json.loads(manifest_path.read_text()), expected_sha256, expected_bytes)
    if parts_dir.is_symlink() or not parts_dir.is_dir() or not destination.parent.is_dir() or destination.parent.is_symlink():
        raise ValueError('real existing operation directories required')
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    # Reject unexpected numbered members and non-files, rather than silently dropping coverage.
    actual = {p.name for p in parts_dir.iterdir() if p.name.startswith('part-')}
    if actual != {p['name'] for p in manifest['parts']}:
        raise ValueError('part directory coverage differs')
    temporary = destination.with_name(destination.name + '.assembling')
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    created = False
    try:
        h, total = sha256(), 0
        with temporary.open('xb') as output:
            created = True
            for descriptor in manifest['parts']:
                path = regular(parts_dir / descriptor['name'])
                if path.stat().st_size != descriptor['bytes'] or digest(path) != descriptor['sha256']:
                    raise ValueError('part hash/size mismatch: ' + descriptor['name'])
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(PART_BYTES), b''):
                        output.write(block)
                        h.update(block)
                        total += len(block)
            output.flush()
            os.fsync(output.fileno())
        if h.hexdigest() != expected_sha256 or total != expected_bytes:
            raise ValueError('assembled source hash/size mismatch')
        os.link(temporary, destination)  # Atomic, refuses existing destination even on a race.
        return {'status':'PASS','sha256':h.hexdigest(),'bytes':total,'parts':len(manifest['parts']),'manifest_sha256':expected_manifest_sha256,'destination':str(destination)}
    finally:
        if created:
            temporary.unlink(missing_ok=True)
