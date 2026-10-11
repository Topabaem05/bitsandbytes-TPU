"""Lossless exact evidence codec. No numerical normalization."""
import gzip
import hashlib
from pathlib import Path
import re
import zlib

MiB = 1024 * 1024
PROFILE = {'kind': 'NESTED_EXACT_LOSSLESS_GRAPH_ARCHIVE_V1',
           'cpu_raw_bytes': 24 * MiB, 'graph_gzip_bytes': 56 * MiB,
           'science_json_gzip_bytes': 12 * MiB, 'other_bytes': 4 * MiB,
           'outer_bytes': 96 * MiB, 'members': 1600,
           'object_gzip_bytes': 8 * MiB, 'graph_raw_bytes': 32 * MiB,
           'json_raw_bytes': 4 * MiB, 'total_graph_raw_bytes': 4 * 1024 * MiB,
           'total_json_raw_bytes': 64 * MiB}

def require(value, message):
    if not value: raise ValueError(message)

def digest(raw): return hashlib.sha256(raw).hexdigest()

def encode(raw, kind):
    require(type(raw) is bytes and kind in ('graph', 'json'), 'OBJECT_TYPE')
    require(0 < len(raw) <= PROFILE['graph_raw_bytes' if kind == 'graph' else 'json_raw_bytes'], 'OBJECT_RAW_SIZE')
    compressed = gzip.compress(raw, mtime=0)
    require(len(compressed) <= PROFILE['object_gzip_bytes'], 'OBJECT_GZIP_SIZE')
    return compressed, {'codec': 'gzip-single-member', 'kind': kind,
                        'path': 'objects/' + digest(raw) + '.gz',
                        'raw_sha256': digest(raw), 'raw_bytes': len(raw),
                        'gzip_sha256': digest(compressed), 'gzip_bytes': len(compressed)}

def metadata(record):
    require(type(record) is dict and set(record) == {'codec', 'kind', 'path', 'raw_sha256', 'raw_bytes', 'gzip_sha256', 'gzip_bytes'}, 'OBJECT_METADATA_SCHEMA')
    require(record.get('codec') == 'gzip-single-member' and record.get('kind') in ('graph', 'json'), 'OBJECT_CODEC')
    require(re.fullmatch('[0-9a-f]{64}', record.get('raw_sha256', '')) is not None
            and record.get('path') == 'objects/' + record['raw_sha256'] + '.gz', 'OBJECT_ADDRESS')
    maximum = PROFILE['graph_raw_bytes' if record['kind'] == 'graph' else 'json_raw_bytes']
    require(type(record.get('raw_bytes')) is int and 0 < record['raw_bytes'] <= maximum
            and type(record.get('gzip_bytes')) is int and 0 < record['gzip_bytes'] <= PROFILE['object_gzip_bytes'], 'OBJECT_DECLARED_SIZE')
    return maximum

def decode(compressed, record):
    metadata(record)
    require(type(compressed) is bytes and len(compressed) == record['gzip_bytes']
            and digest(compressed) == record.get('gzip_sha256'), 'OBJECT_GZIP_SHA')
    stream = zlib.decompressobj(31)
    raw = stream.decompress(compressed, record['raw_bytes'] + 1)
    require(len(raw) == record['raw_bytes'] and stream.eof and not stream.unused_data
            and not stream.unconsumed_tail and digest(raw) == record['raw_sha256'], 'OBJECT_LOSSLESS_RAW')
    return raw

def inventory(records):
    unique = {}; totals = {'graph': 0, 'json': 0}; raw_totals = {'graph': 0, 'json': 0}
    for record in records:
        metadata(record)
        path = record['path']
        if path in unique:
            require(unique[path] == record, 'OBJECT_ADDRESS_KIND_OR_CONTENT_CONFLICT')
            continue
        unique[path] = record
        kind = record['kind']; require(kind in totals, 'OBJECT_TYPE')
        totals[kind] += record['gzip_bytes']; raw_totals[kind] += record['raw_bytes']
    require(len(unique) <= PROFILE['members'] and totals['graph'] <= PROFILE['graph_gzip_bytes']
            and totals['json'] <= PROFILE['science_json_gzip_bytes']
            and raw_totals['graph'] <= PROFILE['total_graph_raw_bytes']
            and raw_totals['json'] <= PROFILE['total_json_raw_bytes'], 'OBJECT_ARCHIVE_QUOTA')
    return {'unique_objects': len(unique), 'gzip_bytes': totals, 'raw_bytes': raw_totals,
            'scope': 'EXACT_BYTE_DEDUP_ONLY_NO_SEMANTIC_NORMALIZATION'}
