"""Separate compiler-mode admission and launch seal. Stdlib only."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import time

HERE = Path(__file__).resolve().parent
HELPERS = ('compiler_contract.py', 'compiler_exec.py', 'compiler_verify.py',
           'dump_observation.py', 'dump_inventory.py', 'analyzer.py')

DUMP_FLAGS = ['--xla_dump_hlo_as_text=true', '--xla_dump_hlo_as_proto=true',
              '--xla_dump_module_metadata=true', '--xla_dump_hlo_pass_re=pipeline-start|pipeline-end',
              '--xla_dump_include_timestamp=true', '--xla_dump_max_hlo_modules=32',
              '--xla_dump_compress_protos=false', '--xla_dump_hlo_snapshots=false',
              '--xla_dump_hlo_unoptimized_snapshots=false', '--xla_dump_full_hlo_config=false',
              '--xla_dump_large_constants=false', '--xla_dump_hlo_as_dot=false',
              '--xla_dump_hlo_as_html=false', '--xla_dump_hlo_as_url=false']

def require(test, label):
    if not test:
        raise ValueError(label)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def unique_pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'DUPLICATE_COMPILER_JSON_KEY')
        result[key] = value
    return result

def read(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_pairs)

def finite(value):
    return type(value) in (int, float) and math.isfinite(value)

def sources():
    return {name: {'sha256': sha(HERE / name), 'bytes': (HERE / name).stat().st_size}
            for name in HELPERS + ('coordinator.py', 'state_child_owner.py')}

def verify_manifest(path, expected_sha, base=None):
    require(sha(path) == expected_sha, 'COMPILER_FULL_MANIFEST_HASH')
    manifest = read(path); base = HERE if base is None else Path(base)
    require(manifest['compiler_mode'] == 'R6_EXPLICIT_COMPILER_CONTENT_DIAGNOSTIC_V1',
            'COMPILER_EXPLICIT_MANIFEST_MODE')
    for name, entry in manifest['sources'].items():
        part = Path(name)
        require(not part.is_absolute() and '..' not in part.parts and name == part.as_posix(),
                'COMPILER_MANIFEST_PATH_SCOPE')
        actual = base / part
        require(actual.resolve() == actual.absolute() and not actual.is_symlink() and
                actual.is_file() and actual.stat().st_size == entry['bytes'] and sha(actual) == entry['sha256'],
                'COMPILER_FULL_SOURCE_AUDIT')
    return manifest['sources']

def admit(path, expected_sha):
    require(sha(path) == expected_sha, 'COMPILER_POLICY_HASH')
    policy = read(path)
    require(policy['kind'] == 'R6_EXPLICIT_COMPILER_POLICY_V1', 'COMPILER_POLICY_KIND')
    require(policy['sources'] == sources(), 'COMPILER_POLICY_EXACT_SOURCE')
    cfg = policy['limits']
    require(set(cfg) == {'entries', 'file_bytes', 'total_bytes'}, 'COMPILER_LIMIT_SCHEMA')
    caps = {'entries': 1024, 'file_bytes': 16777216, 'total_bytes': 134217728}
    require(all(type(cfg[k]) is int and 0 < cfg[k] <= caps[k] for k in caps), 'COMPILER_LIMIT_SCOPE')
    require(cfg['file_bytes'] >= policy['scientific_record_budget_bytes'] >= 8388608,
            'COMPILER_SCIENCE_RECORD_BUDGET')
    require(finite(policy['poll_seconds']) and .001 <= policy['poll_seconds'] <= .25,
            'COMPILER_POLL_SCOPE')
    require(finite(policy['shutdown_grace']) and 0 <= policy['shutdown_grace'] <= 1,
            'COMPILER_GRACE_SCOPE')
    require(type(policy['minimum_free_bytes']) is int and policy['minimum_free_bytes'] >= 268435456,
            'COMPILER_SPACE_SCOPE')
    require(policy['hard_aggregate_quota'] == 'UNAVAILABLE' and
            policy['polling_overshoot'] == 'POSSIBLE_UNBOUNDED_BY_WRITER_RATE', 'COMPILER_LIMIT_TRUTH')
    return policy

def seal(policy, policy_sha, root, parent, argv, child_token, deadline, native_expected_sha):
    root = Path(root).absolute()
    require(root.resolve() == root and not root.exists(), 'FRESH_COMPILER_SIBLING_ROOT')
    require(not any(c.isspace() for c in str(root)), 'COMPILER_FLAG_PATH_WHITESPACE')
    root.mkdir(mode=0o700)
    dump = root / 'raw-private'
    dump.mkdir(mode=0o700)
    free = shutil.disk_usage(root).free
    require(free >= policy['minimum_free_bytes'], 'COMPILER_AVAILABLE_SPACE')
    require(os.getpid() == parent['pid'] == parent['pgid'] == os.getpgrp(), 'COMPILER_DIRECT_PARENT_LEADER')
    require(finite(deadline) and time.time() < deadline <= parent['deadline_epoch'], 'COMPILER_ORIGINAL_DEADLINE')
    require(re.fullmatch('[0-9a-f]{32}', child_token), 'COMPILER_CHILD_TOKEN')
    s = os.lstat(dump)
    require(stat.S_ISDIR(s.st_mode), 'COMPILER_DUMP_DIRECTORY')
    return {'kind': 'R6_COMPILER_PRELAUNCH_V1', 'policy_sha256': policy_sha, 'sources': sources(),
            'parent_pid': parent['pid'], 'parent_pgid': parent['pgid'],
            'parent_process_token': parent['process_token'], 'process_token': child_token,
            'deadline_epoch': deadline, 'parent_deadline_epoch': parent['deadline_epoch'],
            'argv': list(argv), 'target_source_sha256': sha(argv[2]),
            'native_expected_sha256': native_expected_sha,
            'native_manifest_sources': parent['compiler_source_pre'],
            'manifest_sha256': parent['manifest_sha256'], 'oracle_sha256': parent['oracle_sha256'],
            'source_admission_sha256': parent['source_admission_sha256'],
            'dump_directory': str(dump), 'dump_identity': [s.st_dev, s.st_ino],
            'evidence_directory': str(root), 'limits': policy['limits'],
            'poll_seconds': policy['poll_seconds'], 'shutdown_grace': policy['shutdown_grace'],
            'space_admission': {'free_bytes': free, 'minimum_free_bytes': policy['minimum_free_bytes'],
                                'status': 'PASS', 'continuous_capacity_guarantee': False},
            'hard_aggregate_quota': 'UNAVAILABLE', 'polling_overshoot': 'POSSIBLE_UNBOUNDED_BY_WRITER_RATE'}
