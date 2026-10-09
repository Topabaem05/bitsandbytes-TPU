"""Post-closure compiler supplement. Native science verifier remains unchanged."""
import argparse
import json
import os
import re
from pathlib import Path
import tempfile
import time

import compiler_contract as C
import dump_inventory as D
import analyzer as A
import compiler_flags as F

def evidence_files(root):
    root = Path(root)
    return {name: {'sha256': C.sha(root / name), 'bytes': (root / name).stat().st_size}
            for name in ('expected.json', 'exec.json', 'monitor.json')}

def require_closure(outer):
    cleanup = outer.get('cleanup', {})
    C.require(cleanup.get('status') == 'CLEANUP_VERIFIED' and cleanup.get('leader_reaped') is True and
              cleanup.get('group_absence') == 'OBSERVED_NO_SUCH_GROUP' and cleanup.get('errors') == [],
              'COMPILER_EXTERNAL_GROUP_CLOSURE')

def postclosure(root, outer):
    """Run by the existing external owner, only after its registered-group cleanup."""
    root = Path(root)
    require_closure(outer)
    expected = C.read(root / 'expected.json')
    C.require(outer['pid'] == outer['pgid'] == expected['parent_pid'], 'POSTCLOSURE_OWNED_LEADER')
    cfg = expected['limits']
    snapshot = D.capture(root / 'raw-private', max_entries=cfg['entries'],
                         max_file_bytes=cfg['file_bytes'], max_total_bytes=cfg['total_bytes'])
    record = {'kind': 'R6_COMPILER_POSTCLOSURE_V1', 'outer_pid': outer['pid'],
              'outer_pgid': outer['pgid'], 'cleanup': outer['cleanup'],
              'captured_epoch': time.time(), 'captured_monotonic': time.monotonic(),
              'evidence_files': evidence_files(root), 'snapshot': snapshot,
              'selected_executable_link': 'UNKNOWN', 'allocator_peak': 'UNKNOWN'}
    # Caller durably saves and independently seals this receipt with recovered archive bindings.
    return record

def capture_verified(root, expected_sha, post_sha, outer, parent, native_expected, receipt):
    root = Path(root)
    require_closure(outer)
    C.require(C.sha(root / 'expected.json') == expected_sha, 'COMPILER_INDEPENDENT_EXPECTED_HASH')
    C.require(C.sha(root / 'postclosure.json') == post_sha, 'COMPILER_INDEPENDENT_POSTCLOSURE_HASH')
    expected = C.read(root / 'expected.json'); monitor = C.read(root / 'monitor.json')
    prelude = C.read(root / 'exec.json'); post = C.read(root / 'postclosure.json')
    C.require(prelude.get('kind') == 'R6_SAME_PID_EXEC_PRELUDE' and
              monitor.get('kind') == 'R6_DIRECT_CHILD_DUMP_MONITOR', 'COMPILER_EXEC_MONITOR_KINDS')
    C.require(expected['kind'] == 'R6_COMPILER_PRELAUNCH_V1' and expected['sources'] == C.sources(),
              'COMPILER_EXPECTED_EXACT_SOURCE')
    C.require(post['kind'] == 'R6_COMPILER_POSTCLOSURE_V1' and
              post['evidence_files'] == evidence_files(root), 'COMPILER_CLOSED_RECORD_INVENTORY')
    C.require(post['outer_pid'] == post['outer_pgid'] == outer['pid'] == outer['pgid'] ==
              parent['pid'] == parent['pgid'] == expected['parent_pid'] == expected['parent_pgid'],
              'COMPILER_ACTUAL_OUTER_PARENT')
    C.require(post['cleanup'] == outer['cleanup'], 'COMPILER_POSTCLOSURE_OWNER_BINDING')
    C.require(expected['native_expected_sha256'] == parent['expected_sha256'] and
              expected_sha == parent['compiler_expected_sha256'], 'COMPILER_NATIVE_EXPECTED_BINDING')
    C.require(expected['native_manifest_sources'] == parent['compiler_source_pre'] == parent['compiler_source_post'] ==
              C.verify_manifest(C.HERE / 'manifest.json', parent['manifest_sha256']), 'COMPILER_FULL_PRE_POST_SOURCE')
    for key in ('manifest_sha256', 'oracle_sha256', 'source_admission_sha256'):
        C.require(expected[key] == parent[key], 'COMPILER_SOURCE_ORACLE_SEALS')
    def outer_flag(name):
        argv = outer['argv']; flag = '--' + name
        C.require(argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv), 'COMPILER_OUTER_ARGV_FLAG')
        return argv[argv.index(flag) + 1]
    C.require(outer_flag('compiler-dumps') == 'request' and
              outer_flag('compiler-policy-sha256') == expected['policy_sha256'] == parent['compiler_policy_sha256'],
              'COMPILER_OUTER_POLICY_ADMISSION')
    launched_output = Path(outer_flag('output'))
    C.require(launched_output.is_absolute() and expected['evidence_directory'] ==
              str(launched_output.with_name(launched_output.name + '.compiler-private')) and
              expected['dump_directory'] == str(Path(expected['evidence_directory']) / 'raw-private') and
              parent['compiler_dump_request']['sibling_evidence'] == expected['evidence_directory'],
              'COMPILER_SIBLING_LAUNCH_BINDING')
    C.require(expected['parent_process_token'] == parent['process_token'] and
              expected['parent_deadline_epoch'] == parent['deadline_epoch'], 'COMPILER_PARENT_TOKEN_DEADLINE')
    child = parent['children'][0]
    C.require(len(parent['children']) == 1 and monitor['child_terminal'] ==
              {k: v for k, v in child.items() if k not in ('receipt', 'receipt_sha256')},
              'COMPILER_EXACT_NATIVE_CHILD_DESCRIPTOR')
    identity = {'pid': child['pid'], 'pgid': parent['pgid'], 'parent_pid': parent['pid'],
                'process_token': child['process_token'], 'parent_process_token': parent['process_token'],
                'deadline_epoch': child['deadline_epoch']}
    C.require(child['pid'] == child['launched_pid'] != parent['pid'] and
              child['pgid'] == parent['pgid'] and all(prelude.get(k) == receipt.get(k) == v
                                                    for k, v in identity.items()), 'COMPILER_SAME_PID_DIRECT_EXEC_CHAIN')
    C.require(native_expected['launch'] == {'parent_pid': parent['pid'], 'parent_pgid': parent['pgid'],
              'parent_process_token': parent['process_token'], 'process_token': child['process_token'],
              'deadline_epoch': child['deadline_epoch']}, 'COMPILER_NATIVE_LAUNCH_BINDING')
    argv = child['argv']; split = argv.index('--')
    C.require(split == 7 and argv[1] == '-B' and Path(argv[2]).name == 'compiler_exec.py' and
              argv[3] == '--contract' and argv[4] == str(Path(expected['evidence_directory']) / 'expected.json') and
              argv[5:7] == ['--contract-sha256', expected_sha] and argv[split + 1:] == expected['argv'] ==
              prelude['argv'], 'COMPILER_EXACT_WRAPPER_AND_EXEC_ARGV')
    C.require(prelude['wrapper_source_sha256'] == C.sha(C.HERE / 'compiler_exec.py') and
              prelude['target_source_sha256'] == expected['target_source_sha256'], 'COMPILER_EXEC_SOURCE_BINDING')
    C.require(expected['sources'] == monitor['source_pre'] == monitor['source_post'] and
              expected_sha == prelude['prelaunch_sha256'] == monitor['prelaunch_sha256'],
              'COMPILER_PRE_POST_SOURCE_SEALS')
    C.require(monitor['pid'] == monitor['pgid'] == parent['pid'] and
              monitor['child_pid'] == child['pid'], 'COMPILER_MONITOR_IS_COORDINATOR')
    C.require(expected['deadline_epoch'] == monitor['deadline_epoch'] and expected['limits'] == monitor['limits'] and
              prelude['file_limit_soft_hard'] == [expected['limits']['file_bytes']] * 2,
              'COMPILER_FILE_LIMIT_READBACK')
    C.require(prelude['xla_flags'] == expected['xla_flags'] == parent['compiler_dump_request']['flags'] and
              parent['compiler_dump_request']['mode'] == 'request' and
              parent['compiler_dump_request']['policy_sha256'] == expected['policy_sha256'],
              'COMPILER_EXACT_FLAG_REQUEST')
    C.require(expected['space_admission']['status'] == 'PASS' and
              expected['space_admission']['free_bytes'] >= expected['space_admission']['minimum_free_bytes'] and
              expected['space_admission']['continuous_capacity_guarantee'] is False, 'COMPILER_SPACE_ADMISSION')
    C.require(expected['hard_aggregate_quota'] == monitor['hard_aggregate_quota'] == 'UNAVAILABLE' and
              expected['polling_overshoot'] == monitor['polling_overshoot'] ==
              'POSSIBLE_UNBOUNDED_BY_WRITER_RATE', 'COMPILER_AGGREGATE_LIMIT_TRUTH')
    C.require(child['reaped'] is True and child['child_absent'] is True and child['cleanup_errors'] == [] and
              child['timeout'] is False and child['exit_code'] == 0 and
              monitor['status'] == 'COMPLETE_BOUNDED_CAPTURE' and not monitor.get('error') and
              monitor['failure_reason'] is None and monitor['overflow_observation'] is None,
              'COMPILER_UNTRUNCATED_CHILD_TERMINAL')
    C.require(type(monitor['observations']) is int and monitor['observations'] > 0 and
              monitor['last_observation']['status'] == 'WITHIN_OBSERVED_LIMITS', 'COMPILER_POSITIVE_OBSERVATION')
    for item, keys in ((child, ('started_epoch', 'finished_epoch', 'started_monotonic', 'finished_monotonic', 'deadline_epoch')),
                       (prelude, ('started_epoch', 'started_monotonic')),
                       (monitor, ('started_epoch', 'finished_epoch', 'started_monotonic', 'finished_monotonic',
                                  'last_observed_epoch', 'last_observed_monotonic')),
                       (post, ('captured_epoch', 'captured_monotonic'))):
        C.require(all(C.finite(item.get(k)) for k in keys), 'COMPILER_FINITE_INTERVALS')
    C.require(monitor['started_epoch'] <= child['started_epoch'] <= prelude['started_epoch'] <=
              monitor['last_observed_epoch'] <= child['finished_epoch'] <=
              monitor['finished_epoch'] <= post['captured_epoch'] and
              child['finished_epoch'] <= child['deadline_epoch'] == expected['deadline_epoch'] <=
              expected['parent_deadline_epoch'] and
              monitor['started_monotonic'] <= child['started_monotonic'] <= prelude['started_monotonic'] <=
              monitor['last_observed_monotonic'] <= child['finished_monotonic'] <=
              monitor['finished_monotonic'] <= post['captured_monotonic'], 'COMPILER_ORIGINAL_ORDER_DEADLINE')
    cfg = expected['limits']
    actual = D.capture(root / 'raw-private', max_entries=cfg['entries'], max_file_bytes=cfg['file_bytes'],
                       max_total_bytes=cfg['total_bytes'])
    C.require(actual == post['snapshot'], 'COMPILER_POSTCLOSURE_DUMP_CONTENT')
    C.require(actual == monitor['terminal_snapshot'], 'COMPILER_LATE_DUMP_CHANGE')
    observation = monitor['last_observation']
    C.require(set(observation) == {'entries', 'total_bytes', 'largest_file_bytes', 'status'} and
              all(type(observation[k]) is int and 0 <= observation[k] <= cfg[limit] for k, limit in
                  [('entries','entries'),('total_bytes','total_bytes'),('largest_file_bytes','file_bytes')]) and
              observation['entries'] == actual['entries'] and observation['total_bytes'] == actual['total_bytes'] and
              observation['largest_file_bytes'] == max((f['bytes'] for f in actual['files'].values()), default=0),
              'COMPILER_RECOMPUTED_OBSERVATION')
    return {'status': 'OWNED_BOUNDED_DUMP_CAPTURE_ONLY', 'raw_files': len(actual['files']),
            'raw_bytes': actual['total_bytes'], 'compiler_content_chain': 'NOT_EVALUATED',
            'selected_executable_link': 'UNKNOWN', 'backend_alias_dataflow': 'UNKNOWN',
            'physical_memory': 'UNKNOWN', 'allocator_peak': 'UNKNOWN', 'm6': 'NOT_QUALIFIED'}

def verify(actual, evidence, oracle, oracle_sha, admission_sha, expected_sha, compiler_expected_sha,
           post_sha, outer, selection=None, selection_sha=None, *, mosaic_audit_python, audit_deadline_epoch):
    # No bypass flag or monkeypatch: actual native recovery must pass the unchanged verifier first.
    import verify_run as N
    native = N.verify(actual, oracle, oracle_sha, admission_sha, expected_sha, outer,
                      mosaic_audit_python=mosaic_audit_python, audit_deadline_epoch=audit_deadline_epoch)
    root = Path(actual)
    parent=C.read(Path(actual)/'parent.json');flag=parent['compiler_flag_preflight'];archived=Path(actual).parent/'compiler-flag-preflight.json'
    C.require(C.sha(archived)==flag['sha256']and C.read(archived)==flag['record'],'FLAG_PREFLIGHT_ARCHIVED_HASH')
    F.owned(flag['record'],C.read(Path(actual).parent/'steps/11d-compiler-flags/ownership.json'),C.read(Path(actual).parent/'steps/11d-compiler-flags/result.json'),manifest_sha=parent['manifest_sha256'],policy_sha=parent['compiler_policy_sha256'],dump_directory=flag['record']['dump_directory'])
    supplement = capture_verified(evidence, compiler_expected_sha, post_sha, outer,
                                  C.read(root / 'parent.json'), C.read(root / 'expected.json'),
                                  C.read(root / 'device/receipt.json'))
    if selection is not None:
        import protocol as S
        selected = C.read(selection)
        C.require(selected['case'] in S.spec()['native_case_ids'], 'COMPILER_SELECTED_NATIVE_CASE')
        supplement['compiler_content_chain'] = selected_content(actual, evidence, selection, selection_sha)
    return {'native': native, 'compiler': supplement, 'm6': 'NOT_QUALIFIED'}

def selected_content(actual, evidence, selection, selection_sha):
    """After full native/capture verification, bind the selected body to actual boundary bytes.

    This helper alone is not an actual execution verifier. Synthetic source fixtures use it
    without claiming native science. The production verify() always checks native first.
    """
    C.require(C.sha(selection) == selection_sha, 'COMPILER_INDEPENDENT_SELECTION_HASH')
    selected = C.read(selection)
    C.require(selected['kind'] == 'R6_INDEPENDENT_COMPILER_SELECTION_V1', 'COMPILER_SELECTION_KIND')
    C.require(set(selected['files']) == {'before.hlo.pb', 'after.hlo.pb', 'metadata.textproto'},
              'COMPILER_SELECTION_SCHEMA')
    C.require(isinstance(selected.get('case'), str) and re.fullmatch('[a-z0-9-]{1,64}', selected['case']), 'COMPILER_SELECTION_CASE_SCOPE')
    root = Path(evidence); post = C.read(root / 'postclosure.json')
    raw = root / 'raw-private'; record = C.read(Path(actual) / 'device/raw' / (selected['case'] + '.boundary-record.json'))
    boundary_path = Path(actual) / 'device/raw' / (selected['case'] + '.hlo.pb')
    boundary = boundary_path.read_bytes()
    C.require(0 < len(boundary) <= 16777216, 'COMPILER_BOUNDARY_SIZE')
    values = {'boundary.hlo.pb': boundary}
    for target, entry in selected['files'].items():
        name = entry['path']; path = Path(name)
        C.require(not path.is_absolute() and path.parts and all(p not in ('..', '.') for p in path.parts) and
                  name == path.as_posix() and name in post['snapshot']['files'], 'COMPILER_SELECTED_PATH_SCOPE')
        inventory = post['snapshot']['files'][name]
        C.require(entry['sha256'] == inventory['sha256'] and entry['bytes'] == inventory['bytes'],
                  'COMPILER_SELECTED_CLOSED_INVENTORY')
        source = raw / path
        C.require(source.resolve() == source.absolute() and not source.is_symlink(), 'COMPILER_SELECTED_PATH_TYPE')
        # Open each parent through descriptors; no followed final or ancestor symlinks.
        fd = os.open(raw, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in path.parts[:-1]:
                new = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd); fd = new
            with os.fdopen(os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd), 'rb') as f:
                data = f.read(16777217)
        finally:
            os.close(fd)
        C.require(len(data) == entry['bytes'] and A.sha(data) == entry['sha256'], 'COMPILER_SELECTED_EXACT_BYTES')
        values[target] = data
    expected = {'boundary_proto_sha256': A.sha(boundary), 'payload': record['native_boundary']['payload'],
                'parameters': {k: {'dtype': v['dtype'], 'shape': v['shape']} for k, v in record['parameters'].items()},
                'ordered_operand_shapes': [{'dtype': v['dtype'], 'shape': v['shape']}
                                           for v in record['native_boundary']['operands']],
                'result': {'dtype': record['output']['dtype'], 'shape': record['output']['shape']},
                'stage_dump_basenames': {'before': Path(selected['files']['before.hlo.pb']['path']).name,
                                         'after': Path(selected['files']['after.hlo.pb']['path']).name},
                'compiler_artifact_inventory': {name: {'sha256': A.sha(data), 'bytes': len(data)}
                                                for name, data in values.items()}}
    with tempfile.TemporaryDirectory(prefix='r6-compiler-analysis-') as temporary:
        for name, data in values.items():
            (Path(temporary) / name).write_bytes(data)
        return A.analyze(temporary, expected)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('actual', 'evidence', 'oracle', 'oracle-sha256', 'admission-sha256',
                 'expected-sha256', 'compiler-expected-sha256', 'postclosure-sha256',
                 'outer-ownership', 'outer-ownership-sha256', 'output', 'mosaic-audit-python'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--audit-deadline-epoch', type=float, required=True)
    p.add_argument('--selection'); p.add_argument('--selection-sha256')
    a = p.parse_args()
    try:
        C.require((a.selection is None) == (a.selection_sha256 is None), 'COMPILER_SELECTION_PAIR')
        C.require(C.sha(a.outer_ownership) == a.outer_ownership_sha256, 'COMPILER_OUTER_INDEPENDENT_HASH')
        result = verify(a.actual, a.evidence, a.oracle, a.oracle_sha256, a.admission_sha256,
                        a.expected_sha256, a.compiler_expected_sha256, a.postclosure_sha256,
                        C.read(a.outer_ownership), a.selection, a.selection_sha256,
                        mosaic_audit_python=a.mosaic_audit_python, audit_deadline_epoch=a.audit_deadline_epoch); code = 0
    except Exception as e:
        result = {'status': 'REJECTED', 'error': type(e).__name__ + ':' + str(e), 'm6': 'NOT_QUALIFIED'}; code = 2
    Path(a.output).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    raise SystemExit(code)
