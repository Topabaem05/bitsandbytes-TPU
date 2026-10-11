"""Read-only static allocation audit; select HloProto assignments by outer field 3."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def inventory(directory):
    return [{'path': p.relative_to(directory).as_posix(), 'bytes': p.stat().st_size,
             'sha256': sha(p.read_bytes())} for p in sorted(directory.rglob('*.pb'))]


def analyze(parser, old, new, directory):
    before = inventory(directory)
    rows, skipped = [], []
    for item in before:
        raw = (directory / item['path']).read_bytes()
        outer = parser.fields(raw)
        # Never infer the assignment stage from a path or filename.
        if 3 not in outer:
            skipped.append(item)
            continue
        prior = old.audit_bytes(parser, raw)
        result = new.audit_bytes(parser, raw)
        module_raw = new.Auditor(parser).scalar(outer, 1, None, wire=2)
        module_id = None if module_raw is None else new.Auditor(parser).integer(parser.fields(module_raw), 5)
        row = dict(item, compiler_module_id=module_id, outer_field3_present=True,
                   v1_status=prior['status'], v1_reason=prior.get('reason'),
                   v2_status=result['status'], v2_reason=result.get('reason'))
        if result['status'] == 'ALLOCATION_PLAN_BOUNDS_PASS':
            plan = result['compiler_allocation_plan']
            assert module_id == plan['compiler_module_id']
            types = Counter()
            for item_buffer in plan['logical_buffers'].values():
                shape = item_buffer['location']['shape']
                types[shape.get('dtype', shape['kind'])] += 1
            row.update({'logical_buffer_count': len(plan['logical_buffers']),
                        'alias_count': len(plan['buffer_aliases']), 'logical_storage_types': dict(types),
                        'allocation_count': len(plan['allocations']),
                        'allocation_size_sum_bytes': plan['allocation_size_sum_bytes']})
        rows.append(row)
    after = inventory(directory)
    if before != after:
        raise ValueError('READ_ONLY_INPUT_INVENTORY_CHANGED')
    canonical_inventory = json.dumps(before, sort_keys=True, separators=(',', ':')).encode()
    return {'proto_count': len(before), 'proto_bytes': sum(item['bytes'] for item in before),
            'selected_assignment_count': len(rows), 'without_outer_field3_count': len(skipped),
            'selection': 'PARSED_OUTER_FIELD_3_NOT_FILENAME',
            'all_proto_inventory_sha256': sha(canonical_inventory), 'input_inventory_unchanged': True,
            'v1_status_counts': dict(Counter(row['v1_status'] for row in rows)),
            'v2_status_counts': dict(Counter(row['v2_status'] for row in rows)), 'results': rows}


def run(baseline, frozen_auditor, directory, packet_sha, output):
    new = load('allocation_v2', HERE / 'allocation_audit_v2.py')
    if sha(frozen_auditor.read_bytes()) != new.FROZEN_V1_SHA256:
        raise ValueError('FROZEN_V1_AUDITOR_SHA256')
    if not re.fullmatch('[0-9a-f]{64}', packet_sha):
        raise ValueError('CAPTURE_PACKET_SHA256')
    if output.resolve().is_relative_to(directory.resolve()):
        raise ValueError('OUTPUT_MUST_BE_OUTSIDE_READ_ONLY_INPUT')
    old = load('frozen_allocation_v1', frozen_auditor)
    parser = new.load_parser(baseline)
    report = analyze(parser, old, new, directory)
    report.update({'schema_version': 2, 'audit_contract': new.AUDIT_CONTRACT,
        'audit_contract_version': 2, 'analysis_source_sha256': sha((HERE / 'allocation_audit_v2.py').read_bytes()),
        'directory_reader_sha256': sha(Path(__file__).read_bytes()),
        'frozen_v1_analysis_source_sha256': new.FROZEN_V1_SHA256,
        'baseline_analyzer_sha256': new.PARSER_SHA256, 'schema_reference_revision': new.SCHEMA_REVISION,
        'capture_packet_sha256': packet_sha,
        'capture_packet_binding': 'CALLER_SUPPLIED_REQUIRES_INDEPENDENT_ARCHIVE_REVIEW',
        'input_directory': str(directory.resolve()), 'scope': 'STATIC_ALLOCATION_BOUNDS_ONLY',
        'actual_selected_executable_link': 'UNKNOWN', 'allocator_peak': 'UNKNOWN',
        'native_body_internal_memory': 'UNKNOWN', 'numerical_result': 'NOT_EVALUATED', 'm6': 'NOT_QUALIFIED'})
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: report[key] for key in ('proto_count', 'selected_assignment_count',
          'v1_status_counts', 'v2_status_counts', 'input_inventory_unchanged')}))
    return 0 if report['results'] and all(row['v2_status'] == 'ALLOCATION_PLAN_BOUNDS_PASS'
                                         for row in report['results']) else 1


if __name__ == '__main__':
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('--baseline-analyzer', type=Path, required=True)
    args.add_argument('--frozen-auditor', type=Path, required=True)
    args.add_argument('--input-directory', type=Path, required=True)
    args.add_argument('--capture-packet-sha256', required=True)
    args.add_argument('--output', type=Path, required=True)
    options = args.parse_args()
    raise SystemExit(run(options.baseline_analyzer.resolve(), options.frozen_auditor.resolve(),
        options.input_directory.resolve(), options.capture_packet_sha256, options.output))
