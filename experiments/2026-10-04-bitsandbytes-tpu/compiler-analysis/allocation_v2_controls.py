"""Independent version 2 storage fixtures and unchanged version 1 regressions."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ORIGINAL_CONTROLS_SHA256 = '0d135543f1fa6c5be0c0f0b7855503b79928fedf2e2226fbcb4c35de060d4624'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(baseline, original_controls, frozen, output):
    A = load('allocation_v2_under_test', HERE / 'allocation_audit_v2.py')
    F = load('original_fixture_builders', original_controls)
    assert digest(original_controls.parent / 'allocation_audit.py') == A.FROZEN_V1_SHA256
    assert digest(original_controls) == ORIGINAL_CONTROLS_SHA256
    parser = A.load_parser(baseline)
    rows = []
    frozen_before = digest(frozen)
    with tempfile.TemporaryDirectory(prefix='allocation-v2-controls-') as directory:
        temporary = Path(directory)
        # Keep the original 65-control source unchanged. Only the auditor under
        # its historical filename differs in this isolated compatibility copy.
        compatibility = temporary / 'compatibility'
        compatibility.mkdir()
        shutil.copyfile(original_controls, compatibility / 'allocation_controls.py')
        shutil.copyfile(HERE / 'allocation_audit_v2.py', compatibility / 'allocation_audit.py')
        original65 = temporary / 'original65.json'
        child = subprocess.run([sys.executable, '-B', str(compatibility / 'allocation_controls.py'),
            '--baseline-analyzer', str(baseline), '--frozen-fixture', str(frozen),
            '--output', str(original65)], capture_output=True, text=True)
        assert child.returncode == 0, child.stderr
        regression = json.loads(original65.read_text())
        assert regression['controls_count'] == 65 and regression['status'] == 'PASS'
        assert digest(compatibility / 'allocation_controls.py') == digest(original_controls)

        def case(name, raw, expected='ALLOCATION_PLAN_BOUNDS_PASS', reason=None, lower=None):
            result = A.audit_bytes(parser, raw)
            assert result['audit_contract_version'] == 2
            assert result['audit_contract'] == A.AUDIT_CONTRACT
            assert result['status'] == expected, (name, result)
            if reason:
                assert result['reason'] == reason, (name, result)
            if lower is not None:
                assert result['compiler_allocation_plan']['logical_buffers'][10]['location']['shape'][
                    'array_storage_lower_bound_bytes'] == lower
            if expected == 'ALLOCATION_PLAN_BOUNDS_PASS':
                plan = result['compiler_allocation_plan']
                assert plan['allocator_peak'] == plan['native_body_internal_memory'] == 'UNKNOWN'
                assert plan['meaning'] == 'COMPILER_ALLOCATION_PLAN_NOT_RUNTIME_PEAK'
            rows.append({'case': name, 'status': 'PASS', 'observed_status': result['status'],
                         'reason': result.get('reason'), 'fixture_sha256': A.sha(raw),
                         'fixture_bytes': len(raw)})
            return result

        def plain(dtype, width, dims=(3, 7), layout=None, size=None, offset=0, allocation=None):
            import math
            natural = math.prod(dims) * width
            size = natural if size is None else size
            allocation = size + offset if allocation is None else allocation
            return F.fixture(shapes={1: F.array(dtype, dims, layout=layout)},
                logical=[(10, size, 1, (), 0)],
                allocations=[(0, allocation, [(10, offset, size)], 0)])

        # Expected sizes are independently declared, never read from A.WIDTHS.
        for dtype, width, name in ((1, 1, 'PRED'), (4, 4, 'S32'), (5, 8, 'S64'), (8, 4, 'U32')):
            size = 21 * width
            for mode, layout in (('absent', None), ('zero', F.integer(7, 0)),
                                 ('natural', F.integer(7, 8 * width))):
                case(name + '-packing-' + mode, plain(dtype, width, layout=layout), lower=size)
            for bits in (1, width * 8 + 8):
                case(name + '-valid-nonnatural-packing-' + str(bits),
                     plain(dtype, width, layout=F.integer(7, bits)),
                     'UNSUPPORTED_REPRESENTATION', 'UNSUPPORTED_ELEMENT_PACKING')
            case(name + '-negative-packing', plain(dtype, width, layout=F.integer(7, -1)),
                 'INVALID_ALLOCATION_PLAN', 'NONNEGATIVE_SIGNED_INT64')
            case(name + '-minimum-minus-one', plain(dtype, width, size=size - 1),
                 'INVALID_ALLOCATION_PLAN', 'LOGICAL_BUFFER_ARRAY_LOWER_BOUND')
            case(name + '-scalar', plain(dtype, width, dims=()), lower=width)
            case(name + '-scalar-zero-invalid', plain(dtype, width, dims=(), size=0),
                 'INVALID_ALLOCATION_PLAN', 'LOGICAL_BUFFER_ARRAY_LOWER_BOUND')
            case(name + '-zero-dimension', plain(dtype, width, dims=(3, 0, 7)), lower=0)
            case(name + '-slice-exact-allocation-end', plain(dtype, width, offset=17), lower=size)
            case(name + '-slice-exceeds-allocation-end', plain(dtype, width, offset=17,
                 allocation=size + 16), 'INVALID_ALLOCATION_PLAN', 'ALLOCATION_SLICE_BOUNDS')
            case(name + '-insufficient-assigned-slice', F.fixture(
                 shapes={1: F.array(dtype, (3, 7))}, logical=[(10, size, 1, (), 0)],
                 allocations=[(0, size, [(10, 0, size - 1)], 0)]),
                 'INVALID_ALLOCATION_PLAN', 'ASSIGNED_SLICE_LOGICAL_LOWER_BOUND')
            nested = F.tuple_shape(F.array(6, ()), F.tuple_shape(F.array(dtype, (3, 7))))
            case(name + '-nested-tuple-leaf', F.fixture(shapes={1: nested},
                 logical=[(10, size, 1, (1, 0), 0)], allocations=[(0, size, [(10, 0, size)], 0)]),
                 lower=size)
            descriptor = case(name + '-tuple-descriptor', F.fixture(shapes={1: nested},
                 logical=[(10, 0, 1, (), 0)], allocations=[(0, 0, [(10, 0, 0)], 0)]))
            assert descriptor['compiler_allocation_plan']['logical_buffers'][10]['location']['shape'][
                'descriptor_storage_lower_bound'] == 'UNKNOWN_BACKEND_POINTER_SIZE'
            alias_shapes = {1: F.array(dtype, (1,)), 2: F.array(dtype, (3, 7))}
            for suffix, assigned, expected, reason in (
                ('valid', size, 'ALLOCATION_PLAN_BOUNDS_PASS', None),
                ('insufficient', size - 1, 'INVALID_ALLOCATION_PLAN', 'ALIAS_ARRAY_SLICE_LOWER_BOUND')):
                case(name + '-alias-' + suffix, F.fixture(shapes=alias_shapes,
                     logical=[(10, width, 1, (), 0)],
                     allocations=[(0, size, [(10, 0, assigned)], 0)], aliases=[(10, 2, ())]),
                     expected, reason)
            for suffix, shape, reason in (
                ('dynamic', F.array(dtype, (3, 7), dynamic=(1, 0)), 'UNSUPPORTED_DYNAMIC_SHAPE'),
                ('sparse', F.array(dtype, (3, 7), layout=F.sequence(9, (1, 0))),
                 'UNSUPPORTED_SPARSE_LAYOUT'),
                ('combine', F.array(dtype, (3, 7), layout=F.message(6,
                 F.sequence(1, (-(1 << 63), 7)))), 'UNSUPPORTED_COMBINE_DIMENSION_TILE')):
                case(name + '-' + suffix + '-still-unsupported', F.fixture(shapes={1: shape}),
                     'UNSUPPORTED_REPRESENTATION', reason)
            dump = temporary / (name + '.pb')
            report = temporary / (name + '.json')
            dump.write_bytes(plain(dtype, width))
            child = subprocess.run([sys.executable, '-B', str(HERE / 'allocation_audit_v2.py'),
                '--baseline-analyzer', str(baseline), '--hlo-proto', str(dump),
                '--hlo-proto-sha256', digest(dump), '--capture-packet-sha256', '0' * 64,
                '--output', str(report)], capture_output=True, text=True)
            assert child.returncode == 0, child.stderr
            result = json.loads(report.read_text())
            assert result['schema_version'] == result['audit_contract_version'] == 2
            assert result['status'] == 'ALLOCATION_PLAN_BOUNDS_PASS'
            assert result['frozen_v1_analysis_source_sha256'] == A.FROZEN_V1_SHA256
            assert result['actual_selected_executable_link'] == result['allocator_peak'] == 'UNKNOWN'
            assert result['native_body_internal_memory'] == 'UNKNOWN' and result['m6'] == 'NOT_QUALIFIED'
            rows.append({'case': name + '-CLI-contract-v2', 'status': 'PASS'})
        assert set(A.WIDTHS) == {1, 4, 5, 6, 8, 11, 16}
        for dtype in (2, 3, 7, 9, 10, 12):
            case('other-valid-primitive-' + str(dtype) + '-still-unsupported',
                 F.fixture(shapes={1: F.array(dtype)}),
                 'UNSUPPORTED_REPRESENTATION', 'UNSUPPORTED_ARRAY_DTYPE')
        reader = load('versioned_directory_reader', HERE / 'allocation_directory_v2.py')
        old = F.A
        scan = temporary / 'scan'
        scan.mkdir()
        # Misleading names prove field-based selection in both directions.
        (scan / 'before-without-assignment.pb').write_bytes(plain(5, 8))
        (scan / 'after-with-assignment.pb').write_bytes(F.message(1, F.message(3, b'')))
        report = reader.analyze(parser, old, A, scan)
        assert report['proto_count'] == 2 and report['selected_assignment_count'] == 1
        assert report['results'][0]['path'] == 'before-without-assignment.pb'
        assert report['results'][0]['compiler_module_id'] == 42
        assert report['results'][0]['sha256'] == A.sha(plain(5, 8))
        assert report['v2_status_counts'] == {'ALLOCATION_PLAN_BOUNDS_PASS': 1}
        assert report['input_inventory_unchanged'] is True
        rows.append({'case': 'directory-selects-field3-despite-misleading-names', 'status': 'PASS'})
        bad = F.message(1, F.integer(5, 42) + F.message(3, F.message(2,
            F.integer(35, 1) + F.message(3, F.array())))) + F.integer(3, 1)
        (scan / 'wrong-wire.pb').write_bytes(bad)
        report = reader.analyze(parser, old, A, scan)
        rejected = [row for row in report['results'] if row['path'] == 'wrong-wire.pb'][0]
        assert rejected['v2_status'] == 'INVALID_ALLOCATION_PLAN'
        assert rejected['v2_reason'] == 'PROTO_FIELD_WIRE'
        rows.append({'case': 'directory-field3-wrong-wire-invalid-not-skipped', 'status': 'PASS'})
        # The output must never be created under the input directory.
        try:
            reader.run(baseline, original_controls.parent / 'allocation_audit.py',
                       scan, '0' * 64, scan / 'output.json')
        except ValueError as error:
            assert str(error) == 'OUTPUT_MUST_BE_OUTSIDE_READ_ONLY_INPUT'
        else:
            raise AssertionError('read-only directory output accepted')
        assert not (scan / 'output.json').exists()
        rows.append({'case': 'directory-rejects-output-inside-read-only-input', 'status': 'PASS'})
        disposable_files = [p for p in temporary.rglob('*') if p.is_file()]
        disposal = {'files': len(disposable_files), 'bytes': sum(p.stat().st_size for p in disposable_files),
                    'temporary_directory': str(temporary)}
    assert not temporary.exists()
    assert digest(frozen) == frozen_before
    disposal.update({'directory_exists_after': temporary.exists(), 'bytes_after': 0})
    summary = {'status': 'PASS', 'scope': 'OFFLINE_SYNTHETIC_STATIC_ALLOCATION_BOUNDS_ONLY',
        'audit_contract_version': 2, 'auditor_sha256': digest(HERE / 'allocation_audit_v2.py'),
        'controls_sha256': digest(Path(__file__)), 'original_controls_sha256': digest(original_controls),
        'original65_against_v2': regression, 'new_controls_count': len(rows), 'new_controls': rows,
        'disposal': disposal, 'retained_fixture_sha256': frozen_before,
        'numerical_result': 'NOT_EVALUATED', 'm6': 'NOT_QUALIFIED'}
    with output.open('x') as stream:
        stream.write(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': 'PASS', 'original_controls': 65, 'new_controls': len(rows),
                      'disposable_files_deleted': disposal['files'], 'disposable_bytes_deleted': disposal['bytes']}))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('--baseline-analyzer', type=Path, required=True)
    args.add_argument('--original-controls', type=Path, required=True)
    args.add_argument('--frozen-fixture', type=Path, required=True)
    args.add_argument('--output', type=Path, required=True)
    options = args.parse_args()
    run(options.baseline_analyzer.resolve(), options.original_controls.resolve(),
        options.frozen_fixture.resolve(), options.output)
