"""Independent small binary fixtures for the offline allocation bounds auditor."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('allocation_audit', HERE / 'allocation_audit.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def varint(value):
    if value < 0:
        value += 1 << 64
    out = bytearray()
    while value > 127:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)


def integer(number, value):
    return varint(number << 3) + varint(value)


def message(number, raw):
    return varint((number << 3) | 2) + varint(len(raw)) + raw


def sequence(number, values):
    return b''.join(integer(number, value) for value in values)


def array(dtype=11, dims=(128, 256), layout=None, dynamic=()):
    raw = integer(2, dtype) + sequence(3, dims) + sequence(6, dynamic)
    if layout is not None:
        raw += message(5, layout)
    return raw


def tuple_shape(*children):
    return integer(2, 13) + b''.join(message(4, child) for child in children)


def location(instruction, path=()):
    return integer(4, instruction) + sequence(3, path)


def fixture(shapes=None, logical=None, allocations=None, aliases=(), allocation_flags=None):
    shapes = {1: array()} if shapes is None else shapes
    logical = [(10, 131072, 1, (), 0)] if logical is None else logical
    allocations = [(0, 131072, [(10, 0, 131072)], 0)] if allocations is None else allocations
    instructions = b''.join(message(2, integer(35, index) + message(3, shape))
                            for index, shape in shapes.items())
    module = integer(5, 42) + message(3, instructions)
    buffers = b''.join(message(1, integer(1, index) + integer(2, size) +
                              message(3, location(instruction, path)) + integer(4, color))
                       for index, size, instruction, path, color in logical)
    allocated = b''
    allocation_flags = allocation_flags or {}
    for index, size, assigned, color in allocations:
        slices = b''.join(message(9, integer(1, lid) + integer(2, offset) + integer(3, count))
                          for lid, offset, count in assigned)
        flags = b''.join(integer(number, value)
                         for number, value in allocation_flags.get(index, {}).items())
        allocated += message(3, integer(1, index) + integer(2, size) + integer(8, color) + slices + flags)
    alias_raw = b''.join(message(2, integer(1, source) + message(2, location(instruction, path)))
                         for source, instruction, path in aliases)
    return message(1, module) + message(3, buffers + allocated + alias_raw)


def run(baseline, output, frozen):
    parser = A.load_parser(baseline)
    rows = []

    def case(name, raw, expected='ALLOCATION_PLAN_BOUNDS_PASS', reason=None):
        result = A.audit_bytes(parser, raw)
        assert result['status'] == expected, (name, result)
        if reason:
            assert result['reason'] == reason, (name, result)
        if expected == 'ALLOCATION_PLAN_BOUNDS_PASS':
            plan = result['compiler_allocation_plan']
            assert plan['allocator_peak'] == plan['native_body_internal_memory'] == 'UNKNOWN'
            assert plan['meaning'] == 'COMPILER_ALLOCATION_PLAN_NOT_RUNTIME_PEAK'
        rows.append({'case': name, 'status': 'PASS', 'observed_status': result['status'],
                     'reason': result.get('reason'), 'fixture_sha256': A.sha(raw),
                     'fixture_bytes': len(raw)})
        return result

    good = fixture()
    case('coherent-f32-128x256-131072-bytes', good)
    case('padded-logical-buffer-and-larger-assigned-slice', fixture(
        logical=[(10, 131200, 1, (), 0)], allocations=[(0, 132000, [(10, 32, 131500)], 0)]))
    for dtype, width in ((6, 1), (16, 2), (11, 4)):
        size = 3 * 7 * width
        case('natural-width-' + str(dtype), fixture(shapes={1: array(dtype, (3, 7))},
             logical=[(10, size, 1, (), 0)], allocations=[(0, size, [(10, 0, size)], 0)]))
        case('shape-minimum-minus-one-' + str(dtype), fixture(shapes={1: array(dtype, (3, 7))},
             logical=[(10, size - 1, 1, (), 0)], allocations=[(0, size, [(10, 0, size)], 0)]),
             'INVALID_ALLOCATION_PLAN', 'LOGICAL_BUFFER_ARRAY_LOWER_BOUND')
    case('scalar-f32', fixture(shapes={1: array(11, ())}, logical=[(10, 4, 1, (), 0)],
         allocations=[(0, 4, [(10, 0, 4)], 0)]))
    case('empty-array-zero-storage', fixture(shapes={1: array(6, (0, 7))},
         logical=[(10, 0, 1, (), 0)], allocations=[(0, 0, [(10, 0, 0)], 0)]))
    case('positive-dense-tiles-padding-natural-width', fixture(shapes={1: array(layout=
         sequence(1, (1, 0)) + message(6, sequence(1, (8, 128))) + integer(7, 32) +
         integer(15, 32) + integer(16, 128))}))
    combine_fixture = fixture(shapes={1: array(11, (128, 256), layout=
         sequence(1, (1, 0)) + message(6, sequence(1, (-(1 << 63), 128))))})
    assert len(combine_fixture) == 79
    case('legitimate-combine-dimension-tile-outside-scope', combine_fixture,
         'UNSUPPORTED_REPRESENTATION', 'UNSUPPORTED_COMBINE_DIMENSION_TILE')
    case('other-negative-tile-dimension-rejected', fixture(shapes={1: array(layout=
         message(6, sequence(1, (-1, 128))))}), 'INVALID_ALLOCATION_PLAN',
         'NONNEGATIVE_SIGNED_INT64')
    case('nonzero-color-no-hbm-claim', fixture(logical=[(10, 131072, 1, (), 7)],
         allocations=[(0, 131072, [(10, 0, 131072)], 7)]))
    case('overlap-disjoint-liveness-structurally-valid', fixture(
         shapes={1: array(), 2: array()}, logical=[(10, 131072, 1, (), 0), (11, 131072, 2, (), 0)],
         allocations=[(0, 131072, [(10, 0, 131072), (11, 0, 131072)], 0)]))
    case('alias-larger-shape-valid-max-sized-slice', fixture(
         shapes={1: array(11, (4,)), 2: array(11, (8,))}, logical=[(10, 16, 1, (), 0)],
         allocations=[(0, 32, [(10, 0, 32)], 0)], aliases=[(10, 2, ())]))
    nested = tuple_shape(array(), tuple_shape(array(16, (2, 3))))
    case('tuple-descriptor-not-child-array-sum', fixture(shapes={1: nested},
         logical=[(10, 16, 1, (), 0)], allocations=[(0, 16, [(10, 0, 16)], 0)]))
    case('nested-tuple-leaf-index', fixture(shapes={1: nested}, logical=[(10, 12, 1, (1, 0), 0)],
         allocations=[(0, 12, [(10, 0, 12)], 0)]))
    case('allzero-nonempty-f32-plan-rejected', fixture(logical=[(10, 0, 1, (), 0)],
         allocations=[(0, 0, [(10, 0, 0)], 0)]), 'INVALID_ALLOCATION_PLAN',
         'LOGICAL_BUFFER_ARRAY_LOWER_BOUND')
    case('assignment-logical-minus-one', fixture(allocations=[(0, 131072, [(10, 0, 131071)], 0)]),
         'INVALID_ALLOCATION_PLAN', 'ASSIGNED_SLICE_LOGICAL_LOWER_BOUND')
    case('allocation-out-of-bounds', fixture(allocations=[(0, 131072, [(10, 1, 131072)], 0)]),
         'INVALID_ALLOCATION_PLAN', 'ALLOCATION_SLICE_BOUNDS')
    case('int64-addition-overflow-range', fixture(allocations=[(0, A.INT64_MAX,
         [(10, A.INT64_MAX - 1, 131072)], 0)]), 'INVALID_ALLOCATION_PLAN', 'ALLOCATION_SLICE_BOUNDS')
    for path in ((2,), (0, 0), (1, 1)):
        case('invalid-tuple-index-' + str(path), fixture(shapes={1: nested},
             logical=[(10, 131072, 1, path, 0)]), 'INVALID_ALLOCATION_PLAN',
             'LOGICAL_BUFFER_SHAPE_INDEX')
    negative_cases = {
        'logical-size': {'logical': [(10, -1, 1, (), 0)]},
        'logical-id': {'logical': [(-1, 131072, 1, (), 0)]},
        'instruction-id': {'logical': [(10, 131072, -1, (), 0)]},
        'shape-index': {'logical': [(10, 131072, 1, (-1,), 0)]},
        'allocation-size': {'allocations': [(0, -1, [(10, 0, 131072)], 0)]},
        'allocation-id': {'allocations': [(-1, 131072, [(10, 0, 131072)], 0)]},
        'offset': {'allocations': [(0, 131072, [(10, -1, 131072)], 0)]},
        'assigned-size': {'allocations': [(0, 131072, [(10, 0, -1)], 0)]},
        'assigned-id': {'allocations': [(0, 131072, [(-1, 0, 131072)], 0)]},
        'dimension': {'shapes': {1: array(11, (-1, 256))}},
        'color': {'logical': [(10, 131072, 1, (), -1)]},
        'layout-memory-space': {'shapes': {1: array(layout=integer(8, -1))}},
        'layout-element-bits': {'shapes': {1: array(layout=integer(7, -1))}},
    }
    for name, kwargs in negative_cases.items():
        case('signed-int64-negative-' + name, fixture(**kwargs), 'INVALID_ALLOCATION_PLAN',
             'NONNEGATIVE_SIGNED_INT64')
    case('array-product-overflow', fixture(shapes={1: array(11, (A.INT64_MAX, 2))}),
         'INVALID_ALLOCATION_PLAN', 'ARRAY_STORAGE_INT64_OVERFLOW')
    case('invalid-layout-permutation', fixture(shapes={1: array(layout=sequence(1, (0, 0)))}),
         'INVALID_ALLOCATION_PLAN', 'SHAPE_LAYOUT')
    case('ordinary-parameter-color-mismatch', fixture(
         allocations=[(0, 131072, [(10, 0, 131072)], 8)], allocation_flags={0: {5: 1}}),
         'INVALID_ALLOCATION_PLAN', 'ALLOCATION_LOGICAL_COLOR')
    exception = case('nonzero-preallocated-temp-color-exception-valid', fixture(
         allocations=[(0, 131072, [(10, 0, 131072)], 8)]))
    observed_allocation = exception['compiler_allocation_plan']['allocations'][0]
    assert observed_allocation['preallocated_temp_buffer'] is True
    assert observed_allocation['color_match_required'] is False
    case('zero-preallocated-temp-color-mismatch', fixture(logical=[(10, 131072, 1, (), 8)]),
         'INVALID_ALLOCATION_PLAN', 'ALLOCATION_LOGICAL_COLOR')
    for name, number in (('live-out', 7), ('thread-local', 3), ('constant', 12)):
        case(name + '-color-mismatch', fixture(
             allocations=[(0, 131072, [(10, 0, 131072)], 8)], allocation_flags={0: {number: 1}}),
             'INVALID_ALLOCATION_PLAN', 'ALLOCATION_LOGICAL_COLOR')
    case('alias-storage-minimum-exceeds-slice', fixture(
         shapes={1: array(11, (4,)), 2: array(11, (8,))}, logical=[(10, 16, 1, (), 0)],
         allocations=[(0, 16, [(10, 0, 16)], 0)], aliases=[(10, 2, ())]),
         'INVALID_ALLOCATION_PLAN', 'ALIAS_ARRAY_SLICE_LOWER_BOUND')
    case('unassigned-buffer', fixture(allocations=[(0, 131072, [], 0)]),
         'INVALID_ALLOCATION_PLAN', 'ALL_LOGICAL_BUFFERS_ASSIGNED')
    case('duplicate-assigned-buffer', fixture(allocations=[(0, 131072,
         [(10, 0, 131072), (10, 0, 131072)], 0)]), 'INVALID_ALLOCATION_PLAN',
         'LOGICAL_BUFFER_ASSIGNMENT_UNIQUE')
    unsupported = {
        'packing': array(layout=integer(7, 8)),
        'sparse': array(layout=sequence(9, (1, 0))),
        'split-memory': array(layout=message(17, integer(1, 0))),
        'dynamic': array(dynamic=(1, 0)),
        'other-dtype': array(12),
        'unknown-layout-field': array(layout=integer(99, 1)),
    }
    for name, shape in unsupported.items():
        case('unsupported-' + name, fixture(shapes={1: shape}), 'UNSUPPORTED_REPRESENTATION')
    case('missing-buffer-assignment', message(1, message(3, b'')),
         'UNSUPPORTED_REPRESENTATION', 'BUFFER_ASSIGNMENT_ABSENT')
    case('truncated-wire', b'\x0a\x80', 'INVALID_ALLOCATION_PLAN', 'PROTO_TRUNCATED')
    # Retain the frozen artifact unchanged; record that its previous pass is not memory evidence.
    raw = frozen.read_bytes()
    old = parser.allocation_summary(parser.module(raw, True))
    assert old['allocation_size_sum_bytes'] == 100
    case('frozen-incoherent-100-byte-positive-rejected', raw, 'INVALID_ALLOCATION_PLAN',
         'LOGICAL_BUFFER_ARRAY_LOWER_BOUND')
    def encode_fields(values):
        pieces = []
        for number, entries in values.items():
            for wire, value in entries:
                if wire == 0:
                    pieces.append(integer(number, value))
                elif wire == 2:
                    pieces.append(message(number, value))
                else:
                    pieces.append(varint((number << 3) | wire) + value)
        return b''.join(pieces)

    # Recreate the independently reported zero-slice defect on the frozen graph.
    envelope = parser.fields(raw)
    assignment = parser.fields(parser.one(envelope, 3))
    for number in (1, 3):
        changed = []
        for wire, value in assignment[number]:
            item = parser.fields(value)
            item[2] = [(0, 0)]
            if number == 3:
                slices = []
                for slice_wire, slice_raw in item.get(9, []):
                    part = parser.fields(slice_raw)
                    part[2] = [(0, 0)]
                    part[3] = [(0, 0)]
                    slices.append((slice_wire, encode_fields(part)))
                item[9] = slices
            changed.append((wire, encode_fields(item)))
        assignment[number] = changed
    envelope[3] = [(2, encode_fields(assignment))]
    zero_raw = encode_fields(envelope)
    old_zero = parser.allocation_summary(parser.module(zero_raw, True))
    assert old_zero['allocation_size_sum_bytes'] == 0
    case('frozen-allzero-mutant-old-pass-new-reject', zero_raw, 'INVALID_ALLOCATION_PLAN',
         'LOGICAL_BUFFER_ARRAY_LOWER_BOUND')
    # Establish the separate assigned-size defect on a coherent array fixture.
    undersized = fixture(allocations=[(0, 131072, [(10, 0, 131071)], 0)])
    after = {'envelope': parser.fields(undersized),
             'nodes': {1: {3: [(2, array())]}}}
    old_small = parser.allocation_summary(after)
    assert old_small['allocations'][0]['assigned'][0]['bytes'] == 131071
    case('coherent-logical-minus-one-old-pass-new-reject', undersized,
         'INVALID_ALLOCATION_PLAN', 'ASSIGNED_SLICE_LOGICAL_LOWER_BOUND')
    # Exercise the command boundary, exit classifications, SHA binding, and automatic cleanup.
    with tempfile.TemporaryDirectory(prefix='bnb-tpu-memory-controls-') as directory:
        temp = Path(directory)
        def cli_case(name, raw, expected_code, expected_status, dump_sha=None):
            path = temp / (name + '.pb')
            result = temp / (name + '.json')
            path.write_bytes(raw)
            command = [sys.executable, '-B', str(HERE / 'allocation_audit.py'),
                       '--baseline-analyzer', str(baseline), '--hlo-proto', str(path),
                       '--hlo-proto-sha256', dump_sha or A.sha(raw),
                       '--capture-packet-sha256', '0' * 64, '--output', str(result)]
            child = subprocess.run(command, capture_output=True, text=True)
            assert child.returncode == expected_code, (name, child.stderr)
            if expected_status:
                report = json.loads(result.read_text())
                assert report['status'] == expected_status
                assert report['numerical_result'] == 'NOT_EVALUATED_EXISTING_RESULT_RETAINED'
                assert report['m6'] == 'NOT_QUALIFIED'
            else:
                assert not result.exists()
            rows.append({'case': name, 'status': 'PASS', 'exit_code': child.returncode})
        cli_case('cli-valid', good, 0, 'ALLOCATION_PLAN_BOUNDS_PASS')
        cli_case('cli-invalid', fixture(allocations=[(0, 131072, [(10, 0, 0)], 0)]),
                 1, 'INVALID_ALLOCATION_PLAN')
        cli_case('cli-unsupported', fixture(shapes={1: array(12)}), 2, 'UNSUPPORTED_REPRESENTATION')
        cli_case('cli-wrong-dump-sha', good, 1, None, 'f' * 64)
        fake = temp / 'fake_parser.py'
        fake.write_text('raise AssertionError("must not execute altered source")\n')
        try:
            A.load_parser(fake)
        except ValueError as error:
            assert str(error) == 'BASELINE_PARSER_SHA256'
        else:
            raise AssertionError('unbound parser accepted')
        rows.append({'case': 'wrong-baseline-source-rejected-before-import', 'status': 'PASS'})
        temporary_path = str(temp)
    assert not Path(temporary_path).exists()
    summary = {'status': 'PASS', 'scope': 'OFFLINE_SYNTHETIC_CONTROLS_ONLY',
               'baseline_revision': A.BASELINE_REVISION, 'baseline_analyzer_sha256': A.PARSER_SHA256,
               'auditor_sha256': A.sha((HERE / 'allocation_audit.py').read_bytes()),
               'controls_sha256': A.sha(Path(__file__).read_bytes()),
               'controls': rows, 'controls_count': len(rows),
               'temporary_artifacts_deleted': True, 'temporary_directory_exists': False,
               'retained_frozen_fixture_sha256': A.sha(frozen.read_bytes()),
               'numerical_result': 'NOT_EVALUATED', 'm6': 'NOT_QUALIFIED'}
    with output.open('x') as record:
        record.write(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': 'PASS', 'controls': len(rows), 'temporary_artifacts_deleted': True}))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('--baseline-analyzer', type=Path, required=True)
    args.add_argument('--frozen-fixture', type=Path, required=True)
    args.add_argument('--output', type=Path, required=True)
    options = args.parse_args()
    run(options.baseline_analyzer, options.output, options.frozen_fixture)
