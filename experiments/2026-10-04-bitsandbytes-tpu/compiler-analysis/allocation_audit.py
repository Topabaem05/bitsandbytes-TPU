"""Offline allocation-plan bounds audit. No TPU execution or numerical qualification."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

BASELINE_REVISION = '5dbfd05128fe741ba19c513d43a87c09b7cca017'
PARSER_SHA256 = 'c44ecb15696031f326a7abe64ea3d8407b9bc21f527cb94f8b5f149c456fcd8d'
SCHEMA_REVISION = '31eb6029e8973337ef6c99810c27e1d807791c11'
INT64_MAX = (1 << 63) - 1
WIDTHS = {6: ('uint8', 1), 16: ('bfloat16', 2), 11: ('float32', 4)}


class Unsupported(ValueError):
    """A representation outside the declared storage scope."""


def require(value, code):
    if not value:
        raise ValueError(code)


def supported(value, code):
    if not value:
        raise Unsupported(code)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_parser(path):
    path = Path(path)
    require(sha(path.read_bytes()) == PARSER_SHA256, 'BASELINE_PARSER_SHA256')
    spec = importlib.util.spec_from_file_location('fixed_baseline_allocation_parser', path)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    return parser


class Auditor:
    def __init__(self, parser):
        self.parser = parser

    def fields(self, raw):
        return self.parser.fields(raw)

    def scalar(self, fields, number, default=0, wire=0):
        values = fields.get(number, [])
        require(len(values) <= 1, 'PROTO_DUPLICATE_SINGULAR')
        if not values:
            return default
        require(values[0][0] == wire, 'PROTO_FIELD_WIRE')
        return values[0][1]

    def integer(self, fields, number, default=0):
        value = self.scalar(fields, number, default)
        require(isinstance(value, int) and 0 <= value <= INT64_MAX,
                'NONNEGATIVE_SIGNED_INT64')
        return value

    def integers(self, fields, number):
        values = self.parser.ints(fields, number)
        require(all(0 <= value <= INT64_MAX for value in values),
                'NONNEGATIVE_SIGNED_INT64')
        return values

    def messages(self, fields, number):
        values = fields.get(number, [])
        require(all(wire == 2 for wire, _ in values), 'PROTO_MESSAGE_WIRE')
        return [value for _, value in values]

    def boolean(self, fields, number):
        value = self.integer(fields, number)
        require(value in (0, 1), 'PROTO_BOOLEAN')
        return bool(value)

    def shape(self, raw):
        fields = self.fields(raw)
        supported(set(fields) <= {2, 3, 4, 5, 6}, 'UNSUPPORTED_SHAPE_FIELDS')
        dtype = self.integer(fields, 2)
        dims = self.integers(fields, 3)
        children = self.messages(fields, 4)
        dynamic = self.integers(fields, 6)
        require(not dynamic or len(dynamic) == len(dims), 'SHAPE_DYNAMIC_RANK')
        require(all(value in (0, 1) for value in dynamic), 'PROTO_BOOLEAN')
        if dtype == 13:
            require(not dims and not dynamic and 5 not in fields, 'TUPLE_SHAPE_FIELDS')
            return {'kind': 'tuple_descriptor', 'child_count': len(children),
                    'array_storage_lower_bound_bytes': None,
                    'descriptor_storage_lower_bound': 'UNKNOWN_BACKEND_POINTER_SIZE'}
        supported(dtype in WIDTHS, 'UNSUPPORTED_ARRAY_DTYPE')
        require(not children, 'ARRAY_TUPLE_CHILDREN')
        supported(not any(dynamic), 'UNSUPPORTED_DYNAMIC_SHAPE')
        name, width = WIDTHS[dtype]
        layout = self.fields(self.scalar(fields, 5, b'', wire=2))
        supported(set(layout) <= {1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17},
                  'UNSUPPORTED_LAYOUT_FIELDS')
        minor = self.integers(layout, 1)
        require(not minor or sorted(minor) == list(range(len(dims))), 'SHAPE_LAYOUT')
        levels = self.integers(layout, 9)
        require(not levels or len(levels) == len(dims), 'LAYOUT_LEVEL_RANK')
        supported(not any(levels) and not layout.get(10) and
                  not self.integer(layout, 11) and not self.integer(layout, 12),
                  'UNSUPPORTED_SPARSE_LAYOUT')
        for number in (13, 14):
            values = self.integers(layout, number)
            require(not values or len(values) == len(levels), 'LAYOUT_LEVEL_ATTRIBUTE_RANK')
            require(all(value in (0, 1) for value in values), 'PROTO_BOOLEAN')
            supported(all(values), 'UNSUPPORTED_SPARSE_LAYOUT')
        supported(not layout.get(17), 'UNSUPPORTED_SPLIT_MEMORY_LAYOUT')
        bits = self.integer(layout, 7)
        supported(bits in (0, width * 8), 'UNSUPPORTED_ELEMENT_PACKING')
        # Positive dense tiles and tail padding cannot reduce this lower bound.
        # No attempt is made to calculate the padded physical storage size.
        for tile in self.messages(layout, 6):
            tile_fields = self.fields(tile)
            supported(set(tile_fields) <= {1}, 'UNSUPPORTED_TILE_FIELDS')
            tile_dims = self.parser.ints(tile_fields, 1)
            # INT64_MIN is Tile::kCombineDimension, not a negative storage size.
            require(all(0 <= value <= INT64_MAX or value == (1 << 63)
                        for value in tile_dims), 'NONNEGATIVE_SIGNED_INT64')
            supported((1 << 63) not in tile_dims, 'UNSUPPORTED_COMBINE_DIMENSION_TILE')
            supported(tile_dims and all(value > 0 for value in tile_dims),
                      'UNSUPPORTED_TILE_REPRESENTATION')
        memory_space = self.integer(layout, 8)
        self.integer(layout, 15)
        self.integer(layout, 16)
        lower = math.prod(dims) * width
        require(lower <= INT64_MAX, 'ARRAY_STORAGE_INT64_OVERFLOW')
        return {'kind': 'array', 'dtype': name, 'shape': dims,
                'array_storage_lower_bound_bytes': lower,
                'layout_memory_space': memory_space,
                'layout_sha256': sha(self.scalar(fields, 5, b'', wire=2)),
                'padding_storage': 'NOT_CALCULATED'}

    def location(self, raw, nodes):
        fields = self.fields(raw)
        instruction = self.integer(fields, 4)
        require(instruction in nodes, 'LOGICAL_BUFFER_INSTRUCTION')
        path = self.integers(fields, 3)
        selected = self.scalar(nodes[instruction], 3, None, wire=2)
        require(selected is not None, 'INSTRUCTION_SHAPE_REQUIRED')
        for index in path:
            current = self.fields(selected)
            children = self.messages(current, 4)
            require(self.integer(current, 2) == 13 and index < len(children),
                    'LOGICAL_BUFFER_SHAPE_INDEX')
            selected = children[index]
        return {'instruction_id': instruction, 'shape_index': path,
                'shape': self.shape(selected)}

    def audit(self, raw):
        outer = self.fields(raw)
        module_raw = self.scalar(outer, 1, None, wire=2)
        supported(module_raw is not None, 'HLO_PROTO_MODULE_ABSENT')
        module = self.fields(module_raw)
        computations = self.messages(module, 3)
        supported(computations, 'HLO_PROTO_COMPUTATIONS_ABSENT')
        assignment_raw = self.scalar(outer, 3, None, wire=2)
        supported(assignment_raw is not None, 'BUFFER_ASSIGNMENT_ABSENT')
        nodes = {}
        for computation_raw in computations:
            computation = self.fields(computation_raw)
            for instruction_raw in self.messages(computation, 2):
                instruction = self.fields(instruction_raw)
                index = self.integer(instruction, 35)
                require(index not in nodes, 'INSTRUCTION_ID_UNIQUE')
                nodes[index] = instruction
        require(nodes, 'INSTRUCTIONS_REQUIRED')
        assignment = self.fields(assignment_raw)
        logical = {}
        locations = set()
        for buffer_raw in self.messages(assignment, 1):
            buffer = self.fields(buffer_raw)
            index, size = self.integer(buffer, 1), self.integer(buffer, 2)
            require(index not in logical, 'LOGICAL_BUFFER_ID_UNIQUE')
            location_raw = self.scalar(buffer, 3, None, wire=2)
            require(location_raw is not None, 'LOGICAL_BUFFER_LOCATION_REQUIRED')
            location = self.location(location_raw, nodes)
            key = (location['instruction_id'], tuple(location['shape_index']))
            require(key not in locations, 'LOGICAL_BUFFER_LOCATION_UNIQUE')
            locations.add(key)
            lower = location['shape']['array_storage_lower_bound_bytes']
            require(lower is None or lower <= size, 'LOGICAL_BUFFER_ARRAY_LOWER_BOUND')
            logical[index] = {'bytes': size, 'color': self.integer(buffer, 4),
                              'location': location}
        require(logical, 'LOGICAL_BUFFERS_REQUIRED')
        aliases = []
        alias_keys = set()
        for alias_raw in self.messages(assignment, 2):
            alias = self.fields(alias_raw)
            index = self.integer(alias, 1)
            require(index in logical, 'BUFFER_ALIAS_SOURCE')
            location_raw = self.scalar(alias, 2, None, wire=2)
            require(location_raw is not None, 'BUFFER_ALIAS_LOCATION_REQUIRED')
            location = self.location(location_raw, nodes)
            key = (index, location['instruction_id'], tuple(location['shape_index']))
            require(key not in alias_keys, 'BUFFER_ALIAS_UNIQUE')
            alias_keys.add(key)
            aliases.append({'source_buffer_id': index, 'location': location})
        allocations = []
        allocation_ids = set()
        assigned = {}
        for allocation_raw in self.messages(assignment, 3):
            allocation = self.fields(allocation_raw)
            index, size = self.integer(allocation, 1), self.integer(allocation, 2)
            color = self.integer(allocation, 8)
            parameter = self.boolean(allocation, 5)
            live_out = self.boolean(allocation, 7)
            thread_local = self.boolean(allocation, 3)
            constant = self.boolean(allocation, 12)
            # Pinned BufferAllocation::IsPreallocatedTempBuffer and AddAssignment.
            preallocated_temp = not (parameter or live_out or thread_local or constant)
            color_match_required = not (preallocated_temp and color != 0)
            require(index not in allocation_ids, 'ALLOCATION_INDEX_UNIQUE')
            allocation_ids.add(index)
            slices = []
            for slice_raw in self.messages(allocation, 9):
                item = self.fields(slice_raw)
                lid = self.integer(item, 1)
                offset, count = self.integer(item, 2), self.integer(item, 3)
                require(lid in logical, 'ALLOCATION_LOGICAL_BUFFER')
                require(lid not in assigned, 'LOGICAL_BUFFER_ASSIGNMENT_UNIQUE')
                require(logical[lid]['bytes'] <= count, 'ASSIGNED_SLICE_LOGICAL_LOWER_BOUND')
                # Subtraction avoids signed int64 overflow in offset + count.
                require(offset <= size and count <= size - offset, 'ALLOCATION_SLICE_BOUNDS')
                require(not color_match_required or color == logical[lid]['color'],
                        'ALLOCATION_LOGICAL_COLOR')
                item_summary = {'logical_buffer_id': lid, 'offset': offset, 'bytes': count}
                slices.append(item_summary)
                assigned[lid] = item_summary
            parameter_index = self.integers(allocation, 10)
            parameter_number = self.integer(allocation, 6)
            allocations.append({'index': index, 'bytes': size, 'assigned': slices,
                                'color': color, 'parameter': parameter,
                                'parameter_number': parameter_number if parameter else None,
                                'parameter_shape_index': parameter_index,
                                'live_out': live_out, 'thread_local': thread_local,
                                'constant': constant,
                                'preallocated_temp_buffer': preallocated_temp,
                                'color_match_required': color_match_required,
                                'is_tuple': self.boolean(allocation, 11),
                                'parameter_aliased_with_output': self.boolean(allocation, 13)})
        require(allocations, 'ALLOCATIONS_REQUIRED')
        require(set(assigned) == set(logical), 'ALL_LOGICAL_BUFFERS_ASSIGNED')
        for alias in aliases:
            lower = alias['location']['shape']['array_storage_lower_bound_bytes']
            require(lower is None or lower <= assigned[alias['source_buffer_id']]['bytes'],
                    'ALIAS_ARRAY_SLICE_LOWER_BOUND')
        return {'compiler_module_id': self.integer(module, 5),
                'allocation_size_sum_bytes': sum(item['bytes'] for item in allocations),
                'logical_buffers': logical, 'buffer_aliases': aliases,
                'allocations': allocations,
                'bounds_rule': 'logical_bytes <= assigned_bytes <= allocation_bytes - offset',
                'array_lower_bound_rule': 'static_dense_product_dimensions_times_natural_element_bytes',
                'tuple_descriptor_size': 'UNKNOWN_NOT_SUM_OF_CHILD_ARRAYS',
                'overlaps': 'PERMITTED_BACKEND_LIVENESS_AND_ALIAS_SEMANTICS_NOT_VERIFIED',
                'color_semantics': 'BACKEND_SPECIFIC_NOT_HBM_IDENTIFICATION',
                'allocator_peak': 'UNKNOWN', 'native_body_internal_memory': 'UNKNOWN',
                'meaning': 'COMPILER_ALLOCATION_PLAN_NOT_RUNTIME_PEAK'}


def audit_bytes(parser, raw):
    try:
        return {'status': 'ALLOCATION_PLAN_BOUNDS_PASS',
                'compiler_allocation_plan': Auditor(parser).audit(raw)}
    except Unsupported as error:
        return {'status': 'UNSUPPORTED_REPRESENTATION', 'reason': str(error)}
    except (ValueError, TypeError, IndexError, OverflowError, RecursionError) as error:
        return {'status': 'INVALID_ALLOCATION_PLAN', 'reason': str(error)}


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('--baseline-analyzer', type=Path, required=True)
    args.add_argument('--hlo-proto', type=Path, required=True)
    args.add_argument('--hlo-proto-sha256', required=True)
    args.add_argument('--capture-packet-sha256', required=True)
    args.add_argument('--output', type=Path, required=True)
    options = args.parse_args()
    parser = load_parser(options.baseline_analyzer)
    raw = options.hlo_proto.read_bytes()
    require(sha(raw) == options.hlo_proto_sha256, 'INPUT_DUMP_SHA256')
    require(re.fullmatch('[0-9a-f]{64}', options.capture_packet_sha256), 'CAPTURE_PACKET_SHA256')
    result = audit_bytes(parser, raw)
    result.update({'schema_version': 1,
                   'analysis_source_sha256': sha(Path(__file__).read_bytes()),
                   'baseline_revision': BASELINE_REVISION,
                   'baseline_analyzer_sha256': PARSER_SHA256,
                   'schema_reference_revision': SCHEMA_REVISION,
                   'capture_packet_sha256': options.capture_packet_sha256,
                   'capture_packet_binding': 'CALLER_SUPPLIED_REQUIRES_INDEPENDENT_ARCHIVE_REVIEW',
                   'artifact': {'path': str(options.hlo_proto.resolve()), 'bytes': len(raw),
                                'sha256': sha(raw)},
                   'numerical_result': 'NOT_EVALUATED_EXISTING_RESULT_RETAINED',
                   'actual_selected_executable_link': 'UNKNOWN',
                   'allocator_peak': 'UNKNOWN', 'native_body_internal_memory': 'UNKNOWN',
                   'color_semantics': 'BACKEND_SPECIFIC_NOT_HBM_IDENTIFICATION',
                   'm6': 'NOT_QUALIFIED'})
    require(options.output.resolve() != options.hlo_proto.resolve(), 'OUTPUT_MUST_NOT_REPLACE_DUMP')
    with options.output.open('x') as output:
        output.write(json.dumps(result, sort_keys=True, indent=2) + '\n')
    return {'ALLOCATION_PLAN_BOUNDS_PASS': 0, 'INVALID_ALLOCATION_PLAN': 1,
            'UNSUPPORTED_REPRESENTATION': 2}[result['status']]


if __name__ == '__main__':
    raise SystemExit(main())
