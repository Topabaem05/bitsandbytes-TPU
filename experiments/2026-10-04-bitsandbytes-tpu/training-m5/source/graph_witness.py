"""Bound actual pre-optimization composite graphs. No executable or algorithm claim."""
import hashlib
import math
import json
import re
import struct
import hlo_parser as H

LIMIT_BYTES = 32 * 1024 * 1024
LIMIT_NODES = 100000

def require(value, message):
    if not value:
        raise ValueError(message)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def varint(data, offset):
    result = 0
    for i in range(10):
        require(offset < len(data), 'PROTO_TRUNCATED')
        byte = data[offset]; offset += 1
        result |= (byte & 127) << (7 * i)
        if byte < 128:
            return result, offset
    raise ValueError('PROTO_VARINT')

def fields(data):
    require(type(data) is bytes and len(data) <= LIMIT_BYTES, 'PROTO_BOUND')
    offset, result = 0, {}
    while offset < len(data):
        tag, offset = varint(data, offset); number, wire = tag >> 3, tag & 7
        require(number > 0, 'PROTO_FIELD')
        if wire == 0:
            value, offset = varint(data, offset)
        elif wire == 2:
            size, offset = varint(data, offset)
            require(size <= len(data) - offset, 'PROTO_TRUNCATED')
            value = data[offset:offset + size]; offset += size
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            require(size <= len(data) - offset, 'PROTO_TRUNCATED')
            value = data[offset:offset + size]; offset += size
        else:
            raise ValueError('PROTO_WIRE')
        result.setdefault(number, []).append((wire, value))
    return result

def one(value, number, default=None):
    found = value.get(number, [])
    require(len(found) <= 1, 'PROTO_DUPLICATE_FIELD')
    return found[0][1] if found else default

def ints(value, number):
    result = []
    for wire, item in value.get(number, []):
        if wire == 0:
            result.append(item)
        else:
            require(wire == 2, 'PROTO_PACKED_FIELD')
            offset = 0
            while offset < len(item):
                number_value, offset = varint(item, offset); result.append(number_value)
    return result

DTYPES = {1: 'pred', 4: 's32', 6: 'u8', 11: 'f32', 13: 'tuple', 16: 'bf16'}
ARRAY_DTYPES = {'bool': 'pred', 'int32': 's32', 'uint8': 'u8', 'float32': 'f32', 'bfloat16': 'bf16'}

def shape(data, depth=0):
    require(depth <= 32, 'PROTO_TUPLE_DEPTH_SCOPE')
    value = fields(data); dtype = one(value, 2, 0)
    require(dtype in DTYPES, 'PROTO_DTYPE_SCOPE')
    if dtype == 13:
        return ('tuple', tuple(shape(item, depth + 1) for wire, item in value.get(4, []) if wire == 2))
    return DTYPES[dtype], tuple(ints(value, 3))

def module(data):
    value = fields(data)
    computations = {}
    for wire, raw in value.get(3, []):
        require(wire == 2, 'COMPUTATION_WIRE')
        comp = fields(raw); ident = one(comp, 5, 0)
        require(ident not in computations, 'COMPUTATION_ID')
        nodes = {}
        for instruction_wire, instruction in comp.get(2, []):
            require(instruction_wire == 2, 'INSTRUCTION_WIRE')
            node = fields(instruction); node_id = one(node, 35, 0)
            require(node_id not in nodes, 'INSTRUCTION_ID')
            nodes[node_id] = node
        require(one(comp, 6, 0) in nodes, 'PROTO_ROOT')
        computations[ident] = {'nodes': nodes, 'root': one(comp, 6, 0)}
    entry = one(value, 6, 0)
    require(entry in computations and 0 < len(computations) <= 256
            and sum(len(c['nodes']) for c in computations.values()) <= LIMIT_NODES, 'PROTO_SCOPE')
    def dependencies(key):
        comp_id, node_id = key
        require(comp_id in computations and node_id in computations[comp_id]['nodes'], 'PROTO_GRAPH_REFERENCE')
        node = computations[comp_id]['nodes'][node_id]
        require(not ints(node, 37), 'CONTROL_PREDECESSORS_UNSUPPORTED')
        called = ints(node, 38)
        require(all(c in computations for c in called), 'PROTO_GRAPH_REFERENCE')
        return [(comp_id, d) for d in ints(node, 36)] + [(c, computations[c]['root']) for c in called]
    order, depths = postorder((entry, computations[entry]['root']), dependencies)
    memo = {}
    for key in order:
        comp_id, node_id = key; node = computations[comp_id]['nodes'][node_id]
        # Preserve semantic attributes. Names and debug metadata do not define graph content.
        kept = {str(k): [(w, v.hex() if isinstance(v, bytes) else v) for w, v in items]
                for k, items in node.items() if k not in (1, 7, 35, 36, 38)}
        kept['operands'] = [memo[(comp_id, dep)] for dep in ints(node, 36)]
        kept['callees'] = [memo[(callee, computations[callee]['root'])] for callee in ints(node, 38)]
        memo[key] = sha(json.dumps(kept, sort_keys=True, separators=(',', ':')).encode())
    require(len(order) == sum(len(c['nodes']) for c in computations.values()), 'PROTO_UNUSED_NODE_OR_COMPUTATION')
    return computations, entry, memo[(entry, computations[entry]['root'])]


def postorder(root, dependencies):
    """Iterative DFS: bounded nodes, cycle rejection, one visit per shared DAG node."""
    states, order, depths, stack = {}, [], {}, [(root, False)]
    edges = {}
    while stack:
        key, exiting = stack.pop()
        if exiting:
            require(states.get(key) == 1 and all(states.get(d) == 2 for d in edges[key]), 'GRAPH_CYCLE')
            states[key] = 2
            depths[key] = 1 + max((depths[d] for d in edges[key]), default=0)
            order.append(key)
            continue
        if states.get(key) == 2:
            continue
        require(states.get(key) != 1, 'GRAPH_CYCLE')
        require(len(states) < LIMIT_NODES, 'GRAPH_NODES')
        states[key] = 1; edges[key] = list(dependencies(key))
        stack.append((key, True))
        stack.extend((dep, False) for dep in reversed(edges[key]))
    return order, depths


def array_signature(array):
    require(array.get('dtype') in ARRAY_DTYPES and type(array.get('shape')) is list
            and all(type(n) is int and n >= 0 for n in array['shape']), 'PARAMETER_ARRAY_SIGNATURE')
    return ARRAY_DTYPES[array['dtype']], tuple(array['shape'])

def structural(hlo, parameters, output_signatures):
    require(type(hlo) is str and 0 < len(hlo.encode()) <= LIMIT_BYTES, 'HLO_BOUND')
    comps, entry = H.parse(hlo)
    require(sum(len(c['nodes']) for c in comps.values()) <= LIMIT_NODES, 'HLO_NODES')
    root = comps[entry]['nodes'][comps[entry]['root']]
    root_names = root['deps'] if root['opcode'] == 'tuple' else [comps[entry]['root']]
    require(len(root_names) == len(output_signatures), 'COMPOSITE_OUTPUT_MATRIX')
    nodes = comps[entry]['nodes']
    require([(nodes[n]['dtype'], tuple(nodes[n]['shape'])) for n in root_names]
            == output_signatures, 'COMPOSITE_OUTPUT_TYPES')
    observed = {n['parameter']: (n['dtype'], tuple(n['shape'])) for n in nodes.values() if n['opcode'] == 'parameter'}
    require(len(observed) == sum(n['opcode'] == 'parameter' for n in nodes.values())
            and observed == {int(k): array_signature(v) for k, v in parameters.items()}, 'GRAPH_PARAMETER_TYPES')
    integer_ops = {'add', 'subtract', 'multiply', 'negate', 'and', 'or', 'xor', 'not',
                   'shift-left', 'shift-right-arithmetic', 'shift-right-logical', 'maximum', 'minimum',
                   'compare', 'select', 'reduce', 'gather', 'iota', 'pad', 'concatenate',
                   'broadcast', 'reshape', 'transpose', 'slice', 'copy', 'convert', 'bitcast-convert',
                   'call', 'get-tuple-element', 'tuple', 'constant', 'parameter'}
    def dependencies(key):
        comp_name, name = key
        require(comp_name in comps and name in comps[comp_name]['nodes'], 'HLO_GRAPH_REFERENCE')
        node = comps[comp_name]['nodes'][name]
        deps = [(comp_name, dep) for dep in node['deps']]
        if node['opcode'] in ('call', 'reduce'):
            match = re.search(r'to_apply\s*=\s*%?([\w.:-]+)', node['body'])
            require(match is not None and match[1] in comps, 'COMPOSITE_CALLEE')
            deps.append((match[1], comps[match[1]]['root']))
        return deps
    order, depths = postorder((entry, comps[entry]['root']), dependencies)
    count, bitcasts = {}, 0
    for comp_name, name in order:
        node = comps[comp_name]['nodes'][name]; op = node['opcode']
        require(op in integer_ops, 'UNSUPPORTED_COMPOSITE_OPCODE:' + op)
        count[op] = count.get(op, 0) + 1
        dtype = node['dtype']
        if op != 'tuple':
            require(dtype in ('f32', 's32', 'u8', 'pred'), 'COMPOSITE_NO_INT64_OR_BF16_ARITHMETIC')
        deps = [comps[comp_name]['nodes'][dep] for dep in node['deps']]
        if dtype == 'f32':
            require(op in ('parameter', 'bitcast-convert', 'call', 'copy', 'reshape', 'get-tuple-element'),
                    'COMPOSITE_FLOAT_ARITHMETIC')
        if op == 'bitcast-convert':
            require(len(deps) == 1 and deps[0]['shape'] == node['shape']
                    and (deps[0]['dtype'], dtype) in (('f32', 's32'), ('s32', 'f32')), 'COMPOSITE_BITCAST_TYPE')
            bitcasts += 1
        if op == 'convert':
            require(len(deps) == 1 and dtype in ('s32', 'u8', 'pred')
                    and deps[0]['dtype'] in ('s32', 'u8', 'pred'), 'COMPOSITE_INTEGER_CONVERT_ONLY')
        integer = ('s32', 'u8', 'pred')
        binary = {'add', 'subtract', 'multiply', 'and', 'or', 'xor', 'shift-left',
                  'shift-right-arithmetic', 'shift-right-logical', 'maximum', 'minimum'}
        if op in binary:
            require(len(deps) == 2 and dtype in integer and all(d['dtype'] == dtype and d['shape'] == node['shape'] for d in deps),
                    'COMPOSITE_INTEGER_BINARY_TYPES')
            if op not in ('and', 'or', 'xor'):
                require(dtype != 'pred', 'COMPOSITE_PRED_ARITHMETIC')
        if op in ('negate', 'not'):
            require(len(deps) == 1 and dtype in integer and deps[0]['dtype'] == dtype and deps[0]['shape'] == node['shape'],
                    'COMPOSITE_INTEGER_UNARY_TYPES')
            require(op != 'negate' or dtype != 'pred', 'COMPOSITE_PRED_ARITHMETIC')
        if op == 'compare':
            require(len(deps) == 2 and dtype == 'pred' and deps[0]['dtype'] in integer
                    and deps[1]['dtype'] == deps[0]['dtype']
                    and all(d['shape'] == node['shape'] for d in deps), 'COMPOSITE_INTEGER_COMPARE_TYPES')
        if op == 'select':
            require(len(deps) == 3 and dtype in integer and deps[0]['dtype'] == 'pred'
                    and all(d['shape'] == node['shape'] for d in deps)
                    and deps[1]['dtype'] == deps[2]['dtype'] == dtype, 'COMPOSITE_INTEGER_SELECT_TYPES')
        if op in ('copy', 'reshape', 'transpose', 'slice', 'broadcast'):
            require(len(deps) == 1 and deps[0]['dtype'] == dtype, 'COMPOSITE_SHAPE_ONLY_TYPES')
        if op == 'gather':
            require(len(deps) == 2 and dtype in integer and deps[0]['dtype'] == dtype
                    and deps[1]['dtype'] == 's32', 'COMPOSITE_INTEGER_GATHER_TYPES')
        if op == 'reduce':
            require(dtype in integer and len(deps) == 2 and all(d['dtype'] == dtype for d in deps),
                    'COMPOSITE_INTEGER_REDUCE_TYPES')
        if op == 'concatenate':
            require(deps and dtype in integer and all(d['dtype'] == dtype for d in deps),
                    'COMPOSITE_INTEGER_CONCATENATE_TYPES')
        if op == 'pad':
            require(len(deps) == 2 and dtype in integer and all(d['dtype'] == dtype for d in deps),
                    'COMPOSITE_INTEGER_PAD_TYPES')
        if op == 'iota':
            require(not deps and dtype == 's32', 'COMPOSITE_INTEGER_IOTA_TYPES')
        if op == 'constant':
            require(dtype in ('s32', 'u8', 'pred'), 'COMPOSITE_NO_FLOAT_LITERAL')
    require(len(order) == sum(len(c['nodes']) for c in comps.values()), 'HLO_UNUSED_NODE_OR_COMPUTATION')
    require(bitcasts > 0, 'COMPOSITE_BITCAST_REQUIRED')
    return {'scope': 'COMPOSITE_ARITHMETIC_PRE_OPTIMIZATION_ONLY', 'opcode_counts': count,
            'bitcast_count': bitcasts, 'node_count': len(order),
            'maximum_dag_depth': depths[(entry, comps[entry]['root'])], 'no_float_arithmetic': True, 'no_int64': True,
            'no_integer_divide': True, 'selected_executable': 'UNKNOWN',
            'whole_gemm_float_scope': 'NOT_THIS_PROOF'}

def bind(context_proto, printer_proto, hlo, parameters, output_signatures):
    context, entry, canonical = module(context_proto)
    printer, printer_entry, printer_canonical = module(printer_proto)
    require(canonical == printer_canonical, 'CONTEXT_PRINTER_PROTO_CONTENT')
    # Bind every textual node to its proto opcode, type and ordered operands.
    comps, hlo_entry = H.parse(hlo)
    def projection_digest(value):
        return sha(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())
    def proto_deps(key):
        comp_id, ident = key; node = context[comp_id]['nodes'][ident]
        return [(comp_id, d) for d in ints(node, 36)] + [(c, context[c]['root']) for c in ints(node, 38)]
    proto_root = entry, context[entry]['root']
    proto_order, _ = postorder(proto_root, proto_deps); proto_cache = {}
    for comp_id, ident in proto_order:
        node = context[comp_id]['nodes'][ident]
        value = {'op': one(node, 2, b'').decode(), 'type': shape(one(node, 3, b'')),
                 'parameter': one(node, 9, 0) if one(node, 2, b'') == b'parameter' else None,
                 'deps': [proto_cache[(comp_id, d)] for d in ints(node, 36)],
                 'callees': [proto_cache[(c, context[c]['root'])] for c in ints(node, 38)]}
        proto_cache[(comp_id, ident)] = projection_digest(value)
    def text_deps(key):
        comp_name, name = key
        require(comp_name in comps and name in comps[comp_name]['nodes'], 'HLO_GRAPH_REFERENCE')
        node = comps[comp_name]['nodes'][name]
        callee = re.search(r'to_apply\s*=\s*%?([\w.:-]+)', node['body'])
        require(callee is None or callee[1] in comps, 'COMPOSITE_CALLEE')
        return [(comp_name, d) for d in node['deps']] + ([(callee[1], comps[callee[1]]['root'])] if callee else [])
    text_root = hlo_entry, comps[hlo_entry]['root']
    text_order, _ = postorder(text_root, text_deps); text_cache = {}
    for comp_name, name in text_order:
        node = comps[comp_name]['nodes'][name]
        if node['opcode'] == 'tuple':
            signature = ('tuple', tuple((comps[comp_name]['nodes'][d]['dtype'], tuple(comps[comp_name]['nodes'][d]['shape'])) for d in node['deps']))
        else:
            signature = node['dtype'], tuple(node['shape'])
        callee = re.search(r'to_apply\s*=\s*%?([\w.:-]+)', node['body'])
        value = {'op': node['opcode'], 'type': signature,
                 'parameter': node.get('parameter') if node['opcode'] == 'parameter' else None,
                 'deps': [text_cache[(comp_name, d)] for d in node['deps']],
                 'callees': [text_cache[(callee[1], comps[callee[1]]['root'])]] if callee else []}
        text_cache[(comp_name, name)] = projection_digest(value)
    require(proto_cache[proto_root] == text_cache[text_root], 'HLO_PROTO_STRUCTURE_BINDING')
    proof = structural(hlo, parameters, output_signatures)
    proof.update(context_proto_sha256=sha(context_proto), printer_proto_sha256=sha(printer_proto),
                 hlo_sha256=sha(hlo.encode()))
    return proof
