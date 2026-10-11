"""Read top-level host program arity from complete pinned HloModuleProto bytes.

This bounded wire reader does not validate executable HLO semantics. It reads
only the pinned structural fields. It never rewrites bytes used by DataHash.
"""
MAX_BYTES = 16777216
MAX_FIELDS = 262144
TUPLE = 13

class ArityUnsupported(ValueError):
    pass

def check(ok, label):
    if not ok:
        raise ArityUnsupported(label)

def varint(raw, offset):
    value = 0
    for i in range(10):
        check(offset < len(raw), 'UNSUPPORTED_PROTO_TRUNCATION')
        byte = raw[offset]; offset += 1
        check(i < 9 or byte <= 1, 'UNSUPPORTED_PROTO_VARINT')
        value |= (byte & 127) << (i * 7)
        if byte < 128:
            check(i == 0 or byte != 0, 'UNSUPPORTED_NONCANONICAL_VARINT')
            return value, offset
    raise ArityUnsupported('UNSUPPORTED_PROTO_VARINT')

def fields(raw):
    check(type(raw) is bytes and len(raw) <= MAX_BYTES, 'UNSUPPORTED_PROTO_BOUND')
    offset = count = 0
    while offset < len(raw):
        count += 1; check(count <= MAX_FIELDS, 'UNSUPPORTED_PROTO_FIELD_LIMIT')
        tag, offset = varint(raw, offset)
        field, wire = tag >> 3, tag & 7
        check(0 < field < (1 << 29), 'UNSUPPORTED_PROTO_TAG')
        if wire == 0:
            value, offset = varint(raw, offset)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            check(offset + size <= len(raw), 'UNSUPPORTED_PROTO_TRUNCATION')
            value = raw[offset:offset+size]; offset += size
        elif wire == 2:
            size, offset = varint(raw, offset)
            check(offset + size <= len(raw), 'UNSUPPORTED_PROTO_TRUNCATION')
            value = raw[offset:offset+size]; offset += size
        else:
            raise ArityUnsupported('UNSUPPORTED_PROTO_WIRE_TYPE')
        yield field, wire, value

def one_message(raw, number, label):
    items = [(wire, value) for field, wire, value in fields(raw) if field == number]
    check(len(items) == 1 and items[0][0] == 2, label)
    return items[0][1]

def shape_outputs(shape, depth=0):
    check(depth <= 64, 'UNSUPPORTED_SHAPE_DEPTH')
    element = []
    children = []
    for field, wire, value in fields(shape):
        if field == 2:
            check(wire == 0, 'UNSUPPORTED_SHAPE_ELEMENT_TYPE'); element.append(value)
        elif field == 4:
            check(wire == 2, 'UNSUPPORTED_TUPLE_SHAPE'); children.append(value)
    check(len(element) == 1 and 1 <= element[0] <= 34, 'UNSUPPORTED_SHAPE_ELEMENT_TYPE')
    # Count top-level tuple children. Do not recursively count their leaves.
    if element[0] == TUPLE:
        for child in children:
            shape_outputs(child, depth + 1)
        return len(children)
    if element[0] == 34:  # BUFFER is not TUPLE; one underlying shape.
        check(len(children) == 1, 'UNSUPPORTED_BUFFER_SHAPE')
        shape_outputs(children[0], depth + 1)
        return 1
    check(not children, 'UNSUPPORTED_NON_TUPLE_CHILDREN')
    return 1

def module_arity(module):
    host = one_message(module, 4, 'UNSUPPORTED_MODULE_HOST_PROGRAM_SHAPE')
    inputs = 0
    for field, wire, value in fields(host):
        if field == 1:
            check(wire == 2, 'UNSUPPORTED_PROGRAM_PARAMETER')
            # Parameter count belongs only to host_program_shape.parameters.
            # Internal computation parameters and tuple leaves are irrelevant.
            shape_outputs(value)
            inputs += 1
    result = one_message(host, 2, 'UNSUPPORTED_PROGRAM_RESULT')
    return {'inputs': inputs, 'outputs': shape_outputs(result)}
