"""Version 3 reference-only typed scalar observer. No native helper widening."""
import math
import contract as C
PC = C.load(C.HERE / 'helpers/proto_contract.py', 'reference_wire_parser_v3')
CONTRACT = 'PUBLIC15_REFERENCE_TYPED_SCALAR_ROLE_BINDING_V3'


def parameter_shape(raw):
    fields = PC.fields(raw)
    dtype, dims = PC.one(fields, 2, 0), PC.ints(fields, 3)
    if dims:
        return PC.shape(raw)
    C.require(dtype in (5, 6), 'REFERENCE_SCALAR_DTYPE_SCOPE')
    C.require(set(fields) <= {2, 3, 5, 6} and not PC.ints(fields, 6),
              'REFERENCE_SCALAR_STATIC_SHAPE')
    layout = PC.fields(PC.one(fields, 5, b''))
    C.require(set(layout) <= {1, 7, 8, 15, 16} and not PC.ints(layout, 1),
              'REFERENCE_SCALAR_LAYOUT_SCOPE')
    width = 64 if dtype == 5 else 8
    C.require(PC.one(layout, 7, 0) in (0, width), 'REFERENCE_SCALAR_PACKING_SCOPE')
    return {'dtype': {5: 'int64', 6: 'uint8'}[dtype], 'shape': [],
            'minor_to_major': [], 'full_shape_sha256': C.sha_bytes(raw)}


def array(value):
    C.require(isinstance(value, dict) and set(value) == {'dtype', 'shape', 'values'},
              'REFERENCE_ARRAY_FIELDS')
    dims, dtype, values = value['shape'], value['dtype'], value['values']
    C.require(isinstance(dims, list) and all(type(v) is int and v > 0 for v in dims),
              'REFERENCE_ARRAY_SHAPE')
    C.require(isinstance(values, list) and len(values) == math.prod(dims),
              'REFERENCE_ARRAY_VALUES')
    if not dims:
        C.require(dtype in ('int64', 'uint8'), 'REFERENCE_SCALAR_DTYPE_SCOPE')
        low, high = (-(1 << 63), (1 << 63) - 1) if dtype == 'int64' else (0, 255)
        C.require(type(values[0]) is int and low <= values[0] <= high,
                  'REFERENCE_SCALAR_EXACT_INTEGER')
    else:
        C.require(dtype in ('uint8', 'float32', 'bfloat16'), 'REFERENCE_ARRAY_DTYPE')
        C.require(all(type(v) in (int, float) and math.isfinite(v) for v in values),
                  'REFERENCE_ARRAY_VALUES')
        if dtype == 'uint8':
            C.require(all(type(v) is int and 0 <= v <= 255 for v in values), 'REFERENCE_ARRAY_UINT8')
    return value


def scalar_roles(case):
    C.require(case in C.read(C.HERE / 'public-protocol.json')['cases'], 'REFERENCE_FIXED_CASE')
    count = math.prod(case['weight_shape'])
    C.require(count % 64 == 0, 'REFERENCE_FIXED_BLOCK_RECIPE')
    return {'blocksize': {'dtype': 'int64', 'shape': [], 'values': [64]},
            'scales_bound': {'dtype': 'int64', 'shape': [], 'values': [count // 64]},
            'codebook_bound': {'dtype': 'int64', 'shape': [], 'values': [16]},
            'nibble_mask': {'dtype': 'uint8', 'shape': [], 'values': [15]}}


def allowed_reference_values(original, case):
    # Keep every original allowed array unchanged. Add only fixed recipe scalars.
    return list(original) + list(scalar_roles(case).values())


def general_graph(proto):
    """Bind the ENTRY and each reachable reduction computation without native calls."""
    f=PC.fields(proto);entry=PC.one(f,6,0);comps={}
    for w,raw in f.get(3,[]):
        C.require(w==2,'REFERENCE_COMPUTATION_WIRE');comp=PC.fields(raw);cid=PC.one(comp,5,0);C.require(cid not in comps,'REFERENCE_COMPUTATION_ID');nodes={}
        for iw,b in comp.get(2,[]):
            n=PC.fields(b);i=PC.one(n,35,0);C.require(iw==2 and i not in nodes,'REFERENCE_NODE_ID');nodes[i]=n
        root=PC.one(comp,6,0);C.require(root in nodes,'REFERENCE_ROOT');comps[cid]=(nodes,root)
    C.require(entry in comps,'REFERENCE_ENTRY');active=set();reached=set();called=set();ops=[]
    def project(cid,i):
        key=(cid,i);nodes,_=comps[cid];C.require(key not in active and i in nodes,'REFERENCE_GRAPH_CYCLE_OR_ID');active.add(key);reached.add(key);n=nodes[i]
        C.require(not PC.ints(n,37),'REFERENCE_CONTROL_DEPENDENCIES_NOT_ADMITTED')
        op=PC.one(n,2,b'').decode();C.require(op!='custom-call','REFERENCE_NO_CUSTOM_CALL');ops.append(op)
        keep={str(k):[(w,v.hex() if isinstance(v,bytes) else v) for w,v in values] for k,values in n.items() if k not in (1,7,35,36,38)}
        keep['operands']=[project(cid,d) for d in PC.ints(n,36)]
        callees=PC.ints(n,38);C.require(not callees or op=='reduce','REFERENCE_CALLEE_OPCODE_SCOPE')
        keep['callees']=[]
        for callee in callees:
            C.require(callee in comps and callee!=cid,'REFERENCE_CALLEE_ID');called.add(callee);keep['callees'].append(project(callee,comps[callee][1]))
        active.remove(key);return keep
    nodes,root=comps[entry];canonical=project(entry,root)
    C.require(set(comps)=={entry}|called,'REFERENCE_UNREACHABLE_COMPUTATION')
    for cid,(cnodes,croot) in comps.items():C.require(all((cid,i) in reached for i in cnodes),'REFERENCE_DISCONNECTED_NODE')
    visible=root
    if PC.one(nodes[root],2,b'')==b'tuple':
        deps=PC.ints(nodes[root],36);C.require(len(deps)==1,'REFERENCE_SINGLE_OUTPUT_TUPLE');visible=deps[0]
    numbers=[PC.one(n,9,0) for i,n in nodes.items() if (entry,i) in reached and PC.one(n,2,b'')==b'parameter']
    C.require(len(numbers)==len(set(numbers)), 'REFERENCE_PARAMETER_NUMBER_UNIQUE')
    return {'canonical':canonical,'opcodes':ops,'shape':PC.shape(PC.one(nodes[visible],3,b'')),'entry_nodes':nodes,'parameters':{str(PC.one(n,9,0)):parameter_shape(PC.one(n,3,b'')) for i,n in nodes.items() if (entry,i) in reached and PC.one(n,2,b'')==b'parameter'}}



def bind_scalar_roles(graph, parameters, allowed, case, fixed_inputs):
    """Bind typed scalar literals to fixed-data consumer roles, independent of IDs."""
    scalar_ids = {key for key, value in parameters.items() if not value['shape']}
    if not scalar_ids:
        return []
    roles = scalar_roles(case)
    C.require(isinstance(fixed_inputs, dict) and {'packed', 'scales'} <= set(fixed_inputs),
              'REFERENCE_FIXED_ROLE_INPUTS')
    for name in ('packed', 'scales'):
        array(fixed_inputs[name])
    codebooks = [v for v in allowed if v['dtype'] == 'float32' and v['shape'] == [16]]
    C.require(codebooks and all(v == codebooks[0] for v in codebooks), 'REFERENCE_FIXED_CODEBOOK')
    targets = {'scales': fixed_inputs['scales'], 'packed': fixed_inputs['packed'],
               'codebook': codebooks[0]}
    nodes = graph['entry_nodes']
    op = lambda i: PC.one(nodes[i], 2, b'').decode()
    deps = lambda i: PC.ints(nodes[i], 36)
    users = {i: [] for i in nodes}
    for i in nodes:
        for slot, dep in enumerate(deps(i)):
            users[dep].append((i, slot))
    def ancestors(i):
        result = set()
        def visit(j):
            if j in result: return
            result.add(j)
            for k in deps(j): visit(k)
        visit(i)
        return result
    def origin(i):
        while op(i) in ('reshape', 'copy', 'broadcast', 'transpose') and len(deps(i)) == 1:
            i = deps(i)[0]
        return i
    def source(i, target):
        i = origin(i)
        if op(i) != 'parameter': return False
        value = parameters[str(PC.one(nodes[i], 9, 0))]
        return value['dtype'] == target['dtype'] and value['values'] == target['values']
    def gathers(consumer, target):
        permitted = {'divide', 'floor', 'convert', 'reshape', 'copy', 'broadcast',
                     'add', 'compare', 'select', 'concatenate', 'and', 'slice'}
        found = []
        for i in nodes:
            if op(i) != 'gather' or len(deps(i)) != 2 or not source(deps(i)[0], targets[target]):
                continue
            path = ancestors(deps(i)[1])
            if consumer not in path: continue
            dependent = [node for node in path if consumer in ancestors(node)]
            C.require(all(op(node) in permitted for node in dependent), 'REFERENCE_SCALAR_INDEX_LINEAGE_SCOPE')
            found.append(i)
        return found
    def scalar_terminals(start, role):
        pending = [(start, False)];visited = set();terminals = []
        source_dtype = 6 if role == 'nibble_mask' else 5
        while pending:
            i, converted = pending.pop()
            if (i, converted) in visited: continue
            visited.add((i, converted))
            for user, slot in users[i]:
                dtype = PC.one(PC.fields(PC.one(nodes[user], 3, b'')), 2, 0)
                if op(user) in ('reshape', 'copy', 'broadcast') and deps(user) == [i]:
                    C.require(dtype == (11 if converted else source_dtype), 'REFERENCE_SCALAR_VIEW_DTYPE')
                    pending.append((user, converted))
                elif op(user) == 'convert' and deps(user) == [i] and role == 'blocksize' and not converted:
                    C.require(dtype == 11, 'REFERENCE_BLOCKSIZE_CONVERT_F32')
                    pending.append((user, True))
                else:
                    terminals.append((user, slot, converted))
        return terminals
    def integer_shape(i):
        fields = PC.fields(PC.one(nodes[i], 3, b''))
        dims = PC.ints(fields, 3)
        if PC.one(fields, 2, 0) != 5 or set(fields) - {2, 3, 5, 6}: return None
        if any(d <= 0 for d in dims) or any(PC.ints(fields, 6)): return None
        layout = PC.fields(PC.one(fields, 5, b''))
        if set(layout) - {1, 7, 8, 15, 16}: return None
        if sorted(PC.ints(layout, 1)) != list(range(len(dims))): return None
        if PC.one(layout, 7, 0) not in (0, 64): return None
        return dims
    def zero(i):
        # Admit scalar broadcast/copy/reshape or an exact materialized S64 zero.
        while op(i) in ('reshape', 'copy', 'broadcast') and len(deps(i)) == 1:
            if integer_shape(i) is None: return False
            i = deps(i)[0]
        if op(i) != 'constant' or deps(i): return False
        dims = integer_shape(i)
        literal = PC.fields(PC.one(nodes[i], 8, b''))
        if dims is None or set(literal) != {1, 5}: return False
        if PC.one(literal, 1, b'') != PC.one(nodes[i], 3, b''): return False
        return PC.ints(literal, 5) == [0] * math.prod(dims)
    def index_origin(i):
        while op(i) in ('reshape', 'copy') and len(deps(i)) == 1:
            if integer_shape(i) is None: return None
            i = deps(i)[0]
        return i
    def preserves_index(i, selected):
        count = math.prod(integer_shape(selected))
        while i != selected:
            if op(i) not in ('reshape', 'copy', 'concatenate') or len(deps(i)) != 1: return False
            dims = integer_shape(i)
            if dims is None or math.prod(dims) != count: return False
            i = deps(i)[0]
        return True
    def bound_use(add, slot, gather):
        if op(add) != 'add' or len(deps(add)) != 2: return False
        index = index_origin(deps(add)[1 - slot])
        dims = integer_shape(index) if index is not None else None
        if dims is None or integer_shape(add) != dims: return False
        for node in ancestors(deps(gather)[1]):
            if op(node) != 'select' or len(deps(node)) != 3: continue
            if not preserves_index(deps(gather)[1], node): continue
            predicate, positive, negative = deps(node)
            if index_origin(positive) != add or index_origin(negative) != index: continue
            if integer_shape(node) != dims: continue
            if op(predicate) != 'compare' or len(deps(predicate)) != 2: continue
            pred_shape = PC.fields(PC.one(nodes[predicate], 3, b''))
            if PC.one(pred_shape, 2, 0) != 1 or PC.ints(pred_shape, 3) != dims: continue
            left, right = deps(predicate)
            if integer_shape(left) != dims or integer_shape(right) != dims: continue
            direction = PC.one(nodes[predicate], 63, b'')
            if PC.one(nodes[predicate], 72, b'') != b'SIGNED': continue
            # WrapIndicesOnce: where(index < S64(0), index + bound, index).
            # zero > index is the equivalent commuted predicate.
            if direction == b'LT' and index_origin(left) == index and zero(right): return True
            if direction == b'GT' and zero(left) and index_origin(right) == index: return True
        return False
    bindings = []
    for key in sorted(scalar_ids, key=int):
        value = parameters[key]
        matched = [role for role, literal in roles.items() if value == literal]
        C.require(len(matched) == 1, 'REFERENCE_FIXED_SCALAR_LITERAL')
        role = matched[0]
        starts = [i for i in nodes if op(i) == 'parameter' and str(PC.one(nodes[i], 9, 0)) == key]
        C.require(len(starts) == 1, 'REFERENCE_SCALAR_PARAMETER_UNIQUE')
        terminals = scalar_terminals(starts[0], role)
        C.require(terminals, 'REFERENCE_SCALAR_ROLE_USE')
        consumers = []
        for user, slot, converted in terminals:
            if role == 'blocksize':
                good = op(user) == 'divide' and slot == 1 and converted and gathers(user, 'scales')
            elif role == 'nibble_mask':
                good = op(user) == 'and' and len(deps(user)) == 2 and source(
                    deps(user)[1 - slot], targets['packed']) and gathers(user, 'codebook')
            else:
                target = 'scales' if role == 'scales_bound' else 'codebook'
                good = any(bound_use(user, slot, gather) for gather in gathers(user, target))
            C.require(good, 'REFERENCE_SCALAR_ROLE_USE:' + role)
            consumers.append({'opcode': op(user), 'operand_index': slot})
        bindings.append({'parameter_number': key, 'role': role, 'typed_literal_sha256': C.digest(value),
                         'consumers': sorted(consumers, key=lambda v: (v['opcode'], v['operand_index']))})
    return bindings


def verify_reference(context_proto, printer_proto, expected_shape, dtype, *, bias_gradient=False,
                     parameters=None, allowed_parameters=None, reference_case=None, fixed_inputs=None):
    a, b = general_graph(context_proto), general_graph(printer_proto)
    C.require(a['canonical'] == b['canonical'], 'REFERENCE_CONTEXT_PRINTER_CONTENT')
    C.require(isinstance(parameters, dict) and set(parameters) == set(a['parameters']), 'REFERENCE_PARAMETER_NUMBERS')
    C.require(isinstance(allowed_parameters, list) and allowed_parameters, 'REFERENCE_FIXED_INPUT_EXPECTATIONS')
    for key, value in parameters.items():
        array(value)
        meta = a['parameters'][key]
        C.require(value['dtype'] == meta['dtype'] and value['shape'] == meta['shape'], 'REFERENCE_PARAMETER_SHAPE_DTYPE')
        C.require(any(value['dtype'] == v['dtype'] and bool(value['shape']) == bool(v['shape']) and value['values'] == v['values'] for v in allowed_parameters),
                  'REFERENCE_FIXED_PARAMETER_CONTENT')
    bindings = bind_scalar_roles(a, parameters, allowed_parameters, reference_case, fixed_inputs)
    shape = a['shape']
    C.require('tuple' not in shape and shape['shape'] == expected_shape and shape['dtype'] == dtype,
              'REFERENCE_ACTUAL_OUTPUT_SHAPE')
    C.require(('reduce' if bias_gradient else 'dot') in a['opcodes'],
              'REFERENCE_REDUCTION_REQUIRED' if bias_gradient else 'REFERENCE_DENSE_DOT_REQUIRED')
    return {'record_validation': 'PASS', 'graph_scope': 'ACTUAL_REFERENCE_OR_GRADIENT_OUTPUT',
            'reference_observer_contract': CONTRACT, 'reference_observer_version': 3,
            'scalar_role_bindings': bindings, 'context_proto_sha256': C.sha_bytes(context_proto),
            'printer_proto_sha256': C.sha_bytes(printer_proto),
            'opcode_counts': {op: a['opcodes'].count(op) for op in sorted(set(a['opcodes']))},
            'executed_executable_link': 'UNKNOWN'}
