"""Compare native XLA precision in fresh processes; never qualify API42 or M3."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import uuid

ROUTE_SHA = 'feae751734e57c741b1bdade004ff7ca3c041ee7eb3b7086b531bc6be433038e'
MODES = ['default', 'high', 'highest']
ROUTES = ['target_constructor', 'from_prequantized']
CASE_IDS = ['linear-float32-rank2-bias1', 'linear-bfloat16-rank2-bias1']
ENV_KEYS = ['XLA_USE_BF16', 'XLA_DOWNCAST_BF16', 'XLA_USE_F32_FOR_BF16', 'XLA_FLAGS']
PROTOCOL = {'kind': 'PRECISION_DIAGNOSTIC_ONLY', 'modes': MODES, 'routes': ROUTES,
            'route_probe_sha256': ROUTE_SHA, 'case_ids': CASE_IDS,
            'plain_operands': 'sealed FP32 X, oracle-decoded FP32 W, sealed bias, fixed dyadic dy',
            'gradients': 'FP32 y/dX/db; BF16 y only, gradients NOT_RUN',
            'hlo_scope': 'PRE_OPTIMIZATION_OBSERVATION_NOT_FINAL_COMPILER_PROOF',
            'reference': 'sealed CPU y; CPU dense gradients with independent scalar checks',
            'processes': 'one parent and three sequential fresh children; one precision setter call per child',
            'qualification': 'API42 and M3 remain NOT_QUALIFIED'}
PROTOCOL_SHA = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def route_probe(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != ROUTE_SHA:
        raise ValueError('ROUTE_SOURCE_IDENTITY')
    spec = importlib.util.spec_from_file_location('precision_routes', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selection(B, R):
    _, cases = R.selection(B)
    B.require([c['id'] for c in cases] == CASE_IDS, 'PRECISION_MATRIX')
    B.require(all(len(c['x_shape']) == 2 and c['bias'] is not None for c in cases), 'PRECISION_CASES')
    return cases


def precision_environment():
    return {key: os.environ.get(key) for key in ENV_KEYS}


def validate_environment(B, environment):
    B.require(isinstance(environment, dict) and set(environment) == set(ENV_KEYS), 'PRECISION_ENVIRONMENT')
    B.require(all(value is None or isinstance(value, str) for value in environment.values()), 'PRECISION_ENVIRONMENT')
    for key in ENV_KEYS[:-1]:
        B.require(environment[key] is None or environment[key].lower() in ('', '0', 'false'), 'PRECISION_ENVIRONMENT_OVERRIDE')
    flags = environment['XLA_FLAGS'] or ''
    B.require(not re.search(r'bf16|precision|float32|matmul|(?:^|_)f32(?:_|=)', flags, re.I), 'PRECISION_ENVIRONMENT_OVERRIDE')


def split_hlo_arguments(value):
    result = []; start = 0; depth = 0
    for index, char in enumerate(value):
        if char in '[{(': depth += 1
        elif char in ']})': depth -= 1
        elif char == ',' and depth == 0:
            result.append(value[start:index].strip()); start = index+1
        if depth < 0: raise ValueError('HLO_ARGUMENTS')
    if depth: raise ValueError('HLO_ARGUMENTS')
    result.append(value[start:].strip())
    return result if value.strip() else []


def hlo_precision(text):
    """Read only single-line ENTRY nodes. Count dots that reach its ROOT outputs."""
    lines = text.splitlines()
    entries = [index for index, line in enumerate(lines) if re.match(r'^ENTRY\s', line)]
    if len(entries) != 1: raise ValueError('HLO_ENTRY')
    nodes = {}; root_name = None; closed = False
    for line in lines[entries[0]+1:]:
        if line.strip() == '}': closed = True; break
        line = re.sub(r'/\*.*?\*/', '', line).strip()
        if not line: continue
        binding = re.match(r'(ROOT\s+)?%?([\w.-]+)\s*=\s*(.*)', line)
        if binding is None: raise ValueError('HLO_NODE_FORMAT')
        name, body = binding.group(2), binding.group(3)
        operation = re.search(r'\b([a-z][a-z0-9-]*)\(', body)
        if operation is None or name in nodes: raise ValueError('HLO_NODE_FORMAT')
        depth = 1; end = operation.end()
        while end < len(body) and depth:
            if body[end] == '(': depth += 1
            elif body[end] == ')': depth -= 1
            end += 1
        if depth: raise ValueError('HLO_NODE_FORMAT')
        arguments = body[operation.end():end-1]
        opcode = operation.group(1)
        if opcode in ('call', 'fusion', 'while', 'conditional'): raise ValueError('HLO_UNSUPPORTED_COMPUTATION')
        shape = re.match(r'([a-z0-9]+)\[([0-9,]*)\]', body)
        dtype = shape.group(1) if shape else None
        dimensions = [int(value) for value in shape.group(2).split(',')] if shape and shape.group(2) else []
        args = split_hlo_arguments(arguments); dependencies = []
        if opcode not in ('constant', 'parameter', 'iota'):
            for arg in args:
                symbol = re.search(r'%?([\w.-]+)\s*$', arg)
                if symbol is None: raise ValueError('HLO_DEPENDENCY')
                dependencies.append(symbol.group(1))
        tuple_index = re.search(r'\bindex\s*=\s*(\d+)', body) if opcode == 'get-tuple-element' else None
        if opcode == 'get-tuple-element' and (tuple_index is None or len(dependencies) != 1): raise ValueError('HLO_TUPLE_INDEX')
        nodes[name] = {'dtype': dtype, 'shape': dimensions, 'opcode': opcode,
                       'dependencies': dependencies, 'line': line, 'body': body,
                       'tuple_index': int(tuple_index.group(1)) if tuple_index else None}
        if binding.group(1):
            if root_name is not None: raise ValueError('HLO_ROOT')
            root_name = name
    if not closed or root_name is None: raise ValueError('HLO_ROOT')
    for node in nodes.values():
        if node['opcode'] == 'get-tuple-element':
            producer = nodes.get(node['dependencies'][0])
            if producer is None: raise ValueError('HLO_DEPENDENCY')
            if producer['opcode'] == 'tuple':
                index = node['tuple_index']
                if index >= len(producer['dependencies']): raise ValueError('HLO_TUPLE_INDEX')
                node['dependencies'] = [producer['dependencies'][index]]
    reachable = set(); pending = [root_name]
    while pending:
        name = pending.pop()
        if name in reachable: continue
        if name not in nodes: raise ValueError('HLO_DEPENDENCY')
        reachable.add(name); pending.extend(nodes[name]['dependencies'])
    result = []
    for name, node in nodes.items():
        if node['opcode'] != 'dot' or name not in reachable: continue
        deps = node['dependencies']
        if len(deps) != 2 or any(dep not in nodes for dep in deps): raise ValueError('HLO_DOT_OPERANDS')
        match = re.search(r'operand_precision\s*=\s*\{([^}]+)\}', node['body'])
        operands = [value.strip().lower() for value in match.group(1).split(',')] if match else ['default', 'default']
        if len(operands) != 2 or any(value not in MODES for value in operands): raise ValueError('HLO_PRECISION_FORMAT')
        dims = {}
        for side in ('lhs', 'rhs'):
            contract = re.search(side+r'_contracting_dims\s*=\s*\{([^}]*)\}', node['body'])
            batch = re.search(side+r'_batch_dims\s*=\s*\{([^}]*)\}', node['body'])
            if contract is None or (batch and batch.group(1).strip()): raise ValueError('HLO_DOT_CONTRACTION')
            try: dims[side] = [int(value.strip()) for value in contract.group(1).split(',') if value.strip()]
            except ValueError: raise ValueError('HLO_DOT_CONTRACTION') from None
        result.append({'line': node['line'], 'node': name, 'entry_root': root_name, 'root_reachable': True,
                       'operands': operands, 'implicit_default': match is None,
                       'result_dtype': node['dtype'], 'result_shape': node['shape'],
                       'operand_dtypes': [nodes[dep]['dtype'] for dep in deps],
                       'operand_shapes': [nodes[dep]['shape'] for dep in deps],
                       'lhs_contracting_dims': dims['lhs'], 'rhs_contracting_dims': dims['rhs']})
    return result


def validate_dot_matrix(B, dot):
    lhs, rhs = dot['operand_shapes']
    ld, rd = dot['lhs_contracting_dims'], dot['rhs_contracting_dims']
    B.require(len(lhs) == len(rhs) == 2 and len(ld) == len(rd) == 1 and ld[0] in (0, 1) and rd[0] in (0, 1), 'HLO_DOT_CONTRACTION')
    B.require(lhs[ld[0]] == rhs[rd[0]] and dot['result_shape'] == [lhs[1-ld[0]], rhs[1-rd[0]]], 'HLO_DOT_CONTRACTION')


def scalar_forward(case, original):
    m, k = case['x_shape']; n = case['weight_shape'][0]
    x, w, bias = case['x'], original['outputs']['decoded']['values'], case['bias']
    return {'shape': [m, n], 'dtype': 'float32',
            'values': [sum(x[i*k+q]*w[j*k+q] for q in range(k))+bias[j] for i in range(m) for j in range(n)]}


def plain_operands(B, case, original):
    dy_shape = case['x_shape'][:-1] + [case['weight_shape'][0]]
    return {'x': {'shape': case['x_shape'], 'dtype': 'float32', 'values': case['x']},
            'weight': original['outputs']['decoded'],
            'bias': {'shape': [case['weight_shape'][0]], 'dtype': 'float32', 'values': case['bias']},
            'dy': {'shape': dy_shape, 'dtype': 'float32',
                   'values': [((i % 5)-2)/8 for i in range(math.prod(dy_shape))]}}


def plain_compute(case, original, torch, device):
    x = torch.tensor(case['x'], dtype=torch.float32, device=device).reshape(case['x_shape']).requires_grad_(True)
    wraw = original['outputs']['decoded']
    w = torch.tensor(wraw['values'], dtype=torch.float32, device=device).reshape(wraw['shape'])
    bias = torch.tensor(case['bias'], dtype=torch.float32, device=device, requires_grad=True)
    y = torch.matmul(x, w.t()) + bias
    dy = torch.tensor([((i % 5)-2)/8 for i in range(y.numel())], dtype=torch.float32, device=device).reshape(y.shape)
    y.backward(dy)
    return {'y': y, 'dx': x.grad, 'db': bias.grad}, {'x': x, 'weight': w, 'bias': bias, 'dy': dy}


def identities(B, args, profile):
    return {'kind': PROTOCOL['kind'], 'status': 'PARTIAL', 'protocol_sha256': PROTOCOL_SHA,
            'probe_sha256': B.sha(Path(__file__)), 'precision_probe_sha256': B.sha(Path(__file__)),
            'route_probe_sha256': ROUTE_SHA, 'backend_probe_sha256': B.sha(Path(args.backend_probe)),
            'profile_sha256': B.PROFILE_SHA, 'inputs_sha256': B.INPUTS_SHA,
            'source_admission_sha256': args.admission_sha256, 'oracle_sha256': args.oracle_sha256,
            'runtime_lock_sha256': profile['runtime_lock_sha256'], 'pid': os.getpid(),
            'pgid': os.getpgid(0), 'process_token': uuid.uuid4().hex, 'cuda_golden': 'NOT_RUN',
            'precision_environment': precision_environment(), 'api42_status': 'NOT_QUALIFIED',
            'm3_status': 'NOT_QUALIFIED', 'case_ids': CASE_IDS, 'routes': ROUTES}


def validate_identity(B, R, rec, args, profile):
    B.require(rec.get('kind') == PROTOCOL['kind'] and rec.get('status') == 'COMPLETE', 'PRECISION_TERMINAL')
    B.require(rec.get('protocol_sha256') == PROTOCOL_SHA, 'PRECISION_PROTOCOL')
    B.require(rec.get('probe_sha256') == rec.get('precision_probe_sha256') == B.sha(Path(__file__)), 'PRECISION_SOURCE')
    B.require(rec.get('route_probe_sha256') == ROUTE_SHA and rec.get('backend_probe_sha256') == R.BACKEND_SHA, 'HELPER_SOURCE')
    B.require(rec.get('source_pre') == rec.get('source_post') == rec.get('source_admission_sha256') == args.admission_sha256, 'SOURCE_BINDING')
    B.require(rec.get('oracle_sha256') == args.oracle_sha256, 'ORACLE_BINDING')
    B.require(rec.get('runtime') == profile['runtime'] and rec.get('runtime_lock_sha256') == profile['runtime_lock_sha256'], 'RUNTIME_PAIR')
    B.require(rec.get('profile_sha256') == B.PROFILE_SHA and rec.get('inputs_sha256') == B.INPUTS_SHA, 'DATA_BINDING')
    B.require(rec.get('case_ids') == CASE_IDS and rec.get('routes') == ROUTES, 'PRECISION_MATRIX')
    B.require(rec.get('cuda_golden') == 'NOT_RUN' and rec.get('api42_status') == rec.get('m3_status') == 'NOT_QUALIFIED', 'QUALIFICATION_SCOPE')
    B.require(type(rec.get('pid')) is int and rec['pid'] > 0 and type(rec.get('pgid')) is int and rec['pgid'] > 0, 'PROCESS_IDENTITY')
    B.require(isinstance(rec.get('process_token'), str) and re.fullmatch('[0-9a-f]{32}', rec['process_token']), 'PROCESS_TOKEN')
    validate_environment(B, rec.get('precision_environment'))


def validate_execution(B, raw):
    counters = raw.get('counters')
    B.require(isinstance(counters, dict) and all(isinstance(k, str) and type(v) is int and v >= 0 for k, v in counters.items()), 'COUNTERS_UNKNOWN')
    B.require(not any(k.startswith('aten::') and v > 0 for k, v in counters.items()), 'CPU_FALLBACK')
    metrics = raw.get('execution_metrics')
    B.require(isinstance(metrics, dict) and set(metrics).issubset({'ExecuteTime', 'ExecuteReplicatedTime'}), 'EXECUTION_METRICS')
    for data in metrics.values():
        B.require(isinstance(data, list) and len(data) == 3 and type(data[0]) is int and data[0] >= 0 and type(data[1]) in (int, float) and math.isfinite(data[1]) and data[1] >= 0, 'EXECUTION_METRICS')
        samples = data[2]
        B.require(isinstance(samples, list) and all(isinstance(pair, list) and len(pair) == 2 and all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in pair) for pair in samples), 'EXECUTION_METRIC_SAMPLES')
        B.require(data[0] == 0 or samples, 'EXECUTION_METRIC_SAMPLES')
    B.require(any(data[0] > 0 for data in metrics.values()), 'EXECUTION_NOT_OBSERVED')


def array_check(B, array, shape, dtype):
    B.require(isinstance(array, dict) and array.get('shape') == shape and array.get('dtype') == dtype, 'ARRAY_IDENTITY')
    values = array.get('values')
    B.require(isinstance(values, list) and len(values) == math.prod(shape) and all(type(v) in (int, float) and math.isfinite(v) for v in values), 'ARRAY_VALUES')
    if dtype == 'uint8': B.require(all(type(v) is int and 0 <= v <= 255 for v in values), 'ARRAY_UINT8')


def row_spec(cases):
    return [('plain', cases[0])] + [(route, case) for case in cases for route in ROUTES]


def verify_child(B, R, args, root, parent, descriptor, cases, profile):
    rec = B.read(root/'receipt.json')
    validate_identity(B, R, rec, args, profile)
    mode = descriptor['mode']
    B.require(rec.get('mode') == mode and rec.get('parent_pid') == parent['pid'] and rec.get('parent_process_token') == parent['process_token'], 'CHILD_PARENT')
    B.require(rec['pid'] == descriptor.get('pid') == descriptor.get('launched_pid') and rec['process_token'] == descriptor.get('process_token'), 'CHILD_LAUNCH_IDENTITY')
    B.require(rec['pgid'] == parent['pgid'] == descriptor.get('pgid'), 'CHILD_PROCESS_GROUP')
    B.require(rec['precision_environment'] == parent['precision_environment'], 'CHILD_ENVIRONMENT')
    B.require(rec.get('precision') == {'requested': mode, 'readback': mode, 'set_calls': 1, 'before_graph': True}, 'PRECISION_READBACK')
    B.require(rec.get('device') == {'type': 'xla', 'hardware': 'TPU', 'pjrt': 'TPU'}, 'TPU_REQUIRED')
    B.require(rec.get('public_api') == {'Linear4bit': 'bitsandbytes.nn.modules.Linear4bit', 'Params4bit': 'bitsandbytes.nn.modules.Params4bit', 'original_class_identity': True}, 'PUBLIC_API')
    B.require(rec.get('dispatch') == {op: True for op in B.DISPATCH_OPS}, 'DISPATCH')
    B.require(rec.get('bf16_gradients') == 'NOT_RUN', 'BF16_GRADIENT_SCOPE')
    B.check_seal(root, 'receipt.json', rec)
    required = {'references.json'}; refs = B.read(root/'references.json')
    B.require(set(refs) == set(CASE_IDS), 'REFERENCE_MATRIX')
    scalar_checks = {}
    for case in cases:
        original = B.read(Path(args.oracle)/'raw'/(case['id']+'.json'))
        witness = R.gradient_math(case, original)
        B.require(set(refs[case['id']]) == set(witness), 'REFERENCE_OUTPUT_SET')
        for name, expected in witness.items():
            array_check(B, refs[case['id']][name], expected['shape'], expected['dtype'])
            B.require(B.numeric(refs[case['id']][name]['values'], expected['values'], profile['tolerances']['fp32_forward_gradient'])['status'] == 'PASS', 'SCALAR_GRADIENT_REFERENCE')
        if case['dtype'] == 'float32':
            expected = scalar_forward(case, original)
            gate = B.numeric(original['outputs']['y']['values'], expected['values'], profile['tolerances']['fp32_forward_gradient'])
            B.require(gate['status'] == 'PASS', 'SCALAR_FORWARD_REFERENCE'); scalar_checks[case['id']] = gate
    rows = []
    for route, case in row_spec(cases):
        name = 'raw/'+route+'-'+case['id']; raw = B.read(root/(name+'.json'))
        required |= {name+suffix for suffix in ('.json', '.hlo.txt', '.metrics.txt')}
        B.require(raw.get('mode') == mode and raw.get('route') == route and raw.get('case_id') == case['id'] and raw.get('input_sha256') == B.case_sha(case), 'ROW_BINDING')
        B.require(raw.get('pid') == rec['pid'] and raw.get('process_token') == rec['process_token'] and raw.get('parent_pid') == parent['pid'] and raw.get('parent_process_token') == parent['process_token'], 'ROW_PROCESS')
        B.require(raw.get('status') == 'PASS' and raw.get('precision') == rec['precision'], 'ROW_PRECISION')
        B.require(raw.get('gradients') == ('EXECUTED' if case['dtype'] == 'float32' else 'NOT_RUN'), 'GRADIENT_SCOPE')
        original = B.read(Path(args.oracle)/'raw'/(case['id']+'.json'))
        expected = {'y': (case['x_shape'][:-1]+[case['weight_shape'][0]], case['dtype'])}
        if case['dtype'] == 'float32': expected.update(dx=(case['x_shape'], 'float32'), db=([case['weight_shape'][0]], 'float32'))
        if route != 'plain':
            expected.update(B.expected_outputs(case))
            base = {k: raw[k] for k in ('case_id', 'input_sha256', 'metadata', 'counters', 'execution_metrics')}
            base['outputs'] = {k: raw['outputs'][k] for k in B.expected_outputs(case)}
            base['placements'] = {k: raw['placements'][k] for k in B.expected_outputs(case)}
            B.validate_record(base, case, True)
            B.require(raw.get('original_classes') is True and raw.get('module_training') is True, 'ORIGINAL_CLASSES')
            B.require(raw.get('module_state_alias') is True or raw.get('module_state_absent') is True, 'STATE_ALIAS')
            B.require(raw.get('weight_requires_grad') is False and raw.get('weight_grad_is_none') is True and raw.get('bnb_quantized') is True, 'FROZEN_WEIGHT')
            B.require(raw.get('packed_before') == raw['outputs']['packed'] == raw.get('packed_after'), 'PACKED_CHANGED')
            B.require(raw.get('quantization_source') == R.PROTOCOL['quantization_source'][route], 'QUANTIZATION_SOURCE')
        else:
            B.require(raw.get('operands') == plain_operands(B, case, original), 'PLAIN_OPERAND_BINDING')
            B.require(set(raw.get('operand_placements', {})) == {'x', 'weight', 'bias', 'dy'} and all(v.startswith('xla:') for v in raw['operand_placements'].values()), 'PLAIN_OPERAND_PLACEMENT')
        B.require(set(raw.get('outputs', {})) == set(expected) and set(raw.get('placements', {})) == set(expected), 'OUTPUT_SET')
        validate_execution(B, raw)
        hlo = (root/(name+'.hlo.txt')).read_text(); parsed = hlo_precision(hlo)
        B.require(parsed and raw.get('hlo_precision') == parsed, 'HLO_RECORD')
        if case['dtype'] == 'float32':
            fp32_dots = [dot for dot in parsed if dot['result_dtype'] == 'f32' and dot['operand_dtypes'] == ['f32', 'f32']]
            for dot in fp32_dots: validate_dot_matrix(B, dot)
            B.require(fp32_dots and all(dot['operands'] == [mode, mode] for dot in fp32_dots), 'HLO_PRECISION')
            shapes = {tuple(dot['result_shape']) for dot in fp32_dots}
            B.require(tuple(case['x_shape']) in shapes and tuple(case['x_shape'][:-1]+[case['weight_shape'][0]]) in shapes, 'HLO_FP32_FORWARD_GRADIENT')
        B.require(raw.get('hlo_scope') == PROTOCOL['hlo_scope'], 'HLO_SCOPE')
        B.require((root/(name+'.metrics.txt')).stat().st_size > 0, 'MISSING_METRICS')
        references = dict(original['outputs'], **refs[case['id']]); gates = {}
        for key, (shape, dtype) in expected.items():
            array_check(B, raw['outputs'][key], shape, dtype)
            B.require(raw['placements'][key].startswith('xla:'), 'PLACEMENT')
            gates[key] = B.numeric(raw['outputs'][key]['values'], references[key]['values'], R.tolerance(profile, case, key))
        rows.append({'route': route, 'case_id': case['id'], 'gates': gates, 'gradients': raw['gradients'], 'hlo_precision': parsed})
    B.require(set(rec['artifacts']) == required, 'CHILD_ARTIFACT_MATRIX')
    return {'mode': mode, 'pid': rec['pid'], 'process_token': rec['process_token'], 'precision': rec['precision'],
            'numerical_status': 'FAIL' if any(gate['status'] == 'FAIL' for row in rows for gate in row['gates'].values()) else 'PASS',
            'scalar_reference_checks': scalar_checks, 'rows': rows}


def verify(B, R, args):
    profile, _ = B.load_spec(); cases = selection(B, R)
    B.verify_oracle(args.oracle, args.oracle_sha256, args.admission_sha256)
    root = Path(args.actual); parent = B.read(root/'receipt.json')
    validate_identity(B, R, parent, args, profile)
    B.require(parent.get('modes') == MODES, 'MODE_MATRIX')
    children = parent.get('child_processes')
    B.require(isinstance(children, list) and len(children) == 3 and [r.get('mode') for r in children] == MODES, 'MODE_MATRIX')
    pids = [parent['pid']] + [r.get('pid') for r in children]; tokens = [parent['process_token']] + [r.get('process_token') for r in children]
    B.require(all(type(pid) is int and pid > 0 for pid in pids) and len(set(pids)) == 4, 'FRESH_PID')
    B.require(all(isinstance(token, str) and re.fullmatch('[0-9a-f]{32}', token) for token in tokens) and len(set(tokens)) == 4, 'FRESH_PROCESS_TOKEN')
    B.check_seal(root, 'receipt.json', parent)
    required = set(); results = []
    for descriptor in children:
        mode = descriptor['mode']; child_root = root/'modes'/mode
        B.require(descriptor.get('receipt') == f'modes/{mode}/receipt.json' and descriptor.get('stdout') == f'processes/{mode}.stdout.log' and descriptor.get('stderr') == f'processes/{mode}.stderr.log', 'CHILD_PATH')
        B.require(descriptor.get('parent_pid') == parent['pid'] and descriptor.get('parent_process_token') == parent['process_token'], 'CHILD_PARENT')
        B.require(type(descriptor.get('exit_code')) is int and descriptor['exit_code'] == 0 and descriptor.get('reaped') is True, 'CHILD_EXIT')
        B.require(B.sha(child_root/'receipt.json') == descriptor.get('receipt_sha256'), 'CHILD_RECEIPT_BINDING')
        results.append(verify_child(B, R, args, child_root, parent, descriptor, cases, profile))
        required |= {descriptor['receipt'], descriptor['stdout'], descriptor['stderr']}
        required |= {f'modes/{mode}/'+p for p in B.read(child_root/'receipt.json')['artifacts']}
    B.require(set(parent['artifacts']) == required, 'PARENT_ARTIFACT_MATRIX')
    return {'record_validation': 'PASS', 'api42_status': 'NOT_QUALIFIED', 'm3_status': 'NOT_QUALIFIED',
            'kind': PROTOCOL['kind'], 'numerical_status': 'FAIL' if any(r['numerical_status'] == 'FAIL' for r in results) else 'PASS',
            'bf16_gradients': 'NOT_RUN', 'hlo_scope': PROTOCOL['hlo_scope'], 'modes': results,
            'scope': 'Precision diagnostic only. Owner supplies external TPU identity and lifecycle records.'}


def child(B, R, args):
    profile, _ = B.load_spec(); cases = selection(B, R)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    rec = identities(B, args, profile); rec.update(mode=args.mode, parent_pid=args.parent_pid, parent_process_token=args.parent_process_token, bf16_gradients='NOT_RUN')
    admission = None; roots = {}
    try:
        B.require(os.getppid() == args.parent_pid, 'ACTUAL_PARENT_PID')
        B.require(time.time() < args.deadline_epoch, 'OWNER_DEADLINE')
        validate_environment(B, rec['precision_environment'])
        B.verify_oracle(args.oracle, args.oracle_sha256, args.admission_sha256)
        admission, roots = B.admit(args.admission, args.admission_sha256); rec['source_pre'] = args.admission_sha256
        rec['runtime'] = B.runtime_check(True)
        import torch
        import torch_xla
        import torch_xla.backends as backends
        # Exactly one call, before device acquisition, model construction, or graph construction.
        backends.set_mat_mul_precision(args.mode)
        rec['precision'] = {'requested': args.mode, 'readback': backends.get_mat_mul_precision(), 'set_calls': 1, 'before_graph': True}
        B.require(rec['precision']['readback'] == args.mode, 'PRECISION_READBACK')
        import torch_xla.core.xla_model as xm
        import torch_xla.debug.metrics as metrics
        B.require(torch.__version__ == '2.9.0+cpu' and torch_xla.__version__.split('+')[0] == '2.9.0', 'RUNTIME_LOADED_PAIR')
        device = xm.xla_device(); B.require(device.type == 'xla' and xm.xla_device_hw(device) == 'TPU', 'ACTUAL_TPU_REQUIRED')
        rec['device'] = {'type': 'xla', 'hardware': 'TPU', 'pjrt': 'TPU'}
        import bitsandbytes as bnb
        rec['public_api'] = B.public_identity(bnb, roots)
        rec['dispatch'] = {op: torch._C._dispatch_has_kernel_for_dispatch_key(op, 'XLA') for op in B.DISPATCH_OPS}
        B.require(all(rec['dispatch'].values()), 'DISPATCH')
        originals = {c['id']: B.read(Path(args.oracle)/'raw'/(c['id']+'.json')) for c in cases}
        references = {c['id']: R.cpu_gradient(c, originals[c['id']], torch, B) for c in cases}
        B.write(output/'references.json', references)
        for route, case in row_spec(cases):
            B.require(time.time() < args.deadline_epoch, 'OWNER_DEADLINE')
            name = 'raw/'+route+'-'+case['id']; prefix = output/name; prefix.parent.mkdir(exist_ok=True)
            raw = {'mode': args.mode, 'route': route, 'case_id': case['id'], 'input_sha256': B.case_sha(case),
                   'pid': rec['pid'], 'process_token': rec['process_token'], 'parent_pid': args.parent_pid,
                   'parent_process_token': args.parent_process_token, 'precision': rec['precision'], 'status': 'PASS',
                   'gradients': 'EXECUTED' if case['dtype'] == 'float32' else 'NOT_RUN', 'hlo_scope': PROTOCOL['hlo_scope']}
            metrics.clear_all()
            if route == 'plain':
                outputs, operands = plain_compute(case, originals[case['id']], torch, device)
            else:
                saved = R.checkpoint(case, originals[case['id']], bnb, torch)
                module = R.make_route(route, case, saved, bnb, torch, device)
                B.require(type(module) is bnb.nn.Linear4bit and type(module.weight) is bnb.nn.Params4bit and type(module.weight.quant_state) is bnb.functional.QuantState, 'ORIGINAL_CLASSES')
                outputs, before = R.compute(case, module, bnb, torch, device)
                state = module.weight.quant_state; raw.update(R.base_raw(case, {'state': state}))
                raw.update(original_classes=True, module_training=module.training, module_state_alias=module.quant_state is state,
                           module_state_absent=module.quant_state is None, weight_requires_grad=module.weight.requires_grad,
                           weight_grad_is_none=module.weight.grad is None, bnb_quantized=module.weight.bnb_quantized,
                           quantization_source=R.PROTOCOL['quantization_source'][route])
            hlo = torch_xla._XLAC._get_xla_tensors_hlo(list(outputs.values()))
            prefix.with_suffix('.hlo.txt').write_text(hlo); raw['hlo_precision'] = hlo_precision(hlo)
            xm.mark_step(wait=True); xm.wait_device_ops()
            raw['counters'] = {key: metrics.counter_value(key) for key in metrics.counter_names()}
            raw['execution_metrics'] = {key: list(data) for key in profile['execution_metrics'] if (data := metrics.metric_data(key)) is not None}
            prefix.with_suffix('.metrics.txt').write_text(metrics.metrics_report())
            raw['placements'] = {key: str(value.device) for key, value in outputs.items()}
            raw['outputs'] = {key: B.tensor_record(value) for key, value in outputs.items()}
            if route == 'plain':
                raw['operands'] = {key: B.tensor_record(value) for key, value in operands.items()}
                raw['operand_placements'] = {key: str(value.device) for key, value in operands.items()}
            else:
                raw['packed_before'] = B.tensor_record(before); raw['packed_after'] = B.tensor_record(module.weight.data)
            B.write(prefix.with_suffix('.json'), raw)
        B.require(not any(name == 'jax' or name.startswith('jax.') or name.startswith('torchax') for name in sys.modules), 'UNEXPECTED_JAX_TORCHAX_IMPORT')
        rec['status'] = 'COMPLETE'
    except Exception:
        rec['status'] = 'FAILED'; (output/'error.log').write_text(traceback.format_exc())
    finally:
        if admission is not None:
            try:
                for package, root in roots.items(): B.verify_source_tree(root, admission[package]['files'])
                rec['source_post'] = args.admission_sha256
            except Exception:
                rec['status'] = 'FAILED'; (output/'source-post.error.log').write_text(traceback.format_exc())
        rec['artifacts'] = B.inventory(output, 'receipt.json'); B.write(output/'receipt.json', rec)
    return 0 if rec['status'] == 'COMPLETE' else 1


def reap(process, timeout):
    """Keep the inherited process group. Terminate and reap the direct child on timeout."""
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.terminate()
        try: process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(timeout=3)
        raise ValueError('CHILD_DEADLINE')
    finally:
        if process.poll() is None:
            process.kill(); process.wait(timeout=3)


def execute(B, R, args):
    profile, _ = B.load_spec(); selection(B, R)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    rec = identities(B, args, profile); rec.update(modes=MODES, child_processes=[])
    admission = None; roots = {}
    try:
        validate_environment(B, rec['precision_environment'])
        B.verify_oracle(args.oracle, args.oracle_sha256, args.admission_sha256)
        admission, roots = B.admit(args.admission, args.admission_sha256); rec['source_pre'] = args.admission_sha256
        rec['runtime'] = B.runtime_check(False)
        (output/'processes').mkdir(); (output/'modes').mkdir()
        for index, mode in enumerate(MODES):
            remaining = args.deadline_epoch-time.time()-10
            B.require(remaining > 0, 'OWNER_DEADLINE')
            duration = remaining/(len(MODES)-index); deadline = time.time()+duration
            child_output = output/'modes'/mode
            command = [sys.executable, '-B', str(Path(__file__).resolve()), 'child', '--mode', mode,
                       '--parent-pid', str(rec['pid']), '--parent-process-token', rec['process_token'],
                       '--backend-probe', str(Path(args.backend_probe).resolve()), '--route-probe', str(Path(args.route_probe).resolve()),
                       '--oracle', str(Path(args.oracle).resolve()), '--oracle-sha256', args.oracle_sha256,
                       '--admission', str(Path(args.admission).resolve()), '--admission-sha256', args.admission_sha256,
                       '--deadline-epoch', str(deadline), '--output', str(child_output.resolve())]
            stdout = f'processes/{mode}.stdout.log'; stderr = f'processes/{mode}.stderr.log'
            with (output/stdout).open('wb') as out, (output/stderr).open('wb') as err:
                process = subprocess.Popen(command, stdout=out, stderr=err)
                descriptor = {'mode': mode, 'pid': process.pid, 'launched_pid': process.pid,
                              'parent_pid': rec['pid'], 'parent_process_token': rec['process_token'],
                              'pgid': rec['pgid'], 'exit_code': None, 'reaped': False,
                              'receipt': f'modes/{mode}/receipt.json', 'stdout': stdout, 'stderr': stderr}
                rec['child_processes'].append(descriptor)
                try:
                    reap(process, max(0.001, deadline-time.time()))
                finally:
                    descriptor['exit_code'] = process.poll()
                    descriptor['reaped'] = process.poll() is not None
                    if (child_output/'receipt.json').is_file():
                        child_rec = B.read(child_output/'receipt.json')
                        descriptor.update(pid=child_rec['pid'], process_token=child_rec['process_token'],
                                          pgid=child_rec['pgid'], receipt_sha256=B.sha(child_output/'receipt.json'))
            B.require(descriptor['exit_code'] == 0, 'CHILD_EXECUTION_FAILED')
        rec['status'] = 'COMPLETE'
    except Exception:
        rec['status'] = 'FAILED'; (output/'error.log').write_text(traceback.format_exc())
    finally:
        if admission is not None:
            try:
                for package, root in roots.items(): B.verify_source_tree(root, admission[package]['files'])
                rec['source_post'] = args.admission_sha256
            except Exception:
                rec['status'] = 'FAILED'; (output/'source-post.error.log').write_text(traceback.format_exc())
        rec['artifacts'] = B.inventory(output, 'receipt.json'); B.write(output/'receipt.json', rec)
    if rec['status'] == 'COMPLETE':
        args.actual = output
        try: result = verify(B, R, args)
        except Exception:
            (output/'verifier.error.log').write_text(traceback.format_exc()); rec['status'] = 'FAILED'
            rec['artifacts'] = B.inventory(output, 'receipt.json'); B.write(output/'receipt.json', rec); return 1
        print(json.dumps(result)); return 0
    print(json.dumps({'record_validation': 'FAIL', 'api42_status': 'NOT_QUALIFIED', 'm3_status': 'NOT_QUALIFIED', 'status': rec['status']})); return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    for command in ('execute', 'verify', 'child'):
        q = sub.add_parser(command)
        for name in ('backend-probe', 'route-probe', 'admission-sha256', 'oracle', 'oracle-sha256'): q.add_argument('--'+name, required=True)
        if command in ('execute', 'child'):
            q.add_argument('--admission', required=True); q.add_argument('--deadline-epoch', type=float, required=True); q.add_argument('--output', required=True)
        else: q.add_argument('--actual', required=True)
        if command == 'child':
            q.add_argument('--mode', choices=MODES, required=True); q.add_argument('--parent-pid', type=int, required=True); q.add_argument('--parent-process-token', required=True)
    args = parser.parse_args(argv)
    try:
        R = route_probe(args.route_probe); B = R.backend(args.backend_probe)
        if args.command != 'verify': B.require(math.isfinite(args.deadline_epoch) and time.time() < args.deadline_epoch, 'OWNER_DEADLINE')
        if args.command == 'execute': return execute(B, R, args)
        if args.command == 'child': return child(B, R, args)
        print(json.dumps(verify(B, R, args))); return 0
    except Exception as error:
        print(json.dumps({'record_validation': 'FAIL', 'api42_status': 'NOT_QUALIFIED', 'm3_status': 'NOT_QUALIFIED', 'reason': str(error)})); return 1


if __name__ == '__main__': raise SystemExit(main())
