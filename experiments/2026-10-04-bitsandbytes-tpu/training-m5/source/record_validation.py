"""Recovered record checks. Numerical failures remain separate from record validity."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import admission as A
import scope_rational_reference as F
import graph_witness as G
import public_trace as T

def require(value, message):
    if not value:
        raise ValueError(message)

def process(receipt, group_leader=True):
    require(type(receipt.get('pid')) is int and receipt['pid'] > 0
            and type(receipt.get('pgid')) is int and receipt['pgid'] > 0
            and type(receipt.get('parent_pid')) is int and receipt['parent_pid'] > 0
            and re.fullmatch('[0-9a-f]{32}', receipt.get('process_token', '')) is not None, 'PROCESS_IDENTITY')
    if group_leader:
        require(receipt['pid'] == receipt['pgid'], 'PROCESS_GROUP_LEADER')

def complete_case(raw):
    require(raw.get('status') == 'PASS' and not any(key in raw for key in
            ('error_type', 'error_message', 'traceback', 'partial_snapshot', 'error_outputs')),
            'ERROR_BEARING_PASS_REJECTED')

def numerical_status(rows):
    require(type(rows) is list and rows and all(row.get('status') in ('PASS', 'FAIL', 'ERROR') for row in rows),
            'NUMERICAL_ROW_STATUS_MATRIX')
    return 'ERROR' if any(row['status'] == 'ERROR' for row in rows) else 'PASS' if all(row['status'] == 'PASS' for row in rows) else 'FAIL'

def validate_receipt(B, receipt, expected_binding, root, filename):
    require(type(receipt) is dict and all(receipt.get(k) == v for k, v in expected_binding.items()),
            'NEW_SCIENCE_BINDINGS')
    require(receipt.get('status') == 'COMPLETE'
            and receipt.get('source_pre') == receipt.get('source_post') == A.INTEGRATION_ADMISSION_SHA,
            'COMPLETE_SOURCE_AUDIT')
    process(receipt)
    times = ('deadline_epoch', 'started_epoch', 'finished_epoch', 'started_monotonic', 'finished_monotonic')
    require(all(type(receipt.get(k)) in (int, float) and math.isfinite(receipt[k]) and receipt[k] > 0 for k in times)
            and receipt['started_epoch'] <= receipt['finished_epoch'] <= receipt['deadline_epoch']
            and receipt['started_monotonic'] <= receipt['finished_monotonic'], 'FINITE_SCIENCE_DEADLINE_RECORD')
    require(receipt.get('public_api') is not None and receipt.get('schemas') == B.read(Path(__file__).with_name('nine-schemas.json')),
            'NINE_PUBLIC_SCHEMAS')
    require(receipt.get('plugin_loaded_code') == {'loaded_plugin_code': 'COMPILED_ADMITTED_SOURCE', 'python_files': 11},
            'LOADED_PLUGIN_COMPILED_CODE_PROOF')
    B.check_seal(root, filename, receipt)
    return receipt

def cpu_runtime(B, seal):
    require(seal.get('runtime') == B.load_spec()[0]['runtime'], 'EXACT_RETAINED_CPU_RUNTIME')

def cpu_evidence(evidence):
    require(evidence.get('python') == '3.12.14' and evidence.get('torch') == '2.9.0+cpu'
            and evidence.get('platform') == 'Linux' and evidence.get('machine') == 'x86_64'
            and evidence.get('torch_git') == '0fabc3ba44823f257e70ce397d989c8de5e362c1'
            and evidence.get('capability') == 'AVX2' and evidence.get('num_threads') == 1
            and evidence.get('xla_or_jax_imported') is False
            and isinstance(evidence.get('build'), str) and evidence['build'].strip(), 'QUALIFIED_CPU_SOURCE_CAPABILITY')

def verify_cpu(B, P, N, root, expected_sha, expected_binding):
    root = Path(root)
    require(B.sha(root / 'oracle-seal.json') == expected_sha, 'FRESH_ORACLE_SHA')
    seal = validate_receipt(B, B.read(root / 'oracle-seal.json'), expected_binding, root, 'oracle-seal.json')
    cpu_runtime(B, seal)
    require(seal.get('public_api') == N.PUBLIC, 'CPU_ORIGINAL_PUBLIC_METHOD_IDENTITY')
    require(seal.get('phase') == 'CPU_PREPARE'
            and seal.get('oracle_method') == 'EXPLICIT_PINNED_DEFAULT_BODIES_NO_NATIVE_DISPATCH', 'CPU_DEFAULT_METHOD')
    admission = A.admission(root / 'source-admission.json', A.INTEGRATION_ADMISSION_SHA)
    evidence = B.read(root / 'cpu-evidence.json')
    cpu_evidence(evidence)
    require(B.sha(root / 'inputs.json') == admission['inputs_sha256'], 'ORIGINAL_INPUT_SNAPSHOT')
    data = B.read(root / 'inputs.json'); rows, total = {}, 0
    required = {'source-admission.json', 'inputs.json', 'cpu-evidence.json'}
    for case in data['cases']:
        relative = 'raw/' + case['id'] + '.json'; required.add(relative)
        raw = B.read(root / relative)
        require(raw.get('case_id') == case['id'] and raw.get('input_sha256') == B.case_sha(case)
                and raw.get('pid') == seal['pid'] and raw.get('process_token') == seal['process_token'], 'CPU_ROW_IDENTITY')
        complete_case(raw)
        N.validate_arrays(B, P, case, raw, False)
        F.audit_row(case, data, raw)
        profile, _ = B.load_spec()
        for name, witness in N.scalar_witness(case, raw).items():
            require(B.numeric(raw['outputs'][name]['values'], witness, N.tolerance(profile, case, name))['status'] == 'PASS',
                    'INDEPENDENT_DENSE_CPU_WITNESS')
        rows[case['id']] = raw; total += len(raw['outputs'])
    require(len(rows) == 79 and total == 540 and set(seal['artifacts']) == required, 'COMPLETE_CPU_79_540_MATRIX')
    return seal, rows, {'record_validation': 'PASS', 'oracle_status': 'INDEPENDENTLY_VERIFIED',
                       'oracle_sha256': expected_sha, 'case_count': 79, 'array_count': 540,
                       'runtime_scope': 'RECORDED_LINUX_CPU_CONTRACT_REQUIRES_SEPARATE_OWNED_LAUNCH_PROOF',
                       'source_variant': A.VARIANT, 'm4_status': 'NOT_QUALIFIED'}

def validate_cpu_launch(seal, ownership, result, expected_argv):
    """Must use recovered outer launch/closure bytes, not a scientific label."""
    require(ownership.get('argv') == result.get('argv') == expected_argv, 'CPU_ACTUAL_ARGV')
    require(expected_argv.count('--process-token') == expected_argv.count('--deadline-epoch') == 1, 'CPU_SCIENCE_ARGV_FIELDS')
    token_index, deadline_index = expected_argv.index('--process-token'), expected_argv.index('--deadline-epoch')
    require(token_index + 1 < len(expected_argv) and deadline_index + 1 < len(expected_argv)
            and expected_argv[token_index + 1] == seal['process_token'], 'CPU_ACTUAL_LAUNCH_TOKEN')
    require(float(expected_argv[deadline_index + 1]) == seal['deadline_epoch'], 'CPU_ACTUAL_LAUNCH_DEADLINE')
    require(seal['pid'] == seal['pgid'] == ownership.get('pid') == ownership.get('pgid')
            and seal['parent_pid'] == ownership.get('owner_pid'), 'CPU_ACTUAL_OUTER_IDENTITY')
    cleanup = ownership.get('cleanup', {})
    require(cleanup.get('pid') == cleanup.get('pgid') == seal['pid']
            and cleanup.get('status') == 'CLEANUP_VERIFIED' and cleanup.get('leader_reaped') is True
            and cleanup.get('group_absence') == 'OBSERVED_NO_SUCH_GROUP' and cleanup.get('errors') == []
            and cleanup.get('exit_status') == ownership.get('exit_status') == 0, 'CPU_OUTER_CLOSURE')
    summary = result.get('cleanup', {})
    require(result.get('status') == 'PASS' and result.get('exit_code') == 0 and not result.get('error')
            and summary.get('errors') == [] and summary.get('groups') == [cleanup]
            and summary.get('restoration', {}).get('handlers_restored') is True
            and summary.get('restoration', {}).get('timer_restored') is True, 'CPU_OUTER_RESULT')

def frame_parameter_roles(raw, expected_inputs, scalar_literals):
    ids = raw.get('input_parameter_ids')
    require(type(ids) is list and len(ids) == len(expected_inputs)
            and all(type(i) is int and i >= 0 for i in ids) and len(set(ids)) == len(ids), 'COMPOSITE_ARGUMENT_PARAMETER_IDS')
    require(type(raw.get('parameters')) is dict and set(map(str, ids)) <= set(raw['parameters']), 'COMPOSITE_PARAMETERS_PRESENT')
    roles, gates = {}, {}
    for index, (parameter, wanted) in enumerate(zip(ids, expected_inputs)):
        got = raw['parameters'][str(parameter)]
        require(G.array_signature(got) == G.array_signature(wanted), 'COMPOSITE_ARGUMENT_PARAMETER_TYPE')
        roles[str(parameter)] = {'role': 'ORIGINAL_POSITIONAL_ARGUMENT', 'argument': index}
        gates['argument_' + str(index)] = 'PASS' if got == wanted else 'FAIL'
    for key in set(raw['parameters']) - set(map(str, ids)):
        got = raw['parameters'][key]
        require(got['dtype'] == 'int32' and got['shape'] == [] and len(got['values']) == 1
                and type(got['values'][0]) is int and got['values'][0] in scalar_literals,
                'UNADMITTED_COMPOSITE_AUXILIARY_PARAMETER')
        roles[key] = {'role': 'ADMITTED_SOURCE_INTEGER_SCALAR', 'value': got['values'][0],
                      'creator_identity': 'NOT_CLAIMED_VALUE_AND_SOURCE_DOMAIN_ONLY'}
    return roles, gates

def literals(plugin_root, count):
    """Source integer literals/closed expressions and declared count-derived scalars only."""
    require(type(count) is int and (1 <= count <= 513 or count == 1024), 'FIXED_COMPOSITE_COUNT')
    def closed(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Invert)):
            v = closed(node.operand)
            return -v if isinstance(node.op, ast.USub) else v if isinstance(node.op, ast.UAdd) else ~v
        if isinstance(node, ast.BinOp):
            a, b = closed(node.left), closed(node.right)
            require(abs(a) < 1 << 63 and abs(b) < 1 << 63, 'SOURCE_LITERAL_BOUND')
            if isinstance(node.op, ast.LShift) and 0 <= b <= 32: return a << b
            if isinstance(node.op, ast.RShift) and 0 <= b <= 32: return a >> b
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.BitAnd): return a & b
            if isinstance(node.op, ast.BitOr): return a | b
            if isinstance(node.op, ast.BitXor): return a ^ b
        raise ValueError('NOT_CLOSED_INTEGER_SOURCE_EXPRESSION')
    log_count = count.bit_length() - 1
    # The only nonliteral full_like scalar in count division is log_count. Threshold
    # and shift are static expressions of that same reviewed positive count.
    result = {count, log_count, 23 - log_count, count << (23 - log_count)}
    # Python loops feed bit positions 0..26 to tensor integer shifts in add/mul/div.
    result.update(range(27))
    for name in ('exact.py', '_integer_clamp.py', '_integer_addsub.py', '_integer_muldiv.py', '_source_order.py'):
        for node in ast.walk(ast.parse((Path(plugin_root) / name).read_text())):
            try:
                value = closed(node)
            except ValueError:
                continue
            if -(1 << 31) <= value < 1 << 31:
                result.add(value)
    return result

def composite_record(B, P, directory, expected_inputs, plugin_map, scalar_literals):
    directory = Path(directory); raw = B.read(directory / 'raw.json')
    require(set(B.inventory(directory, '')) == {'raw.json', 'context.hlo.pb', 'printer.hlo.pb',
                                              'graph.hlo.txt', 'context.textproto', 'metrics.txt'},
            'COMPOSITE_COMPLETE_ARTIFACT_SET')
    require(raw.get('status') == 'OBSERVED' and raw.get('calls') == 1
            and raw.get('method_replacement') is False and raw.get('tensor_order') == 'ORIGINAL_POSITIONAL_ARGUMENTS'
            and raw.get('parameter_mapping_api') == 'LoweringContext.device_parameter_id_tensor_mapping+tensor_parameter_id',
            'OBSERVED_ORIGINAL_COMPOSITE_CALL')
    require(not any(key in raw for key in ('error_type', 'error_message', 'traceback')), 'ERROR_BEARING_OBSERVED_COMPOSITE')
    require(raw.get('tls_before') == raw.get('tls_after') and raw.get('frame_errors') == []
            and raw.get('profile_restored') is True, 'COMPOSITE_TLS_PROFILE_RESTORATION')
    for got, wanted in zip(raw.get('inputs', []), expected_inputs):
        require(G.array_signature(got) == G.array_signature(wanted), 'COMPOSITE_INPUT_STRUCTURE')
    require(len(raw.get('inputs', [])) == len(expected_inputs), 'COMPOSITE_INPUT_COUNT')
    for array in raw['inputs'] + raw['outputs'] + list(raw['parameters'].values()):
        P.array_check(B, array, array['shape'], array['dtype'])
        # N.binary-compatible independent byte hash, including signed zero.
        import struct
        dtype = array['dtype']; formats = {'int32': 'i', 'uint8': 'B', 'float32': 'f', 'bool': '?'}
        require(dtype in formats, 'COMPOSITE_ARRAY_DTYPE')
        encoded = b''.join(struct.pack('<' + formats[dtype], v) for v in array['values'])
        require(hashlib.sha256(encoded).hexdigest() == array['bytes_sha256'], 'COMPOSITE_ARRAY_BYTES')
    roles, gates = frame_parameter_roles(raw, expected_inputs, scalar_literals)
    actual_roles, actual_gates = frame_parameter_roles(raw, raw['inputs'], scalar_literals)
    require(roles == actual_roles == raw.get('live_parameter_roles')
            and all(value == 'PASS' for value in actual_gates.values()), 'ACTUAL_ARGUMENT_GRAPH_VALUE_BINDING')
    profile = T.validate_frames(raw['frames'], raw['schema_name'], plugin_map)
    require(profile == raw.get('live_functionalization'), 'LIVE_RECOVERED_FUNCTIONALIZATION_WITNESS')
    for function, prefix in [('wrapped', 'inputs'), (raw['schema_name'].split('::')[-1], 'inputs')]:
        event = next(x for x in raw['frames'] if x['function'] == function)
        require(len(event[prefix]) == len(raw['inputs']) and all(
                meta['shape'] == array['shape'] and meta['dtype'] == array['dtype']
                and meta['device'].startswith('xla:') for meta, array in zip(event[prefix], raw['inputs'])),
                'SOURCE_FRAME_ARGUMENT_TYPES')
    wrapper = next(x for x in raw['frames'] if x['function'] == 'wrapped')
    require(len(wrapper['outputs']) == len(raw['outputs']) and all(
            meta['shape'] == array['shape'] and meta['dtype'] == array['dtype']
            and meta['device'].startswith('xla:') for meta, array in zip(wrapper['outputs'], raw['outputs'])),
            'SOURCE_FRAME_OUTPUT_TYPES')
    witness = G.bind((directory / 'context.hlo.pb').read_bytes(), (directory / 'printer.hlo.pb').read_bytes(),
                     (directory / 'graph.hlo.txt').read_text(), raw['parameters'],
                     [G.array_signature(a) for a in raw['outputs']])
    require(raw['graph_witness'] == witness, 'RECOVERED_GRAPH_WITNESS')
    P.validate_execution(B, raw)
    execute = raw.get('execution', {})
    require(execute.get('api') == '_xla_sync_multi' and execute.get('targets') == 'ACTUAL_COMPOSITE_OUTPUTS_ONLY'
            and execute.get('wait') is True and execute.get('wait_device_ops') is True
            and execute.get('metrics_before_host_extraction') is True
            and all(type(execute.get(k)) in (int, float) and math.isfinite(execute[k]) for k in ('started_monotonic', 'finished_monotonic'))
            and execute['started_monotonic'] <= execute['finished_monotonic'], 'COMPOSITE_EXECUTION_BOUNDARY')
    return raw, {'record_validation': 'PASS', 'parameter_roles': roles, 'input_gates': gates,
                 'functionalization': profile, 'graph': witness}
