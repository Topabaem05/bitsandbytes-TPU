"""Portable controls. Historical CPU data is numerical comparison only, never resealed."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys
import unittest

parser = argparse.ArgumentParser()
for name in ('repo', 'generation', 'oracle', 'readback-lock', 'mul-reference', 'output'):
    parser.add_argument('--' + name, type=Path, required=True)
parser.add_argument('--old-failed-receipt', type=Path, required=True)
args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exact_admission as A
import rational_reference as F
import graph_witness as G
import public_trace as T
import record_validation as V
import probe_nested_exact as E
attempts = {'network': 0, 'credentials': 0}
def audit(event, values):
    if event in ('socket.connect', 'socket.getaddrinfo'):
        attempts['network'] += 1; raise ValueError('NETWORK_FORBIDDEN')
    if event == 'open' and isinstance(values[0], (str, bytes)) and any(x in os.fsdecode(values[0]) for x in ('credentials.json', 'access_token', '/.kaggle/')):
        attempts['credentials'] += 1; raise ValueError('CREDENTIAL_FORBIDDEN')
sys.addaudithook(audit)

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module); return module
EX = args.repo / 'experiments/2026-10-04-bitsandbytes-tpu'
for key, basename in [('backend_probe', 'probe_backend.py'), ('nested_reference', 'probe_nested.py'),
                      ('precision_probe', 'probe_precision.py'), ('route_probe', 'probe_routes.py'),
                      ('transfer_admission', 'transfer_admission.py')]:
    if A.sha(EX / basename) != E.PINS[key]:
        raise ValueError('CONTROL_IMMUTABLE_HELPER_' + key)
B = load(EX / 'probe_backend.py', 'exact_science_controls_backend')
N = load(EX / 'probe_nested.py', 'exact_science_controls_original_cases')
P = load(EX / 'probe_precision.py', 'exact_science_controls_precision')
data = B.read(HERE / 'nested-inputs.json')
records = []

def vi(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128); value >>= 7
    return bytes(result) + bytes([value])
def scalar(n, value):
    return vi(n << 3) + vi(value)
def blob(n, value):
    return vi((n << 3) | 2) + vi(len(value)) + value
def shape(dtype, dims):
    return scalar(2, dtype) + b''.join(scalar(3, x) for x in dims)
def instruction(ident, opcode, dtype, dims, deps=(), parameter=None):
    result = blob(1, ('n' + str(ident)).encode()) + blob(2, opcode.encode()) + blob(3, shape(dtype, dims)) + scalar(35, ident)
    result += b''.join(scalar(36, x) for x in deps)
    if parameter is not None:
        result += scalar(9, parameter)
    return result
def fixture_graph(float_add=False, int64=False):
    dtype = 5 if int64 else 4
    nodes = [instruction(1, 'parameter', 11, [1], parameter=0),
             instruction(2, 'bitcast-convert', dtype, [1], [1]),
             instruction(3, 'bitcast-convert', 11, [1], [2])]
    if float_add:
        nodes.append(instruction(4, 'add', 11, [1], [3, 3])); root = 4
    else:
        root = 3
    comp = b''.join(blob(2, n) for n in nodes) + scalar(5, 10) + scalar(6, root)
    proto = blob(3, comp) + scalar(6, 10)
    hilo = 'HloModule synthetic\nENTRY main {\n p = f32[1] parameter(0)\n i = ' + ('s64' if int64 else 's32') + '[1] bitcast-convert(p)\n'
    hilo += ' f = f32[1] bitcast-convert(i)\n ROOT result = f32[1] add(f, f)\n}\n' if float_add else ' ROOT f = f32[1] bitcast-convert(i)\n}\n'
    return proto, hilo

def deep_graph(depth, diamond=False, cycle=False, metadata=False):
    nodes = [instruction(1, 'parameter', 11, [1], parameter=0), instruction(2, 'bitcast-convert', 4, [1], [1])]
    lines = ['HloModule synthetic_deep', 'ENTRY main {', ' p = f32[1] parameter(0)', ' i = s32[1] bitcast-convert(p)']
    previous, previous_name = 2, 'i'
    for j in range(depth):
        ident = j + 3; opcode = 'add' if diamond else 'copy'
        deps = [previous, previous] if diamond else [previous]
        if cycle and j == 0:
            deps = [ident]
        nodes.append(instruction(ident, opcode, 4, [1], deps))
        names = [previous_name, previous_name] if diamond else [previous_name]
        if cycle and j == 0: names = ['n' + str(ident)]
        lines.append(' n' + str(ident) + ' = s32[1] ' + opcode + '(' + ', '.join(names) + ')')
        previous, previous_name = ident, 'n' + str(ident)
    nodes.append(instruction(previous + 1, 'bitcast-convert', 11, [1], [previous]))
    if metadata: nodes[0] += blob(7, b'opaque debug metadata ignored by graph-content comparison')
    comp = b''.join(blob(2, n) for n in nodes) + scalar(5, 10) + scalar(6, previous + 1)
    lines.extend([' ROOT result = f32[1] bitcast-convert(' + previous_name + ')', '}'])
    return blob(3, comp) + scalar(6, 10), '\n'.join(lines) + '\n'

class Controls(unittest.TestCase):
    def test_01_literal_source_and_old_admission_rejection(self):
        admission = A.admission(args.generation / 'source-admission.json', A.INTEGRATION_ADMISSION_SHA)
        result = A.verify_installed(admission, {'bitsandbytes': args.generation / 'build-v3/source/bitsandbytes',
                                              'bitsandbytes_tpu': args.generation / 'plugin/src/bitsandbytes_tpu'})
        self.assertEqual(result, {'upstream_package_members': 48, 'upstream_python_members': 47, 'plugin_python_members': 11})
        for change in ('generation', 'plugin_python_files', 'device_acceptance'):
            bad = copy.deepcopy(admission)
            bad[change] = 'wrong' if change == 'generation' else {} if change == 'plugin_python_files' else {'status': 'PASS'}
            p = args.output / (change + '.json'); B.write(p, bad)
            with self.assertRaisesRegex(ValueError, 'EXACT_ADMISSION_LITERAL'):
                A.admission(p, B.sha(p))
        self.assertEqual(A.sha(args.generation / 'adoption-map.json'), A.INTEGRATION_MAP_SHA)
        records.append({'control': 'EXACT48_SOURCE_47PY_PLUS_TYPED_11PY_OLD_WRONG_REJECTION', 'status': 'PASS'})

    def test_02_all79_readonly_independent_rational_comparison(self):
        self.assertEqual(A.sha(args.readback_lock), '7799ac334c5bffcec9c348217ba33ce80a433664bd82dbb621ada77d4079a52c')
        lock = B.read(args.readback_lock)['files']; count = 0
        for case in data['cases']:
            path = args.oracle / 'raw' / (case['id'] + '.json')
            relative = '.work/cloud-20261008-nested-browser-adopted/recovered/cpu-oracle/' + path.relative_to(args.oracle).as_posix()
            self.assertEqual(B.sha(path), lock[relative]['sha256'])
            raw = B.read(path); N.validate_arrays(B, P, case, raw, False)
            result = F.audit_row(case, data, raw); count += result['quantized_array_count']
            for name, expected in N.scalar_witness(case, raw).items():
                self.assertEqual(B.numeric(raw['outputs'][name]['values'], expected, N.tolerance(B.load_spec()[0], case, name))['status'], 'PASS')
        records.append({'control': 'ALL79_READONLY_RATIONAL_QUANTIZATION_AND_SCALAR_DENSE', 'cases': 79,
                        'quantized_arrays': count, 'status': 'PASS', 'new_source_oracle': 'NOT_CREATED', 'TPU': 'NOT_RUN'})

    def test_03_mean_source_order_and_altered_operations(self):
        reference = load(args.mul_reference, 'science_control_fraction')
        self.assertEqual(A.sha(Path(reference.__file__)), '535fc0d060b13c40952e4aec18011c96d0a0337976a5aa02e121dd97f4ef7c38')
        pairs = [(1, 0x3f800000), (0x006ce3ee, 0x3f000000), (0x7fffff, 0x3f000000), (0x00800000, 0x3f800001),
                 (0x3f800001, 0x3f800001), (0x80000000, 0x3f800000)]
        for a, b in pairs:
            unsigned = lambda n: n & 0xffffffff
            self.assertEqual(F.mul(a, b), unsigned(reference.multiply(unsigned(a) if a < 1 << 31 else a - (1 << 32),
                                                                    unsigned(b) if b < 1 << 31 else b - (1 << 32))))
            self.assertEqual(F.div(a, b), unsigned(reference.divide(a if a < 1 << 31 else a - (1 << 32), b)))
        full = F.mul(0x437d0000, F.div(0x3f800000, 0x437f0000))
        direct = F.div(0x437d0000, 0x437f0000)
        self.assertNotEqual(full, direct)
        self.assertEqual(F.add(0x80000000, 0x80000000), 0x80000000)
        self.assertEqual(F.sub(0x80000000, 0x80000000), 0)
        self.assertEqual(F.source_sum([0x3f800000] * 513), F.word(513))
        with self.assertRaises(ValueError):
            F.source_sum([0x3f800000] * 1024)
        case = next(c for c in data['cases'] if c['kind'] == 'nested' and any(x != 0 for x in c['weight']))
        raw = B.read(args.oracle / 'raw' / (case['id'] + '.json')); changed = copy.deepcopy(raw)
        changed['outputs']['scale_codes']['values'][0] ^= 1
        with self.assertRaisesRegex(ValueError, 'INDEPENDENT_RATIONAL_scale_codes'):
            F.audit_row(case, data, changed)
        records.append({'control': 'RNE_ZERO_SUBNORMAL_SOURCE_ORDER_RECIPROCAL_NOT_DIRECT_AND_WRONG_CODE', 'status': 'PASS'})

    def test_04_actual_proto_hlo_binding_and_negative_graphs(self):
        proto, hlo = fixture_graph()
        parameter = E.array_words([0x3f800000], [1])
        report = G.bind(proto, proto, hlo, {'0': parameter}, [('f32', (1,))])
        self.assertTrue(report['no_float_arithmetic']); self.assertEqual(report['bitcast_count'], 2)
        for float_add, int64 in ((True, False), (False, True)):
            bad_proto, bad_hlo = fixture_graph(float_add, int64)
            with self.assertRaises(ValueError):
                G.bind(bad_proto, bad_proto, bad_hlo, {'0': parameter}, [('f32', (1,))])
        with self.assertRaisesRegex(ValueError, 'HLO_PROTO_STRUCTURE_BINDING'):
            G.bind(proto, proto, hlo.replace('bitcast-convert(i)', 'copy(i)'), {'0': parameter}, [('f32', (1,))])
        with self.assertRaisesRegex(ValueError, 'CONTEXT_PRINTER_PROTO_CONTENT'):
            G.bind(proto, fixture_graph(True)[0], hlo, {'0': parameter}, [('f32', (1,))])
        with self.assertRaisesRegex(ValueError, 'GRAPH_PARAMETER_TYPES'):
            G.bind(proto, proto, hlo, {'0': parameter, '1': parameter}, [('f32', (1,))])
        records.append({'control': 'SYNTHETIC_PROTO_HLO_BITCAST_FLOAT_INT64_EXTRA_PARAMETER_NEGATIVES', 'status': 'PASS'})

    def test_04b_deep_iterative_dag_cycle_and_debug_metadata(self):
        parameter = E.array_words([0x3f800000], [1])
        for diamond in (False, True):
            proto, hlo = deep_graph(1500, diamond)
            report = G.bind(proto, proto, hlo, {'0': parameter}, [('f32', (1,))])
            self.assertEqual(report['maximum_dag_depth'], 1503)
            self.assertEqual(report['node_count'], 1503)
            with_metadata, _ = deep_graph(1500, diamond, metadata=True)
            self.assertEqual(G.module(proto)[2], G.module(with_metadata)[2])
            G.bind(proto, with_metadata, hlo, {'0': parameter}, [('f32', (1,))])
        proto, hlo = deep_graph(2, cycle=True)
        with self.assertRaisesRegex(ValueError, 'GRAPH_CYCLE'):
            G.module(proto)
        with self.assertRaisesRegex(ValueError, 'GRAPH_CYCLE'):
            G.structural(hlo, {'0': parameter}, [('f32', (1,))])
        records.append({'control': '1503_DEPTH_ITERATIVE_CHAIN_DIAMOND_CYCLE_AND_DEBUG_METADATA', 'status': 'PASS',
                        'actual_HLO_depth': 'NOT_OBSERVED'})

    def test_05_metrics_tuple_roundtrip_and_original_validator(self):
        evidence = {'counters': {}, 'execution_metrics': {'ExecuteTime': (1, 0.2, ((1.0, 0.2),))}}
        good = T.canonical(evidence); P.validate_execution(B, good)
        for changed in [dict(good, counters={'aten::view': 1}),
                        {'counters': {}, 'execution_metrics': {'ExecuteTime': [1, 0.2, [[1.0, -0.1]]]}}]:
            with self.assertRaises(ValueError):
                P.validate_execution(B, changed)
        with self.assertRaises(ValueError):
            T.canonical({'value': float('nan')})
        records.append({'control': 'COMPLETE_TUPLE_SERIALIZATION_ORIGINAL_METRIC_FALLBACK_GUARDS', 'status': 'PASS'})

    def test_06_parameter_roles_numeric_fail_and_unexpected_auxiliary(self):
        expected = E.array_words([0x3f800000], [1]); altered = E.array_words([0x3f800001], [1])
        raw = {'input_parameter_ids': [0], 'parameters': {'0': expected}}
        _, gates = V.frame_parameter_roles(raw, [expected], {24}); self.assertEqual(gates, {'argument_0': 'PASS'})
        raw['parameters']['0'] = altered
        _, gates = V.frame_parameter_roles(raw, [expected], {24}); self.assertEqual(gates, {'argument_0': 'FAIL'})
        raw['parameters']['1'] = {'shape': [], 'dtype': 'int32', 'values': [24]}
        roles, _ = V.frame_parameter_roles(raw, [expected], {24}); self.assertEqual(roles['1']['role'], 'ADMITTED_SOURCE_INTEGER_SCALAR')
        raw['parameters']['1']['values'] = [25]
        with self.assertRaisesRegex(ValueError, 'UNADMITTED_COMPOSITE_AUXILIARY_PARAMETER'):
            V.frame_parameter_roles(raw, [expected], {24})
        raw['parameters']['1'] = {'shape': [], 'dtype': 'float32', 'values': [24.0]}
        with self.assertRaises(ValueError):
            V.frame_parameter_roles(raw, [expected], {24})
        literals = V.literals(args.generation / 'plugin/src/bitsandbytes_tpu', 513)
        self.assertIn(24, literals); self.assertIn(9, literals); self.assertNotIn(-191, literals)
        records.append({'control': 'ACTUAL_ARGUMENT_WORD_FAIL_RETAINED_AND_AUXILIARY_DOMAIN_REJECTION', 'status': 'PASS'})

    def test_07_cpu_outer_identity_closure_and_wrong_launch(self):
        seal = {'pid': 101, 'pgid': 101, 'parent_pid': 99, 'process_token': 'a' * 32, 'deadline_epoch': 1000.0}
        cleanup = {'pid': 101, 'pgid': 101, 'status': 'CLEANUP_VERIFIED', 'leader_reaped': True,
                   'group_absence': 'OBSERVED_NO_SUCH_GROUP', 'errors': [], 'exit_status': 0}
        ownership = {'pid': 101, 'pgid': 101, 'owner_pid': 99, 'argv': ['python', 'prepare', '--process-token', 'a' * 32, '--deadline-epoch', '1000.0'], 'cleanup': cleanup, 'exit_status': 0}
        result = {'argv': ownership['argv'], 'status': 'PASS', 'exit_code': 0, 'error': None,
                  'cleanup': {'errors': [], 'groups': [cleanup], 'restoration': {'handlers_restored': True, 'timer_restored': True}}}
        V.validate_cpu_launch(seal, ownership, result, ownership['argv'])
        for key, changed in [('pgid', 102), ('owner_pid', 98), ('argv', ['python', 'execute'])]:
            bad = dict(ownership, **{key: changed})
            with self.assertRaises(ValueError):
                V.validate_cpu_launch(seal, bad, result, ownership['argv'])
        for token, deadline in [('b' * 32, 1000.0), ('a' * 32, 1001.0)]:
            with self.assertRaises(ValueError):
                V.validate_cpu_launch(dict(seal, process_token=token, deadline_epoch=deadline), ownership, result, ownership['argv'])
        bad = copy.deepcopy(ownership); bad['cleanup']['errors'] = ['EPERM']
        with self.assertRaisesRegex(ValueError, 'CPU_OUTER_CLOSURE'):
            V.validate_cpu_launch(seal, bad, result, ownership['argv'])
        records.append({'control': 'SYNTHETIC_OUTER_CPU_LAUNCH_PID_PGID_ARGV_EPERM_REJECTION', 'status': 'PASS'})

    def test_08_real_native_profile_observer_restoration(self):
        import torch
        sys.path.insert(0, str(args.generation / 'plugin/src'))
        from bitsandbytes_tpu.functionalization import functionalized_kernel
        original = torch._C._dispatch_tls_local_include_set(), torch._C._dispatch_tls_local_exclude_set()
        body_inputs = []
        def body(value):
            body_inputs.append(torch._is_functional_tensor(value)); return value + 1
        wrapped = functionalized_kernel(body)
        for dtype in (torch.float32, torch.bfloat16):
            value = torch.tensor([1], dtype=dtype)
            with T.SourceFrames(torch, args.generation / 'plugin/src/bitsandbytes_tpu') as frames:
                actual = wrapped(value)
            self.assertEqual(actual.tolist(), [2]); self.assertFalse(torch._is_functional_tensor(actual))
            self.assertIsNone(sys.getprofile()); self.assertFalse(frames.errors); self.assertFalse(frames.active)
            self.assertTrue(any(e['function'] == 'wrapped' and e['status'] == 'RETURN' for e in frames.events))
            with self.assertRaisesRegex(RuntimeError, 'unwrapped tensors'):
                wrapped(torch._to_functional_tensor(value))
        self.assertEqual(body_inputs, [True, True])
        self.assertEqual(original, (torch._C._dispatch_tls_local_include_set(), torch._C._dispatch_tls_local_exclude_set()))
        records.append({'control': 'ACTUAL_CPU_NATIVE_PROFILE_SINGLE_WRAP_FP32_BF16_TLS_AND_DOUBLE_WRAP_REJECTION', 'status': 'PASS', 'XLA': 'NOT_RUN'})

    def test_08b_recovered_functional_frames_and_partial_invalid_numbers(self):
        plugin_map = A.read(HERE / 'integration-admission.json')['plugin_python_files']
        def meta(dtype, functional):
            return {'shape': [1], 'dtype': dtype, 'device': 'xla:0', 'native_functional': functional}
        def event(name, source, inputs, outputs):
            return {'function': name, 'source': source, 'source_sha256': plugin_map[source],
                    'line': 1, 'status': 'RETURN', 'inputs': inputs, 'outputs': outputs}
        frames = [event('wrapped', 'functionalization.py', [meta('float32', False)], [meta('float32', False)]),
                  event('compress_statistics', 'exact.py', [meta('float32', True)], [meta('float32', True)])]
        frames += [event('bitcast', 'exact.py', [meta(a, True)], [meta(b, True)])
                   for a, b in [('float32', 'int32')] * 2 + [('int32', 'float32')] * 2]
        self.assertEqual(T.validate_frames(frames, 'bnb_tpu_exact::compress_statistics', plugin_map)['single_native_wrapper'], 'OBSERVED')
        for mutation in ('double', 'missing_bitcast', 'source', 'functional'):
            bad = copy.deepcopy(frames)
            if mutation == 'double': bad.insert(0, copy.deepcopy(bad[0]))
            elif mutation == 'missing_bitcast': bad.pop()
            elif mutation == 'source': bad[1]['source_sha256'] = 'a' * 64
            else: bad[2]['inputs'][0]['native_functional'] = False
            with self.assertRaises(ValueError): T.validate_frames(bad, 'bnb_tpu_exact::compress_statistics', plugin_map)
        raw = {'status': 'ERROR', 'error_type': 'ValueError', 'outputs': [{'values': [float('nan')]}]}
        retained = T.retained_error(raw)
        self.assertEqual(retained['outputs'][0]['values'], ['INVALID_NONFINITE:nan'])
        self.assertEqual(retained['invalid_nonfinite_paths'], ['/outputs/0/values/0'])
        with self.assertRaises(ValueError): T.retained_error(dict(raw, status='PASS'))
        records.append({'control': 'SYNTHETIC_RECOVERED_SOURCE_FRAME_NEGATIVES_AND_NONFINITE_ERROR_RETENTION', 'status': 'PASS'})

    def test_09_public_case_boundary_is_static_and_old_oracle_rejects(self):
        import ast
        tree = ast.parse((HERE / 'public_cases.py').read_text())
        self.assertFalse(any(isinstance(n, (ast.Import, ast.ImportFrom)) and getattr(n, 'module', '') == 'torch_xla' for n in tree.body))
        text = (HERE / 'public_cases.py').read_text()
        self.assertEqual(text.count('torch.ops.bnb_tpu_exact.restore_scales.default'), 2)
        self.assertIn('module.to(device).train()', text)
        self.assertNotIn('setattr(', text)
        wrong = B.read(args.oracle / 'oracle-seal.json')
        with self.assertRaisesRegex(ValueError, 'NEW_SCIENCE_BINDINGS'):
            V.validate_receipt(B, wrong, {'kind': E.KIND, 'source_variant': A.VARIANT}, args.oracle, 'oracle-seal.json')
        records.append({'control': 'STATIC_ORIGINAL_PUBLIC_CALLS_EXPLICIT_RESTORE_OLD_ORACLE_REJECTS', 'status': 'PASS'})

    def test_09b_real_entry_source_set_and_old_oracle_rejection(self):
        from types import SimpleNamespace
        members = ['probe_nested_exact.py', 'exact_admission.py', 'rational_reference.py', 'record_validation.py',
                   'graph_witness.py', 'hlo_parser.py', 'public_trace.py', 'public_cases.py', 'generation_gate.py',
                   'integration-admission.json', 'nested-inputs.json', 'cpu_gate.py', 'state8_contract.py',
                   'nine-schemas.json', 'original-seven-schemas.json', 'archive_store.py', 'archive_objects.py', 'probe_state8_exact.py', 'state8_public.py']
        manifest = {'kind': 'NESTED_EXACT79_SCIENCE_SOURCES_V1', 'integration_map_sha256': A.INTEGRATION_MAP_SHA,
                    'source_admission_sha256': A.INTEGRATION_ADMISSION_SHA, 'device_acceptance': None,
                    'files': {name: {'sha256': A.sha(HERE / name), 'bytes': (HERE / name).stat().st_size} for name in members}}
        path = args.output / 'current-science-manifest.json'; B.write(path, manifest)
        dependencies = {key: EX / name for key, name in [('backend_probe', 'probe_backend.py'), ('route_probe', 'probe_routes.py'),
                    ('precision_probe', 'probe_precision.py'), ('transfer_admission', 'transfer_admission.py'), ('nested_reference', 'probe_nested.py')]}
        candidate = SimpleNamespace(**dependencies, science_manifest=path, science_manifest_sha256=A.sha(path),
                    admission=HERE / 'integration-admission.json', admission_sha256=A.INTEGRATION_ADMISSION_SHA,
                    patch_manifest=args.generation / 'build-v3/patch-manifest.json')
        self.assertEqual(A.sha(candidate.patch_manifest), A.PATCH_MANIFEST_SHA)
        E.sources(candidate)
        argv = ['verify-cpu']
        for key in (*E.PINS, 'science_manifest', 'science_manifest_sha256', 'admission', 'admission_sha256', 'patch_manifest'):
            argv += ['--' + key.replace('_', '-'), str(getattr(candidate, key))]
        argv += ['--oracle', str(args.oracle), '--oracle-sha256', B.sha(args.oracle / 'oracle-seal.json')]
        import contextlib, io
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(E.main(argv), 1)
        self.assertEqual(json.loads(output.getvalue())['reason'], 'NEW_SCIENCE_BINDINGS')
        bad = copy.deepcopy(manifest); bad['files']['public_cases.py']['sha256'] = 'a' * 64
        changed = args.output / 'incorrect-science-manifest.json'; B.write(changed, bad)
        candidate.science_manifest = changed; candidate.science_manifest_sha256 = A.sha(changed)
        with self.assertRaisesRegex(ValueError, 'SCIENCE_SOURCE_MEMBER_public_cases.py'): E.sources(candidate)
        records.append({'control': 'REAL_SOURCE_LOADER_AND_CLI_OLD_ORACLE_WRONG_SOURCE_REJECTION', 'status': 'PASS',
                        'qualified_runtime_started': False})

    def test_10_post_validation_error_never_retains_pass(self):
        valid = {'status': 'PASS'}; V.complete_case(valid)
        broken = N.case_error_record(dict(valid), RuntimeError('injected-after-array-validation'))
        self.assertEqual(broken['status'], 'ERROR')
        with self.assertRaisesRegex(ValueError, 'ERROR_BEARING_PASS_REJECTED'):
            V.complete_case(dict(broken, status='PASS'))
        self.assertEqual(V.numerical_status([{'status': 'PASS'}, {'status': 'FAIL'}]), 'FAIL')
        self.assertEqual(V.numerical_status([{'status': 'PASS'}, {'status': 'ERROR'}]), 'ERROR')
        with self.assertRaises(ValueError):
            V.numerical_status([])
        records.append({'control': 'POST_VALIDATION_ERROR_PASS_REJECTION_AND_FAIL_ERROR_RETENTION', 'status': 'PASS'})

    def test_10b_exact_retained_cpu_runtime_correct_wrong_omitted(self):
        fixed = copy.deepcopy(B.load_spec()[0]['runtime'])
        V.cpu_runtime(B, {'runtime': fixed})
        wrong = dict(fixed, torch='2.14.1')
        for seal in ({'runtime': wrong}, {}, {'runtime': None}, {'runtime': dict(fixed, extra='unadmitted')}):
            with self.assertRaisesRegex(ValueError, 'EXACT_RETAINED_CPU_RUNTIME'):
                V.cpu_runtime(B, seal)
        records.append({'control': 'EXACT_RETAINED_CPU_RUNTIME_CORRECT_WRONG_OMITTED_EXTRA', 'status': 'PASS',
                        'fixture': 'SYNTHETIC_METADATA_NO_CPU_ORACLE_CREATED'})

    def test_10c_lossless_archive_proposal_correct_wrong_truncated_and_bomb(self):
        import archive_objects as O
        text = ('HloModule synthetic\n' + 'integer graph body\n' * 500).encode()
        compressed, record = O.encode(text, 'graph')
        self.assertEqual(O.decode(compressed, record), text)
        self.assertEqual(O.inventory([record, record])['unique_objects'], 1)
        changed, changed_record = O.encode(text + b'actual distinct body', 'graph')
        self.assertEqual(O.inventory([record, changed_record])['unique_objects'], 2)
        for field, value in [('raw_sha256', 'a' * 64), ('gzip_sha256', 'a' * 64),
                             ('raw_bytes', len(text) - 1), ('raw_bytes', O.PROFILE['graph_raw_bytes'] + 1),
                             ('path', '../outside.gz')]:
            with self.assertRaises(ValueError): O.decode(compressed, dict(record, **{field: value}))
        for incorrect in (compressed[:-1], compressed + compressed):
            resealed = dict(record, gzip_bytes=len(incorrect), gzip_sha256=O.digest(incorrect))
            with self.assertRaises((ValueError, __import__('zlib').error)): O.decode(incorrect, resealed)
        # A declared shorter output cannot trigger unbounded decompression.
        with self.assertRaises(ValueError): O.decode(compressed, dict(record, raw_bytes=1))
        with self.assertRaisesRegex(ValueError, 'OBJECT_ADDRESS_KIND_OR_CONTENT_CONFLICT'):
            O.inventory([record, dict(record, kind='json')])
        too_many = [dict(record, path='objects/' + str(i).zfill(64) + '.gz', raw_sha256=str(i).zfill(64), gzip_bytes=8*O.MiB) for i in range(8)]
        with self.assertRaisesRegex(ValueError, 'OBJECT_ARCHIVE_QUOTA'): O.inventory(too_many)
        records.append({'control': 'PROPOSED_LOSSLESS_OBJECT_FULL_BYTES_DEDUP_WRONG_SHA_TRUNCATED_MULTIGZIP_BOMB_QUOTA',
                        'status': 'PASS', 'production_bridge': 'NOT_IMPLEMENTED_NOT_ADOPTED'})

    def test_11_state8_stage_order_cannot_accept_old_failed79(self):
        import state8_contract as S
        self.assertEqual(len(S.fixed_cases(data)), 8)
        old = args.old_failed_receipt
        # The old record remains unchanged. No accepted-result fixture is constructed.
        with self.assertRaises(ValueError):
            A.state_dependency(B.read(old), old, B.sha(old), 'a' * 64)
        with self.assertRaises(ValueError):
            S.fixed_cases(dict(data, cases=[c for c in data['cases'] if c['kind'] != 'module']))
        records.append({'control': 'FIXED8_STATE_CASES_OLD_FAILED79_AND_MISSING_CASE_REJECTION', 'status': 'PASS', 'state_execution': 'NOT_RUN'})

result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
B.write(args.output / 'review.json', {'status': 'PASS' if result.wasSuccessful() else 'FAIL',
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'controls': records, 'attempts': attempts, 'TPU': 'NOT_RUN', 'new_cpu_oracle': 'NOT_CREATED',
        'scope': 'Local macOS unqualified controls; historical records read only; synthetic graph/launch fixtures'})
raise SystemExit(0 if result.wasSuccessful() else 1)
