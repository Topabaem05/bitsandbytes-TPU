"""Dedicated candidate nested-exact79 bridge. No cloud or source adoption command."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import sys
import time
import traceback
import tempfile
import contextlib
import archive_store as Z

HERE = Path(__file__).resolve().parent
for _name in ('exact_admission', 'rational_reference', 'record_validation', 'public_trace', 'public_cases', 'graph_witness', 'hlo_parser'):
    if _name in sys.modules and Path(sys.modules[_name].__file__).resolve().parent != HERE:
        raise ValueError('SIBLING_MODULE_CACHE_COLLISION_' + _name)
sys.path.insert(0, str(HERE))
try:
    import exact_admission as A
    import rational_reference as F
    import record_validation as V
    import public_trace as T
    import public_cases as C
finally:
    sys.path.remove(str(HERE))

PINS = {'precision_probe': 'e0534955507e5f91e67ecddfffc738a367ad752e69a4eea7ae8052c6d76a8852',
        'route_probe': 'feae751734e57c741b1bdade004ff7ca3c041ee7eb3b7086b531bc6be433038e',
        'backend_probe': 'e8743e33e6a0454d5d05b77075d47a5c9f983cf32f3956c38f2b544f86f5ef6d',
        'transfer_admission': '6a7c925211b14aba990318360c8f0ca2d4d803e4528c634135c8bde86326816b',
        'nested_reference': '9cfbc7c6b9841b365b47ecfd0bf1b184fda754208aec5f214fe3d0cd9198348d'}
INPUT_SHA = 'b7490f4c622aef793d14bd9a7a38727754cb1fb8e6e1dec7ff91ef8c45fbafbc'
KIND = 'PRIVATE_NESTED_EXACT79_PROBE'

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def sources(args):
    A.require(A.sha(args.science_manifest) == args.science_manifest_sha256, 'SCIENCE_MANIFEST_BYTE_SHA')
    manifest = A.read(args.science_manifest)
    A.require(manifest.get('kind') == 'NESTED_EXACT79_SCIENCE_SOURCES_V1'
              and manifest.get('integration_map_sha256') == A.INTEGRATION_MAP_SHA
              and manifest.get('source_admission_sha256') == A.INTEGRATION_ADMISSION_SHA
              and manifest.get('device_acceptance') is None, 'CONTINGENT_SCIENCE_SOURCE_MAP')
    for name, wanted in manifest['files'].items():
        path = HERE / name
        A.require(not Path(name).is_absolute() and '..' not in Path(name).parts
                  and not path.is_symlink() and A.sha(path) == wanted['sha256']
                  and path.stat().st_size == wanted['bytes'], 'SCIENCE_SOURCE_MEMBER_' + name)
    A.require(set(manifest['files']) == {'probe_nested_exact.py', 'exact_admission.py', 'rational_reference.py',
              'record_validation.py', 'graph_witness.py', 'hlo_parser.py', 'public_trace.py', 'public_cases.py',
              'generation_gate.py', 'integration-admission.json', 'nested-inputs.json',
              'cpu_gate.py', 'state8_contract.py', 'nine-schemas.json', 'original-seven-schemas.json',
              'archive_store.py', 'archive_objects.py', 'probe_state8_exact.py', 'state8_public.py'}, 'SCIENCE_EXACT_SOURCE_SET')
    A.admission(args.admission, args.admission_sha256)
    A.require(A.sha(args.patch_manifest) == A.PATCH_MANIFEST_SHA, 'EXACT_PATCH_MANIFEST')
    for key, pin in PINS.items():
        A.require(A.sha(getattr(args, key)) == pin, 'IMMUTABLE_HELPER_' + key)
    P = load(args.precision_probe, 'exact79_precision'); R = P.route_probe(args.route_probe)
    B = R.backend(args.backend_probe); OldA = load(args.transfer_admission, 'exact79_original_public_identity')
    N = load(args.nested_reference, 'exact79_original_fixed_cases')
    A.require(B.sha(HERE / 'nested-inputs.json') == INPUT_SHA, 'ORIGINAL79_INPUT_BYTES')
    return B, R, P, OldA, N

def binding(B, args):
    return {'kind': KIND, 'science_variant': 'nested-exact79-v3-lossless-state8', 'source_variant': A.VARIANT,
            'probe_sha256': B.sha(__file__), 'science_manifest_sha256': args.science_manifest_sha256,
            'source_admission_sha256': A.INTEGRATION_ADMISSION_SHA,
            'integration_map_sha256': A.INTEGRATION_MAP_SHA,
            'patch_manifest_sha256': A.PATCH_MANIFEST_SHA, 'inputs_sha256': INPUT_SHA,
            'schema_sha256': B.sha(HERE / 'nine-schemas.json'), 'helper_sha256': PINS,
            'profile_sha256': B.PROFILE_SHA, 'runtime_lock_sha256': B.load_spec()[0]['runtime_lock_sha256'],
            'case_ids': [c['id'] for c in B.read(HERE / 'nested-inputs.json')['cases']],
            'selected_reference': 'PINNED_DEFAULT_PYTHON',
            'default_body_sha256': 'e39afd16dca6e6f34a14305ade0ba4acbb49dc2c7a9baf937341d7e33b682819',
            'm4_status': 'NOT_QUALIFIED', 'saved_state_status': 'NOT_RUN',
            'm5_scope': '1024_FIRST_SCALES_OUTSIDE_1_TO_513',
            'hlo_scope': 'PRE_OPTIMIZATION_COMPOSITE_NOT_WHOLE_GEMM'}

def schema_readback(torch):
    result = {}
    for key in A.read(HERE / 'nine-schemas.json'):
        namespace, overload_name = key.split('::')
        name, _, overload = overload_name.partition('.')
        result[key] = str(torch._C._dispatch_find_schema_or_throw(namespace + '::' + name, overload).schema())
    A.require(result == A.read(HERE / 'nine-schemas.json'), 'NINE_SCHEMA_READBACK')
    return result

def exact_runtime(B, device):
    versions = B.runtime_check(device)
    A.require(platform.python_version() == '3.12.14' and platform.python_implementation() == 'CPython', 'PYTHON31214')
    A.require(importlib.metadata.version('jax') == importlib.metadata.version('jaxlib') == '0.7.1', 'JAX_METADATA071')
    return versions

def array_words(words, shape, dtype='float32'):
    if dtype == 'float32':
        encoded = b''.join(struct.pack('<I', x & 0xffffffff) for x in words)
        values = [struct.unpack('<f', encoded[i:i + 4])[0] for i in range(0, len(encoded), 4)]
    else:
        encoded = bytes(words); values = list(words)
    return {'dtype': dtype, 'shape': shape, 'values': values, 'bytes_sha256': hashlib.sha256(encoded).hexdigest()}

def expected_trace_inputs(case, data, reference, name):
    outputs = reference['outputs']
    if name == 'bnb_tpu_exact::compress_statistics':
        A.require(case['kind'] in ('nested', 'module'), 'UNEXPECTED_COMPRESS_ROUTE')
        _, first = F.quantize([F.word(x, case['dtype']) for x in case['weight']],
                              [F.word(x) for x in data['nf4_map']], 64)
        return [array_words(first, [len(first)]), outputs['dynamic_map']]
    A.require(name == 'bnb_tpu_exact::restore_scales' and case['kind'] != 'block', 'UNEXPECTED_RESTORE_ROUTE')
    code_key = 'codes' if case['kind'] == 'signed_known' else 'scale_codes'
    return [outputs[code_key], outputs['second_scales'], outputs['dynamic_map'], outputs['offset']]

def verify_device(B, P, N, args):
    with tempfile.TemporaryDirectory(prefix='bnb-m4-recovery-') as scratch:
        return verify_device_recovered(B, P, N, args, Path(scratch))

def verify_device_recovered(B, P, N, args, actual):
    gold, references, _ = V.verify_cpu(B, P, N, args.oracle, args.oracle_sha256, binding(B, args))
    archive_root = Path(args.actual)
    store = Z.Store(archive_root); store.audit()
    rec = V.validate_receipt(B, B.read(archive_root / 'receipt.json'), binding(B, args), archive_root, 'receipt.json')
    A.require(rec.get('public_api') == N.PUBLIC, 'ACTUAL_ORIGINAL_PUBLIC_METHOD_IDENTITY')
    A.require(rec.get('phase') == 'TPU_EXECUTE' and rec.get('oracle_sha256') == args.oracle_sha256
              and rec['pid'] != gold['pid'] and rec['process_token'] != gold['process_token'], 'FRESH_CPU_TPU_PAIR')
    A.require(rec.get('runtime') == B.load_spec()[0]['runtime']
              and rec.get('device') == {'type': 'xla', 'hardware': 'TPU', 'pjrt': 'TPU'}
              and rec.get('precision') == {'requested': 'highest', 'readback': 'highest', 'set_calls': 1, 'before_graph': True},
              'FIXED_DEVICE_RUNTIME_PRECISION')
    P.validate_environment(B, rec['precision_environment'])
    schemas = A.read(HERE / 'nine-schemas.json')
    A.require(rec.get('dispatch') == {key: True for key in schemas}, 'NINE_ACTUAL_XLA_DISPATCH')
    admission = A.admission(args.admission, args.admission_sha256)
    plugin = Path(args.plugin_source)
    A.require(not plugin.is_symlink() and all(not p.is_symlink() for p in plugin.rglob('*'))
              and {p.relative_to(plugin).as_posix(): B.sha(p) for p in plugin.rglob('*.py')}
              == admission['plugin_python_files'], 'RECOVERED_PLUGIN_SOURCE_MAP')
    data = B.read(HERE / 'nested-inputs.json'); profile, _ = B.load_spec(); rows = []
    expected_inventory = set()
    for case in data['cases']:
        shutil.rmtree(actual); actual.mkdir()
        row_names = ['raw/' + case['id'] + suffix for suffix in ('.json', '.hlo.txt', '.metrics.txt')]
        for logical_name in row_names:
            target = actual / logical_name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(store.read(logical_name))
        prefix = 'raw/' + case['id']; raw = B.read(actual / (prefix + '.json'))
        expected_inventory.update(prefix + suffix for suffix in ('.json', '.hlo.txt', '.metrics.txt'))
        trace_root = actual / 'trace' / case['id']
        declared = raw.get('composite_records')
        A.require(type(declared) is list and all(re.fullmatch('[0-9]{3}', x) for x in declared)
                  and len(declared) == len(set(declared)), 'DECLARED_COMPOSITE_RECORDS')
        trace_names = {n for n in store.files if n.startswith('trace/' + case['id'] + '/')}
        expected_inventory.update(trace_names)
        for logical_name in trace_names:
            if logical_name.endswith('/raw.json'):
                target = actual / logical_name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(store.read(logical_name))
        A.require(not trace_root.is_symlink() and all(not p.is_symlink() for p in trace_root.rglob('*')),
                  'TRACE_NO_SYMLINK')
        if trace_root.is_dir():
            A.require({p.name for p in trace_root.iterdir()} == set(declared)
                      and all(p.is_dir() for p in trace_root.iterdir()), 'TRACE_DECLARED_DIRECTORY_SET')
        A.require(raw.get('case_id') == case['id'] and raw.get('input_sha256') == B.case_sha(case)
                  and raw.get('pid') == rec['pid'] and raw.get('process_token') == rec['process_token'], 'DEVICE_ROW_IDENTITY')
        if raw.get('status') == 'ERROR':
            A.require(all(isinstance(raw.get(k), str) and raw[k] for k in ('error_type', 'error_message', 'traceback')), 'RETAINED_CASE_ERROR')
            rows.append({'case_id': case['id'], 'status': 'ERROR', 'gates': {}, 'trace_status': 'INCOMPLETE_RETAINED'})
            continue
        V.complete_case(raw); N.validate_arrays(B, P, case, raw, True); P.validate_execution(B, raw)
        retained_intervals = [B.read(trace_root / name / 'raw.json') for name in declared]
        A.require(raw.get('execution_scope') == 'DISJOINT_ACTUAL_COMPOSITE_INTERVALS_PLUS_FINAL_OUTPUT_SYNC'
                  and T.aggregate_execution([{key: item[key] for key in ('counters', 'execution_metrics')}
                                             for item in retained_intervals] + [raw['final_output_execution']])
                  == {key: raw[key] for key in ('counters', 'execution_metrics')}, 'ROW_RETAINED_EXECUTION_INTERVALS')
        hlo = (actual / (prefix + '.hlo.txt')).read_text()
        A.require(hlo.strip() and (actual / (prefix + '.metrics.txt')).stat().st_size > 0, 'ACTUAL_ROW_GRAPH_METRICS')
        if case['kind'] == 'module' and case['dtype'] == 'float32' and len(case['x_shape']) == 2:
            dots = P.hlo_precision(hlo)
            A.require(raw.get('hlo_precision') == dots, 'ROW_HLO_PRECISION_BINDING')
            f32 = [d for d in dots if d['result_dtype'] == 'f32' and d['operand_dtypes'] == ['f32', 'f32']]
            for dot in f32:
                P.validate_dot_matrix(B, dot)
            shapes = {tuple(d['result_shape']) for d in f32}
            A.require(f32 and all(d['operands'] == ['highest', 'highest'] for d in f32)
                      and tuple(case['x_shape']) in shapes
                      and tuple(case['x_shape'][:-1] + [case['weight_shape'][0]]) in shapes, 'FORWARD_DX_HIGHEST')
        else:
            A.require(raw.get('hlo_precision') is None, 'ROW_HLO_OBSERVER_SCOPE')
        gates = {}
        for name, observed in raw['outputs'].items():
            wanted = references[case['id']]['outputs'][name]
            gate = B.numeric(observed['values'], wanted['values'], N.tolerance(profile, case, name))
            if name in N.EXACT and observed['bytes_sha256'] != wanted['bytes_sha256']:
                gate['status'] = 'FAIL'
            gates[name] = gate
        traces = []
        for name in declared:
            directory = trace_root / name; observed = B.read(directory / 'raw.json')
            selected_names = [n for n in trace_names if n.startswith('trace/' + case['id'] + '/' + name + '/') and not n.endswith('/raw.json')]
            A.require(sum(store.files[n]['raw_bytes'] for n in selected_names) <= 132 * 1024 * 1024, 'COMPOSITE_RECOVERY_QUOTA')
            for logical_name in selected_names:
                (actual / logical_name).write_bytes(store.read(logical_name))
            A.require(observed.get('pid') == rec['pid'] and observed.get('pgid') == rec['pgid'], 'COMPOSITE_PROCESS_BINDING')
            inputs = expected_trace_inputs(case, data, references[case['id']], observed['schema_name'])
            count = len(inputs[0]['values'])
            observed, audit = V.composite_record(B, P, directory, inputs, admission['plugin_python_files'], V.literals(plugin, count))
            wanted_names = ['scale_codes', 'offset', 'second_scales'] if observed['schema_name'].endswith('compress_statistics') else ['restored_scales']
            A.require(len(observed['outputs']) == len(wanted_names), 'COMPOSITE_OUTPUT_ARITY')
            for output_name, got in zip(wanted_names, observed['outputs']):
                wanted = references[case['id']]['outputs'][output_name]
                gate = B.numeric(got['values'], wanted['values'], N.tolerance(profile, case, output_name))
                if output_name in N.EXACT and got['bytes_sha256'] != wanted['bytes_sha256']:
                    gate['status'] = 'FAIL'
                audit['input_gates']['output_' + output_name] = gate['status']
            traces.append({'schema_name': observed['schema_name'], **audit})
            for logical_name in selected_names:
                (actual / logical_name).unlink()
        names = [t['schema_name'] for t in traces]
        if case['kind'] == 'block':
            A.require(not names, 'BLOCK_NO_COMPOSITE_ROUTE')
        else:
            A.require(names.count('bnb_tpu_exact::compress_statistics') == (1 if case['kind'] in ('nested', 'module') else 0)
                      and names.count('bnb_tpu_exact::restore_scales') == (1 if case['kind'] == 'signed_known' else 4 if case['kind'] == 'module' and case['dtype'] == 'float32' else 3), 'PUBLIC_COMPOSITE_ROUTE_MATRIX')
        passed = all(g['status'] == 'PASS' for g in gates.values())
        passed = passed and all(v == 'PASS' for t in traces for v in t['input_gates'].values())
        rows.append({'case_id': case['id'], 'status': 'PASS' if passed else 'FAIL', 'gates': gates, 'composites': traces})
    A.require(set(store.files) == expected_inventory, 'COMPLETE_ACTUAL_LOGICAL_ARTIFACT_MATRIX')
    A.require(set(rec['artifacts']) == {'logical-files.json'} | {r['path'] for r in store.files.values()}, 'COMPLETE_ACTUAL_PHYSICAL_ARTIFACT_MATRIX')
    status = V.numerical_status(rows)
    return {'record_validation': 'PASS', 'numerical_status': status, 'case_count': 79, 'rows': rows,
            'source_variant': A.VARIANT, 'm4_status': 'NOT_QUALIFIED', 'saved_state_status': 'NOT_RUN'}

def numerical_observations(B, P, N, args, actual, selected_cases=None):
    """Retain returned arithmetic after observation failure. This cannot qualify execution."""
    _, references, _ = V.verify_cpu(B, P, N, args.oracle, args.oracle_sha256, binding(B, args))
    store = Z.Store(actual); rows = []; profile, _ = B.load_spec()
    cases = selected_cases if selected_cases is not None else B.read(HERE / 'nested-inputs.json')['cases']
    for case in cases:
        try:
            raw = json.loads(store.read('raw/' + case['id'] + '.json'))
            A.require(raw.get('case_id') == case['id'] and raw.get('input_sha256') == B.case_sha(case), 'DIAGNOSTIC_INPUT_BINDING')
            V.complete_case(raw); N.validate_arrays(B, P, case, raw, True)
            gates = {}
            for key, observed in raw['outputs'].items():
                wanted = references[case['id']]['outputs'][key]
                gate = B.numeric(observed['values'], wanted['values'], N.tolerance(profile, case, key))
                if key in N.EXACT and observed['bytes_sha256'] != wanted['bytes_sha256']:
                    gate['status'] = 'FAIL'
                gates[key] = gate
            rows.append({'case_id': case['id'], 'status': 'PASS' if all(g['status'] == 'PASS' for g in gates.values()) else 'FAIL',
                         'gates': gates, 'observation_status': raw.get('observation_status')})
        except Exception as error:
            rows.append({'case_id': case['id'], 'status': 'ERROR', 'gates': {}, 'reason': str(error)})
    return {'record_validation': 'FAIL', 'numerical_status': V.numerical_status(rows), 'rows': rows,
            'scope': 'RETAINED_RETURNED_NUMERICS_ONLY_NO_DEVICE_OR_GRAPH_QUALIFICATION', 'm4_status': 'NOT_QUALIFIED'}

def run(B, R, P, OldA, N, args):
    prepare = args.command == 'prepare'; output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False); name = 'oracle-seal.json' if prepare else 'receipt.json'
    rec = dict(binding(B, args), status='PARTIAL', phase='CPU_PREPARE' if prepare else 'TPU_EXECUTE',
               pid=os.getpid(), pgid=os.getpgid(0), parent_pid=os.getppid(), process_token=args.process_token,
               deadline_epoch=args.deadline_epoch, started_epoch=time.time(), started_monotonic=time.monotonic())
    admission = None; roots = {}; bnb = None
    if not prepare:
        Z.Store(output, create=True)
    try:
        V.process(rec)
        A.require(math.isfinite(args.deadline_epoch) and time.time() < args.deadline_epoch, 'ABSOLUTE_OWNER_DEADLINE')
        review = A.reviewer_dependency(A.read(args.source_review), args.source_review_sha256, record_path=args.source_review)
        A.require(review['science_map_sha256'] == args.science_manifest_sha256, 'REVIEWED_SCIENCE_MAP')
        versions = exact_runtime(B, not prepare)
        if not prepare:
            gold, _, _ = V.verify_cpu(B, P, N, args.oracle, args.oracle_sha256, binding(B, args))
            gate = A.read(args.cpu_gate)
            required = {'kind': 'ROOT_NESTED_EXACT_CPU_ORACLE_GATE', 'status': 'QUALIFIED_LINUX_CPU_ORACLE_VERIFIED',
                        'oracle_sha256': args.oracle_sha256, 'source_admission_sha256': A.INTEGRATION_ADMISSION_SHA,
                        'science_manifest_sha256': args.science_manifest_sha256, 'case_count': 79, 'array_count': 540,
                        'source_variant': A.VARIANT, 'm4_status': 'NOT_QUALIFIED',
                        'owned_cpu_pid': gold['pid'], 'owned_cpu_pgid': gold['pgid'], 'owned_cpu_token': gold['process_token']}
            A.require(A.sha(args.cpu_gate) == args.cpu_gate_sha256 and all(gate.get(k) == v for k, v in required.items()), 'HOST_RECOVERED_CPU_GATE')
            for key in ('upstream_wheel_sha256', 'plugin_wheel_sha256', 'cpu_step_result_sha256', 'observation_report_sha256'):
                A.require(re.fullmatch('[0-9a-f]{64}', gate.get(key, '')) is not None, 'CPU_GATE_PROOF_' + key)
        admission, roots = A.admit(args.admission, args.admission_sha256, args.patch_manifest)
        rec['source_pre'] = args.admission_sha256
        import torch
        A.require(torch.__version__ == '2.9.0+cpu', 'LOADED_TORCH290_CPU')
        torch.set_num_threads(1)
        if prepare:
            device = 'cpu'; rec.update(runtime=versions, oracle_method='EXPLICIT_PINNED_DEFAULT_BODIES_NO_NATIVE_DISPATCH')
        else:
            import torch_xla as xla
            import torch_xla.backends as backends
            backends.set_mat_mul_precision('highest')
            rec['precision'] = {'requested': 'highest', 'readback': backends.get_mat_mul_precision(), 'set_calls': 1, 'before_graph': True}
            import torch_xla.core.xla_model as xm
            import torch_xla.debug.metrics as metrics
            device = xm.xla_device(); A.require(device.type == 'xla' and xm.xla_device_hw(device) == 'TPU', 'ACTUAL_TPU_REQUIRED')
            rec.update(runtime=versions, precision_environment=P.precision_environment(), device={'type': 'xla', 'hardware': 'TPU', 'pjrt': 'TPU'}, oracle_sha256=args.oracle_sha256)
        import bitsandbytes as bnb
        rec['public_api'] = N.public_methods(B, OldA, bnb, roots); rec['schemas'] = schema_readback(torch)
        # Entry-point registration imports the exact plugin modules. It does not initialize XLA here.
        rec['plugin_loaded_code'] = A.loaded_plugin_functions(admission, roots['bitsandbytes_tpu'])
        if not prepare:
            rec['dispatch'] = {key: torch._C._dispatch_has_kernel_for_dispatch_key(key, 'XLA') for key in rec['schemas']}
            A.require(all(rec['dispatch'].values()), 'ALL_NINE_XLA_KERNELS')
        data = B.read(HERE / 'nested-inputs.json'); functions = N.default_functions(B, roots) if prepare else None
        if prepare:
            shutil.copyfile(HERE / 'nested-inputs.json', output / 'inputs.json')
            shutil.copyfile(args.admission, output / 'source-admission.json')
            evidence = {'python': platform.python_version(), 'torch': torch.__version__, 'torch_git': torch.version.git_version,
                        'platform': platform.system(), 'machine': platform.machine(), 'capability': torch.backends.cpu.get_cpu_capability(),
                        'num_threads': torch.get_num_threads(), 'build': torch.__config__.show(),
                        'xla_or_jax_imported': any(n == 'torch_xla' or n.startswith('torch_xla.') or n == 'jax' or n.startswith('jax.') for n in sys.modules)}
            V.cpu_evidence(evidence)
            B.write(output / 'cpu-evidence.json', evidence)
        for case in data['cases']:
            A.require(time.time() < args.deadline_epoch, 'ABSOLUTE_SCIENCE_DEADLINE')
            raw = {'case_id': case['id'], 'input_sha256': B.case_sha(case), 'pid': rec['pid'], 'process_token': rec['process_token'], 'status': 'ERROR', 'composite_records': []}
            prefix = output / 'raw' / case['id']; prefix.parent.mkdir(exist_ok=True)
            mode = None
            try:
                N.public_methods(B, OldA, bnb, roots)
                A.loaded_plugin_functions(admission, roots['bitsandbytes_tpu'])
                if prepare:
                    tensors, details = C.compute_public(B, N, case, data, torch, bnb, device, functions)
                else:
                    metrics.clear_all()
                    mode = T.make_mode(torch, B, P, xla, xm, metrics, roots['bitsandbytes_tpu'], output / 'trace' / case['id'], args.deadline_epoch)
                    with mode:
                        tensors, details = C.compute_public(B, N, case, data, torch, bnb, device)
                    raw['composite_records'] = ['%03d' % (i + 1) for i in range(len(mode.records))]
                    hlo = xla._XLAC._get_xla_tensors_hlo(list(tensors.values())); prefix.with_suffix('.hlo.txt').write_text(hlo)
                    raw['hlo_precision'] = P.hlo_precision(hlo) if case['kind'] == 'module' and case['dtype'] == 'float32' and len(case['x_shape']) == 2 else None
                    # Composite intervals are separately retained; this interval observes actual row outputs only.
                    metrics.clear_all(); xla._XLAC._xla_sync_multi(list(tensors.values()), devices=[], wait=True, sync_xla_data=True); xm.wait_device_ops()
                    final_evidence = T.canonical({'counters': {key: metrics.counter_value(key) for key in metrics.counter_names()},
                                                 'execution_metrics': {key: value for key in B.load_spec()[0]['execution_metrics'] if (value := metrics.metric_data(key)) is not None}})
                    measured = [{key: item[key] for key in ('counters', 'execution_metrics')} for item in mode.records if all(k in item for k in ('counters', 'execution_metrics'))]
                    raw['observation_errors'] = mode.observation_errors
                    raw['observation_status'] = 'ERROR' if mode.observation_errors else 'COMPLETE'
                    raw['final_output_execution'] = final_evidence
                    raw['execution_scope'] = 'DISJOINT_ACTUAL_COMPOSITE_INTERVALS_PLUS_FINAL_OUTPUT_SYNC'
                    raw.update(T.aggregate_execution(measured + [final_evidence]))
                    prefix.with_suffix('.metrics.txt').write_text(metrics.metrics_report())
                raw.update(N.serialize_details(B, details), outputs={key: N.tensor_record(B, value) for key, value in tensors.items()},
                           placements={key: str(value.device) for key, value in tensors.items()}, status='PASS')
                N.validate_arrays(B, P, case, raw, not prepare)
            except Exception as error:
                raw = N.case_error_record(raw, error)
                if mode is not None:
                    raw['composite_records'] = ['%03d' % (i + 1) for i in range(len(mode.records))]
                if not prepare:
                    for suffix in ('.hlo.txt', '.metrics.txt'):
                        if not prefix.with_suffix(suffix).exists():
                            prefix.with_suffix(suffix).write_text('NOT_RUN_CASE_ERROR\n')
            B.write(prefix.with_suffix('.json'), raw)
            if not prepare:
                store = Z.Store(output)
                for suffix in ('.json', '.hlo.txt', '.metrics.txt'):
                    store.pack(prefix.with_suffix(suffix))
                A.require(mode is None or not mode.evidence_limit, 'EVIDENCE_LIMIT_STOP_AFTER_RETAINED_ROW')
            if prepare:
                A.require(raw['status'] == 'PASS', 'CPU_CASE_ERROR')
        if not prepare:
            rec['lossless_archive'] = Z.seal(output)
        rec['status'] = 'COMPLETE'
    except Exception:
        rec['status'] = 'FAILED'; (output / 'error.log').write_text(traceback.format_exc())
    finally:
        if admission is not None:
            try:
                A.verify_installed(admission, roots)
                if bnb is not None:
                    N.public_methods(B, OldA, bnb, roots)
                    A.loaded_plugin_functions(admission, roots['bitsandbytes_tpu'])
                rec['source_post'] = args.admission_sha256
            except Exception:
                rec['status'] = 'FAILED'; (output / 'source-post.error.log').write_text(traceback.format_exc())
        rec.update(finished_epoch=time.time(), finished_monotonic=time.monotonic())
        rec['artifacts'] = B.inventory(output, name); B.write(output / name, rec)
    if rec['status'] != 'COMPLETE':
        return {'record_validation': 'FAIL', 'numerical_status': 'ERROR', 'm4_status': 'NOT_QUALIFIED'}, 1
    try:
        if prepare:
            _, _, result = V.verify_cpu(B, P, N, output, B.sha(output / name), binding(B, args)); return result, 0
        args.actual = output; result = verify_device(B, P, N, args)
        return result, 0 if result['numerical_status'] == 'PASS' else 2
    except Exception:
        rec['status'] = 'FAILED'; (output / 'postvalidation.error.log').write_text(traceback.format_exc())
        rec.update(finished_epoch=time.time(), finished_monotonic=time.monotonic())
        rec['artifacts'] = B.inventory(output, name); B.write(output / name, rec)
        if not prepare:
            try:
                return numerical_observations(B, P, N, args, output), 1
            except Exception:
                pass
        return {'record_validation': 'FAIL', 'numerical_status': 'ERROR', 'm4_status': 'NOT_QUALIFIED'}, 1

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'execute', 'verify', 'verify-cpu'):
        p = sub.add_parser(name)
        for key in (*PINS, 'science_manifest', 'science_manifest_sha256', 'admission', 'admission_sha256', 'patch_manifest'):
            p.add_argument('--' + key.replace('_', '-'), required=True)
        if name in ('prepare', 'execute'):
            for key in ('output', 'source_review', 'source_review_sha256', 'process_token'):
                p.add_argument('--' + key.replace('_', '-'), required=True)
            p.add_argument('--deadline-epoch', type=float, required=True)
        if name != 'prepare':
            p.add_argument('--oracle', required=True); p.add_argument('--oracle-sha256', required=True)
        if name in ('execute', 'verify'):
            p.add_argument('--plugin-source', required=True)
        if name == 'execute':
            p.add_argument('--cpu-gate', required=True); p.add_argument('--cpu-gate-sha256', required=True)
        if name == 'verify':
            p.add_argument('--actual', required=True)
    args = parser.parse_args(argv)
    try:
        B, R, P, OldA, N = sources(args)
        if args.command == 'verify':
            result = verify_device(B, P, N, args); code = 0 if result['numerical_status'] == 'PASS' else 2
        elif args.command == 'verify-cpu':
            _, _, result = V.verify_cpu(B, P, N, args.oracle, args.oracle_sha256, binding(B, args)); code = 0
        else:
            result, code = run(B, R, P, OldA, N, args)
    except Exception as error:
        result = {'record_validation': 'FAIL', 'numerical_status': 'ERROR', 'reason': str(error), 'm4_status': 'NOT_QUALIFIED'}; code = 1
    print(json.dumps(result, allow_nan=False)); return code

if __name__ == '__main__':
    raise SystemExit(main())
