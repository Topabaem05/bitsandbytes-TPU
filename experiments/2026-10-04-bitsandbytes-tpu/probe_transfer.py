"""Reviewed CPU-to-XLA transfer and the complete fixed API42 matrix.

The external owner launches this process and enforces its deadline. No allocation.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import sys
import time
import traceback

PRECISION_SHA = 'e0534955507e5f91e67ecddfffc738a367ad752e69a4eea7ae8052c6d76a8852'
HERE = Path(__file__).resolve().parent
ROUTES = ['params_to_xla', 'module_to_xla']
PROTOCOL = {'kind': 'TRANSFER_API42_PROBE', 'routes': ROUTES,
            'precision': 'highest', 'hlo_scope': 'PRE_OPTIMIZATION_OUTPUT_GRAPH',
            'matrix': 'fixed API42', 'state_restore': 'NOT_RUN', 'cuda_golden': 'NOT_RUN'}
PROTOCOL_SHA = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def helpers(args):
    if hashlib.sha256(Path(args.precision_probe).read_bytes()).hexdigest() != PRECISION_SHA:
        raise ValueError('PRECISION_SOURCE_IDENTITY')
    P = load(args.precision_probe, 'transfer_precision')
    R = P.route_probe(args.route_probe); B = R.backend(args.backend_probe)
    A = load(HERE/'transfer_admission.py', 'reviewed_transfer_admission')
    A.load_manifest(B, args.patch_manifest)
    return B, R, P, A


def identity(B, R, P, A, args):
    profile, _ = B.load_spec()
    return {'kind': PROTOCOL['kind'], 'status': 'PARTIAL', 'protocol_sha256': PROTOCOL_SHA,
            'pid': os.getpid(), 'pgid': os.getpgid(0), 'process_token': secrets.token_hex(16),
            'transfer_probe_sha256': B.sha(__file__), 'probe_sha256': B.sha(__file__),
            'transfer_admission_sha256': B.sha(HERE/'transfer_admission.py'),
            'precision_probe_sha256': PRECISION_SHA, 'route_probe_sha256': P.ROUTE_SHA,
            'backend_probe_sha256': R.BACKEND_SHA, 'patch_manifest_sha256': A.PATCH_MANIFEST_SHA,
            'profile_sha256': B.PROFILE_SHA, 'inputs_sha256': B.INPUTS_SHA,
            'source_admission_sha256': args.admission_sha256, 'runtime_lock_sha256': profile['runtime_lock_sha256'],
            'source_commit': profile['source_commit'], 'routes': ROUTES, 'case_ids': profile['case_ids'],
            'transfer_case_ids': P.CASE_IDS, 'cuda_golden': 'NOT_RUN', 'm3_status': 'NOT_QUALIFIED',
            'precision_environment': P.precision_environment()}


def check_identity(B, R, P, A, args, rec):
    expected = identity(B, R, P, A, args)
    for key in expected:
        if key not in ('status', 'pid', 'pgid', 'process_token', 'precision_environment'):
            B.require(rec.get(key) == expected[key], 'TRANSFER_BINDING_' + key)
    B.require(rec.get('status') == 'COMPLETE', 'TRANSFER_TERMINAL')
    B.require(type(rec.get('pid')) is int and rec['pid'] > 0 and type(rec.get('pgid')) is int and rec['pgid'] > 0 and isinstance(rec.get('process_token'), str) and re.fullmatch('[0-9a-f]{32}', rec['process_token']), 'PROCESS_IDENTITY')
    P.validate_environment(B, rec.get('precision_environment'))
    profile, _ = B.load_spec()
    B.require(rec.get('runtime') == profile['runtime'], 'RUNTIME_PAIR')
    B.require(rec.get('source_pre') == rec.get('source_post') == args.admission_sha256, 'SOURCE_BINDING')
    B.require(rec.get('oracle_sha256') == args.oracle_sha256, 'ORACLE_BINDING')
    B.require(rec.get('precision') == {'requested': 'highest', 'readback': 'highest', 'set_calls': 1, 'before_graph': True}, 'PRECISION_READBACK')
    B.require(rec.get('device') == {'type': 'xla', 'hardware': 'TPU', 'pjrt': 'TPU'}, 'TPU_REQUIRED')
    B.require(rec.get('public_api') == {'Linear4bit': 'bitsandbytes.nn.modules.Linear4bit', 'Params4bit': 'bitsandbytes.nn.modules.Params4bit', 'original_class_identity': True, 'original_method_identity': True}, 'PUBLIC_API_REPLACEMENT')
    B.require(rec.get('dispatch') == {op: True for op in B.DISPATCH_OPS}, 'DISPATCH')


def verify_oracle(B, A, args):
    seal = B.verify_oracle(args.oracle, args.oracle_sha256, args.admission_sha256)
    B.require(seal.get('patch_manifest_sha256') == A.PATCH_MANIFEST_SHA and seal.get('transfer_admission_sha256') == B.sha(HERE/'transfer_admission.py'), 'ORACLE_OVERLAY_BINDING')
    B.require(seal.get('transfer_probe_sha256') == B.sha(__file__) and seal.get('precision_probe_sha256') == PRECISION_SHA and seal.get('route_probe_sha256') == 'feae751734e57c741b1bdade004ff7ca3c041ee7eb3b7086b531bc6be433038e' and seal.get('backend_probe_sha256') == 'e8743e33e6a0454d5d05b77075d47a5c9f983cf32f3956c38f2b544f86f5ef6d', 'ORACLE_HELPER_BINDING')
    B.require(seal.get('source_pre') == seal.get('source_post') == args.admission_sha256 and seal.get('public_api') == {'Linear4bit': 'bitsandbytes.nn.modules.Linear4bit', 'Params4bit': 'bitsandbytes.nn.modules.Params4bit', 'original_class_identity': True, 'original_method_identity': True}, 'ORACLE_SOURCE_METHODS')
    return seal


def row_identity(B, rec, case, route):
    return {'case_id': case['id'], 'input_sha256': B.case_sha(case), 'route': route,
            'pid': rec['pid'], 'process_token': rec['process_token'], 'precision': rec['precision'],
            'status': 'PASS', 'hlo_scope': PROTOCOL['hlo_scope']}


def validate_row_identity(B, rec, raw, case, route):
    for key, value in row_identity(B, rec, case, route).items():
        if key != 'status': B.require(raw.get(key) == value, 'ROW_BINDING_' + key)
    B.require(raw.get('status') in ('PASS', 'ERROR'), 'ROW_STATUS')
    if raw['status'] == 'ERROR':
        B.require(all(isinstance(raw.get(key), str) and raw[key] for key in ('error_type', 'error_message', 'traceback')), 'ERROR_RECORD')


def validate_hlo(B, P, prefix, raw, case, gradient=False):
    text = prefix.with_suffix('.hlo.txt').read_text()
    B.require(text.strip() and prefix.with_suffix('.metrics.txt').stat().st_size > 0, 'MISSING_DIAGNOSTIC')
    P.validate_execution(B, raw)
    # Rank-three API cases retain HLO without imposing this rank-two dot observer.
    if case['op'] == 'linear' and case['dtype'] == 'float32' and len(case['x_shape']) == 2:
        parsed = P.hlo_precision(text)
        B.require(parsed == raw.get('hlo_precision'), 'HLO_RECORD')
        dots = [d for d in parsed if d['result_dtype'] == 'f32' and d['operand_dtypes'] == ['f32', 'f32']]
        for dot in dots: P.validate_dot_matrix(B, dot)
        B.require(dots and all(d['operands'] == ['highest', 'highest'] for d in dots), 'HLO_PRECISION')
        shapes = {tuple(d['result_shape']) for d in dots}
        B.require(tuple(case['x_shape'][:-1] + [case['weight_shape'][0]]) in shapes, 'HLO_FORWARD')
        if gradient: B.require(tuple(case['x_shape']) in shapes, 'HLO_GRADIENT')
    else:
        B.require(raw.get('hlo_precision') is None, 'HLO_OBSERVER_SCOPE')


def compare(B, R, profile, case, outputs, reference):
    return {name: B.numeric(value['values'], reference[name]['values'], R.tolerance(profile, case, name)) for name, value in outputs.items()}


def verify(B, R, P, A, args):
    profile, inputs = B.load_spec(); verify_oracle(B, A, args)
    root = Path(args.actual); rec = B.read(root/'receipt.json')
    check_identity(B, R, P, A, args, rec); B.check_seal(root, 'receipt.json', rec)
    cases = P.selection(B, R); refs = B.read(root/'references.json')
    B.require(set(refs) == set(P.CASE_IDS), 'REFERENCE_MATRIX')
    for case in cases:
        original = B.read(Path(args.oracle)/'raw'/(case['id']+'.json'))
        witness = R.gradient_math(case, original)
        B.require(set(refs[case['id']]) == set(witness), 'REFERENCE_OUTPUT_SET')
        for name, value in witness.items():
            P.array_check(B, refs[case['id']][name], value['shape'], value['dtype'])
            B.require(B.numeric(refs[case['id']][name]['values'], value['values'], R.tolerance(profile, case, name))['status'] == 'PASS', 'SCALAR_GRADIENT_REFERENCE')
        if case['dtype'] == 'float32':
            witness = P.scalar_forward(case, original)
            B.require(B.numeric(original['outputs']['y']['values'], witness['values'], R.tolerance(profile, case, 'y'))['status'] == 'PASS', 'SCALAR_FORWARD_REFERENCE')
    required = {'references.json'}; matrix = []; transfers = []
    for group, route, case in [('matrix', 'api42', c) for c in inputs['cases']] + [('transfer', r, c) for c in cases for r in ROUTES]:
        prefix = root/group/(route+'-'+case['id']); raw = B.read(prefix.with_suffix('.json'))
        required |= {prefix.relative_to(root).as_posix()+suffix for suffix in ('.json', '.hlo.txt', '.metrics.txt')}
        validate_row_identity(B, rec, raw, case, route)
        if raw['status'] == 'ERROR':
            (matrix if group == 'matrix' else transfers).append({'case_id': case['id'], 'route': route, 'status': 'ERROR', 'error_type': raw['error_type'], 'error_message': raw['error_message'], 'gates': {}})
            continue
        original = B.read(Path(args.oracle)/'raw'/(case['id']+'.json'))
        expected = B.expected_outputs(case)
        if route == 'params_to_xla': expected.pop('y')
        if route == 'module_to_xla' and case['dtype'] == 'float32': expected.update(dx=(case['x_shape'], 'float32'), db=([case['weight_shape'][0]], 'float32'))
        B.require(set(raw.get('outputs', {})) == set(expected) and set(raw.get('placements', {})) == set(expected), 'OUTPUT_SET')
        for name, (shape, dtype) in expected.items():
            P.array_check(B, raw['outputs'][name], shape, dtype)
            B.require(isinstance(raw['placements'][name], str) and raw['placements'][name].startswith('xla:'), 'PLACEMENT')
        metadata = {'shape': case['weight_shape'], 'dtype': case['dtype'], 'blocksize': 64, 'quant_type': 'nf4', 'nested': False, 'packing_format_for_cpu': False}
        B.require(raw.get('metadata') == metadata, 'OUTPUT_METADATA')
        if group == 'transfer':
            B.require(raw.get('original_classes') is True and raw.get('parameter_identity') is True and raw.get('custom_attribute_retained') is True and raw.get('quant_state_class') == 'bitsandbytes.functional.QuantState', 'TRANSFER_IDENTITY')
            B.require(raw.get('parameter_attrs') == {'blocksize': 64, 'compress_statistics': False, 'quant_type': 'nf4', 'quant_storage': 'uint8', 'bnb_quantized': True}, 'PARAMETER_ATTRS')
            B.require(raw.get('weight_requires_grad') is False and raw.get('weight_grad_is_none') is True, 'FROZEN_WEIGHT')
            B.require(raw.get('quant_state_placements') == {name: raw['placements'][name] for name in ('absmax', 'code')}, 'STATE_PLACEMENT')
            B.require(raw.get('packed_before') == raw['outputs']['packed'] == raw.get('packed_after'), 'PACKED_CHANGED')
            if route == 'module_to_xla':
                B.require(raw.get('module_identity') is True and raw.get('module_state_alias') is True and raw.get('parameter_module_alias') is True and raw.get('module_training') is True, 'MODULE_ALIAS_IDENTITY')
                B.require(raw.get('bias') == {'shape': [case['weight_shape'][0]], 'dtype': case['dtype'], 'values': case['bias']} and raw.get('bias_placement', '').startswith('xla:') and raw.get('bias_class') == 'torch.nn.parameter.Parameter', 'BIAS')
                B.require(raw.get('gradients') == ('EXECUTED' if case['dtype'] == 'float32' else 'NOT_RUN'), 'GRADIENT_SCOPE')
            else:
                B.require(raw.get('parameter_module_alias') is True and raw.get('gradients') == 'NOT_RUN', 'DIRECT_PARAMETER_SCOPE')
        observe_case = dict(case)
        if route == 'params_to_xla': observe_case['op'] = 'quantize'
        validate_hlo(B, P, prefix, raw, observe_case, route == 'module_to_xla' and case['dtype'] == 'float32')
        reference = dict(original['outputs'], **refs.get(case['id'], {}))
        gates = compare(B, R, profile, case, raw['outputs'], reference)
        (matrix if group == 'matrix' else transfers).append({'case_id': case['id'], 'route': route, 'status': 'PASS' if all(g['status'] == 'PASS' for g in gates.values()) else 'FAIL', 'gates': gates})
    B.require(set(rec['artifacts']) == required, 'ARTIFACT_MATRIX')
    status = 'PASS' if all(row['status'] == 'PASS' for row in matrix + transfers) else 'FAIL'
    return {'kind': PROTOCOL['kind'], 'record_validation': 'PASS', 'byte_integrity': 'PASS', 'numerical_status': status,
            'api42_status': status, 'transfer_status': 'PASS' if all(row['status'] == 'PASS' for row in transfers) else 'FAIL',
            'm3_status': 'NOT_QUALIFIED', 'cuda_golden': 'NOT_RUN', 'matrix': matrix, 'transfers': transfers,
            'precision': rec['precision'], 'hlo_scope': PROTOCOL['hlo_scope'], 'state_restore': 'NOT_RUN'}


def transfer_case(B, R, case, route, torch, bnb, device):
    dtype = getattr(torch, case['dtype'])
    weight = torch.tensor(case['weight'], dtype=dtype).reshape(case['weight_shape'])
    marker = {'probe': 'reviewed-public-transfer', 'case_id': case['id']}
    if route == 'params_to_xla':
        module = None
        parameter = bnb.nn.Params4bit(weight, requires_grad=False, blocksize=64, compress_statistics=False, quant_type='nf4', quant_storage=torch.uint8)
    else:
        bias = torch.tensor(case['bias'], dtype=dtype)
        module = R.bnb_module_cpu(case, weight, bias, bnb, torch)
        parameter = module.weight
    parameter.transfer_probe_marker = marker
    old_id = id(parameter); old_module_id = id(module)
    if module is None:
        transferred = parameter.to(device)
        B.require(transferred is parameter, 'PARAMETER_IDENTITY')
        state = parameter.quant_state
        outputs = {'packed': parameter.data, 'absmax': state.absmax, 'code': state.code,
                   'decoded': bnb.functional.dequantize_4bit(parameter.data, quant_state=state)}
        before = parameter.detach().clone()
    else:
        transferred = module.to(device).train()
        B.require(transferred is module, 'MODULE_IDENTITY')
        parameter = module.weight; state = parameter.quant_state
        outputs, before = R.compute(case, module, bnb, torch, device)
    B.require(type(parameter) is bnb.nn.Params4bit and type(state) is bnb.functional.QuantState and (module is None or type(module) is bnb.nn.Linear4bit), 'ORIGINAL_CLASSES')
    raw = R.base_raw(case, {'state': state})
    raw.update(original_classes=True, parameter_identity=id(parameter) == old_id,
               custom_attribute_retained=getattr(parameter, 'transfer_probe_marker', None) is marker,
               quant_state_class='bitsandbytes.functional.QuantState',
               parameter_attrs={'blocksize': parameter.blocksize, 'compress_statistics': parameter.compress_statistics,
                                'quant_type': parameter.quant_type, 'quant_storage': str(parameter.quant_storage).removeprefix('torch.'), 'bnb_quantized': parameter.bnb_quantized},
               weight_requires_grad=parameter.requires_grad, weight_grad_is_none=parameter.grad is None,
               parameter_module_alias=parameter.module is module,
               quant_state_placements={'absmax': str(state.absmax.device), 'code': str(state.code.device)},
               gradients='EXECUTED' if module is not None and case['dtype'] == 'float32' else 'NOT_RUN')
    if module is not None:
        raw.update(module_identity=id(module) == old_module_id, module_state_alias=module.quant_state is state,
                   module_training=module.training, bias_placement=str(module.bias.device),
                   bias_class=type(module.bias).__module__+'.'+type(module.bias).__name__)
    return outputs, raw, before, parameter, module


def run(B, R, P, A, args):
    profile, inputs = B.load_spec(); output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    rec = identity(B, R, P, A, args); admission = None; roots = {}; bnb = None
    prepare = args.command == 'prepare'; receipt_name = 'oracle-seal.json' if prepare else 'receipt.json'
    try:
        if not prepare:
            B.require(time.time() < args.deadline_epoch, 'OWNER_DEADLINE')
            verify_oracle(B, A, args); rec['oracle_sha256'] = args.oracle_sha256
            P.validate_environment(B, rec['precision_environment'])
        admission, roots = A.admit(B, args.admission, args.admission_sha256, args.patch_manifest)
        rec['source_pre'] = args.admission_sha256; versions = B.runtime_check(not prepare)
        import torch
        B.require(torch.__version__ == '2.9.0+cpu', 'RUNTIME_LOADED_TORCH')
        if not prepare:
            import torch_xla
            import torch_xla.backends as backends
            backends.set_mat_mul_precision('highest')
            rec['precision'] = {'requested': 'highest', 'readback': backends.get_mat_mul_precision(), 'set_calls': 1, 'before_graph': True}
            B.require(rec['precision']['readback'] == 'highest', 'PRECISION_READBACK')
            import torch_xla.core.xla_model as xm
            import torch_xla.debug.metrics as metrics
            B.require(torch_xla.__version__.split('+')[0] == '2.9.0', 'RUNTIME_LOADED_XLA')
            device = xm.xla_device(); B.require(device.type == 'xla' and xm.xla_device_hw(device) == 'TPU', 'ACTUAL_TPU_REQUIRED')
            rec.update(runtime=versions, device={'type': 'xla', 'hardware': 'TPU', 'pjrt': 'TPU'})
        else: device = 'cpu'
        import bitsandbytes as bnb
        rec['public_api'] = A.public_methods(B, bnb, roots)
        if prepare:
            functions = B.original_cpu_functions(roots['bitsandbytes'])
            for case in inputs['cases']:
                outputs, raw = B.run_case(case, torch, bnb, device, functions)
                raw['outputs'] = {k: B.tensor_record(v) for k, v in outputs.items()}; raw['counters'] = {}
                B.validate_record(raw, case, False); B.write(output/'raw'/(case['id']+'.json'), raw)
            rec.update(kind='CPU_ORACLE', runtime={'python': versions['python'], 'torch': versions['torch']}, oracle_method=B.ORACLE_METHOD)
        else:
            rec['dispatch'] = {op: torch._C._dispatch_has_kernel_for_dispatch_key(op, 'XLA') for op in B.DISPATCH_OPS}
            B.require(all(rec['dispatch'].values()), 'DISPATCH')
            cases = P.selection(B, R)
            originals = {c['id']: B.read(Path(args.oracle)/'raw'/(c['id']+'.json')) for c in cases}
            B.write(output/'references.json', {c['id']: R.cpu_gradient(c, originals[c['id']], torch, B) for c in cases})
            rows = [('transfer', route, case) for case in cases for route in ROUTES] + [('matrix', 'api42', case) for case in inputs['cases']]
            for group, route, case in rows:
                B.require(time.time() < args.deadline_epoch, 'OWNER_DEADLINE')
                A.public_methods(B, bnb, roots)
                prefix = output/group/(route+'-'+case['id']); prefix.parent.mkdir(exist_ok=True)
                raw = row_identity(B, rec, case, route)
                metrics.clear_all()
                try:
                    if group == 'transfer':
                        outputs, details, before, parameter, module = transfer_case(B, R, case, route, torch, bnb, device)
                        raw.update(details)
                    else: outputs, details = B.run_case(case, torch, bnb, device); raw.update(details)
                    hlo = torch_xla._XLAC._get_xla_tensors_hlo(list(outputs.values()))
                    prefix.with_suffix('.hlo.txt').write_text(hlo)
                    raw['hlo_precision'] = P.hlo_precision(hlo) if route != 'params_to_xla' and case['op'] == 'linear' and case['dtype'] == 'float32' and len(case['x_shape']) == 2 else None
                    xm.mark_step(wait=True); xm.wait_device_ops()
                    raw['counters'] = {key: metrics.counter_value(key) for key in metrics.counter_names()}
                    raw['execution_metrics'] = {key: list(data) for key in profile['execution_metrics'] if (data := metrics.metric_data(key)) is not None}
                    prefix.with_suffix('.metrics.txt').write_text(metrics.metrics_report())
                    raw['outputs'] = {k: B.tensor_record(v) for k, v in outputs.items()}
                    raw['placements'] = {k: str(v.device) for k, v in outputs.items()}
                    if group == 'transfer':
                        raw.update(packed_before=B.tensor_record(before), packed_after=B.tensor_record(parameter.data))
                        if module is not None: raw['bias'] = B.tensor_record(module.bias)
                except Exception as error:
                    raw.update(status='ERROR', error_type=type(error).__name__, error_message=str(error), traceback=traceback.format_exc())
                    for suffix in ('.hlo.txt', '.metrics.txt'):
                        if not prefix.with_suffix(suffix).exists(): prefix.with_suffix(suffix).write_text('NOT_RUN_CASE_ERROR\n')
                B.write(prefix.with_suffix('.json'), raw)
        A.public_methods(B, bnb, roots)
        B.require(not any(n == 'jax' or n.startswith('jax.') or n.startswith('torchax') for n in sys.modules), 'UNEXPECTED_JAX_TORCHAX_IMPORT')
        rec['status'] = 'COMPLETE'
    except Exception:
        rec['status'] = 'FAILED'; (output/'error.log').write_text(traceback.format_exc())
    finally:
        if admission is not None:
            try:
                A.verify_installed(B, admission, roots)
                if bnb is not None: A.public_methods(B, bnb, roots)
                rec['source_post'] = args.admission_sha256
            except Exception:
                rec['status'] = 'FAILED'; (output/'source-post.error.log').write_text(traceback.format_exc())
        rec['artifacts'] = B.inventory(output, receipt_name); B.write(output/receipt_name, rec)
    if rec['status'] != 'COMPLETE':
        print(json.dumps({'record_validation': 'FAIL', 'api42_status': 'FAIL', 'reason': 'See retained error.log'})); return 1
    if prepare:
        print(json.dumps({'oracle_status': 'SEALED', 'oracle_sha256': B.sha(output/receipt_name), 'cases': 42})); return 0
    args.actual = output
    audit = verify(B, R, P, A, args); print(json.dumps(audit, allow_nan=False))
    return 0 if audit['numerical_status'] == 'PASS' else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'execute', 'verify'):
        cmd = sub.add_parser(command)
        for option in ('backend-probe', 'route-probe', 'precision-probe', 'patch-manifest', 'admission-sha256'):
            cmd.add_argument('--'+option, required=True)
        if command != 'verify':
            cmd.add_argument('--admission', required=True); cmd.add_argument('--output', required=True)
        else: cmd.add_argument('--actual', required=True)
        if command != 'prepare':
            cmd.add_argument('--oracle', required=True); cmd.add_argument('--oracle-sha256', required=True)
        if command == 'execute': cmd.add_argument('--deadline-epoch', type=float, required=True)
    args = parser.parse_args(argv)
    try:
        B, R, P, A = helpers(args)
        if args.command == 'verify':
            audit = verify(B, R, P, A, args); print(json.dumps(audit, allow_nan=False))
            return 0 if audit['numerical_status'] == 'PASS' else 2
        return run(B, R, P, A, args)
    except Exception as error:
        print(json.dumps({'record_validation': 'FAIL', 'api42_status': 'FAIL', 'm3_status': 'NOT_QUALIFIED', 'reason': str(error)})); return 1


if __name__ == '__main__':
    raise SystemExit(main())
