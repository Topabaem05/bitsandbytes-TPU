"""Keep independent NF4 route records. This diagnostic cannot qualify M2 or M3."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

BACKEND_SHA = 'e8743e33e6a0454d5d05b77075d47a5c9f983cf32f3956c38f2b544f86f5ef6d'
CASE_IDS = ['linear-float32-rank2-bias1', 'linear-bfloat16-rank2-bias1']
ROUTES = ['cpu_to_xla', 'target_constructor', 'from_prequantized']
KNOWN_ERROR = 'Attempted to call `variable.set_data(tensor)`, but `variable` and `tensor` have incompatible tensor type.'
PROTOCOL = {'kind': 'DEVICE_ROUTE_DIAGNOSTIC_ONLY', 'backend_probe_sha256': BACKEND_SHA,
            'route_case_ids': CASE_IDS, 'routes': ROUTES, 'nonlinear': 'all34 fixed quantize/decode cases',
            'gradient': 'FP32 only; fixed dyadic dy and CPU dense diagnostic from sealed decoded weights',
            'roundtrip': 'same PID only; exact state/packed, unchanged gates for floating outputs; NOT M3 new-process evidence',
            'quantization_source': {'cpu_to_xla':'TPU_PUBLIC_MODULE_QUANTIZE','target_constructor':'TPU_PUBLIC_MODULE_QUANTIZE','from_prequantized':'CPU_ORACLE_CHECKPOINT'}, 'cuda_golden': 'NOT_RUN'}
PROTOCOL_SHA = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def backend(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != BACKEND_SHA:
        raise ValueError('BACKEND_SOURCE_IDENTITY')
    spec = importlib.util.spec_from_file_location('route_backend', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selection(B):
    _, inputs = B.load_spec()
    nonlinear = [c for c in inputs['cases'] if c['op'] != 'linear']
    cases = {c['id']: c for c in inputs['cases']}
    B.require(len(nonlinear) == 34 and all(k in cases for k in CASE_IDS), 'DIAGNOSTIC_MATRIX')
    return nonlinear, [cases[k] for k in CASE_IDS]


def target_module(case, bnb, torch, device):
    dtype = getattr(torch, case['dtype'])
    n, k = case['weight_shape']
    module = bnb.nn.Linear4bit(k, n, bias=case['bias'] is not None, compute_dtype=dtype,
                              compress_statistics=False, quant_type='nf4', quant_storage=torch.uint8, device=device)
    return module.to(dtype=dtype).train()


def checkpoint(case, raw, bnb, torch):
    def tensor(value):
        return torch.tensor(value['values'], dtype=getattr(torch, value['dtype']), device='cpu').reshape(value['shape'])
    outputs = raw['outputs']
    state = bnb.functional.QuantState(absmax=tensor(outputs['absmax']), shape=torch.Size(case['weight_shape']),
        dtype=getattr(torch, case['dtype']), blocksize=64, code=tensor(outputs['code']), quant_type='nf4')
    result = {'weight': tensor(outputs['packed']), **{'weight.' + k: v for k, v in state.as_dict(packed=True).items()}}
    if case['bias'] is not None:
        result['bias'] = torch.tensor(case['bias'], dtype=getattr(torch, case['dtype']), device='cpu')
    return result


def restore_target(case, saved, bnb, torch, device):
    module = target_module(case, bnb, torch, device)
    stats = {k.removeprefix('weight.'): v for k, v in saved.items() if k.startswith('weight.')}
    # Public reconstruction creates a NEW target TensorImpl. No cross-device .data assignment.
    module.weight = bnb.nn.Params4bit.from_prequantized(saved['weight'], dict(stats), requires_grad=False,
                                                       device=device, module=module)
    module.load_state_dict({k: saved[k] for k in ('weight', 'bias') if k in saved}, strict=True)
    return module


def make_route(route, case, saved, bnb, torch, device):
    if route == 'from_prequantized':
        return restore_target(case, saved, bnb, torch, device)
    dtype = getattr(torch, case['dtype'])
    weight = torch.tensor(case['weight'], dtype=dtype, device='cpu').reshape(case['weight_shape'])
    bias = None if case['bias'] is None else torch.tensor(case['bias'], dtype=dtype, device='cpu')
    if route == 'cpu_to_xla':
        # Keep the required workflow, including any known error. Do not rescue it.
        return bnb_module_cpu(case, weight, bias, bnb, torch).to(device).train()
    if route != 'target_constructor':
        raise ValueError('UNKNOWN_ROUTE')
    module = target_module(case, bnb, torch, device)
    with torch.no_grad():
        module.weight.copy_(weight.to(device))
        if bias is not None:
            module.bias.copy_(bias.to(device))
    return module.to(device).train()


def bnb_module_cpu(case, weight, bias, bnb, torch):
    dtype = getattr(torch, case['dtype'])
    n, k = case['weight_shape']
    module = bnb.nn.Linear4bit(k, n, bias=bias is not None, compute_dtype=dtype,
        compress_statistics=False, quant_type='nf4', quant_storage=torch.uint8).to(dtype=dtype)
    with torch.no_grad():
        module.weight.copy_(weight)
        if bias is not None:
            module.bias.copy_(bias)
    return module


def compute(case, module, bnb, torch, device):
    x = torch.tensor(case['x'], dtype=getattr(torch, case['dtype']), device=device).reshape(case['x_shape'])
    x.requires_grad_(case['dtype'] == 'float32')
    before = module.weight.detach().clone()
    y = module(x)
    if x.requires_grad:
        dy = torch.tensor([((i % 5) - 2) / 8 for i in range(y.numel())], dtype=y.dtype, device=device).reshape(y.shape)
        y.backward(dy)
    state = module.weight.quant_state
    outputs = {'packed': module.weight.data, 'absmax': state.absmax, 'code': state.code,
               'decoded': bnb.functional.dequantize_4bit(module.weight.data, quant_state=state), 'y': y}
    if x.requires_grad:
        outputs['dx'] = x.grad
        if module.bias is not None:
            outputs['db'] = module.bias.grad
    return outputs, before


def cpu_gradient(case, raw, torch, B):
    dtype = getattr(torch, case['dtype'])
    result = {}
    if case['dtype'] == 'float32':
        x = torch.tensor(case['x'], dtype=dtype, device='cpu').reshape(case['x_shape']).requires_grad_(True)
        weights = raw['outputs']['decoded']
        w = torch.tensor(weights['values'], dtype=dtype, device='cpu').reshape(weights['shape'])
        bias = torch.tensor(case['bias'], dtype=dtype, device='cpu', requires_grad=True)
        y = torch.nn.functional.linear(x, w, bias)
        dy = torch.tensor([((i % 5) - 2) / 8 for i in range(y.numel())], dtype=dtype).reshape(y.shape)
        y.backward(dy)
        result = {k: B.tensor_record(v) for k, v in {'dx': x.grad, 'db': bias.grad}.items()}
    return result


def gradient_math(case, raw):
    """Independent real-valued dX/db witness; validate FP32 references with fixed gates."""
    if case['dtype'] != 'float32': return {}
    m, k = case['x_shape']; n = case['weight_shape'][0]
    weights = raw['outputs']['decoded']['values']
    dy = [((i % 5) - 2) / 8 for i in range(m*n)]
    return {'dx': {'shape':[m,k], 'dtype':'float32',
                   'values':[sum(dy[i*n+j]*weights[j*k+q] for j in range(n)) for i in range(m) for q in range(k)]},
            'db': {'shape':[n], 'dtype':'float32', 'values':[sum(dy[i*n+j] for i in range(m)) for j in range(n)]}}


def tolerance(profile, case, name):
    return ({'exact':True} if name=='packed' else
            profile['tolerances']['fp32_forward_gradient' if case['dtype']=='float32' else 'bf16_forward_unit_scale'] if name in ('y','dx','db') else
            profile['tolerances']['fp32_decode'])


def base_raw(case, outputs):
    state = outputs['state'] if 'state' in outputs else None
    return {'metadata': {'shape': list(state.shape), 'dtype': str(state.dtype).removeprefix('torch.'),
        'blocksize': state.blocksize, 'quant_type': state.quant_type, 'nested': state.nested,
        'packing_format_for_cpu': bool(getattr(state, 'packing_format_for_cpu', False))}} if state else {}


def captured(route, case, module, saved, torch, bnb, device, B):
    if type(module) is not bnb.nn.Linear4bit or type(module.weight) is not bnb.nn.Params4bit:
        raise ValueError('ORIGINAL_MODULE_CLASSES')
    if type(module.weight.quant_state) is not bnb.functional.QuantState:
        raise ValueError('ORIGINAL_QUANTSTATE_CLASS')
    outputs, before = compute(case, module, bnb, torch, device)
    state = module.weight.quant_state
    raw = base_raw(case, {'state': state})
    raw.update(original_classes=True, module_training=module.training,
        module_state_alias=module.quant_state is state, module_state_absent=module.quant_state is None,
        weight_requires_grad=module.weight.requires_grad, weight_grad_is_none=module.weight.grad is None, bnb_quantized=module.weight.bnb_quantized)
    checkpoint_tensors = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
    # This is a local-process diagnostic, not M3's distinct-process restoration.
    reconstructed = restore_target(case, checkpoint_tensors, bnb, torch, device)
    if type(reconstructed) is not bnb.nn.Linear4bit or type(reconstructed.weight) is not bnb.nn.Params4bit or type(reconstructed.weight.quant_state) is not bnb.functional.QuantState:
        raise ValueError('ORIGINAL_RESTORED_CLASSES')
    if reconstructed.weight.quant_state is module.weight.quant_state or reconstructed.weight.quant_state is not reconstructed.quant_state:
        raise ValueError('FRESH_QUANTSTATE_ALIAS')
    again, _ = compute(case, reconstructed, bnb, torch, device)
    raw['state_dict'] = {k: B.tensor_record(v) for k, v in checkpoint_tensors.items()}
    raw['restored_state_dict'] = {k:B.tensor_record(v) for k,v in reconstructed.state_dict().items()}
    return outputs, raw, {'packed_before': before, 'packed_after': module.weight.data}, again


def verify_raw(B, raw, case, route=None):
    B.require(raw['case_id'] == case['id'] and raw['input_sha256'] == B.case_sha(case), 'INPUT_BINDING')
    B.require(raw['status'] in ('PASS', 'ERROR'), 'RECORD_STATUS')
    B.require(raw.get('quantization_source') == (PROTOCOL['quantization_source'][route] if route else 'TPU_FUNCTIONAL_QUANTIZE_OR_DIRECT_DECODE'), 'QUANTIZATION_SOURCE')
    if raw['status'] == 'ERROR':
        B.require(isinstance(raw.get('error_type'), str) and isinstance(raw.get('error_message'), str) and raw.get('traceback'), 'ERROR_RECORD')
        expected = route == 'cpu_to_xla' and raw.get('stage') == 'construct' and raw['error_type'] == 'RuntimeError' and raw['error_message'] == KNOWN_ERROR
        B.require(raw.get('expected_error_observed') is expected, 'EXPECTED_ERROR_CLASSIFICATION')
        return
    B.require(raw.get('expected_error_observed') is False, 'EXPECTED_ERROR_CLASSIFICATION')
    B.require(set(B.expected_outputs(case)).issubset(raw.get('outputs', {})), 'OUTPUT_SET')
    base = {k: raw[k] for k in ('case_id','input_sha256','metadata','placements','counters','execution_metrics','outputs')}
    expected = B.expected_outputs(case)
    base['outputs'] = {k: raw['outputs'][k] for k in expected}
    base['placements'] = {k: raw['placements'][k] for k in expected}
    B.validate_record(base, case, True)
    keys = set(expected)
    if route:
        if case['dtype'] == 'float32': keys |= {'dx', 'db'}
        B.require(raw.get('original_classes') is True and raw.get('module_training') is True, 'ORIGINAL_MODULE_CLASSES')
        B.require(raw.get('module_state_alias') is True or raw.get('module_state_absent') is True, 'STATE_ALIAS')
        B.require(raw.get('weight_requires_grad') is False and raw.get('weight_grad_is_none') is True and raw.get('bnb_quantized') is True, 'FROZEN_WEIGHT')
        B.require(raw.get('packed_before') == raw['outputs']['packed'] == raw.get('packed_after'), 'PACKED_CHANGED')
        B.require(raw.get('roundtrip_scope') == 'SAME_PID_DIAGNOSTIC_ONLY' and set(raw.get('roundtrip_outputs', {})) == keys, 'STATE_ROUNDTRIP')
        B.require(raw['roundtrip_outputs']['packed'] == raw['outputs']['packed'], 'STATE_PACKED_ROUNDTRIP')
        state = raw.get('state_dict', {})
        B.require(raw.get('restored_state_dict') == state, 'STATE_VALUE_ROUNDTRIP')
        B.require(set(state) == {'weight','bias','weight.absmax','weight.quant_map','weight.quant_state.bitsandbytes__nf4'}, 'STATE_KEYS')
        B.require(state['weight'] == raw['outputs']['packed'] and state['weight.absmax'] == raw['outputs']['absmax'] and state['weight.quant_map'] == raw['outputs']['code'], 'STATE_VALUES')
        B.require(state['bias'] == {'shape':[case['weight_shape'][0]],'dtype':case['dtype'],'values':case['bias']}, 'STATE_BIAS')
        blob = state['weight.quant_state.bitsandbytes__nf4']
        B.require(blob['dtype'] == 'uint8' and blob['shape'] == [len(blob['values'])] and json.loads(bytes(blob['values']).decode()) == {'quant_type':'nf4','blocksize':64,'dtype':case['dtype'],'shape':case['weight_shape']}, 'STATE_METADATA')
    B.require(set(raw['outputs']) == keys and set(raw['placements']) == keys, 'OUTPUT_SET')
    for k in keys:
        array = raw['outputs'][k]
        shape = case['x_shape'] if k == 'dx' else [case['weight_shape'][0]] if k == 'db' else expected[k][0]
        dtype = case['dtype'] if k in ('dx','db') else expected[k][1]
        B.require(array['shape'] == shape and array['dtype'] == dtype and len(array['values']) == math.prod(shape), 'ARRAY_IDENTITY')
        B.require(all(type(v) in (int,float) and math.isfinite(v) for v in array['values']), 'NONFINITE_OUTPUT')
        B.require(raw['placements'][k].startswith('xla:'), 'PLACEMENT')
        if route:
            again = raw['roundtrip_outputs'][k]
            B.require(again['shape']==shape and again['dtype']==dtype and len(again['values'])==math.prod(shape) and all(type(v) in (int,float) and math.isfinite(v) for v in again['values']), 'ROUNDTRIP_ARRAY')


def verify(B, args):
    profile, _ = B.load_spec()
    B.verify_oracle(args.oracle, args.oracle_sha256, args.admission_sha256)
    root = Path(args.actual)
    rec = B.read(root/'receipt.json')
    nonlinear, cases = selection(B)
    B.require(rec.get('kind') == PROTOCOL['kind'] and rec.get('status') == 'COMPLETE' and rec.get('protocol_sha256') == PROTOCOL_SHA, 'DIAGNOSTIC_TERMINAL')
    B.require(rec.get('source_pre') == rec.get('source_post') == rec.get('source_admission_sha256') == args.admission_sha256, 'SOURCE_BINDING')
    B.require(rec.get('oracle_sha256') == args.oracle_sha256 and rec.get('backend_probe_sha256') == BACKEND_SHA and rec.get('probe_sha256') == B.sha(Path(__file__)), 'PARENT_BINDING')
    B.require(rec.get('runtime') == profile['runtime'] and rec.get('runtime_lock_sha256') == profile['runtime_lock_sha256'], 'RUNTIME_PAIR')
    B.require(rec.get('device') == {'type':'xla','hardware':'TPU','pjrt':'TPU'}, 'TPU_REQUIRED')
    B.require(rec.get('public_api') == {'Linear4bit':'bitsandbytes.nn.modules.Linear4bit','Params4bit':'bitsandbytes.nn.modules.Params4bit','original_class_identity':True}, 'PUBLIC_API')
    B.require(rec.get('dispatch') == {op:True for op in B.DISPATCH_OPS}, 'DISPATCH')
    B.require(type(rec.get('pid')) is int and rec['pid'] > 0 and isinstance(rec.get('process_token'),str) and len(rec['process_token']) == 32, 'PROCESS_IDENTITY')
    B.require(rec.get('profile_sha256') == B.PROFILE_SHA and rec.get('inputs_sha256') == B.INPUTS_SHA and rec.get('cuda_golden') == 'NOT_RUN', 'SPEC_IDENTITY')
    B.require(rec.get('nonlinear_case_ids') == [c['id'] for c in nonlinear] and rec.get('route_case_ids') == CASE_IDS and rec.get('routes') == ROUTES, 'DIAGNOSTIC_MATRIX')
    B.check_seal(root, 'receipt.json', rec)
    rows = []; required={'references.json'}
    refs = B.read(root/'references.json')
    B.require(set(refs) == set(CASE_IDS), 'REFERENCE_MATRIX')
    for case in cases:
        original = B.read(Path(args.oracle)/'raw'/(case['id']+'.json'))
        witness = gradient_math(case,original)
        expected_refs = {'dx':case['x_shape'], 'db':[case['weight_shape'][0]]} if case['dtype']=='float32' else {}
        B.require(set(refs[case['id']]) == set(expected_refs),'GRADIENT_REFERENCE_SET')
        for name, shape in expected_refs.items():
            array = refs[case['id']][name]
            B.require(array['shape'] == shape and array['dtype']=='float32' and len(array['values'])==math.prod(shape) and all(type(v) in (int,float) and math.isfinite(v) for v in array['values']), 'GRADIENT_REFERENCE_ARRAY')
            B.require(B.numeric(array['values'],witness[name]['values'],profile['tolerances']['fp32_forward_gradient'])['status']=='PASS', 'GRADIENT_REFERENCE_WITNESS')
    for collection, route, case in [('nonlinear',None,c) for c in nonlinear]+[('routes',r,c) for c in cases for r in ROUTES]:
        name = f'raw/{collection}-' + (route+'-' if route else '') + case['id']
        raw = B.read(root/(name+'.json'));required.add(name+'.json')
        B.require(raw.get('collection') == collection and raw.get('route') == route and raw.get('pid') == rec['pid'] and raw.get('process_token') == rec['process_token'], 'ROW_ORIGIN')
        verify_raw(B,raw,case,route)
        if raw['status'] == 'ERROR':
            B.require(raw['traceback'] == name+'.error.log', 'ERROR_PATH');required.add(raw['traceback'])
            B.require((root/raw['traceback']).stat().st_size > 0, 'EMPTY_ERROR')
            for suffix in ('.hlo.txt','.metrics.txt'):
                if (root/(name+suffix)).exists(): required.add(name+suffix)
            rows.append({'collection':collection,'route':route,'case_id':case['id'],'status':'ERROR','expected_error_observed':raw['expected_error_observed']});continue
        required |= {name+'.hlo.txt',name+'.metrics.txt'}
        for suffix in ('.hlo.txt','.metrics.txt'): B.require((root/(name+suffix)).stat().st_size>0,'EMPTY_EXECUTION_RECORD')
        reference = B.read(Path(args.oracle)/'raw'/(case['id']+'.json'))['outputs']
        if route: reference = dict(reference, **refs[case['id']])
        gates={}
        for key,array in raw['outputs'].items():
            gate_tolerance = tolerance(profile,case,key)
            B.require(array['shape'] == reference[key]['shape'] and array['dtype'] == reference[key]['dtype'], 'REFERENCE_SHAPE')
            gates[key] = B.numeric(array['values'],reference[key]['values'],gate_tolerance)
        roundtrip_gates = {key:B.numeric(raw['roundtrip_outputs'][key]['values'],array['values'],tolerance(profile,case,key)) for key,array in raw['outputs'].items()} if route else {}
        rows.append({'collection':collection,'route':route,'case_id':case['id'],'status':'PASS','gates':gates,'roundtrip_gates':roundtrip_gates,'quantization_source':raw['quantization_source'],'expected_error_observed':False})
    B.require(set(rec['artifacts']) == required, 'ARTIFACT_MATRIX')
    return {'record_validation':'PASS','api42_status':'NOT_QUALIFIED','m3_status':'NOT_QUALIFIED',
        'numerical_status':'FAIL' if any(g['status']=='FAIL' for row in rows for g in [*row.get('gates',{}).values(),*row.get('roundtrip_gates',{}).values()]) else 'INCOMPLETE' if any(r['status']=='ERROR' for r in rows) else 'PASS',
        'scope':'Diagnostic only. Per-route PASS is returned execution, not required CPU-to-XLA feature qualification. Owner supplies external device/process/lifecycle proof.', 'rows':rows}


def execute(B,args):
    profile,_=B.load_spec();nonlinear,cases=selection(B)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    rec={'kind':PROTOCOL['kind'],'status':'PARTIAL','protocol_sha256':PROTOCOL_SHA,'probe_sha256':B.sha(Path(__file__)),
        'backend_probe_sha256':BACKEND_SHA,'profile_sha256':B.PROFILE_SHA,'inputs_sha256':B.INPUTS_SHA,
        'source_admission_sha256':args.admission_sha256,'oracle_sha256':args.oracle_sha256,'runtime_lock_sha256':profile['runtime_lock_sha256'],
        'pid':os.getpid(),'process_token':uuid.uuid4().hex,'cuda_golden':'NOT_RUN','routes':ROUTES,'route_case_ids':CASE_IDS,
        'nonlinear_case_ids':[c['id'] for c in nonlinear]}
    roots={};admission=None
    try:
        B.verify_oracle(args.oracle,args.oracle_sha256,args.admission_sha256)
        admission,roots=B.admit(args.admission,args.admission_sha256);rec['source_pre']=args.admission_sha256
        rec['runtime']=B.runtime_check(True)
        import torch
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.debug.metrics as metrics
        B.require(torch.__version__=='2.9.0+cpu' and torch_xla.__version__.split('+')[0]=='2.9.0','RUNTIME_LOADED_PAIR')
        device=xm.xla_device();B.require(device.type=='xla' and xm.xla_device_hw(device)=='TPU','ACTUAL_TPU_REQUIRED')
        rec['device']={'type':'xla','hardware':'TPU','pjrt':'TPU'}
        import bitsandbytes as bnb
        rec['public_api']=B.public_identity(bnb,roots)
        rec['dispatch']={op:torch._C._dispatch_has_kernel_for_dispatch_key(op,'XLA') for op in B.DISPATCH_OPS}
        B.require(all(rec['dispatch'].values()),'DISPATCH')
        references={c['id']:cpu_gradient(c,B.read(Path(args.oracle)/'raw'/(c['id']+'.json')),torch,B) for c in cases}
        B.write(output/'references.json',references)
        (output/'raw').mkdir()
        for collection,route,case in [('nonlinear',None,c) for c in nonlinear]+[('routes',r,c) for c in cases for r in ROUTES]:
            B.require(time.time()<args.deadline_epoch,'OWNER_DEADLINE')
            name='raw/'+collection+'-'+(route+'-' if route else '')+case['id'];prefix=output/name
            raw={'collection':collection,'route':route,'case_id':case['id'],'input_sha256':B.case_sha(case),
                 'pid':rec['pid'],'process_token':rec['process_token'],'status':'ERROR','stage':'construct','expected_error_observed':False,
                 'quantization_source':PROTOCOL['quantization_source'][route] if route else 'TPU_FUNCTIONAL_QUANTIZE_OR_DIRECT_DECODE'}
            try:
                metrics.clear_all()
                if route:
                    original=B.read(Path(args.oracle)/'raw'/(case['id']+'.json'))
                    saved=checkpoint(case,original,bnb,torch)
                    module=make_route(route,case,saved,bnb,torch,device)
                    raw['stage']='forward_gradient_state'
                    outputs,details,packed,again=captured(route,case,module,saved,torch,bnb,device,B)
                    raw.update(details)
                else:
                    outputs,details=B.run_case(case,torch,bnb,device)
                    raw.update(details)
                raw['stage']='synchronize'
                prefix.with_suffix('.hlo.txt').write_text(torch_xla._XLAC._get_xla_tensors_hlo(list(outputs.values()) + (list(again.values()) if route else [])))
                xm.mark_step(wait=True);xm.wait_device_ops()
                raw['counters']={k:metrics.counter_value(k) for k in metrics.counter_names()}
                raw['execution_metrics']={k:list(data) for k in profile['execution_metrics'] if (data:=metrics.metric_data(k)) is not None}
                prefix.with_suffix('.metrics.txt').write_text(metrics.metrics_report())
                raw['placements']={k:str(v.device) for k,v in outputs.items()}
                raw['outputs']={k:B.tensor_record(v) for k,v in outputs.items()}
                if route:
                    raw.update({k:B.tensor_record(v) for k,v in packed.items()})
                    raw['roundtrip_outputs']={k:B.tensor_record(v) for k,v in again.items()}
                    raw['roundtrip_scope']='SAME_PID_DIAGNOSTIC_ONLY'
                raw['status']='PASS';raw['stage']='record_validation'
                verify_raw(B,raw,case,route)
            except Exception as error:
                raw.update(status='ERROR',error_type=type(error).__name__,error_message=str(error),traceback=name+'.error.log',
                    expected_error_observed=route=='cpu_to_xla' and raw['stage']=='construct' and type(error).__name__=='RuntimeError' and str(error)==KNOWN_ERROR)
                prefix.with_suffix('.error.log').write_text(traceback.format_exc())
            B.write(prefix.with_suffix('.json'),raw)
        B.require(not any(n=='jax' or n.startswith('jax.') or n.startswith('torchax') for n in sys.modules),'UNEXPECTED_JAX_TORCHAX_IMPORT')
        rec['status']='COMPLETE'
    except Exception:
        rec['status']='FAILED';(output/'error.log').write_text(traceback.format_exc())
    finally:
        if admission is not None:
            try:
                for package,root in roots.items():B.verify_source_tree(root,admission[package]['files'])
                rec['source_post']=args.admission_sha256
            except Exception:
                rec['status']='FAILED';(output/'source-post.error.log').write_text(traceback.format_exc())
        rec['artifacts']=B.inventory(output,'receipt.json');B.write(output/'receipt.json',rec)
    if rec['status']=='COMPLETE':
        args.actual=output
        try:result=verify(B,args)
        except Exception:
            (output/'verifier.error.log').write_text(traceback.format_exc());rec['status']='FAILED';rec['artifacts']=B.inventory(output,'receipt.json');B.write(output/'receipt.json',rec);return 1
        print(json.dumps(result));return 0
    print(json.dumps({'record_validation':'FAIL','api42_status':'NOT_QUALIFIED','status':rec['status']}));return 1


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for command in ('execute','verify'):
        q=sub.add_parser(command);q.add_argument('--backend-probe',required=True);q.add_argument('--admission-sha256',required=True)
        q.add_argument('--oracle',required=True);q.add_argument('--oracle-sha256',required=True)
        if command=='execute':q.add_argument('--admission',required=True);q.add_argument('--deadline-epoch',type=float,required=True);q.add_argument('--output',required=True)
        else:q.add_argument('--actual',required=True)
    a=p.parse_args(argv);B=backend(a.backend_probe)
    if a.command=='execute':
        B.require(math.isfinite(a.deadline_epoch) and time.time()<a.deadline_epoch,'OWNER_DEADLINE')
        return execute(B,a)
    try:print(json.dumps(verify(B,a)));return 0
    except Exception as error:print(json.dumps({'record_validation':'FAIL','api42_status':'NOT_QUALIFIED','reason':str(error)}));return 1

if __name__=='__main__':raise SystemExit(main())
