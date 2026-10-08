"""Private M3 candidate. Device commands require separate root adoption and owner review."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import time
import traceback
import types

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SCIENCE = HERE if (HERE/'probe_backend.py').is_file() else PROJECT/'experiments/2026-10-04-bitsandbytes-tpu'
PINS = {
    'probe_backend.py':'e8743e33e6a0454d5d05b77075d47a5c9f983cf32f3956c38f2b544f86f5ef6d',
    'probe_routes.py':'feae751734e57c741b1bdade004ff7ca3c041ee7eb3b7086b531bc6be433038e',
    'probe_precision.py':'e0534955507e5f91e67ecddfffc738a367ad752e69a4eea7ae8052c6d76a8852',
    'probe_state.py':'006604d18d10c202583c5d42dfa361a83da441ea99d625241101b58d5948da45',
    'transfer_admission.py':'6a7c925211b14aba990318360c8f0ca2d4d803e4528c634135c8bde86326816b',
    'probe_transfer.py':'0179eb9cf2781c9c44ef98a472d5da937c8ca74670229dabf7f83ebbf782a5f8',
}
PROTOCOL = {'kind':'R3_PRIVATE_STATE_PREPARATION','cases':'all8 fixed linear',
    'save':'target_constructor public TPU quantization',
    'restore':'target_constructor, public from_prequantized, strict load_state_dict(assign=False)',
    'precision':'highest once before device acquisition and graphs',
    'oracle':'unchanged CPU42 seal; derived generic scalar FP32 witness',
    'roundtrip':'exact saved/restored tensor state and all outputs',
    'process':'distinct launch-bound PID and token; children inherit externally owned parent PGID; each child reaped before next launch',
    'hlo_scope':'PRE_OPTIMIZATION_ROOT_REACHABLE_DOT_OBSERVATION',
    'bf16_gradients':'NOT_RUN','qualification':'NOT_QUALIFIED'}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
def read(path): return json.loads(Path(path).read_text())
def require(condition, name):
    if not condition: raise ValueError(name)


def helpers():
    result={}
    for filename, expected in PINS.items():
        path=SCIENCE/filename;require(sha(path)==expected,'HELPER_SOURCE_'+filename)
        spec=importlib.util.spec_from_file_location('r3_'+filename.replace('.','_'),path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);result[filename]=module
    return tuple(result[name] for name in ('probe_backend.py','probe_routes.py','probe_precision.py','probe_state.py','transfer_admission.py','probe_transfer.py'))


B,R,P,S,A,T=helpers()
CASES=S.cases()
PROTOCOL_SHA=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()


def scalar_reference(case, original):
    """Generic rank-two/rank-three real-valued witness from fixed decoded weights."""
    k=case['x_shape'][-1];m=math.prod(case['x_shape'][:-1]);n=case['weight_shape'][0]
    x,w=case['x'],original['outputs']['decoded']['values']
    bias=case['bias'] or [0.0]*n
    dy=[((i%5)-2)/8 for i in range(m*n)]
    result={'y':{'shape':case['x_shape'][:-1]+[n],'dtype':case['dtype'],
                     'values':[sum(x[i*k+q]*w[j*k+q] for q in range(k))+bias[j] for i in range(m) for j in range(n)]}}
    if case['dtype']=='float32':
        result['dx']={'shape':case['x_shape'],'dtype':'float32','values':[sum(dy[i*n+j]*w[j*k+q] for j in range(n)) for i in range(m) for q in range(k)]}
        if case['bias'] is not None:
            result['db']={'shape':[n],'dtype':'float32','values':[sum(dy[i*n+j] for i in range(m)) for j in range(n)]}
    return result


def state_methods(bnb, roots):
    """Supplement reviewed admission without replacing any loaded method."""
    observed=A.public_methods(B,bnb,roots)
    for relative, classname, methods in (
        ('nn/modules.py','Params4bit',('from_prequantized',)),
        ('nn/modules.py','Linear4bit',('_save_to_state_dict',)),
        ('functional.py','QuantState',('as_dict','from_dict'))):
        path=(roots['bitsandbytes']/relative).resolve();compiled=compile(path.read_text(),str(path),'exec')
        cls=getattr(bnb.functional if classname=='QuantState' else bnb.nn,classname)
        candidates=[code for code in compiled.co_consts if isinstance(code,types.CodeType) and code.co_name==classname]
        require(len(candidates)==1,'STATE_SOURCE_CLASS')
        for method in methods:
            expected=[code for code in candidates[0].co_consts if isinstance(code,types.CodeType) and code.co_name==method]
            function=getattr(cls,method);function=getattr(function,'__func__',function)
            require(expected and isinstance(function,types.FunctionType) and function.__code__==expected[-1] and Path(function.__code__.co_filename).resolve()==path,'STATE_METHOD_REPLACEMENT')
    return dict(observed,original_state_method_identity=True)


def checkpoint_records(torch, path, case):
    state=torch.load(path,map_location='cpu',weights_only=True)
    require(type(state) is dict and set(state)==S.state_keys(case),'CHECKPOINT_STATE_KEYS')
    require(all(type(value) is torch.Tensor and value.device.type=='cpu' for value in state.values()),'CHECKPOINT_PLAIN_CPU_TENSORS')
    return state,{key:B.tensor_record(value) for key,value in state.items()}


def restore(case, state, bnb, torch, device):
    require(set(state)==S.state_keys(case),'RESTORE_STATE_KEYS')
    module=R.target_module(case,bnb,torch,device)
    original_parameter=module.weight;original_state=module.quant_state
    stats={key.removeprefix('weight.'):value for key,value in state.items() if key.startswith('weight.')}
    parameter=bnb.nn.Params4bit.from_prequantized(state['weight'],dict(stats),requires_grad=False,device=device,module=module)
    module.weight=parameter
    require(parameter is not original_parameter and parameter.quant_state is not original_state,'FRESH_PARAMETER_STATE')
    result=module.load_state_dict({key:state[key] for key in ('weight','bias') if key in state},strict=True,assign=False)
    require(not result.missing_keys and not result.unexpected_keys,'STRICT_STATE_LOAD')
    require(module.weight is parameter and module.weight.quant_state is module.quant_state,'RESTORED_PARAMETER_ALIAS')
    require(type(module) is bnb.nn.Linear4bit and type(parameter) is bnb.nn.Params4bit and type(parameter.quant_state) is bnb.functional.QuantState,'ORIGINAL_RESTORED_CLASSES')
    return module.train()


def validate_hlo(path, raw, case):
    dots=P.hlo_precision(Path(path).read_text());require(dots==raw['hlo_precision'],'HLO_BINDING')
    require(dots,'HLO_ROOT_DOT')
    if case['dtype']=='float32':
        fp32=[dot for dot in dots if dot['result_dtype']=='f32' and dot['operand_dtypes']==['f32','f32']]
        require(fp32 and all(dot['operands']==['highest','highest'] for dot in fp32),'HLO_PRECISION')
        for dot in fp32:
            lhs,rhs=dot['operand_shapes'];ld,rd=dot['lhs_contracting_dims'],dot['rhs_contracting_dims']
            require(len(ld)==len(rd)==1 and 0<=ld[0]<len(lhs) and 0<=rd[0]<len(rhs),'HLO_CONTRACTION')
            require(lhs[ld[0]]==rhs[rd[0]] and dot['result_shape']==[v for i,v in enumerate(lhs) if i!=ld[0]]+[v for i,v in enumerate(rhs) if i!=rd[0]],'HLO_CONTRACTION')
        m=math.prod(case['x_shape'][:-1]);n,k=case['weight_shape'];shapes={tuple(dot['result_shape']) for dot in fp32}
        require(any(shape in shapes for shape in (tuple(case['x_shape'][:-1]+[n]),(m,n))),'HLO_FORWARD')
        require(any(shape in shapes for shape in (tuple(case['x_shape']),(m,k))),'HLO_GRADIENT')
    P.validate_execution(B,raw)


def expected_gate_names(case):
    names=set(B.expected_outputs(case))
    if case['dtype']=='float32':
        names.add('dx')
        if case['bias'] is not None:names.add('db')
    return names


def validate_device_gates(raw, case):
    gates=raw.get('numerical_gates')
    require(type(gates) is dict and set(gates)==expected_gate_names(case),'DEVICE_GATE_INVENTORY')
    for gate in gates.values():
        require(type(gate) is dict and set(gate)=={'status','max_abs','rmse','nrmse'} and gate['status'] in ('PASS','FAIL'),'DEVICE_GATE_STATUS')
        require(all(type(gate[name]) in (int,float) and math.isfinite(gate[name]) and gate[name]>=0 for name in ('max_abs','rmse','nrmse')),'DEVICE_GATE_METRICS')
    return gates


def device_gates(case, raw, original, profile):
    witness=scalar_reference(case,original) if case['dtype']=='float32' else {}
    reference=dict(original['outputs'],**{name:value for name,value in witness.items() if name in ('dx','db')})
    gates={name:B.numeric(array['values'],reference[name]['values'],R.tolerance(profile,case,name)) for name,array in raw['outputs'].items()}
    validate_device_gates(dict(raw,numerical_gates=gates),case)
    return gates


def validate_phase(root, phase, parent, launch, local=False):
    root=Path(root);receipt=read(root/'receipt.json')
    require(receipt['status']=='COMPLETE' and receipt['phase']==phase,'PHASE_TERMINAL')
    require(receipt['protocol_sha256']==PROTOCOL_SHA and receipt['probe_sha256']==sha(__file__) and receipt['helper_pins']==PINS,'PHASE_SOURCE_PROTOCOL')
    require(receipt['pid']==launch['pid']==launch['launched_pid'] and receipt['pgid']==launch['pgid']==parent['pgid'],'ACTUAL_LAUNCH_IDENTITY')
    require(receipt['process_token']==launch['process_token'] and receipt['parent_pid']==parent['pid'] and receipt['parent_process_token']==parent['process_token'],'PARENT_CHILD_IDENTITY')
    require(receipt['deadline_epoch']==launch['deadline_epoch']<=parent['deadline_epoch'] and math.isfinite(receipt['deadline_epoch']),'CHILD_DEADLINE_BINDING')
    require(launch['exit_code'] in (0,2) and launch['reaped'] is True and launch['child_absent'] is True,'CHILD_TERMINAL')
    require(launch['cleanup_errors']==[],'CHILD_CLEANUP_ERRORS')
    require(receipt['profile_sha256']==B.PROFILE_SHA and receipt['inputs_sha256']==B.INPUTS_SHA,'FIXED_INPUTS')
    require(receipt['case_ids']==[case['id'] for case in CASES] and receipt['scope']==('LOCAL_CPU_SMOKE' if local else 'TPU_CANDIDATE'),'CASE_MATRIX_SCOPE')
    require(receipt['precision']==({'requested':'highest','readback':None,'set_calls':0,'before_graph':None} if local else {'requested':'highest','readback':'highest','set_calls':1,'before_graph':True}),'PRECISION')
    require(receipt['public_api']=={'Linear4bit':'bitsandbytes.nn.modules.Linear4bit','Params4bit':'bitsandbytes.nn.modules.Params4bit','original_class_identity':True,'original_method_identity':True,'original_state_method_identity':True},'PUBLIC_STATE_METHODS')
    if not local:
        profile,_=B.load_spec()
        require(receipt['runtime']==profile['runtime'] and receipt['runtime_lock_sha256']==profile['runtime_lock_sha256'],'RUNTIME_BINDING')
        require(receipt['source_pre']==receipt['source_post']==parent['source_admission_sha256'],'SOURCE_BINDING')
        require(receipt['oracle_sha256']==parent['oracle_sha256'] and receipt['patch_manifest_sha256']==A.PATCH_MANIFEST_SHA,'ORACLE_OVERLAY_BINDING')
        require(receipt['device']=={'type':'xla','hardware':'TPU','pjrt':'TPU'} and receipt['dispatch']=={op:True for op in B.DISPATCH_OPS},'TPU_DISPATCH')
        P.validate_environment(B,receipt['precision_environment'])
        require(receipt['precision_environment']==parent['precision_environment'],'CHILD_ENVIRONMENT')
    B.check_seal(root,'receipt.json',receipt)
    required=set();allowed=set()
    for case in CASES:
        raw=read(root/'raw'/(case['id']+'.json'))
        stem='raw/'+case['id'];required.add(stem+'.json')
        allowed.update(stem+suffix for suffix in ('.json','.state.pt','.hlo.txt','.metrics.txt','.error.log'))
        require(raw['case_id']==case['id'] and raw['input_sha256']==B.case_sha(case) and raw['pid']==receipt['pid'] and raw['process_token']==receipt['process_token'],'ROW_BINDING')
        require(raw['status'] in ('PASS','ERROR'),'CASE_STATUS')
        if raw['status']=='ERROR':
            require(raw['error_type'] and raw['error_message'] and raw['traceback'],'RETAINED_CASE_ERROR')
            required.add(stem+'.error.log');continue
        require(not any(key in raw for key in ('error_type','error_message','traceback')),'PASS_WITH_ERROR_METADATA')
        if not local:validate_device_gates(raw,case)
        required.update(stem+suffix for suffix in (('.state.pt',) if local else ('.state.pt','.hlo.txt','.metrics.txt')))
        S.validate_raw(raw,case,not local)
        require(raw['original_classes'] is True and raw['module_training'] is True and raw['module_state_alias'] is True and raw['parameter_module_alias'] is True,'CLASS_AND_STATE_ALIAS')
        require(raw['gradients']==('EXECUTED' if case['dtype']=='float32' else 'NOT_RUN'),'GRADIENT_SCOPE')
        if not local:validate_hlo(root/'raw'/(case['id']+'.hlo.txt'),raw,case)
    require(required.issubset(receipt['artifacts']) and set(receipt['artifacts']).issubset(allowed),'PHASE_ARTIFACT_MATRIX')
    return receipt


def verify_pair(root, local=False, oracle_path=None, oracle_sha=None, admission_sha=None, parent_sha=None):
    root=Path(root);parent=read(root/'parent.json');require(parent['protocol_sha256']==PROTOCOL_SHA,'PARENT_PROTOCOL')
    require(type(parent['pid']) is int and parent['pid']>0 and type(parent['pgid']) is int and parent['pgid']>0 and re.fullmatch('[0-9a-f]{32}',parent['process_token']),'PARENT_IDENTITY')
    require(type(parent.get('deadline_epoch')) in (int,float) and math.isfinite(parent['deadline_epoch']),'FINITE_PARENT_DEADLINE')
    require(parent.get('kind')=='STATE_ROUNDTRIP_PARENT' and parent.get('status')=='COMPLETE' and parent.get('probe_sha256')==sha(__file__) and parent.get('helper_pins')==PINS,'PARENT_TERMINAL_SOURCE')
    if parent_sha is not None:require(sha(root/'parent.json')==parent_sha,'PARENT_RECEIPT_HASH')
    B.check_seal(root,'parent.json',parent)
    if not local:
        require(oracle_path is not None and oracle_sha==parent['oracle_sha256'] and admission_sha==parent['source_admission_sha256'] and parent_sha is not None,'EXPLICIT_RECOVERED_BINDINGS')
        oracle=T.verify_oracle(B,A,types.SimpleNamespace(oracle=oracle_path,oracle_sha256=oracle_sha,admission_sha256=admission_sha))
    launches=parent['children'];require([row['phase'] for row in launches]==['save','restore'],'PHASE_ORDER')
    require(len({parent['pid'],*[row['pid'] for row in launches]})==3,'FRESH_PID')
    require(all(row['pgid']==parent['pgid'] for row in launches),'OWNED_PARENT_GROUP')
    if not local:require(parent['pid']==parent['pgid'],'PARENT_GROUP_LEADER')
    require(len({parent['process_token'],*[row['process_token'] for row in launches]})==3,'FRESH_TOKEN')
    for launch in launches:
        require(type(launch['pid']) is int and launch['pid']>0 and re.fullmatch('[0-9a-f]{32}',launch['process_token']),'CHILD_PID_TOKEN')
        require(all(type(launch[key]) in (int,float) and math.isfinite(launch[key]) for key in ('started_monotonic','finished_monotonic','started_epoch','finished_epoch','deadline_epoch')),'FINITE_LAUNCH_TIMES')
        require(launch['started_monotonic']<=launch['finished_monotonic'] and launch['started_epoch']<=launch['finished_epoch']<=launch['deadline_epoch']<=parent['deadline_epoch'],'LAUNCH_DEADLINES')
        require(launch['receipt']==launch['phase']+'/receipt.json' and launch['receipt_sha256']==sha(root/launch['receipt']) and launch['parent_pid']==parent['pid'] and launch['parent_process_token']==parent['process_token'] and launch['timeout'] is False,'OWNER_LAUNCH_BINDING')
    require(launches[0]['finished_monotonic']<=launches[1]['started_monotonic'],'SEQUENTIAL_GROUP_CLOSURE')
    receipts={phase:validate_phase(root/phase,phase,parent,launch,local) for phase,launch in zip(('save','restore'),launches)}
    require(receipts['restore']['saved_seal_sha256']==sha(root/'save/receipt.json') and receipts['restore']['saved_pid']==receipts['save']['pid'] and receipts['restore']['saved_process_token']==receipts['save']['process_token'],'SAVE_PROCESS_SEAL_LINK')
    import torch
    rows=[]
    for case in CASES:
        name=case['id'];saved=read(root/'save/raw'/(name+'.json'));restored=read(root/'restore/raw'/(name+'.json'))
        if saved['status']=='ERROR' or restored['status']=='ERROR':rows.append({'case_id':name,'status':'ERROR'});continue
        save_path=root/'save/raw'/(name+'.state.pt');restore_path=root/'restore/raw'/(name+'.state.pt')
        require(sha(save_path)==sha(restore_path)==restored['loaded_checkpoint_sha256'],'CHECKPOINT_BYTES')
        require(restored['saved_receipt_sha256']==sha(root/'save/receipt.json'),'SAVE_RECEIPT_LINK')
        _,actual_state=checkpoint_records(torch,save_path,case)
        require(actual_state==saved['state_dict']==restored['loaded_state_before']==restored['state_dict'],'CHECKPOINT_TENSOR_BINDING')
        require(restored['fresh_reconstruction'] is True,'FRESH_RECONSTRUCTION')
        gates=S.compare_records(restored,saved,case,exact=True)
        oracle=read(Path(oracle_path)/'raw'/(name+'.json')) if not local else saved
        if not local:
            for raw in (saved,restored):require(raw['numerical_gates']==device_gates(case,raw,oracle,B.load_spec()[0]),'DEVICE_GATE_BINDING')
        scalar=scalar_reference(case,oracle) if case['dtype']=='float32' else {}
        profile,_=B.load_spec()
        for output in ('y','dx','db'):
            if output in scalar:
                gates['scalar_'+output]=B.numeric(saved['outputs'][output]['values'],scalar[output]['values'],R.tolerance(profile,case,output))
        if not local:
            gates.update({'cpu_'+key:B.numeric(value['values'],oracle['outputs'][key]['values'],R.tolerance(profile,case,key)) for key,value in saved['outputs'].items() if key in oracle['outputs']})
            gates.update({'restore_cpu_'+key:B.numeric(value['values'],oracle['outputs'][key]['values'],R.tolerance(profile,case,key)) for key,value in restored['outputs'].items() if key in oracle['outputs']})
        rows.append({'case_id':name,'status':'PASS' if all(gate['status']=='PASS' for gate in gates.values()) else 'FAIL','gates':gates})
    return {'record_validation':'PASS','numerical_status':'PASS' if all(row['status']=='PASS' for row in rows) else 'FAIL','m3_status':'NOT_QUALIFIED','scope':'LOCAL_CPU_SMOKE' if local else 'PRIVATE_TPU_CANDIDATE_OWNER_REVIEW_REQUIRED','rows':rows}


def case_loop(args, bnb, torch, device, receipt, xla=None):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    for case in CASES:
        prefix=output/'raw'/case['id'];prefix.parent.mkdir(exist_ok=True)
        raw={'case_id':case['id'],'input_sha256':B.case_sha(case),'pid':receipt['pid'],'process_token':receipt['process_token'],'status':'ERROR'}
        try:
            require(time.time()<args.deadline_epoch,'OWNER_DEADLINE')
            if args.phase=='save':module=R.make_route('target_constructor',case,None,bnb,torch,device)
            else:
                source=Path(args.saved)/'raw'/(case['id']+'.state.pt')
                saved_receipt=read(Path(args.saved)/'receipt.json');require(sha(source)==saved_receipt['artifacts']['raw/'+case['id']+'.state.pt']['sha256'],'CHECKPOINT_SOURCE_SEAL')
                state,state_raw=checkpoint_records(torch,source,case)
                module=restore(case,state,bnb,torch,device)
                prefix.with_suffix('.state.pt').write_bytes(source.read_bytes())
                raw.update(loaded_state_before=state_raw,loaded_checkpoint_sha256=sha(source),saved_receipt_sha256=sha(Path(args.saved)/'receipt.json'),fresh_reconstruction=True)
            if xla:xla[2].clear_all()
            outputs,before=S.compute(case,module,torch,bnb,device)
            if xla:
                txla,xm,metrics=xla;hlo=txla._XLAC._get_xla_tensors_hlo(list(outputs.values()))
                prefix.with_suffix('.hlo.txt').write_text(hlo);raw['hlo_precision']=P.hlo_precision(hlo)
                xm.mark_step(wait=True);xm.wait_device_ops()
                profile,_=B.load_spec();raw['counters']={key:metrics.counter_value(key) for key in metrics.counter_names()}
                raw['execution_metrics']={key:list(data) for key in profile['execution_metrics'] if (data:=metrics.metric_data(key)) is not None}
                prefix.with_suffix('.metrics.txt').write_text(metrics.metrics_report())
            else:raw.update(counters={},execution_metrics={},hlo_precision=None)
            state=module.weight.quant_state
            raw.update(R.base_raw(case,{'state':state}),outputs={key:B.tensor_record(value) for key,value in outputs.items()},placements={key:str(value.device) for key,value in outputs.items()},packed_before=B.tensor_record(before),packed_after=B.tensor_record(module.weight.data),state_dict={key:B.tensor_record(value) for key,value in module.state_dict().items()},original_quant_state_class=type(state) is bnb.functional.QuantState,module_state_alias=module.quant_state is state,module_state_absent=module.quant_state is None,weight_requires_grad=module.weight.requires_grad,weight_grad_is_none=module.weight.grad is None,original_classes=type(module) is bnb.nn.Linear4bit and type(module.weight) is bnb.nn.Params4bit,parameter_module_alias=module.weight.module is module,module_training=module.training,gradients='EXECUTED' if case['dtype']=='float32' else 'NOT_RUN')
            if args.phase=='save':torch.save({key:value.detach().cpu().clone() for key,value in module.state_dict().items()},prefix.with_suffix('.state.pt'))
            S.validate_raw(raw,case,bool(xla))
            if xla:
                original=read(Path(args.oracle)/'raw'/(case['id']+'.json'))
                raw['numerical_gates']=device_gates(case,raw,original,profile)
            raw['status']='PASS'
        except Exception as error:
            raw.update(status='ERROR',error_type=type(error).__name__,error_message=str(error),traceback=traceback.format_exc())
            prefix.with_suffix('.error.log').write_text(raw['traceback'])
        write(prefix.with_suffix('.json'),raw)
    rows=[read(output/'raw'/(case['id']+'.json')) for case in CASES]
    passed=all(row['status']=='PASS' and (not xla or all(gate['status']=='PASS' for gate in validate_device_gates(row,case).values())) for row,case in zip(rows,CASES))
    receipt.update(status='COMPLETE',numerical_status='PASS' if passed else 'FAIL',artifacts=B.inventory(output,'receipt.json'));write(output/'receipt.json',receipt)
    return 0 if passed else 2


def identity(args, local):
    require(os.getppid()==args.parent_pid,'ACTUAL_PARENT')
    require(os.getpid()!=args.parent_pid and os.getpgid(0)==os.getpgid(args.parent_pid),'INHERITED_OWNER_GROUP')
    require(args.phase in ('save','restore') and re.fullmatch('[0-9a-f]{32}',args.process_token) and re.fullmatch('[0-9a-f]{32}',args.parent_process_token),'PHASE_TOKENS')
    require(math.isfinite(args.deadline_epoch) and time.time()<args.deadline_epoch,'OWNER_DEADLINE')
    receipt={'phase':args.phase,'status':'PARTIAL','pid':os.getpid(),'pgid':os.getpgid(0),'process_token':args.process_token,'parent_pid':args.parent_pid,'parent_process_token':args.parent_process_token,'deadline_epoch':args.deadline_epoch,'protocol_sha256':PROTOCOL_SHA,'probe_sha256':sha(__file__),'helper_pins':PINS,'profile_sha256':B.PROFILE_SHA,'inputs_sha256':B.INPUTS_SHA,'case_ids':[case['id'] for case in CASES],'scope':'LOCAL_CPU_SMOKE' if local else 'TPU_CANDIDATE','precision':{'requested':'highest','readback':None,'set_calls':0,'before_graph':None}}
    if args.phase=='restore':
        require(sha(Path(args.saved)/'receipt.json')==args.saved_sha256,'SAVE_SEAL_BEFORE_RESTORE')
        saved=read(Path(args.saved)/'receipt.json')
        receipt.update(saved_seal_sha256=args.saved_sha256,saved_pid=saved['pid'],saved_process_token=saved['process_token'])
    return receipt


def device_child(args):
    """Unexecuted device adapter. Promotion needs fixed runtime/source/oracle receipt validation."""
    receipt=identity(args,False);P.validate_environment(B,P.precision_environment())
    oracle=T.verify_oracle(B,A,args)
    require(not args.phase=='restore' or sha(Path(args.saved)/'receipt.json')==args.saved_sha256,'SAVE_SEAL_BEFORE_RESTORE')
    admission,roots=A.admit(B,args.admission,args.admission_sha256,args.patch_manifest)
    profile,_=B.load_spec()
    receipt.update(source_pre=args.admission_sha256,runtime=B.runtime_check(True),runtime_lock_sha256=profile['runtime_lock_sha256'],oracle_sha256=args.oracle_sha256,patch_manifest_sha256=A.PATCH_MANIFEST_SHA,precision_environment=P.precision_environment())
    import torch,torch_xla
    import torch_xla.backends as backends
    backends.set_mat_mul_precision('highest')
    require(backends.get_mat_mul_precision()=='highest','HIGHEST_READBACK')
    receipt['precision']={'requested':'highest','readback':'highest','set_calls':1,'before_graph':True}
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as metrics
    require(torch.__version__=='2.9.0+cpu' and torch_xla.__version__.split('+')[0]=='2.9.0','RUNTIME_PAIR')
    device=xm.xla_device();require(device.type=='xla' and xm.xla_device_hw(device)=='TPU','ACTUAL_TPU')
    receipt['device']={'type':'xla','hardware':'TPU','pjrt':'TPU'}
    import bitsandbytes as bnb
    receipt['public_api']=state_methods(bnb,roots)
    require(not torch.__future__.get_overwrite_module_params_on_conversion() and not torch.__future__.get_swap_module_params_on_conversion(),'DEFAULT_FLAGS')
    receipt['dispatch']={op:torch._C._dispatch_has_kernel_for_dispatch_key(op,'XLA') for op in B.DISPATCH_OPS}
    require(all(receipt['dispatch'].values()),'XLA_DISPATCH')
    try:return case_loop(args,bnb,torch,device,receipt,(torch_xla,xm,metrics))
    finally:
        A.verify_installed(B,admission,roots);state_methods(bnb,roots)
        receipt['source_post']=args.admission_sha256
        if Path(args.output).exists():receipt['artifacts']=B.inventory(Path(args.output),'receipt.json');write(Path(args.output)/'receipt.json',receipt)


def local_child(args):
    receipt=identity(args,True)
    require(not args.phase=='restore' or sha(Path(args.saved)/'receipt.json')==args.saved_sha256,'SAVE_SEAL_BEFORE_RESTORE')
    import importlib.metadata
    discover=importlib.metadata.entry_points
    def fixture_discover(*positional,**keywords):return [] if keywords.get('group')=='bitsandbytes.backends' else discover(*positional,**keywords)
    importlib.metadata.entry_points=fixture_discover
    try:import torch,bitsandbytes as bnb
    finally:importlib.metadata.entry_points=discover
    roots={'bitsandbytes':Path(bnb.__file__).resolve().parent};require(roots['bitsandbytes']==Path(args.source).resolve()/'bitsandbytes','CPU_FIXTURE_IMPORT')
    manifest=A.load_manifest(B,PROJECT/'patches/params4bit-xla-v1.json')
    B.verify_source_tree(roots['bitsandbytes'],A.python_files(manifest['post_patch_package_files']))
    receipt['public_api']=state_methods(bnb,roots)
    require(not torch.__future__.get_overwrite_module_params_on_conversion() and not torch.__future__.get_swap_module_params_on_conversion(),'DEFAULT_FLAGS')
    result=case_loop(args,bnb,torch,'cpu',receipt)
    state_methods(bnb,roots);B.verify_source_tree(roots['bitsandbytes'],A.python_files(manifest['post_patch_package_files']))
    require(not any(name=='torch_xla' or name.startswith('torch_xla.') for name in sys.modules),'CPU_SMOKE_NO_XLA')
    return result


def run_owned(command, timeout, stdout, stderr, env=None, on_launch=None):
    """Local direct-child fixture. The outer owner retains the shared group."""
    started=time.monotonic()
    with Path(stdout).open('w') as out,Path(stderr).open('w') as err:
        process=subprocess.Popen(command,stdout=out,stderr=err,env=env)
        record={'pid':process.pid,'launched_pid':process.pid,'pgid':os.getpgid(0),'started_monotonic':started,'started_epoch':time.time(),'reaped':False,'child_absent':False,'cleanup_errors':[],'timeout':False}
        try:
            if on_launch:on_launch(record)
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            record['timeout']=True;process.terminate()
            try:process.wait(timeout=3)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=3)
        finally:
            if process.poll() is None:process.kill();process.wait(timeout=3)
            try:os.kill(process.pid,0)
            except ProcessLookupError:record['child_absent']=True
            record.update(reaped=process.poll() is not None,exit_code=process.returncode,finished_monotonic=time.monotonic(),finished_epoch=time.time())
    return record


def local_parent(source, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    parent={'kind':'STATE_ROUNDTRIP_PARENT','status':'PARTIAL','pid':os.getpid(),'pgid':os.getpgid(0),'process_token':secrets.token_hex(16),'protocol_sha256':PROTOCOL_SHA,'probe_sha256':sha(__file__),'helper_pins':PINS,'deadline_epoch':time.time()+180,'children':[],'m3_status':'NOT_QUALIFIED'}
    for phase in ('save','restore'):
        token=secrets.token_hex(16)
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'local-child','--source',str(Path(source).resolve()),'--phase',phase,'--output',str((output/phase).resolve()),'--saved',str((output/'save').resolve()),'--saved-sha256',sha(output/'save/receipt.json') if phase=='restore' else 'NOT_APPLICABLE','--parent-pid',str(parent['pid']),'--parent-process-token',parent['process_token'],'--process-token',token,'--deadline-epoch',str(time.time()+90)]
        env=dict(os.environ,PYTHONPATH=str(Path(source).resolve()),PYTHONDONTWRITEBYTECODE='1',TORCHINDUCTOR_CACHE_DIR=str(HERE/'compiler-cache'),XDG_CACHE_HOME=str(HERE/'cache'))
        def retain_launch(record):
            record.update(phase=phase,process_token=token,deadline_epoch=float(command[-1]));parent['children'].append(record);write(output/'parent.json',parent)
        launch=run_owned(command,90,output/(phase+'.stdout'),output/(phase+'.stderr'),env,retain_launch)
        launch.update(parent_pid=parent['pid'],parent_process_token=parent['process_token'],receipt=phase+'/receipt.json',receipt_sha256=sha(output/phase/'receipt.json'))
        write(output/'parent.json',parent)
        require(launch['reaped'] and launch['child_absent'] and not launch['timeout'],'PHASE_CHILD_CLEANUP')
        require(launch['exit_code'] in (0,2),'CHILD_GLOBAL_FAILURE')
    parent.update(status='COMPLETE',artifacts=B.inventory(output,'parent.json'));write(output/'parent.json',parent)
    result=verify_pair(output,True);write(output.with_suffix('.review.json'),result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    child=sub.add_parser('local-child')
    for name in ('source','phase','output','saved','saved-sha256','parent-process-token','process-token'):child.add_argument('--'+name,required=True)
    child.add_argument('--parent-pid',type=int,required=True);child.add_argument('--deadline-epoch',type=float,required=True)
    parent=sub.add_parser('local-smoke');parent.add_argument('--source',required=True);parent.add_argument('--output',required=True)
    device=sub.add_parser('device-child')
    for name in ('phase','output','admission','admission-sha256','patch-manifest','oracle','oracle-sha256','parent-process-token','process-token'):device.add_argument('--'+name,required=True)
    device.add_argument('--saved');device.add_argument('--saved-sha256');device.add_argument('--parent-pid',type=int,required=True);device.add_argument('--deadline-epoch',type=float,required=True)
    verify=sub.add_parser('verify')
    for name in ('actual','oracle','oracle-sha256','admission-sha256','parent-receipt-sha256','patch-manifest'):verify.add_argument('--'+name,required=True)
    args=parser.parse_args()
    if args.command=='local-child':return local_child(args)
    if args.command=='device-child':
        if args.phase=='restore' and (not args.saved or not args.saved_sha256):parser.error('Restore requires --saved and --saved-sha256')
        return device_child(args)
    if args.command=='verify':
        try:
            A.load_manifest(B,args.patch_manifest)
            result=verify_pair(args.actual,False,args.oracle,args.oracle_sha256,args.admission_sha256,args.parent_receipt_sha256)
        except Exception as error:
            print(json.dumps({'record_validation':'FAIL','m3_status':'NOT_QUALIFIED','reason':str(error)}));return 1
        print(json.dumps(result));return 0 if result['numerical_status']=='PASS' else 2
    result=local_parent(args.source,args.output);print(json.dumps(result));return 0 if result['numerical_status']=='PASS' else 2


if __name__=='__main__':raise SystemExit(main())
