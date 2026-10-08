"""Fixed actual public API probe. Source-admitted separate generation; no API replacement."""
import argparse
import base64
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import contract as C
import oracle_math as M
import public_graph as G
import verify_public_oracle as O
from prepare_public_oracle import source_args


def sanitize_error(v):
    if isinstance(v,float) and not math.isfinite(v):return {'forensic_nonfinite':repr(v)}
    if isinstance(v,dict):return {k:sanitize_error(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [sanitize_error(x) for x in v]
    return v


def runtime(a):
    import platform
    import torch
    actual={'platform':platform.system(),'machine':platform.machine(),'python':platform.python_version(),'torch':str(torch.__version__)}
    for dist,key in [('torch-xla','torch_xla'),('libtpu','libtpu'),('jax','jax'),('jaxlib','jaxlib')]:actual[key]=importlib.metadata.version(dist).split('+')[0] if key=='torch_xla' else importlib.metadata.version(dist)
    expected=dict(a['runtime']);expected.pop('precision');C.require(actual==expected,'PUBLIC_FIXED_DEVICE_RUNTIME');return actual


def capture(value,root,name,case,expected,xla,xm,metrics,B,P,*,native,bias_gradient=False,native_record=None):
    """Read actual output graph, then sync only that output. No replay factory."""
    context=xla._XLAC.lowering.LoweringContext('public-'+name);context.build([value]);proto=bytes(context.hlo());printer=bytes(xla._XLAC._get_xla_tensors_hlo_proto([value]));hlo=xla._XLAC._get_xla_tensors_hlo([value])
    parameters={str(k):B.tensor_record(v) for k,v in context.parameter_id_tensor_mapping().items()}
    prefix=root/'graphs'/name;prefix.parent.mkdir(parents=True,exist_ok=True)
    prefix.with_suffix('.context.pb').write_bytes(proto);prefix.with_suffix('.printer.pb').write_bytes(printer);prefix.with_suffix('.hlo.txt').write_text(hlo);prefix.with_suffix('.context.textproto').write_text(context.hlo_text());C.write(prefix.with_suffix('.parameters.json'),parameters)
    if native:
        proof=G.verify_native(hlo,proto,printer,parameters,expected);_,_,_,payload,_=G.payload(hlo)
        G.verify_boundary(native_record,payload,expected)
        retain_conversion(prefix,native_record)
    else:proof=G.verify_reference(proto,printer,list(value.shape),str(value.dtype).removeprefix('torch.'),bias_gradient=bias_gradient,parameters=parameters,allowed_parameters=expected['allowed_parameters'])
    graph_hash=xla._XLAC._get_graph_hash([value]).hex();metrics.clear_all();started=time.monotonic();xla._XLAC._xla_sync_multi([value],devices=[],wait=True,sync_xla_data=True);xm.wait_device_ops();finished=time.monotonic()
    evidence=G.E.execution_evidence(metrics,B,P)
    return {'graph':proof,'graph_files_prefix':'graphs/'+name,'parameters_sha256':C.digest(parameters),'output':B.tensor_record(value),'native_boundary':native_record if native else None,'execution':{'api':'_xla_sync_multi','targets':['actual'],'wait':True,'wait_device_ops':True,'metrics_after_sync_before_output':True,'other_execution_in_interval':False,'graph_hash':graph_hash,'graph_hash_binding':'PRE_SYNC_IDENTIFIER_NOT_EXECUTABLE_LINK','started_monotonic':started,'finished_monotonic':finished},**evidence}


def retain_conversion(prefix,row):
    """Retain before sync; errors keep the original trace and zero/one call count."""
    for key,suffix,body_suffix in [('original_payload','.original-native-config.json','.original-mosaic-body.bin'),('payload','.payload.json','.mosaic-body.bin')]:
        if key in row:
            prefix.with_suffix(suffix).parent.mkdir(parents=True,exist_ok=True)
            prefix.with_suffix(suffix).write_text(row[key]);prefix.with_suffix(body_suffix).write_bytes(base64.b64decode(json.loads(row[key])['custom_call_config']['body'],validate=True))
    if 'payload_conversion'in row:C.write(prefix.with_suffix('.conversion-audit.json'),row['payload_conversion'])
    C.write(prefix.with_suffix('.native-call.json'),row)


def object_state(a,weight,bias,B):
    def parameter(v):return None if v is None else {'id':id(v),'class':type(v).__module__+'.'+type(v).__name__,'device':str(v.device),'requires_grad':v.requires_grad,'value_sha256':C.digest(B.tensor_record(v)),'grad_is_none':v.grad is None,'byte_sha256':M.byte_sha(v)}
    result={'activation':{'id':id(a),'is_leaf':a.is_leaf,'shape':list(a.shape),'stride':list(a.stride()),'dtype':str(a.dtype),'device':str(a.device),'requires_grad':a.requires_grad},'weight':parameter(weight),'bias':parameter(bias)}
    if weight is not None:
        q=weight.quant_state;C.require(q.shape==(256,512) and q.blocksize==64 and q.quant_type=='nf4' and q.nested is False,'ORIGINAL_NON_NESTED_STATE_CONTRACT');result['weight']['quant_state']=M.state_contract(q.absmax,q.code,str(q.dtype).removeprefix('torch.'))
    return result


def call_public(case,bnb,torch,a,weight,qs,module,out,flags):
    name=case['id']
    if name.startswith('direct-'):return torch.ops.bitsandbytes.gemm_4bit.default(a,weight,(256,512),qs.absmax,64,'nf4')
    if name.startswith('module-'):return module(a)
    if name=='public-functionalized-fp32':
        def original_caller(x):
            flags['caller_activation_functional']=torch._is_functional_tensor(x)
            return bnb.matmul_4bit(x,weight,qs)
        return torch.func.functionalize(original_caller)(a)
    if name=='public-out-fp32':return bnb.matmul_4bit(a,weight,qs,out=out)
    return bnb.matmul_4bit(a,weight,qs)


def execute(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False);start=time.time()
    r={'kind':'PUBLIC_PALLAS_DEVICE_PROBE','status':'PARTIAL','scope':'ACTUAL_PUBLIC_API','scientific_variant':C.VARIANT,'source_variant':'pallas-forward-mosaic7-gather-bf16-fp32-v1','pid':os.getpid(),'pgid':os.getpgid(0),'parent_pid':os.getppid(),'process_token':args.process_token,'deadline_epoch':args.deadline_epoch,'started_epoch':start,'admission_sha256':C.ADMISSION_SHA,'protocol_sha256':C.PROTOCOL_SHA,'oracle_sha256':args.oracle_sha256,'native_acceptance_sha256':args.native_acceptance_sha256,'cases':[],'m4_status':'NOT_QUALIFIED','m6_memory_status':'NOT_QUALIFIED','performance':'NOT_MEASURED'}
    C.write(root/'receipt.json',r);S=bnb=None
    try:
        C.owner(os.getpid(),os.getpgid(0),os.getppid(),args.process_token,args.deadline_epoch);C.require(start<args.deadline_epoch,'DEVICE_DEADLINE')
        a,p,upstream=C.admission(args.admission,args.protocol,args.package_root,args.upstream_source,args.repo_root);C.accepted_native(args.native_acceptance,args.native_acceptance_sha256,a)
        C.require(not any(n in sys.modules for n in ('torch_xla','jax','bitsandbytes','bitsandbytes_tpu_pallas')),'FRESH_DEVICE_PACKAGE_CLIENTS')
        _,rows=O.verify(args,qualified=True);O.verify_owned_cpu(args,C.read(Path(args.oracle)/'oracle-seal.json'))
        r.update(runtime_wheel_records=C.runtime_wheels(a,args.repo_root,args.runtime_wheels,qualified=True),installed_runtime_metadata=C.installed_runtime_metadata(args.repo_root),scientific_sources=C.scientific_sources(),source_pre=C.ADMISSION_SHA,wheel_records=C.wheels(a,args.plugin_wheel,args.upstream_wheel),runtime=runtime(a))
        import torch,torch_xla
        import torch_xla.backends as backends
        backends.set_mat_mul_precision('highest');C.require(backends.get_mat_mul_precision()=='highest','DEVICE_PRECISION_READBACK');r['precision']={'requested':'highest','readback':'highest','set_calls':1,'before_graph':True}
        import torch_xla.core.xla_model as xm
        import torch_xla.debug.metrics as metrics
        B=C.load(C.HERE/'helpers/probe_backend.py','public_device_B');P=C.load(C.HERE/'helpers/probe_precision.py','public_device_P');P.validate_environment(B,P.precision_environment());r['precision_environment']=P.precision_environment()
        eps=list(importlib.metadata.entry_points(group='bitsandbytes.backends'));C.require(len(eps)==1 and eps[0].name==a['entry_point']['name'] and eps[0].value==a['entry_point']['value'],'EXCLUSIVE_PUBLIC_PLUGIN_ENTRY_POINT')
        import bitsandbytes as bnb
        S=importlib.import_module('bitsandbytes_tpu_pallas.source_admission');L=importlib.import_module('bitsandbytes_tpu_pallas.frontend');S.validate_registration(bnb)
        C.require(Path(bnb.__file__).resolve().parent==upstream.resolve() and Path(S.__file__).resolve().parent==Path(args.package_root).resolve(),'ACTUAL_INSTALLED_SOURCE_ROOTS')
        device=xm.xla_device();C.require(device.type=='xla' and xm.xla_device_hw(device)=='TPU','ACTUAL_TPU_REQUIRED');r['device']={'type':'xla','hardware':'TPU','pjrt':os.environ.get('PJRT_DEVICE')}
        decode,code=M.decode_original(upstream)
        first_failure=None
        for case in p['cases']:
            name=case['id'];raw={'case':name,'status':'ERROR','pid':os.getpid(),'process_token':args.process_token};prefix=root/'rows'/(name+'.json')
            native_before=len(L.gemm_4bit.native_calls)
            if first_failure is not None:
                raw.update(status='NOT_RUN',reason='SHARED_XLA_STATE_AFTER_FIRST_FAILURE',first_failure=first_failure)
                C.write(prefix,raw);r['cases'].append({'case':name,'status':'NOT_RUN'});C.write(root/'receipt.json',r);continue
            try:
                C.require(time.time()<args.deadline_epoch,'DEVICE_DEADLINE');cpu_a,cpu_b,cpu_s,cpu_dy,cpu_bias=M.inputs(case);dtype=getattr(torch,case['dtype']);train=case['gradients']!='NOT_RUN'
                act=cpu_a.to(device)
                if case['layout']=='transpose-copy-transpose':act=act.t().contiguous().t()
                act=act.detach().requires_grad_(train);packed=cpu_b.to(device);scales=cpu_s.to(device)
                qs=bnb.functional.QuantState(absmax=scales,shape=(256,512),code=code.to(device),blocksize=64,quant_type='nf4',dtype=dtype)
                weight=bnb.nn.Params4bit.from_prequantized(packed,qs.as_dict(packed=True),requires_grad=False,device=device)
                module=None;bias=None
                if name.startswith('module-'):
                    module=bnb.nn.Linear4bit(512,256,bias=case['bias'],compute_dtype=dtype,compress_statistics=False,quant_type='nf4',device=device).to(dtype=dtype);module.weight=weight;module.quant_state=weight.quant_state
                    if case['bias']:
                        with torch.no_grad():module.bias.copy_(cpu_bias.to(device))
                        bias=module.bias
                out=torch.empty((128,256),device=device,dtype=dtype) if name=='public-out-fp32' else None
                dy=cpu_dy.to(device);before=object_state(act,weight,bias,B);xm.mark_step(wait=True);xm.wait_device_ops();before_event=len(L.gemm_4bit.events);flags={}
                y=call_public(case,bnb,torch,act,weight,weight.quant_state,module,out,flags)
                calls=L.gemm_4bit.native_calls[native_before:]
                C.require(len(calls)==(1 if case['expected_path']=='PALLAS_CANDIDATE' else 0),'PUBLIC_NATIVE_CALL_COUNT')
                native_record=calls[0] if calls else None
                if native_record:retain_conversion(root/'graphs'/(name+'-y'),native_record)
                if out is not None:C.require(y is out,'ORIGINAL_PUBLIC_OUT_IDENTITY')
                if name=='public-functionalized-fp32':C.require(flags.get('caller_activation_functional') is True and not torch._is_functional_tensor(y),'NORMAL_FUNCTIONALIZED_CALLER_EVIDENCE')
                selected=list(L.gemm_4bit.events)[before_event:];C.require(len(selected)==1 and selected[0].get('path')==case['expected_path'] and selected[0].get('reason')==case['expected_reason'],'ACTUAL_PUBLIC_SELECTION_REASON')
                expected={**rows[name],'inputs':M.input_record(case),'allowed_parameters':M.allowed_reference_values(case,decode,code,rows[name]['outputs'])};raw.update(inputs_sha256=expected['inputs_sha256'],selected_events=selected,functionalization=flags,out_same_object=y is out if out is not None else None,forward=capture(y,root,name+'-y',case,expected,torch_xla,xm,metrics,B,P,native=case['expected_path']=='PALLAS_CANDIDATE',native_record=native_record))
                raw['outputs']={'y':raw['forward']['output']};raw['gradients']={}
                if train:
                    y.backward(dy,retain_graph=True);g1=id(act.grad);raw['gradients']['dX']=capture(act.grad,root,name+'-dX',case,expected,torch_xla,xm,metrics,B,P,native=False);raw['outputs']['dX']=raw['gradients']['dX']['output']
                    if bias is not None:
                        bg=id(bias.grad);raw['gradients']['db']=capture(bias.grad,root,name+'-db',case,expected,torch_xla,xm,metrics,B,P,native=False,bias_gradient=True);raw['outputs']['db']=raw['gradients']['db']['output']
                    y.backward(dy);C.require(id(act.grad)==g1,'ACTIVATION_GRAD_BUFFER_IDENTITY');raw['gradients']['dX_accumulated']=capture(act.grad,root,name+'-dX2',case,expected,torch_xla,xm,metrics,B,P,native=False);raw['outputs']['dX_accumulated']=raw['gradients']['dX_accumulated']['output']
                    if bias is not None:
                        C.require(id(bias.grad)==bg,'BIAS_GRAD_BUFFER_IDENTITY');raw['gradients']['db_accumulated']=capture(bias.grad,root,name+'-db2',case,expected,torch_xla,xm,metrics,B,P,native=False,bias_gradient=True);raw['outputs']['db_accumulated']=raw['gradients']['db_accumulated']['output']
                    raw['gradient_identity']={'activation_leaf_before':before['activation']['id'],'activation_leaf_after':id(act),'first_buffer':g1,'second_buffer':id(act.grad),'bias_first_buffer':bg if bias is not None else None,'bias_second_buffer':id(bias.grad) if bias is not None else None}
                after=object_state(act,weight,bias,B);C.require(before['activation']==after['activation'] and before['weight']==after['weight'],'PUBLIC_INPUT_PARAMETER_IDENTITY_CONTENT');C.require(weight.grad is None and weight.requires_grad is False,'FROZEN_PACKED_BASE')
                if bias is not None:C.require({k:v for k,v in before['bias'].items() if k!='grad_is_none'}=={k:v for k,v in after['bias'].items() if k!='grad_is_none'},'BIAS_PARAMETER_IDENTITY_CONTENT')
                tol=p['tolerances']['fp32_forward_gradient' if dtype==torch.float32 else 'bf16_forward_unit_scale'];raw['gates']={key:B.numeric(raw['outputs'][key]['values'],rows[name]['outputs'][key]['values'],tol) for key in rows[name]['outputs']};raw.update(objects_before=before,objects_after=after,status='PASS' if all(v['status']=='PASS' for v in raw['gates'].values()) else 'FAIL')
            except Exception as error:
                first_failure=name
                for index,row in enumerate(L.gemm_4bit.native_calls[native_before:]):retain_conversion(root/'graphs'/(name+'-failure-'+str(index)),row)
                raw.update(status='ERROR',error_type=type(error).__name__,error_message=str(error));(root/'errors').mkdir(exist_ok=True);(root/'errors'/(name+'.log')).write_text(traceback.format_exc())
                # ERROR remains recoverable even if output values are nonfinite.
                raw=sanitize_error(raw)
            C.write(prefix,raw);r['cases'].append({'case':name,'status':raw['status']});C.write(root/'receipt.json',r)
        C.admission(args.admission,args.protocol,args.package_root,args.upstream_source,args.repo_root);S.validate_registration(bnb);r.update(status='COMPLETE',source_post=C.ADMISSION_SHA,numerical_status='ERROR' if any(v['status']=='ERROR' for v in r['cases']) else 'FAIL' if any(v['status']=='FAIL' for v in r['cases']) else 'PASS')
    except Exception as error:r.update(status='FAILED',error_type=type(error).__name__,error_message=str(error));(root/'global.error.log').write_text(traceback.format_exc())
    finally:
        r['finished_epoch']=time.time();r['artifacts']=C.inventory(root,('receipt.json',));C.write(root/'receipt.json',r)
    return 0 if r.get('status')=='COMPLETE' and r.get('numerical_status')=='PASS' else 2 if r.get('status')=='COMPLETE' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);source_args(p)
    for n in ('oracle','oracle-sha256','cpu-ownership','cpu-ownership-sha256','output','process-token','native-acceptance','native-acceptance-sha256'):p.add_argument('--'+n,required=True)
    p.add_argument('--deadline-epoch',type=float,required=True);raise SystemExit(execute(p.parse_args()))
