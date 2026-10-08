"""Independent saved public records: structural validity and numeric outcome are separate."""
import argparse
import json
import math
from pathlib import Path
import contract as C
import oracle_math as M
import public_graph as G
import verify_public_oracle as O
from prepare_public_oracle import source_args


def execution(record,B,P):
    P.validate_execution(B,record);e=record.get('execution',{})
    C.require(e.get('api')=='_xla_sync_multi' and e.get('targets')==['actual'] and e.get('wait') is True and e.get('wait_device_ops') is True and e.get('metrics_after_sync_before_output') is True and e.get('other_execution_in_interval') is False,'ACTUAL_ONLY_EXECUTION_INTERVAL')
    C.require(e.get('graph_hash_binding')=='PRE_SYNC_IDENTIFIER_NOT_EXECUTABLE_LINK' and isinstance(e.get('graph_hash'),str) and e['graph_hash'],'PRE_SYNC_GRAPH_ID_SCOPE')
    C.require(all(type(e.get(k))in(int,float) and math.isfinite(e[k]) for k in ('started_monotonic','finished_monotonic')) and e['started_monotonic']<=e['finished_monotonic'],'EXECUTION_FINITE_INTERVAL')


def graph_record(root,rec,expected,B,P,*,native,bias_gradient=False):
    prefix=root/rec['graph_files_prefix'];C.require(not Path(rec['graph_files_prefix']).is_absolute() and '..' not in Path(rec['graph_files_prefix']).parts and str(rec['graph_files_prefix']).startswith('graphs/'),'RELATIVE_GRAPH_ARTIFACT')
    context=prefix.with_suffix('.context.pb').read_bytes();printer=prefix.with_suffix('.printer.pb').read_bytes();hlo=prefix.with_suffix('.hlo.txt').read_text();parameters=C.read(prefix.with_suffix('.parameters.json'))
    C.require(C.digest(parameters)==rec.get('parameters_sha256'),'PARAMETER_RECORD_SHA')
    if native:
        proof=G.verify_native(hlo,context,printer,parameters,expected);_,_,_,payload,_=G.payload(hlo)
        C.require(prefix.with_suffix('.payload.json').read_text()==payload and prefix.with_suffix('.mosaic-body.bin').read_bytes()==__import__('base64').b64decode(json.loads(payload)['custom_call_config']['body'],validate=True),'ACTUAL_GRAPH_PAYLOAD_BODY_ARTIFACT')
        boundary=G.verify_boundary(rec.get('native_boundary'),payload,expected)
        C.require(prefix.with_suffix('.original-native-config.json').read_text()==boundary['original_payload'] and prefix.with_suffix('.original-mosaic-body.bin').read_bytes()==__import__('base64').b64decode(json.loads(boundary['original_payload'])['custom_call_config']['body'],validate=True),'ORIGINAL_PUBLIC_PAYLOAD_ARTIFACT')
        C.require(C.read(prefix.with_suffix('.conversion-audit.json'))==boundary['payload_conversion'] and C.read(prefix.with_suffix('.native-call.json'))==boundary,'PUBLIC_NATIVE_BOUNDARY_ARTIFACT')
    else:proof=G.verify_reference(context,printer,rec['output']['shape'],rec['output']['dtype'],bias_gradient=bias_gradient,parameters=parameters,allowed_parameters=expected['allowed_parameters'])
    C.require(rec.get('graph')==proof,'RECOMPUTED_OUTPUT_GRAPH_PROOF');G.V.array(rec['output']);execution(rec,B,P)


def verify(args,*,synthetic=False):
    root=Path(args.actual);C.require(C.sha(root/'receipt.json')==args.receipt_sha256,'DEVICE_RECEIPT_SHA');r=C.read(root/'receipt.json')
    a,p,upstream=C.admission(args.admission,args.protocol,args.package_root,args.upstream_source,args.repo_root)
    C.require(r.get('kind')=='PUBLIC_PALLAS_DEVICE_PROBE' and r.get('status')=='COMPLETE' and not any(k in r for k in ('error_type','error_message')),'DEVICE_TERMINAL_RECORD')
    C.require(r.get('scope')==('SYNTHETIC_PROTOCOL_FIXTURE' if synthetic else 'ACTUAL_PUBLIC_API'),'DEVICE_EVIDENCE_SCOPE')
    C.require(r.get('scientific_variant')==C.VARIANT and r.get('source_variant')=='pallas-forward-mosaic7-gather-bf16-fp32-v1' and r.get('scientific_sources')==C.scientific_sources(),'DEVICE_EXACT_SOURCE_GENERATION')
    C.require(r.get('admission_sha256')==r.get('source_pre')==r.get('source_post')==C.ADMISSION_SHA and r.get('protocol_sha256')==C.PROTOCOL_SHA,'DEVICE_SOURCE_POST_PROTOCOL')
    C.require(r.get('oracle_sha256')==args.oracle_sha256,'DEVICE_CPU_ORACLE_LINK');_,oracle=O.verify(args,qualified=not synthetic)
    if not synthetic:O.verify_owned_cpu(args,C.read(Path(args.oracle)/'oracle-seal.json'));C.accepted_native(args.native_acceptance,args.native_acceptance_sha256,a)
    C.require(r.get('native_acceptance_sha256')==args.native_acceptance_sha256,'DEVICE_NATIVE_DEPENDENCY_LINK')
    C.require(r.get('wheel_records')==C.wheels(a,args.plugin_wheel,args.upstream_wheel),'DEVICE_BUILT_WHEEL_RECORDS')
    C.require(r.get('runtime_wheel_records')==C.runtime_wheels(a,args.repo_root,args.runtime_wheels,qualified=not synthetic),'DEVICE_FIXED_RUNTIME_WHEEL_BODIES')
    if not synthetic:C.require(r.get('installed_runtime_metadata')=={w['name']:{'version':w['version'],'metadata_sha256':w['metadata_sha256']} for w in C.read(Path(args.repo_root)/'experiments/2026-10-04-bitsandbytes-tpu/runtime/requirements.lock.json')['wheels']},'DEVICE_INSTALLED_RUNTIME_METADATA')
    C.owner(r.get('pid'),r.get('pgid'),r.get('parent_pid'),r.get('process_token'),r.get('deadline_epoch'))
    C.require(all(type(r.get(n))in(int,float) and math.isfinite(r[n]) for n in ('started_epoch','finished_epoch')) and r['started_epoch']<=r['finished_epoch']<=r['deadline_epoch'],'DEVICE_FINITE_DEADLINE')
    C.require(C.sha(args.ownership)==args.ownership_sha256,'DEVICE_OUTER_OWNERSHIP_SHA');C.outer_owned(C.read(args.ownership),r,'probe_public.py')
    expected_runtime=dict(a['runtime']);expected_runtime.pop('precision');C.require(r.get('runtime')==expected_runtime,'DEVICE_PINNED_RUNTIME')
    C.require(r.get('device',{}).get('type')=='xla' and r['device'].get('hardware')=='TPU','DEVICE_TPU')
    C.require(r.get('precision')=={'requested':'highest','readback':'highest','set_calls':1,'before_graph':True},'DEVICE_HIGHEST_BEFORE_GRAPHS')
    B=C.load(C.HERE/'helpers/probe_backend.py','public_verify_B');P=C.load(C.HERE/'helpers/probe_precision.py','public_verify_P');P.validate_environment(B,r['precision_environment'])
    C.require(C.inventory(root,('receipt.json',))==r.get('artifacts'),'DEVICE_FULL_ARTIFACT_INVENTORY')
    C.require(r.get('m4_status')==r.get('m6_memory_status')=='NOT_QUALIFIED' and r.get('performance')=='NOT_MEASURED','DEVICE_CLAIM_SCOPE')
    C.require(isinstance(r.get('cases'),list) and [v.get('case') for v in r['cases']]==[c['id'] for c in p['cases']],'PUBLIC_FIXED_COMPLETE_CASE_MATRIX')
    decode,code=M.decode_original(upstream);rows=[];conversion_rows={};first_failure=None
    for case,short in zip(p['cases'],r['cases']):
        name=case['id'];raw=C.read(root/'rows'/(name+'.json'));C.require(raw.get('case')==name and raw.get('pid')==r['pid'] and raw.get('process_token')==r['process_token'] and raw.get('status')==short.get('status'),'ROW_CASE_PROCESS_TERMINAL')
        if first_failure is not None:
            C.require(raw.get('status')=='NOT_RUN' and raw.get('reason')=='SHARED_XLA_STATE_AFTER_FIRST_FAILURE' and raw.get('first_failure')==first_failure and set(raw)=={'case','status','pid','process_token','reason','first_failure'},'PUBLIC_FIRST_ERROR_NOT_RUN')
            rows.append({'case':name,'status':'NOT_RUN'});continue
        if raw['status']=='ERROR':
            C.require(isinstance(raw.get('error_type'),str) and raw['error_type'] and isinstance(raw.get('error_message'),str) and (root/'errors'/(name+'.log')).is_file(),'RETAINED_CASE_ERROR');rows.append({'case':name,'status':'ERROR'});first_failure=name;continue
        C.require(raw.get('status') in ('PASS','FAIL') and not any(k in raw for k in ('error','error_type','error_message')),'ERROR_CANNOT_BECOME_PASS')
        expected={**oracle[name],'inputs':M.input_record(case),'allowed_parameters':M.allowed_reference_values(case,decode,code,oracle[name]['outputs'])};C.require(raw.get('inputs_sha256')==expected['inputs_sha256'],'ROW_FIXED_RECIPE_INPUTS')
        selected=raw.get('selected_events');C.require(isinstance(selected,list) and len(selected)==1 and selected[0].get('path')==case['expected_path'] and selected[0].get('reason')==case['expected_reason'],'ROW_EXPLICIT_OBSERVED_SELECTION')
        C.require(set(raw['outputs'])==set(expected['outputs']),'ROW_OUTPUT_GRADIENT_INVENTORY')
        graph_record(root,raw['forward'],expected,B,P,native=case['expected_path']=='PALLAS_CANDIDATE');C.require(raw['forward']['output']==raw['outputs']['y'],'PUBLIC_ACTUAL_Y_RECORD')
        if case['expected_path']=='PALLAS_CANDIDATE':conversion_rows[name]=raw['forward']['native_boundary']
        needed=set(expected['outputs'])-{'y'};C.require(set(raw.get('gradients',{}))==needed,'ROW_GRADIENT_GRAPH_INVENTORY')
        for key in needed:
            graph_record(root,raw['gradients'][key],expected,B,P,native=False,bias_gradient=key.startswith('db'));C.require(raw['gradients'][key]['output']==raw['outputs'][key],'PUBLIC_ACTUAL_GRADIENT_RECORD')
        before,after=raw['objects_before'],raw['objects_after'];C.require(before['activation']==after['activation'] and before['weight']==after['weight'],'ROW_INPUT_WEIGHT_IDENTITY')
        C.require(before['weight']['class']=='bitsandbytes.nn.modules.Params4bit' and before['weight']['requires_grad'] is False and before['weight']['grad_is_none'] is True,'ROW_ORIGINAL_FROZEN_PARAMETER')
        C.require(before['weight']['value_sha256']==C.digest(expected['inputs']['packed']) and before['weight']['byte_sha256']==M.byte_sha(M.materialize(expected['inputs']['packed'])),'ROW_PACKED_PARAMETER_CONTENT')
        C.require(before['weight']['quant_state']==M.state_contract(M.materialize(expected['inputs']['scales']),code,case['dtype']),'ROW_EXACT_ORIGINAL_QUANT_STATE_BYTES')
        C.require(before['activation']['shape']==case['activation_shape'] and before['activation']['dtype']=='torch.'+case['dtype'] and before['activation']['is_leaf'] is True and before['activation']['requires_grad']==(case['gradients']!='NOT_RUN'),'ROW_ACTIVATION_LEAF')
        if case['bias']:
            C.require(before['bias']['class']=='torch.nn.parameter.Parameter' and before['bias']['value_sha256']==C.digest(expected['inputs']['bias']) and before['bias']['byte_sha256']==M.byte_sha(M.materialize(expected['inputs']['bias'])) and {k:v for k,v in before['bias'].items() if k!='grad_is_none'}=={k:v for k,v in after['bias'].items() if k!='grad_is_none'},'ROW_BIAS_IDENTITY_CONTENT')
        else:C.require(before['bias'] is None and after['bias'] is None,'ROW_NO_BIAS')
        if needed:
            ids=raw['gradient_identity'];C.require(ids['activation_leaf_before']==ids['activation_leaf_after']==before['activation']['id'] and type(ids['first_buffer'])is int and ids['first_buffer']>0 and ids['first_buffer']==ids['second_buffer'],'ROW_GRADIENT_IDENTITY')
            if case['bias']:C.require(type(ids['bias_first_buffer'])is int and ids['bias_first_buffer']>0 and ids['bias_first_buffer']==ids['bias_second_buffer'],'ROW_BIAS_GRADIENT_IDENTITY')
        if name=='public-out-fp32':C.require(raw.get('out_same_object') is True,'ROW_PUBLIC_OUT_IDENTITY')
        if name=='public-functionalized-fp32':C.require(raw.get('functionalization')=={'caller_activation_functional':True},'ROW_NORMAL_FUNCTIONALIZED_CALLER')
        tolerance=p['tolerances']['fp32_forward_gradient' if case['dtype']=='float32' else 'bf16_forward_unit_scale'];gates={}
        for key,value in raw['outputs'].items():
            G.V.array(value);target=expected['outputs'][key];C.require(value['shape']==target['shape'] and value['dtype']==target['dtype'],'ROW_OUTPUT_METADATA');gates[key]=B.numeric(value['values'],target['values'],tolerance)
        status='PASS' if all(v['status']=='PASS' for v in gates.values()) else 'FAIL';C.require(raw.get('gates')==gates and raw['status']==status,'ROW_INDEPENDENT_NUMERICAL_STATUS');rows.append({'case':name,'status':status,'gates':gates})
    numeric='ERROR' if any(v['status']=='ERROR' for v in rows) else 'FAIL' if any(v['status']=='FAIL' for v in rows) else 'PASS';C.require(r.get('numerical_status')==numeric,'PARENT_NUMERICAL_STATUS')
    # Every successful actual native graph is independently replayed on the explicit
    # pinned CPU frontend. Four-case batches retain the reviewed helper bounds.
    replay=[]
    names=list(conversion_rows)
    for start in range(0,max(1,len(names)),4):
        subset={n:conversion_rows[n]for n in names[start:start+4]}
        replay.append(G.CR.replay(subset,args.mosaic_audit_python,args.audit_deadline_epoch))
    return {'record_validation':'PASS','numerical_status':numeric,'case_count':15,'native_case_count':11,'reference_case_count':4,'rows':rows,'scope':r['scope'],'independent_conversion_replay':replay,'m4_status':'NOT_QUALIFIED','m6_memory_status':'NOT_QUALIFIED','performance':'NOT_MEASURED'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);source_args(p)
    for n in ('actual','receipt-sha256','oracle','oracle-sha256','cpu-ownership','cpu-ownership-sha256','ownership','ownership-sha256','native-acceptance','native-acceptance-sha256'):p.add_argument('--'+n,required=True)
    p.add_argument('--mosaic-audit-python',required=True);p.add_argument('--audit-deadline-epoch',type=float,required=True)
    args=p.parse_args();result=verify(args);print(json.dumps(result,sort_keys=True));raise SystemExit(0 if result['numerical_status']=='PASS' else 2)
