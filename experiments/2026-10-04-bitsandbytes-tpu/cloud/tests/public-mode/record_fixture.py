"""Synthetic transport records; CPU values and frontend graphs are not device evidence."""
import base64
import copy
import json
from pathlib import Path
import contract as C
import oracle_math as M
import public_graph as G
import verify_public as V
import probe_public as Probe

HERE=Path(__file__).resolve().parent

def graph(root,name,case,expected,output,decode,code,*,kind):
    prefix=root/'graphs'/name;prefix.parent.mkdir(parents=True,exist_ok=True);inputs=expected['inputs'];parameters={};boundary=None
    if kind=='native':
        dtype='bf16' if case['dtype']=='bfloat16' else 'f32';proto=(HERE/'fixtures'/('native-'+dtype+'.pb')).read_bytes();hlo=(HERE/'fixtures'/('native-'+dtype+'.hlo.txt')).read_text()
        parameters={str(i):copy.deepcopy(inputs[k]) for i,k in enumerate(('activation','packed','scales'))}
        parameters['0']['shape']=[128,512]
        if case['rank']==3:
            proto=(HERE/'fixtures'/('native-'+dtype+'-rank3.pb')).read_bytes();hlo=(HERE/'fixtures'/('native-'+dtype+'-rank3.hlo.txt')).read_text()
        proof=G.verify_native(hlo,proto,proto,parameters,expected);_,_,_,payload,_=G.payload(hlo);prefix.with_suffix('.payload.json').write_text(payload);prefix.with_suffix('.mosaic-body.bin').write_bytes(base64.b64decode(json.loads(payload)['custom_call_config']['body']))
        item=C.read(HERE/'fixtures/mosaic-metadata.json')['cases'][dtype];a=C.read(C.HERE/'package-admission.json');target=G.grouped(inputs)
        boundary={'status':'RETURNED','native_api':'_xla_tpu_custom_call','calls':1,'original_payload':item['original'],'original_payload_sha256':C.sha_bytes(item['original'].encode()),'payload':payload,'payload_sha256':C.sha_bytes(payload.encode()),'payload_conversion':item['audit'],'metadata':[{'shape':v['shape'],'dtype':v['dtype']}for v in target],'operand_object_ids':[4001,4002,4003],'output_shapes':[[128,256]],'output_dtypes':[case['dtype']],'factory_sha256':a['plugin_python_files']['native_factory.py'],'converter_sha256':C.sha(C.HERE/'helpers/mosaic_compatibility.py')}
        G.verify_boundary(boundary,payload,expected);Probe.retain_conversion(prefix,boundary)
    else:
        rank='rank'+str(case['rank'])
        if kind.startswith('dX'):key=('dx2-' if kind=='dX_accumulated' else 'dx-')+rank;parameters={'0':inputs['dy'],'1':M.tensor(decode(M.materialize(inputs['packed']).flatten(),M.materialize(inputs['scales']),code,64,(256,512),__import__('torch').float32))}
        elif kind.startswith('db'):key=('db2-' if kind=='db_accumulated' else 'db-')+rank;parameters={'0':inputs['dy']}
        else:
            key='reference-tail' if 'tail' in case['id'] else 'reference-noncontiguous' if 'noncontiguous' in case['id'] else 'reference-'+rank
            dense=M.tensor(decode(M.materialize(inputs['packed']).flatten(),M.materialize(inputs['scales']),code,64,(256,512),__import__('torch').float32));parameters={'0':inputs['activation'],'1':dense}
            if case['bias']:parameters['2']=inputs['bias']
        proto=(HERE/'fixtures'/(key+'.pb')).read_bytes();hlo=(HERE/'fixtures'/(key+'.hlo.txt')).read_text();proof=G.verify_reference(proto,proto,output['shape'],output['dtype'],bias_gradient=kind.startswith('db'),parameters=parameters,allowed_parameters=expected['allowed_parameters'])
    prefix.with_suffix('.context.pb').write_bytes(proto);prefix.with_suffix('.printer.pb').write_bytes(proto);prefix.with_suffix('.hlo.txt').write_text(hlo);prefix.with_suffix('.context.textproto').write_text('fixture_scope: "SYNTHETIC_PROTOCOL_FIXTURE"\n');C.write(prefix.with_suffix('.parameters.json'),parameters)
    return {'graph':proof,'native_boundary':boundary,'graph_files_prefix':'graphs/'+name,'parameters_sha256':C.digest(parameters),'output':copy.deepcopy(output),'execution':{'api':'_xla_sync_multi','targets':['actual'],'wait':True,'wait_device_ops':True,'metrics_after_sync_before_output':True,'other_execution_in_interval':False,'graph_hash':'aabbccdd','graph_hash_binding':'PRE_SYNC_IDENTIFIER_NOT_EXECUTABLE_LINK','started_monotonic':1.,'finished_monotonic':2.},'counters':{},'execution_metrics':{'ExecuteTime':[1,0.1,[[1.0,0.1]]]}}


def make_fixture(base,args):
    root=Path(base);root.mkdir(parents=True,exist_ok=False);a,p,upstream=C.admission(args.admission,args.protocol,args.package_root,args.upstream_source,args.repo_root);decode,code=M.decode_original(upstream);B=C.load(C.HERE/'helpers/probe_backend.py','public_fixture_B');rows={c['id']:C.read(Path(args.oracle)/'rows'/(c['id']+'.json')) for c in p['cases']};cases=[]
    for case in p['cases']:
        name=case['id'];expected={**rows[name],'inputs':M.input_record(case),'allowed_parameters':M.allowed_reference_values(case,decode,code,rows[name]['outputs'])};objects={'activation':{'id':1001,'is_leaf':True,'shape':case['activation_shape'],'stride':[512,1] if case['rank']==2 else [32768,512,1],'dtype':'torch.'+case['dtype'],'device':'xla:0','requires_grad':case['gradients']!='NOT_RUN'},'weight':{'id':1002,'class':'bitsandbytes.nn.modules.Params4bit','device':'xla:0','requires_grad':False,'value_sha256':C.digest(expected['inputs']['packed']),'grad_is_none':True,'byte_sha256':M.byte_sha(M.materialize(expected['inputs']['packed'])),'quant_state':M.state_contract(M.materialize(expected['inputs']['scales']),code,case['dtype'])},'bias':None}
        if case['bias']:objects['bias']={'id':1003,'class':'torch.nn.parameter.Parameter','device':'xla:0','requires_grad':True,'value_sha256':C.digest(expected['inputs']['bias']),'byte_sha256':M.byte_sha(M.materialize(expected['inputs']['bias'])),'grad_is_none':True}
        after=copy.deepcopy(objects)
        if case['bias']:after['bias']['grad_is_none']=False
        raw={'case':name,'status':'PASS','pid':33000,'process_token':'2'*32,'inputs_sha256':expected['inputs_sha256'],'selected_events':[{'path':case['expected_path'],'reason':case['expected_reason']}],'functionalization':{'caller_activation_functional':True} if name=='public-functionalized-fp32' else {},'out_same_object':True if name=='public-out-fp32' else None,'objects_before':objects,'objects_after':after,'outputs':copy.deepcopy(expected['outputs']),'gradients':{}}
        raw['forward']=graph(root,name+'-y',case,expected,raw['outputs']['y'],decode,code,kind='native' if case['expected_path']=='PALLAS_CANDIDATE' else 'reference')
        for key in set(expected['outputs'])-{'y'}:raw['gradients'][key]=graph(root,name+'-'+key,case,expected,raw['outputs'][key],decode,code,kind=key)
        if raw['gradients']:raw['gradient_identity']={'activation_leaf_before':1001,'activation_leaf_after':1001,'first_buffer':2001,'second_buffer':2001,'bias_first_buffer':2002 if case['bias'] else None,'bias_second_buffer':2002 if case['bias'] else None}
        tol=p['tolerances']['fp32_forward_gradient' if case['dtype']=='float32' else 'bf16_forward_unit_scale'];raw['gates']={key:B.numeric(v['values'],expected['outputs'][key]['values'],tol) for key,v in raw['outputs'].items()};C.write(root/'rows'/(name+'.json'),raw);cases.append({'case':name,'status':'PASS'})
    runtime=dict(a['runtime']);runtime.pop('precision');r={'kind':'PUBLIC_PALLAS_DEVICE_PROBE','status':'COMPLETE','scope':'SYNTHETIC_PROTOCOL_FIXTURE','scientific_variant':C.VARIANT,'source_variant':'pallas-forward-mosaic7-gather-bf16-fp32-v1','scientific_sources':C.scientific_sources(),'pid':33000,'pgid':33000,'parent_pid':32000,'process_token':'2'*32,'deadline_epoch':2000.,'started_epoch':1000.,'finished_epoch':1500.,'admission_sha256':C.ADMISSION_SHA,'source_pre':C.ADMISSION_SHA,'source_post':C.ADMISSION_SHA,'protocol_sha256':C.PROTOCOL_SHA,'oracle_sha256':args.oracle_sha256,'native_acceptance_sha256':None,'wheel_records':C.wheels(a,args.plugin_wheel,args.upstream_wheel),'runtime_wheel_records':C.runtime_wheels(a,args.repo_root,None,qualified=False),'installed_runtime_metadata':None,'runtime':runtime,'precision':{'requested':'highest','readback':'highest','set_calls':1,'before_graph':True},'precision_environment':{'XLA_USE_BF16':None,'XLA_DOWNCAST_BF16':None,'XLA_USE_F32_FOR_BF16':None,'XLA_FLAGS':None},'device':{'type':'xla','hardware':'TPU','pjrt':'TPU'},'cases':cases,'numerical_status':'PASS','m4_status':'NOT_QUALIFIED','m6_memory_status':'NOT_QUALIFIED','performance':'NOT_MEASURED','artifacts':{}}
    C.write(root/'receipt.json',r);C.write(root.parent/'outer.json',{'pid':33000,'pgid':33000,'argv':['python','probe_public.py','--process-token','2'*32,'--deadline-epoch','2000.0'],'cleanup':{'status':'CLEANUP_VERIFIED','leader_reaped':True,'group_absence':'OBSERVED_NO_SUCH_GROUP','errors':[]}});reseal(root);return root

def reseal(root):
    root=Path(root);r=C.read(root/'receipt.json');r['artifacts']=C.inventory(root,('receipt.json',));C.write(root/'receipt.json',r)
