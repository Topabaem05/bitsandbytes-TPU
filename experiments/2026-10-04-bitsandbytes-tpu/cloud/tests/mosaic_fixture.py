"""Explicit synthetic owner/record fixture. Never used as a CPU oracle or actual TPU result."""
import base64
import copy
import json
from pathlib import Path
import protocol as S
import verifier as V
import graph_binding as G
FIXTURES=Path(__file__).resolve().parent/'fixtures'


def reseal(root):
    root=Path(root);receipt=S.read(root/'device/receipt.json');receipt['artifacts']=S.inventory(root/'device',('receipt.json',));S.write(root/'device/receipt.json',receipt)
    parent=S.read(root/'parent.json');parent['children'][0]['receipt_sha256']=S.sha(root/'device/receipt.json');parent['artifacts']=S.inventory(root,('parent.json',));S.write(root/'parent.json',parent)

def make_fixture(root,oracle,admission='a'*64):
    root=Path(root);root.mkdir(parents=True,exist_ok=False);device=root/'device';(device/'raw').mkdir(parents=True)
    oracle_sha=S.sha(Path(oracle)/'oracle-seal.json');seal,cases=S.oracle(oracle,oracle_sha,qualified=False);parent_token='1'*32;child_token='2'*32;pid=33000;child_pid=33001;deadline=1800.0
    owner={'pid':child_pid,'pgid':pid,'parent_pid':pid,'process_token':child_token,'parent_process_token':parent_token,'deadline_epoch':deadline}
    launch={'parent_pid':pid,'parent_pgid':pid,'parent_process_token':parent_token,'process_token':child_token,'deadline_epoch':deadline}
    expected={'kind':'R6_EXPECTED_PRELAUNCH','source_variant':S.spec()['source_variant'],'sources':S.sources(),'protocol_sha256':S.sha(S.HERE/'protocol.json'),'oracle_sha256':oracle_sha,'oracle_artifacts':seal['artifacts'],'input_sha256':{name:row['input_sha256'] for name,row in cases.items()},'source_admission_sha256':admission,'launch':launch};S.write(root/'expected.json',expected);expected_sha=S.sha(root/'expected.json')
    config=S.read(FIXTURES/'printer/metadata.json')['cases']['f32']['converted']
    for name in S.spec()['native_case_ids']:
        target=S.expected_case(name,cases[name],owner);dtype='bf16' if name.endswith('bf16') else 'f32';a,b,c=target['grouped'];inputs=cases[name]['inputs']
        hlo=f'''HloModule synthetic_r6
ENTRY %synthetic_r6 {{
  %a = {dtype}[128,512]{{1,0}} parameter(0)
  %b = u8[65536,1]{{1,0}} parameter(1)
  %s = f32[2048]{{0}} parameter(2)
  %br = u8[256,2,128]{{2,1,0}} reshape(u8[65536,1]{{1,0}} %b)
  %bt = u8[2,256,128]{{2,1,0}} transpose(u8[256,2,128]{{2,1,0}} %br), dimensions={{1,0,2}}
  %sr = f32[256,2,4]{{2,1,0}} reshape(f32[2048]{{0}} %s)
  %st = f32[2,256,4]{{2,1,0}} transpose(f32[256,2,4]{{2,1,0}} %sr), dimensions={{1,0,2}}
  ROOT %cc = {dtype}[128,256]{{1,0}} custom-call({dtype}[128,512]{{1,0}} %a, u8[2,256,128]{{2,1,0}} %bt, f32[2,256,4]{{2,1,0}} %st), custom_call_target="tpu_custom_call", operand_layout_constraints={{{dtype}[128,512]{{1,0}}, u8[2,256,128]{{2,1,0}}, f32[2,256,4]{{2,1,0}}}}, backend_config={config}
}}
'''
        dtype_key='bf16' if name.endswith('bf16') else 'f32';conversion=S.read(FIXTURES/'printer/metadata.json')['cases'][dtype_key];config=conversion['converted'];proto=(FIXTURES/'printer'/(dtype_key+'.hlo.pb')).read_bytes();hlo=(FIXTURES/'printer'/(dtype_key+'.hlo.txt')).read_text()
        graph_binding=G.bind(proto,proto,hlo,{'0':copy.deepcopy(inputs['activation']),'1':copy.deepcopy(inputs['packed']),'2':copy.deepcopy(inputs['scales'])},config)
        graph_hash='aabbccdd';boundary={'original_payload':conversion['original'],'original_payload_sha256':V.digest(conversion['original']),'payload_conversion':conversion['audit'],'status':'RETURNED','native_api':'_xla_tpu_custom_call','calls':1,'payload':config,'payload_sha256':V.digest(config),'recorder_sha256':S.sources()['recorder.py'],'output_shapes':[[128,256]],'output_dtypes':[a['dtype']],'graph_hash':graph_hash,'operands':[{'shape':value['shape'],'dtype':value['dtype'],'value_sha256':V.digest(value)} for value in (a,b,c)]}
        execution={'api':'_xla_sync_multi','targets':['actual'],'wait':True,'wait_device_ops':True,'metrics_after_sync_before_output':True,'other_execution_in_interval':False,'graph_hash':graph_hash,'graph_hash_binding':'PRE_SYNC_IDENTIFIER_NOT_EXECUTABLE_LINK','started_monotonic':1.2,'finished_monotonic':1.8}
        raw={'protocol':'R6_NATIVE_BOUNDARY_V1','scope':'SYNTHETIC_PROTOCOL_FIXTURE','status':'COMPLETE','source_pins':target['source_pins'],'owner':owner,'case':name,'inputs':inputs,'hlo_sha256':V.digest(hlo),'selected_path':{'path':'PALLAS_CANDIDATE','calls':1,'fallback':False},'native_boundary':boundary,'parameters':{'0':copy.deepcopy(inputs['activation']),'1':copy.deepcopy(inputs['packed']),'2':copy.deepcopy(inputs['scales'])},'execution':execution,'counters':{},'execution_metrics':{'ExecuteTime':[1,0.1,[[1.0,0.1]]]},'output':cases[name]['cpu_reference'],'same_device_reference':cases[name]['cpu_reference'],'graph_binding':graph_binding}
        prefix=device/'raw'/name;prefix.with_suffix('.hlo.txt').write_text(hlo);prefix.with_suffix('.hlo.pb').write_bytes(proto);prefix.with_suffix('.printer.hlo.pb').write_bytes(proto);prefix.with_suffix('.context.textproto').write_text('name: "source-backed-protobuf-text-fixture"\n');prefix.with_suffix('.native-config.json').write_text(config);prefix.with_suffix('.mosaic-body.bin').write_bytes(base64.b64decode(json.loads(config)['custom_call_config']['body']))
        prefix.with_suffix('.original-native-config.json').write_text(conversion['original']);prefix.with_suffix('.original-mosaic-body.bin').write_bytes(base64.b64decode(json.loads(conversion['original'])['custom_call_config']['body']));S.write(prefix.with_suffix('.conversion-audit.json'),conversion['audit'])
        S.write(prefix.with_suffix('.boundary-record.json'),raw);raw['status']='PASS';S.write(prefix.with_suffix('.json'),raw)
    name='tail-reference-fp32';S.write(device/'raw'/f'{name}.json',{'status':'PASS','inputs':cases[name]['inputs'],'output':cases[name]['cpu_reference'],'selected_events':[{'path':'SAME_DEVICE_REFERENCE','reason':'UNSUPPORTED_TILE_REFERENCE'}],'counters':{},'execution_metrics':{'ExecuteTime':[1,0.1,[[1.0,0.1]]]}})
    receipt={'status':'COMPLETE','numerical_status':'PASS',**owner,'source_variant':S.spec()['source_variant'],'m4_source_compatibility':'NOT_QUALIFIED','oracle_sha256':oracle_sha,'expected_sha256':expected_sha,'source_pre':admission,'source_post':admission,'source_admission_sha256':admission,'precision':{'requested':'highest','readback':'highest','set_calls':1,'before_graph':True},'precision_environment':{'XLA_USE_BF16':None,'XLA_DOWNCAST_BF16':None,'XLA_USE_F32_FOR_BF16':None,'XLA_FLAGS':None},'runtime':S.read(S.HERE/'probe-profile.json')['runtime'],'device':{'type':'xla','hardware':'TPU'},'public_api':{'original_class_identity':True,'original_method_identity':True},'m6_status':'NOT_QUALIFIED','registration':'NOT_EXECUTED','backward_memory_performance':'NOT_RUN','cases':[{'name':name,'status':'PASS'} for name in S.spec()['case_ids']],'compiler_dumps':{'status':'NO_ARTIFACTS','executable_link':'UNKNOWN'},'artifacts':{}}
    S.write(device/'receipt.json',receipt)
    argv=['python','device_diagnostic.py']
    for key,value in [('parent-pid',pid),('parent-process-token',parent_token),('process-token',child_token),('deadline-epoch',deadline),('oracle-sha256',oracle_sha),('expected-sha256',expected_sha),('admission-sha256',admission)]:argv+=['--'+key,str(value)]
    child={'phase':'native-boundary','pid':child_pid,'launched_pid':child_pid,'pgid':pid,'parent_pid':pid,'parent_process_token':parent_token,'process_token':child_token,'deadline_epoch':deadline,'started_epoch':1000.0,'finished_epoch':1500.0,'started_monotonic':1.0,'finished_monotonic':2.0,'reaped':True,'child_absent':True,'cleanup_errors':[],'timeout':False,'exit_code':0,'argv':argv,'receipt':'device/receipt.json','receipt_sha256':S.sha(device/'receipt.json')}
    parent={'kind':'R6_NATIVE_BOUNDARY_PARENT','status':'COMPLETE','pid':pid,'pgid':pid,'process_token':parent_token,'deadline_epoch':2000.0,'source_variant':S.spec()['source_variant'],'protocol_sha256':S.sha(S.HERE/'protocol.json'),'manifest_sha256':'f'*64,'source_pre':admission,'source_post':admission,'source_admission_sha256':admission,'oracle_sha256':oracle_sha,'expected_sha256':expected_sha,'children':[child],'child_science_exit_code':0,'artifacts':{}};S.write(root/'parent.json',parent);reseal(root)
    outer={'pid':pid,'pgid':pid,'argv':['python','coordinator.py','--process-token',parent_token,'--deadline-epoch','2000.0','--manifest-sha256','f'*64,'--oracle-sha256',oracle_sha,'--admission-sha256',admission],'cleanup':{'status':'CLEANUP_VERIFIED','leader_reaped':True,'group_absence':'OBSERVED_NO_SUCH_GROUP','errors':[]}}
    return {'oracle_sha':oracle_sha,'admission_sha':admission,'expected_sha':expected_sha,'outer':outer}
