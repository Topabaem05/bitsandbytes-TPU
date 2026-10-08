"""Offline recovery verifier. Requires independent oracle and actual outer ownership record."""
import argparse
import math
from pathlib import Path
import re
import traceback
import protocol as S
import verifier as V
import graph_binding as G


def flag(argv,name):
    S.require(isinstance(argv,list) and argv.count('--'+name)==1,'ARGV_FLAG_'+name);i=argv.index('--'+name);S.require(i+1<len(argv),'ARGV_FLAG_'+name);return argv[i+1]

def verify(actual,oracle,oracle_sha,admission_sha,expected_sha,outer,*,actual_device=True):
    root=Path(actual);parent=S.read(root/'parent.json');expected=S.read(root/'expected.json');seal,cases=S.oracle(oracle,oracle_sha,qualified=actual_device)
    S.require(parent.get('kind')=='R6_NATIVE_BOUNDARY_PARENT' and parent.get('status')=='COMPLETE' and not parent.get('error'),'PARENT_TERMINAL')
    S.require(parent.get('source_variant')==S.spec()['source_variant'] and parent.get('source_pre')==parent.get('source_post')==parent.get('source_admission_sha256')==admission_sha,'PARENT_SOURCE_VARIANT')
    S.require(parent.get('oracle_sha256')==oracle_sha and parent.get('expected_sha256')==expected_sha==S.sha(root/'expected.json'),'INDEPENDENT_EXPECTED_SEAL')
    S.check_inventory(root,parent['artifacts'],('parent.json',))
    S.require(outer.get('pid')==outer.get('pgid')==parent.get('pid')==parent.get('pgid') and type(parent['pid'])is int and parent['pid']>0,'ACTUAL_OUTER_PARENT_PID_PGID')
    cleanup=outer.get('cleanup',{});S.require(cleanup.get('status')=='CLEANUP_VERIFIED' and cleanup.get('leader_reaped') is True and cleanup.get('group_absence')=='OBSERVED_NO_SUCH_GROUP' and cleanup.get('errors')==[],'OUTER_GROUP_CLOSURE')
    S.require(re.fullmatch('[0-9a-f]{32}',parent.get('process_token','')) and flag(outer['argv'],'process-token')==parent['process_token'] and float(flag(outer['argv'],'deadline-epoch'))==parent['deadline_epoch'],'OUTER_LAUNCH_TOKEN_DEADLINE')
    S.require(parent['manifest_sha256']==flag(outer['argv'],'manifest-sha256') and parent['oracle_sha256']==flag(outer['argv'],'oracle-sha256') and parent['source_admission_sha256']==flag(outer['argv'],'admission-sha256'),'OUTER_SOURCE_ORACLE_MANIFEST')
    S.require(expected.get('kind')=='R6_EXPECTED_PRELAUNCH' and expected.get('source_variant')==S.spec()['source_variant'] and expected.get('sources')==S.sources() and expected.get('protocol_sha256')==parent.get('protocol_sha256')==S.sha(S.HERE/'protocol.json'),'EXPECTED_SOURCE_PROTOCOL')
    S.require(expected.get('oracle_sha256')==oracle_sha and expected.get('source_admission_sha256')==admission_sha and expected.get('oracle_artifacts')==seal['artifacts'] and expected.get('input_sha256')=={name:row['input_sha256'] for name,row in cases.items()},'EXPECTED_INDEPENDENT_ORACLE')
    children=parent.get('children');S.require(isinstance(children,list) and len(children)==1,'ONE_DEVICE_CHILD');child=children[0]
    S.require(child.get('phase')=='native-boundary' and child.get('pid')==child.get('launched_pid') and type(child.get('pid'))is int and child['pid']>0 and child['pid']!=parent['pid'] and child.get('pgid')==parent['pgid'],'FRESH_INHERITED_CHILD')
    S.require(child.get('parent_pid')==parent['pid'] and child.get('parent_process_token')==parent['process_token'] and re.fullmatch('[0-9a-f]{32}',child.get('process_token','')),'CHILD_PARENT_LINK')
    S.require(child.get('reaped') is True and child.get('child_absent') is True and child.get('cleanup_errors')==[] and child.get('timeout') is False and child.get('exit_code')==0 and parent.get('child_science_exit_code')==0,'CHILD_TERMINAL')
    for key in ('started_monotonic','finished_monotonic','started_epoch','finished_epoch','deadline_epoch'):S.require(type(child.get(key))in(int,float) and math.isfinite(child[key]),'CHILD_TIMESTAMPS')
    S.require(math.isfinite(parent['deadline_epoch']) and child['started_epoch']<=child['finished_epoch']<=child['deadline_epoch']<=parent['deadline_epoch'] and child['started_monotonic']<=child['finished_monotonic'],'CHILD_DEADLINE_INTERVAL')
    launch={'parent_pid':parent['pid'],'parent_pgid':parent['pgid'],'parent_process_token':parent['process_token'],'process_token':child['process_token'],'deadline_epoch':child['deadline_epoch']};S.require(expected.get('launch')==launch,'SEALED_PRELAUNCH_IDENTITY')
    for name,value in [('parent-pid',str(parent['pid'])),('parent-process-token',parent['process_token']),('process-token',child['process_token']),('deadline-epoch',str(child['deadline_epoch'])),('oracle-sha256',oracle_sha),('expected-sha256',expected_sha),('admission-sha256',admission_sha)]:S.require(flag(child['argv'],name)==value,'ACTUAL_CHILD_ARGV')
    receipt=S.read(root/'device/receipt.json');S.require(child.get('receipt')=='device/receipt.json' and child.get('receipt_sha256')==S.sha(root/'device/receipt.json'),'CHILD_RECEIPT_HASH')
    S.require(receipt.get('status')=='COMPLETE' and receipt.get('numerical_status')=='PASS' and not receipt.get('error_type') and receipt.get('source_variant')==S.spec()['source_variant'] and receipt.get('m4_source_compatibility')=='NOT_QUALIFIED','DEVICE_TERMINAL_VARIANT')
    owner={'pid':child['pid'],'pgid':parent['pgid'],'parent_pid':parent['pid'],'process_token':child['process_token'],'parent_process_token':parent['process_token'],'deadline_epoch':child['deadline_epoch']}
    for key,value in owner.items():S.require(receipt.get(key)==value,'CHILD_ACTUAL_IDENTITY')
    S.require(receipt.get('oracle_sha256')==oracle_sha and receipt.get('expected_sha256')==expected_sha and receipt.get('source_pre')==receipt.get('source_post')==receipt.get('source_admission_sha256')==admission_sha,'DEVICE_SOURCE_ORACLE_SEALS')
    S.require(receipt.get('precision')=={'requested':'highest','readback':'highest','set_calls':1,'before_graph':True},'DEVICE_HIGHEST')
    profile=S.read(S.HERE/'probe-profile.json');S.require(receipt.get('runtime')==profile['runtime'] and receipt.get('device',{}).get('type')=='xla' and receipt.get('device',{}).get('hardware')=='TPU','DEVICE_RUNTIME_HARDWARE_RECORD')
    identity=receipt.get('public_api',{});S.require(identity.get('original_class_identity') is True and identity.get('original_method_identity') is True,'PUBLIC_METHOD_IDENTITY')
    S.require(receipt.get('m6_status')=='NOT_QUALIFIED' and receipt.get('registration')=='NOT_EXECUTED' and receipt.get('backward_memory_performance')=='NOT_RUN','QUALIFICATION_SCOPE')
    S.require(receipt.get('cases')==[{'name':name,'status':'PASS'} for name in S.spec()['case_ids']],'COMPLETE_CASE_MATRIX')
    S.check_inventory(root/'device',receipt['artifacts'],('receipt.json',))
    B=S.load(S.HERE/'probe_backend.py','r6verifyB');P=S.load(S.HERE/'probe_precision.py','r6verifyP');P.validate_environment(B,receipt['precision_environment']);reports=[]
    for name in S.spec()['native_case_ids']:
        raw=S.read(root/'device/raw'/(name+'.json'));record=S.read(root/'device/raw'/(name+'.boundary-record.json'))
        S.require(record.get('scope')==('ACTUAL_NATIVE_BOUNDARY' if actual_device else 'SYNTHETIC_PROTOCOL_FIXTURE'),'CASE_ACTUAL_SCOPE')
        S.require(raw.get('status')=='PASS' and not raw.get('error_type') and record.get('status')=='COMPLETE' and not record.get('error_type'),'CASE_TERMINAL')
        S.require(record['native_boundary']==raw['native_boundary'] and record['output']==raw['output'] and record['parameters']==raw['parameters'] and record['execution']==raw['execution'] and record['inputs']==raw['inputs'],'CASE_CAPTURE_BOUNDARY_PAIR')
        S.require(child['started_monotonic']<=record['execution']['started_monotonic']<=record['execution']['finished_monotonic']<=child['finished_monotonic'],'CASE_OWNED_EXECUTION_INTERVAL')
        target=S.expected_case(name,cases[name],owner);V.array(raw['same_device_reference']);S.require(raw['same_device_reference']['shape']==target['cpu_reference']['shape'] and raw['same_device_reference']['dtype']==target['cpu_reference']['dtype'],'REFERENCE_ARRAY_IDENTITY');reference_gate=B.numeric(raw['same_device_reference']['values'],target['cpu_reference']['values'],target['tolerance']);S.require(reference_gate['status']=='PASS','REFERENCE_INDEPENDENT_ORACLE');hlo=(root/'device/raw'/(name+'.hlo.txt')).read_text();reports.append({'case':name,**V.verify(record,hlo,target,B,P)})
        S.require((root/'device/raw'/(name+'.native-config.json')).read_text()==record['native_boundary']['payload'],'EXACT_NATIVE_CONFIG_ARTIFACT')
        import base64,json
        body=base64.b64decode(json.loads(record['native_boundary']['payload'])['custom_call_config']['body'],validate=True)
        S.require((root/'device/raw'/(name+'.mosaic-body.bin')).read_bytes()==body and (root/'device/raw'/(name+'.hlo.pb')).stat().st_size>0,'RETAINED_GRAPH_PROTO_BODY_BYTES')
        binding=G.bind((root/'device/raw'/(name+'.hlo.pb')).read_bytes(),(root/'device/raw'/(name+'.printer.hlo.pb')).read_bytes(),hlo,record['parameters'],record['native_boundary']['payload'])
        S.require(binding==record.get('graph_binding')==raw.get('graph_binding'),'RETAINED_MAPPED_PRINTER_BINDING')
        S.require((root/'device/raw'/(name+'.context.textproto')).stat().st_size>0,'RETAINED_CONTEXT_PROTO_TEXT')
    tail=S.read(root/'device/raw/tail-reference-fp32.json');V.array(tail['output']);S.require(tail['output']['shape']==cases['tail-reference-fp32']['cpu_reference']['shape'] and tail['output']['dtype']==cases['tail-reference-fp32']['cpu_reference']['dtype'],'TAIL_ARRAY_IDENTITY');P.validate_execution(B,tail);S.require(tail.get('status')=='PASS' and tail['inputs']==cases['tail-reference-fp32']['inputs'] and tail['selected_events'][0]['path']=='SAME_DEVICE_REFERENCE','EXPLICIT_TAIL_REFERENCE')
    gate=B.numeric(tail['output']['values'],cases['tail-reference-fp32']['cpu_reference']['values'],profile['tolerances']['fp32_forward_gradient']);S.require(gate['status']=='PASS','TAIL_INDEPENDENT_ORACLE')
    return {'status':'BOUNDED_NATIVE_DIAGNOSTIC_PASS' if actual_device else 'OFFLINE_SHAPED_FIXTURE_PASS','source_variant':S.spec()['source_variant'],'native_cases':reports,'tail_gate':gate,'oracle_sha256':oracle_sha,'expected_sha256':expected_sha,'actual_device':actual_device,'optimized_executable_link':'UNKNOWN','compiler_body_memory_allocator':'NOT_QUALIFIED','public_integration_backward':'NOT_QUALIFIED','m4_source_compatibility':'NOT_QUALIFIED','m6':'NOT_QUALIFIED'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('actual','oracle','oracle-sha256','admission-sha256','expected-sha256','outer-ownership','outer-ownership-sha256','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    try:S.require(S.sha(args.outer_ownership)==args.outer_ownership_sha256,'INDEPENDENT_OUTER_OWNER_HASH');report=verify(args.actual,args.oracle,args.oracle_sha256,args.admission_sha256,args.expected_sha256,S.read(args.outer_ownership));code=0
    except Exception as error:report={'status':'REJECTED','error':str(error),'traceback':traceback.format_exc(),'m6':'NOT_QUALIFIED'};code=2
    S.write(args.output,report);raise SystemExit(code)
