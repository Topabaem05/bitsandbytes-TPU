"""Recover exact source/CPU/device proof; acceptance is an explicit root command."""
import argparse,json,sys,time,types
from pathlib import Path
import contract as C
import remote as D

def modules(packet):
 n=Path(packet)/'nested-exact';sys.path.insert(0,str(n/'source'));import probe_nested_exact as E;import probe_state8_exact as S;import cpu_gate as G;import record_validation as V
 common=D.common(Path(packet));a=types.SimpleNamespace(**{common[i][2:].replace('-','_'):common[i+1]for i in range(0,len(common),2)})
 B,R,P,A,N=E.sources(a);return E,S,G,V,B,R,P,A,N,a

def arguments(packet,records,remote_base,phase,rec):
 base=Path(remote_base);rp=base/'payload';rr=base/'records';state=phase.endswith('state');device=phase.startswith('device')
 argv=[str(base/'venv/bin/python'),'-B',str(rp/'nested-exact/source'/('probe_state8_exact.py'if state else'probe_nested_exact.py')),'execute'if device else'prepare',*D.common(rp)]
 if state or device:argv+=['--oracle',str(rr/'cpu79'),'--oracle-sha256',C.sha(records/'cpu79/oracle-seal.json')]
 if device:argv+=['--plugin-source',str(rp/'nested-exact/integration/plugin/src/bitsandbytes_tpu'),'--cpu-gate',str(rr/'cpu-gate.json'),'--cpu-gate-sha256',C.sha(records/'cpu-gate.json')]
 if state and device:argv+=['--state-oracle',str(rr/'cpustate'),'--state-oracle-sha256',C.sha(records/'cpustate/oracle-seal.json'),'--accepted79',str(rr/'accepted79.json'),'--accepted79-sha256',C.sha(records/'accepted79.json')]
 argv+=['--output',str(rr/phase),'--source-review',str(rp/'source-review.json'),'--source-review-sha256',C.sha(packet/'source-review.json'),'--process-token',rec['process_token'],'--deadline-epoch',str(rec['deadline_epoch'])]
 return argv

def owned_device(rec,ownership,result,argv):
 C.require(ownership.get('argv')==result.get('argv')==argv and rec['pid']==rec['pgid']==ownership.get('pid')==ownership.get('pgid')and rec['parent_pid']==ownership.get('owner_pid'),'DEVICE_ACTUAL_LAUNCH_IDENTITY')
 cleanup=ownership.get('cleanup',{});summary=result.get('cleanup',{})
 C.require(result.get('status')in('PASS','CHILD_FAILED')and result.get('exit_code')in(0,2)and not result.get('error')and cleanup.get('exit_status')==ownership.get('exit_status')==result['exit_code']and cleanup.get('status')=='CLEANUP_VERIFIED'and cleanup.get('leader_reaped')is True and cleanup.get('group_absence')=='OBSERVED_NO_SUCH_GROUP'and cleanup.get('errors')==[]and summary.get('errors')==[]and summary.get('groups')==[cleanup]and summary.get('restoration',{}).get('handlers_restored')is True and summary.get('restoration',{}).get('timer_restored')is True,'DEVICE_OUTER_CLOSURE')

def preserve_observations(verify,diagnose):
 try:return verify()
 except Exception as error:
  result={'record_validation':'FAIL','numerical_status':'ERROR','record_validation_reason':str(error),'m4_status':'NOT_QUALIFIED'}
  try:result.update(diagnose())
  except Exception as diagnostic:result['diagnostic_error']=str(diagnostic)
  result.update(record_validation='FAIL',record_validation_reason=str(error),m4_status='NOT_QUALIFIED');return result

def audit(packet,records,remote_base,phase,packet_sha=None):
 packet=Path(packet);records=Path(records);C.admit(packet);E,S,G,V,B,R,P,A,N,a=modules(packet);allocation=C.read(records/'allocation.json');C.require(packet_sha is not None and allocation['packet_sha256']==packet_sha,'HOST_EXTERNAL_PACKET_BINDING');birth=allocation['birth_epoch'];deadline=allocation['science_deadline_epoch'];C.require(deadline==birth+3600-C.CLOSURE_RESERVE,'ORIGINAL_DEADLINE_NO_RESET')
 a.oracle=str(records/'cpu79');a.oracle_sha256=C.sha(records/'cpu79/oracle-seal.json');a.plugin_source=str(packet/'nested-exact/integration/plugin/src/bitsandbytes_tpu');a.source_review=str(packet/'source-review.json');a.source_review_sha256=C.sha(a.source_review)
 gold,_,_=V.verify_cpu(B,P,N,a.oracle,a.oracle_sha256,E.binding(B,a));expected=arguments(packet,records,remote_base,'cpu79',gold)
 gate,observations=G.audit(B,P,N,oracle=a.oracle,oracle_sha256=a.oracle_sha256,binding=E.binding(B,a),ownership=C.read(records/'steps/cpu79/ownership.json'),step_result=C.read(records/'steps/cpu79/result.json'),expected_argv=expected,installed_source=C.read(records/'installed-source.json'),upstream_wheel=records/'wheels/upstream.whl',plugin_wheel=records/'wheels/plugin.whl',admission_path=packet/'nested-exact/integration/source-admission.json')
 C.require(gold['deadline_epoch']==deadline and birth<=gold['started_epoch']<=gold['finished_epoch']<=deadline,'CPU79_ORIGINAL_ALLOCATION_TIMELINE')
 if phase=='cpu79':return gate
 C.require(C.read(records/'cpu-gate.json')==gate,'RECOVERED_UPLOADED_CPU_GATE')
 a.cpu_gate=str(records/'cpu-gate.json');a.cpu_gate_sha256=C.sha(a.cpu_gate)
 if phase=='device79':
  a.actual=str(records/'device79')
  def verify79():
   rec=C.read(records/'device79/receipt.json');C.require(rec['deadline_epoch']==deadline and gold['finished_epoch']<=rec['started_epoch']<=rec['finished_epoch']<=deadline,'DEVICE79_ORIGINAL_ALLOCATION_TIMELINE')
   launch=C.read(records/'steps/device79/ownership.json');result=C.read(records/'steps/device79/result.json');owned_device(rec,launch,result,arguments(packet,records,remote_base,'device79',rec));return E.verify_device(B,P,N,a)
  return preserve_observations(verify79,lambda:E.numerical_observations(B,P,N,a,Path(a.actual)))
 a.state_oracle=str(records/'cpustate');a.state_oracle_sha256=C.sha(records/'cpustate/oracle-seal.json');state,rows=S.verify_state_cpu(B,P,N,a)
 V.validate_cpu_launch(state,C.read(records/'steps/cpustate/ownership.json'),C.read(records/'steps/cpustate/result.json'),arguments(packet,records,remote_base,'cpustate',state))
 accepted=C.read(records/'accepted79.json');A.state_dependency(accepted,records/'accepted79.json',C.sha(records/'accepted79.json'),C.SCIENCE_SHA)
 actual=C.read(records/'device79/receipt.json');C.require(actual['finished_epoch']<=state['started_epoch']<=state['finished_epoch']<=deadline and state['deadline_epoch']==deadline,'STATE_CPU_AFTER_ACCEPTED79_ORIGINAL_DEADLINE')
 request=C.read(records/'root-review-boundary.json');C.require(request['accepted79_sha256']==C.sha(records/'accepted79.json')and request['actual79_finished_epoch']==actual['finished_epoch']and actual['finished_epoch']<=request['accepted_epoch']<=state['started_epoch']and request['birth_epoch']==birth and request['accepted_epoch']+C.STATE_RESERVE<birth+3600,'ACTUAL_ROOT_REVIEW_BOUNDARY')
 stateaudit={'kind':'ROOT_NESTED_EXACT_STATE_CPU_LAUNCH_AUDIT','status':'ACTUAL_STATE_CPU_LAUNCH_VERIFIED','state_oracle_sha256':a.state_oracle_sha256,'nested_oracle_sha256':a.oracle_sha256,'owned_cpu_pid':state['pid'],'owned_cpu_pgid':state['pgid'],'owned_cpu_token':state['process_token'],'ownership_sha256':C.sha(records/'steps/cpustate/ownership.json'),'step_result_sha256':C.sha(records/'steps/cpustate/result.json'),'root_acceptance_sha256':C.sha(records/'accepted79.json'),'m4_status':'NOT_QUALIFIED'}
 if phase=='cpustate':return stateaudit
 C.require(C.read(records/'state-cpu-launch-audit.json')==stateaudit,'STATE_CPU_SEPARATE_AUDIT_BYTES');a.accepted79=str(records/'accepted79.json');a.accepted79_sha256=C.sha(a.accepted79);a.actual=str(records/'devicestate')
 def verify8():
  parent=C.read(records/'devicestate/receipt.json');C.require(parent['deadline_epoch']==deadline and state['finished_epoch']<=parent['started_epoch']<=parent['finished_epoch']<=deadline,'STATE_DEVICE_ORIGINAL_TIMELINE');owned_device(parent,C.read(records/'steps/devicestate/ownership.json'),C.read(records/'steps/devicestate/result.json'),arguments(packet,records,remote_base,'devicestate',parent));return S.verify(B,R,P,A,N,a)
 def diagnose8():
  observations={phase:E.numerical_observations(B,P,N,a,Path(a.actual)/phase,selected_cases=S.fixed_cases(B))for phase in('save','restore')}
  return {'numerical_status':V.numerical_status([{'status':r['numerical_status']}for r in observations.values()]),'numerical_observations':observations,'scope':'RETURNED_NUMERICS_ONLY_STATE_AND_DEVICE_NOT_QUALIFIED'}
 return preserve_observations(verify8,diagnose8)

def accept(packet,records,remote_base,packet_sha,decision):
 C.require(decision=='ACCEPT_AFTER_ROOT_RECORD_REVIEW','EXPLICIT_ROOT_REVIEW_DECISION');r=audit(packet,records,remote_base,'device79',packet_sha);C.require(r.get('record_validation')==r.get('numerical_status')=='PASS'and r.get('case_count')==79,'ACTUAL79_NOT_ACCEPTABLE')
 n=Path(packet)/'nested-exact';rec=C.read(Path(records)/'device79/receipt.json')
 return {'kind':'ROOT_NESTED_EXACT79_DEVICE_ACCEPTANCE','status':'ACCEPTED_NESTED_EXACT79_DEVICE_RECORDS','record_validation':'PASS','numerical_status':'PASS','case_count':79,'source_variant':'nested-exact-v1','source_admission_sha256':C.ADMISSION_SHA,'integration_map_sha256':C.MAP_SHA,'science_map_sha256':C.SCIENCE_SHA,'inputs_sha256':rec['inputs_sha256'],'profile_sha256':rec['profile_sha256'],'runtime_lock_sha256':rec['runtime_lock_sha256'],'saved_state_status':'NOT_RUN','m4_status':'NOT_QUALIFIED','packet_sha256':packet_sha,'actual_receipt_sha256':C.sha(Path(records)/'device79/receipt.json'),'fresh_cpu_oracle_sha256':C.sha(Path(records)/'cpu79/oracle-seal.json')}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=('cpu79','device79','cpustate','devicestate','accept79'));p.add_argument('--packet',type=Path,required=True);p.add_argument('--records',type=Path,required=True);p.add_argument('--remote-base',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--packet-sha256');p.add_argument('--reviewer-decision');a=p.parse_args()
 try:r=accept(a.packet,a.records,a.remote_base,a.packet_sha256,a.reviewer_decision)if a.phase=='accept79'else audit(a.packet,a.records,a.remote_base,a.phase,a.packet_sha256);C.write(a.output,r);print(json.dumps({'phase':a.phase,'output':str(a.output),'sha256':C.sha(a.output),'record_validation':r.get('record_validation'),'status':r.get('status'),'numerical_status':r.get('numerical_status')}))
 except Exception as e:print(json.dumps({'record_validation':'FAIL','reason':str(e),'m4_status':'NOT_QUALIFIED'}));raise
