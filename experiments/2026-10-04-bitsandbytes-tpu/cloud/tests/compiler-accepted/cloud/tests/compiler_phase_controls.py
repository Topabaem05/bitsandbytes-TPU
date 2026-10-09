"""Real owned fixture processes at compiler remote phase; no compiler or native science."""
import argparse,copy,json,os,shutil,sys,time
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;CLOUD=HERE.parent;PREP=CLOUD.parent;sys.path.insert(0,str(CLOUD));import remote,compiler_cloud_contract as CC

def run(packet,output):
 packet=Path(packet).resolve();output=Path(output).resolve();output.mkdir(exist_ok=False);m=CC.read(packet/'manifest.json');rows=[];groups=[]
 for fault in (None,'gate-old-compiler','gate-manifest','gate-policy','gate-oracle','gate-kind','wrong-mode','missing-parent','count-overflow','gate-native-null','gate-native-hash','gate-native-generation','gate-native-result','gate-native-revision','receipt-native-hash','receipt-old-generation','flag-blocked','flag-timeout'):
  base=output/(fault or 'correct');base.mkdir();shutil.copytree(packet,base/'payload');shutil.copyfile(packet/'payload.zip',base/'payload.zip');(base/'venv/bin').mkdir(parents=True);(base/'venv/bin/python').symlink_to(sys.executable);out=base/'records';out.mkdir();epoch=time.time()
  receipt={**CC.NATIVE_DEPENDENCY_BINDINGS,'compiler_manifest_sha256':CC.MANIFEST_SHA,'compiler_generation':CC.GENERATION,'status':'CPU_ORACLE_READY_TPU_NOT_RUN','experiment':CC.MODE,'precision':'highest','tpu_status':'NOT_RUN','cpu_status':'PASS','packet_sha256':CC.sha(base/'payload.zip'),'manifest_sha256':CC.sha(base/'payload/manifest.json'),'source_admission_sha256':m['source_admission_sha256'],'runtime_lock_sha256':m['runtime_lock_sha256'],'allocation_epoch':epoch,'science_deadline_epoch':epoch+400,'steps':[],'error':None,'oracle_sha256':'a'*64,'compiler_cpu_inventory_sha256':'b'*64,'compiler_cpu_evidence_sha256':'c'*64,**{k:m[k]for k in remote.TRANSFER_BINDINGS}}
  gate={**CC.NATIVE_DEPENDENCY_BINDINGS,'compiler_generation':CC.GENERATION,'kind':'ROOT_COMPILER_CPU_ORACLE_GATE','status':'QUALIFIED_LINUX_CPU_ORACLE_VERIFIED','oracle_sha256':receipt['oracle_sha256'],'source_admission_sha256':m['source_admission_sha256'],'compiler_manifest_sha256':CC.MANIFEST_SHA,'compiler_policy_sha256':CC.POLICY_SHA,'compiler_cpu_inventory_sha256':receipt['compiler_cpu_inventory_sha256'],'compiler_cpu_evidence_sha256':receipt['compiler_cpu_evidence_sha256'],'source_variant':CC.VARIANT}
  if fault=='gate-old-compiler':gate['compiler_manifest_sha256']=CC.read(PREP/'previous-compiler-manifest.json')and 'b46a21cf9d5fa8133df456c22b31ac31ec03984fd22bc61cb93f910ef87e3dbd'
  if fault=='gate-manifest':gate['compiler_manifest_sha256']='0'*64
  if fault=='gate-policy':gate['compiler_policy_sha256']='0'*64
  if fault=='gate-oracle':gate['oracle_sha256']='0'*64
  if fault=='gate-kind':gate['kind']='ROOT_NATIVE_CPU_ORACLE_GATE'
  if fault=='gate-native-null':gate['actual_native_acceptance']=None
  if fault=='gate-native-hash':gate['native_dependency_sha256']='0'*64
  if fault=='gate-native-generation':gate['base_native_generation']='mosaic-serde7-gather-v1'
  if fault=='gate-native-result':gate['native_accepted_result_sha256']='0'*64
  if fault=='gate-native-revision':gate['native_adoption_revision']='0'*40
  if fault=='receipt-native-hash':receipt['native_dependency_sha256']='0'*64
  if fault=='receipt-old-generation':receipt['compiler_generation']='compiler-mosaic-serde7-gather-bf16-fp32-v1'

  CC.write(out/'receipt.json',receipt);CC.write(base/'launch.json',gate);calls=[];real=remote.run_step
  def bounded_fixture(out,label,argv,deadline,limit,**kw):
   calls.append(label)
   if label=='12-native-parent':
    assert Path(argv[2])==base/'payload/native/coordinator.py' and argv[argv.index('--compiler-dumps')+1]=='request' and argv[argv.index('--compiler-policy-sha256')+1]==CC.POLICY_SHA
    assert float(argv[argv.index('--deadline-epoch')+1])==deadline-10 and deadline<=receipt['science_deadline_epoch']
    mode='count-overflow' if fault=='count-overflow' else 'correct'
    fixture=[sys.executable,'-B',str(PREP/'fixtures/topology_parent.py'),'--output',str(out/'native'),'--mode',mode,'--process-token',argv[argv.index('--process-token')+1],'--deadline-epoch',str(time.time()+5),'--compiler-dumps','request','--compiler-policy-sha256','e'*64]
    r=real(out,label,fixture,time.time()+8,8,cwd=out)
    o=CC.read(out/'steps/12-native-parent/ownership.json');groups.append(o)
    if fault=='missing-parent':(out/'native/parent.json').unlink()
    return r
   if label=='12b-compiler-postclosure':
    assert groups[-1]['cleanup']['group_absence']=='OBSERVED_NO_SUCH_GROUP' and groups[-1]['cleanup']['leader_reaped'] is True
   return real(out,label,argv,deadline,limit,**kw)
  def flag_fixture(*args,**kw):
   # Synthetic parser-admission stand-in. Dedicated flag controls verify real contract/owner rejection.
   assert args[0]==base and args[1]==base/'payload' and args[2]==out
   calls.append('11d-compiler-flags-FIXTURE_ONLY')
   if fault=='flag-blocked':raise ValueError('FLAG_PREFLIGHT_STEP_PASS')
   if fault=='flag-timeout':raise TimeoutError('FLAG_PREFLIGHT_OWNED_DEADLINE')
   (out/'steps/11d-compiler-flags').mkdir(parents=True)
   CC.write(out/'steps/11d-compiler-flags/result.json',{'status':'SYNTHETIC_PHASE_FIXTURE_NOT_LIBTPU_QUALIFICATION'})
   return {'compiler_flag_preflight_sha256':'d'*64,'compiler_flag_preflight_status':'SYNTHETIC_PHASE_FIXTURE_NOT_LIBTPU_QUALIFICATION'}
  try:
   with patch.object(remote,'run_step',bounded_fixture),patch.object(remote.FP,'execute',flag_fixture):code=remote.execute(base,CC.sha(base/'payload.zip'),epoch,'native' if fault=='wrong-mode' else 'compiler-native')
  except ValueError as e:
   assert fault in ('receipt-native-hash','receipt-old-generation') and str(e)in ('COMPILER_ACCEPTED_NATIVE_BINDING','COMPILER_RECEIPT_GENERATION') and calls==[]
   rows.append({'case':fault,'status':'PASS','requested_steps':calls,'rejected':str(e),'scope':'ACTUAL_OWNED_FIXTURE_WRITER_NOT_SCIENCE'});continue

  r=CC.read(out/'receipt.json')
  if fault in ('gate-old-compiler','gate-manifest','gate-policy','gate-oracle','gate-kind','wrong-mode','gate-native-null','gate-native-hash','gate-native-generation','gate-native-result','gate-native-revision'):assert code==1 and calls==[] and r['status']=='BLOCKED'
  elif fault in ('flag-blocked','flag-timeout'):
   assert code==1 and r['status']=='BLOCKED'and calls==['11d-compiler-flags-FIXTURE_ONLY']and not(out/'native').exists()
   assert r['tpu_status']=='NOT_RUN'
  elif fault=='missing-parent':
   assert code==1 and r['status']=='BLOCKED' and r['error']['type']=='FileNotFoundError';assert calls==['11d-compiler-flags-FIXTURE_ONLY','12-native-parent','12b-compiler-postclosure'] and (out/'compiler-evidence.zip').is_file()
  else:
   assert code==0 and calls==['11d-compiler-flags-FIXTURE_ONLY','12-native-parent','12b-compiler-postclosure']
   assert r['compiler_capture_status']==('METADATA_ONLY_CAPTURE_FAILED' if fault=='count-overflow' else 'SEALED_BOUNDED_RAW_CAPTURE')
   assert r['status']=='NATIVE_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED' and r['native_outer_owner_sha256']==CC.sha(out/'steps/12-native-parent/ownership.json')
   if fault=='count-overflow':assert CC.read(out/'native.compiler-private/monitor.json')['overflow_observation'] is not None
  rows.append({'case':fault or 'correct-phase-order-source-argv-deadline','status':'PASS','requested_steps':calls,'remote_status':r['status'],'scope':'ACTUAL_OWNED_FIXTURE_WRITER_NOT_SCIENCE'})
 CC.write(output/'results.json',{'status':'PASS','controls':rows,'registered_groups':[{'pid':o['pid'],'pgid':o['pgid'],'cleanup':o['cleanup']}for o in groups],'provider_calls':0,'actual_compiler':'NOT_RUN','actual_TPU':'NOT_RUN'});print(json.dumps({'status':'PASS','controls':len(rows),'groups':len(groups)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--packet',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.packet,a.output)
