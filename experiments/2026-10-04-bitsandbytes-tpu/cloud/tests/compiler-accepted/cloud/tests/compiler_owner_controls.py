"""Complete simulated host failure recovery. Never dispatch or certify a device result."""
import argparse,copy,json,shutil,sys,time
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;CLOUD=HERE.parent;sys.path.insert(0,str(CLOUD))
import remote,owner,transport,compiler_cloud_contract as CC

def run(packet,cpu_proofs,closed_fixture,output,jax_python):
 packet=Path(packet).resolve();cpu_proofs=Path(cpu_proofs).resolve();closed_fixture=Path(closed_fixture).resolve();out=Path(output).resolve();out.mkdir(exist_ok=False);m=CC.read(packet/'manifest.json');rows=[]
 for fault in (None,'old-cpu-source','compiler-inventory-corrupt','closed-metadata-failure'):
  case=out/(fault or 'primary-remote-failure');case.mkdir();base=case/'remote';base.mkdir();records=base/'records';records.mkdir()
  for n in ('cpu-oracle','steps/11-cpu-oracle'):shutil.copytree(cpu_proofs/n,records/n)
  for n in ('source-controls.json','installed-source.json','built-source.json'):shutil.copyfile(cpu_proofs/n,records/n)
  owned=CC.read(records/'steps/11-cpu-oracle/ownership.json');owned['cleanup']['group_absence']='OBSERVED_NO_SUCH_GROUP';CC.write(records/'steps/11-cpu-oracle/ownership.json',owned)
  if fault=='old-cpu-source':
   seal=CC.read(records/'cpu-oracle/oracle-seal.json');seal['sources']['coordinator.py']='0'*64;CC.write(records/'cpu-oracle/oracle-seal.json',seal)
  built=CC.read(records/'built-source.json');receipt={**CC.NATIVE_DEPENDENCY_BINDINGS,'compiler_manifest_sha256':CC.MANIFEST_SHA,'compiler_generation':CC.GENERATION,'status':'CPU_ORACLE_READY_TPU_NOT_RUN','error':None,'experiment':CC.MODE,'precision':'highest','packet_sha256':CC.sha(packet/'payload.zip'),'manifest_sha256':CC.sha(packet/'manifest.json'),'source_admission_sha256':m['source_admission_sha256'],'runtime_lock_sha256':m['runtime_lock_sha256'],'runtime_status':'PASS_TPU_RUNTIME_PROBE_ONLY','cpu_status':'PASS','tpu_status':'NOT_RUN','built_source_status':'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS','installed_source_status':'POST_PATCH_PYTHON_SOURCE_PASS','source_controls_status':'PASS_QUALIFIED_LINUX_SOURCE_CONTROLS','built_wheels':[{'name':built['wheel_name'],'sha256':built['wheel_sha256']}],'oracle_sha256':CC.sha(records/'cpu-oracle/oracle-seal.json'),'steps':[],**{k:m[k]for k in remote.TRANSFER_BINDINGS}}
  receipt.update(CC.cpu_archive(records));transport.split(records/'compiler-cpu-evidence.zip',base/'compiler-cpu-parts',expected_sha256=receipt['compiler_cpu_evidence_sha256'],expected_bytes=receipt['compiler_cpu_evidence_bytes']);CC.write(records/'receipt.json',receipt)
  if fault in ('compiler-inventory-corrupt','closed-metadata-failure'):
   for n in ('compiler-inventory.json','compiler-evidence.zip'):shutil.copyfile(closed_fixture/n,records/n)
   capture=CC.read(closed_fixture/'compiler-capture.json');receipt.update({k:v for k,v in capture.items()if k.startswith('compiler_')});receipt['native_outer_owner_sha256']=CC.read(records/'compiler-inventory.json')['outer_ownership_sha256']
   transport.split(records/'compiler-evidence.zip',base/'compiler-parts',expected_sha256=receipt['compiler_evidence_sha256'],expected_bytes=receipt['compiler_evidence_bytes'])
  host=case/'host';identity=case/'cli-identity.json';CC.write(identity,{})
  gate={**CC.NATIVE_DEPENDENCY_BINDINGS,'status':'ACTUAL_DISPATCH_AUTHORIZED','packet_sha256':CC.sha(packet/'payload.zip'),'driver_sha256':CC.sha(CLOUD/'owner.py'),'output':str(host.resolve()),'plugin_source_manifest_sha256':m['plugin_source_manifest_sha256'],'runtime_lock_sha256':m['runtime_lock_sha256'],'budget':m['budget'],'one_allocation_only':True,'provider_or_solver':'FORBIDDEN','cli_identity_sha256':CC.sha(identity),'experiment':CC.MODE,'precision':'highest',**{k:m[k]for k in remote.TRANSFER_BINDINGS},'mosaic_audit_python':str(Path(jax_python).absolute()),'mosaic_audit_python_sha256':CC.sha(Path(jax_python).absolute()),'compiler_generation':m['compiler_generation'],'compiler_manifest_sha256':CC.MANIFEST_SHA,'compiler_policy_sha256':CC.POLICY_SHA,'m2_dependency_sha256':m['m2_dependency_sha256'],'compiler_source_variant':CC.VARIANT,'compiler_scope':m['compiler_scope']};CC.write(case/'gate.json',gate)
  calls=[];session=[]
  def api(label,argv,timeout):
   calls.append(label)
   if label=='03-one-allocation':session.append(argv[argv.index('-s')+1]);receipt['allocation_epoch']=CC.read(host/'owner.json')['allocation_epoch'];CC.write(records/'receipt.json',receipt)
   if label in ('04-sessions-after','90-before-stop'):return {'status':'PASS'},'['+session[0]+']\nHardware: V6E1'
   if label=='13-tpu':
    assert "'compiler-native'" in Path(argv[-1]).read_text();assert CC.read(host/'launch.json')['kind']=='ROOT_COMPILER_CPU_ORACLE_GATE'
    receipt.update(status='BLOCKED',tpu_status='ATTEMPTED_BLOCKED',error={'type':'NotImplementedError','message':'SYNTHETIC_PRIMARY_COMPILER_FAILURE'});CC.write(records/'receipt.json',receipt)
   if label=='21-export':CC.write(records/'receipt.json',{**receipt,**remote.package(records)})
   if label=='24-export-parts':transport.split(records/'evidence.zip',base/'result-parts',expected_sha256=CC.sha(records/'evidence.zip'),expected_bytes=(records/'evidence.zip').stat().st_size)
   if argv[0]=='download':
    destination=Path(argv[-1])
    if label in ('09-install-receipt','10-cpu-receipt','11-native-cpu-receipt'):CC.write(destination,{**receipt,'status':'INSTALLED_NOT_QUALIFIED'if label=='09-install-receipt'else'CPU_ORACLE_READY_TPU_NOT_RUN'})
    else:
     rel=argv[-2].removeprefix('content/bnb-tpu-first/');source=records/rel.removeprefix('records/')if rel.startswith('records/')else base/rel;shutil.copyfile(source,destination)
     if fault=='compiler-inventory-corrupt' and label=='27-compiler-inventory':destination.write_bytes(destination.read_bytes()+b'bad')
   return {'status':'PASS'},'Active assignments: 0\nUsage rate: 0.00/hr'if argv==['usage']else'No active sessions found on server.'
  with patch.object(owner,'verify_cli_identity',lambda *a:None):result=owner.drive(packet,host,CC.sha(packet/'payload.zip'),case/'gate.json',CC.sha(case/'gate.json'),Path('SYNTHETIC_CLI_PYTHON'),identity,api=api,simulated=True,mosaic_audit_python=Path(jax_python).absolute())
  assert result['mode']=='SIMULATED_NO_CLOUD' and result['status']=='BLOCKED' and result['server_empty_observed'] and result['usage_zero_observed'];assert calls[-4:]==['90-before-stop','91-stop-exact','92-after-stop','93-usage-after'];assert result.get('local_verifier_status')=='NOT_RUN_INCOMPLETE_NATIVE_RECORDS'
  if fault=='old-cpu-source':assert '13-tpu' not in calls and result['original_error']['message']=='ORACLE_CANDIDATE_SOURCE'
  else:
   assert '13-tpu' in calls and result['original_error']['message']=='REMOTE_PHASE_BLOCKED:13-tpu';assert CC.read(host/'recovered/receipt.json')['error']=={'type':'NotImplementedError','message':'SYNTHETIC_PRIMARY_COMPILER_FAILURE'}
  if fault=='compiler-inventory-corrupt':assert result['compiler_retrieval']=='BLOCKED' and result['compiler_retrieval_error']['message']=='COMPILER_RECOVERY_SEAL'
  if fault=='closed-metadata-failure':assert result['compiler_retrieval']=='SEALED_SEPARATE_ARCHIVE' and result['compiler_capture_status']=='METADATA_ONLY_CAPTURE_FAILED'
  rows.append({'case':fault or 'remote-primary-failure-no-secondary-verifier','status':'PASS','owner_status':result['status'],'simulated_allocation_requests':calls.count('03-one-allocation'),'provider_calls':0})
 CC.write(out/'results.json',{'status':'PASS','controls':rows,'scope':'SIMULATED_NO_CLOUD_NO_DEVICE_SCIENCE','provider_calls':0,'qualified_CPU':'SYNTHETIC_PROOFS_ONLY'});print(json.dumps({'status':'PASS','controls':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ('packet','cpu-proofs','closed-fixture','output','jax-python'):p.add_argument('--'+n,required=True)
 a=p.parse_args();run(a.packet,a.cpu_proofs,a.closed_fixture,a.output,a.jax_python)
