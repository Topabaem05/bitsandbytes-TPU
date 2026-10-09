"""Actual dependency admission and isolated incorrect controls. Never dispatch a provider."""
import argparse,copy,json,shutil,sys,zipfile
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;ROOT=next(p for p in HERE.parents if(p/'packages/bitsandbytes-tpu/pyproject.toml').is_file());sys.path.insert(0,str(HERE/'cloud'));import compiler_cloud_contract as C,owner,remote,compiler_recovery as CR
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--packet',required=True);p.add_argument('--second-packet',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False);packet=Path(a.packet);second=Path(a.second_packet);m=C.read(packet/'manifest.json');dep=C.read(packet/'native-dependency.json');result=C.read(packet/'accepted-native-result.json');rows=[]
def yes(name,fn):fn();rows.append({'name':name,'status':'PASS'})
def no(name,fn,reason):
 try:fn()
 except (ValueError,PermissionError)as e:assert str(e)==reason,(name,str(e),reason);rows.append({'name':name,'status':'EXPECTED_REJECT','reason':str(e)})
 else:raise AssertionError('FALSE_ACCEPT:'+name)
yes('actual_accepted_dependency_record_and_result',lambda:C.dependency_records(dep,result))
yes('actual_dependency_payload_and_preflight',lambda:owner.preflight(packet,C.sha(packet/'payload.zip')))
for key in C.NATIVE_DEPENDENCY_BINDINGS:
 bad=copy.deepcopy(m);bad.pop(key);no('missing_'+key,lambda bad=bad:C.verify_manifest(bad),'COMPILER_BASE_NATIVE_MANIFEST'if key=='base_native_manifest_sha256'else'COMPILER_ACCEPTED_NATIVE_BINDING')
 bad=copy.deepcopy(m);bad[key]=None;no('null_'+key,lambda bad=bad:C.verify_manifest(bad),'COMPILER_BASE_NATIVE_MANIFEST'if key=='base_native_manifest_sha256'else'COMPILER_ACCEPTED_NATIVE_BINDING')
bad=copy.deepcopy(m);bad['compiler_generation']='compiler-mosaic-serde7-gather-bf16-fp32-v1';no('old_pending_generation',lambda:C.verify_manifest(bad),'NATIVE_MOSAIC_GENERATION')
for key,val,reason in [('status','FAILED','COMPILER_NATIVE_ACCEPTANCE_OUTCOME'),('native_manifest_sha256','0'*64,'COMPILER_NATIVE_DEPENDENCY_IDENTITY:native_manifest_sha256'),('native_generation','mosaic-serde7-gather-v1','COMPILER_NATIVE_DEPENDENCY_IDENTITY:native_generation'),('native_adoption_revision','0'*40,'COMPILER_NATIVE_DEPENDENCY_IDENTITY:native_adoption_revision'),('accepted_result_sha256','0'*64,'COMPILER_NATIVE_DEPENDENCY_IDENTITY:accepted_result_sha256'),('closure_status','UNKNOWN','COMPILER_NATIVE_DEPENDENCY_PROOFS')]:
 bad=copy.deepcopy(dep);bad[key]=val;no('wrong_dependency_'+key,lambda bad=bad:C.dependency_records(bad,result),reason)
for key,val,reason in [('status','FAILED','COMPILER_NATIVE_RESULT_OUTCOME'),('native_generation','mosaic-serde7-gather-v1','COMPILER_NATIVE_RESULT_IDENTITY:native_generation'),('cpu_gate_status','NOT_RUN','COMPILER_NATIVE_RESULT_PROOFS'),('resource_closure','FAIL','COMPILER_NATIVE_RESULT_PROOFS'),('actual_device_attempted',False,'COMPILER_NATIVE_RESULT_PROOFS'),('device_case_count',4,'COMPILER_NATIVE_RESULT_COUNTS')]:
 bad=copy.deepcopy(result);bad[key]=val;no('wrong_result_'+key,lambda bad=bad:C.dependency_records(dep,bad),reason)
bad=copy.deepcopy(result);bad['cases'][2]['status']='FAIL';no('failed_BF16_case',lambda:C.dependency_records(dep,bad),'COMPILER_NATIVE_RESULT_CASES')
# A complete, correctly bound dispatch gate reaches identity admission. It executes no provider or audit.
identity=out/'synthetic-cli-identity.json';C.write(identity,{})
jax=ROOT/'.work/r6-pallas-preparation/venv-jax071/bin/python';gate={'status':'ACTUAL_DISPATCH_AUTHORIZED','packet_sha256':C.sha(packet/'payload.zip'),'driver_sha256':C.sha(HERE/'cloud/owner.py'),'output':str((out/'never-created-host').resolve()),'plugin_source_manifest_sha256':m['plugin_source_manifest_sha256'],'runtime_lock_sha256':m['runtime_lock_sha256'],'budget':m['budget'],'one_allocation_only':True,'provider_or_solver':'FORBIDDEN','experiment':C.MODE,'precision':'highest',**{k:m[k]for k in remote.TRANSFER_BINDINGS},**C.NATIVE_DEPENDENCY_BINDINGS,'compiler_generation':C.GENERATION,'compiler_manifest_sha256':C.MANIFEST_SHA,'compiler_policy_sha256':C.POLICY_SHA,'m2_dependency_sha256':m['m2_dependency_sha256'],'compiler_source_variant':C.VARIANT,'compiler_scope':m['compiler_scope'],'mosaic_audit_python':str(jax),'mosaic_audit_python_sha256':C.sha(jax),'cli_identity_sha256':C.sha(identity)}
calls=[]
def api(*args,**kw):calls.append(args);raise AssertionError('PROVIDER_CALL_FORBIDDEN')
def drive(value):
 f=out/'synthetic-root-gate.json';C.write(f,value)
 with patch.object(owner,'verify_cli_identity',side_effect=ValueError('CORRECT_GATE_REACHED_IDENTITY_ADMISSION')):
  return owner.drive(packet,out/'never-created-host',C.sha(packet/'payload.zip'),f,C.sha(f),Path(sys.executable),identity,api=api,simulated=True,mosaic_audit_python=jax)
no('correct_root_gate_before_provider',lambda:drive(gate),'CORRECT_GATE_REACHED_IDENTITY_ADMISSION')
for key in C.NATIVE_DEPENDENCY_BINDINGS:
 bad=copy.deepcopy(gate);bad[key]=None;no('root_gate_missing_'+key,lambda bad=bad:drive(bad),'EXACT_ROOT_ACCEPTANCE_REQUIRED')
assert calls==[]and not(out/'never-created-host').exists()
# Hash/type/name readback of both genuine packets, with the actual admitted dependency bytes.
assert(packet/'payload.zip').read_bytes()==(second/'payload.zip').read_bytes()
with zipfile.ZipFile(packet/'payload.zip')as z:
 assert len(z.infolist())==len(m['files'])+1 and len(z.namelist())==len(set(z.namelist()))and set(z.namelist())=={*m['files'],'manifest.json'}
 for i in z.infolist():assert i.create_system==3 and i.external_attr==0o100644<<16 and z.read(i)==(packet/i.filename).read_bytes()
rows.append({'name':'two_identical_current_member_genuine_packets','status':'PASS'})
for name,key in [('native-dependency.json','native_dependency_sha256'),('accepted-native-result.json','native_accepted_result_sha256')]:
 mutant=out/('changed-'+name);shutil.copytree(packet,mutant);q=mutant/name;q.write_bytes(q.read_bytes()+b' ')
 bad=copy.deepcopy(m);bad['files'][name]={'bytes':q.stat().st_size,'sha256':C.sha(q)};bad[key]=C.sha(q);no('resealed_changed_'+name,lambda mutant=mutant,bad=bad:C.payload(mutant,bad),'COMPILER_ACCEPTED_NATIVE_BINDING')
# Recovered receipt and archive admission reject stale native acceptance before archive access.
for name,fn in [('cpu',lambda r:C.recover_cpu(out/'nonexistent',r)),('compiler',lambda r:CR.recover(out/'nonexistent',r))]:
 no('recovery_'+name+'_missing_dependency',lambda fn=fn:fn({}),'COMPILER_ACCEPTED_NATIVE_BINDING')
# Every native scientific file and all policy-bound helpers remain exact pending bytes.
native=C.read(HERE/'compiler-native/manifest.json')
for n,r in native['sources'].items():assert C.sha(HERE/'compiler-native'/n)==r['sha256'] and (HERE/'compiler-native'/n).stat().st_size==r['bytes']
assert C.sha(HERE/'compiler-native/compiler-policy.json')==C.POLICY_SHA;rows.append({'name':'native36_policy10_sources_exact_admitted_bytes','status':'PASS'})
C.write(out/'results.json',{'status':'PASS','count':len(rows),'controls':rows,'provider_calls':0,'actual_native_dependency':'ROOT_ACCEPTED_EXACT_BYTES','actual_compiler':'NOT_RUN','dispatch':'NOT_AUTHORIZED','M6':'NOT_QUALIFIED'});print(json.dumps({'status':'PASS','count':len(rows)}))
