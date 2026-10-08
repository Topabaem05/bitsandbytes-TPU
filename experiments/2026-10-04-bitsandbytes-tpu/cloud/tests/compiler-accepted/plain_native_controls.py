"""Test ordinary-native admission with genuine packet bytes and an owned stdlib child."""
import argparse,copy,json,shutil,sys,time
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'cloud'));import remote,native_contract as NC,compiler_cloud_contract as CC
p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);m=NC.read(a.packet/'manifest.json');rows=[];groups=[]
fields=tuple(CC.NATIVE_DEPENDENCY_BINDINGS)+('compiler_generation','compiler_manifest_sha256','compiler_policy_sha256','compiler_scope','compiler_source_variant')
for field in (None,*fields):
 base=a.output/(field or 'correct');base.mkdir();shutil.copytree(a.packet,base/'payload');shutil.copyfile(a.packet/'payload.zip',base/'payload.zip');(base/'venv/bin').mkdir(parents=True);(base/'venv/bin/python').symlink_to(sys.executable);out=base/'records';out.mkdir();epoch=time.time()
 receipt={'status':'CPU_ORACLE_READY_TPU_NOT_RUN','experiment':NC.MODE,'precision':'highest','tpu_status':'NOT_RUN','cpu_status':'PASS','packet_sha256':NC.sha(base/'payload.zip'),'manifest_sha256':NC.sha(base/'payload/manifest.json'),'source_admission_sha256':m['source_admission_sha256'],'runtime_lock_sha256':m['runtime_lock_sha256'],'allocation_epoch':epoch,'science_deadline_epoch':epoch+400,'steps':[],'error':None,'oracle_sha256':'a'*64,'native_cpu_inventory_sha256':'b'*64,'native_cpu_evidence_sha256':'c'*64,**{k:m[k]for k in remote.TRANSFER_BINDINGS}}
 gate={'kind':'ROOT_NATIVE_CPU_ORACLE_GATE','status':'QUALIFIED_LINUX_CPU_ORACLE_VERIFIED','oracle_sha256':receipt['oracle_sha256'],'source_admission_sha256':m['source_admission_sha256'],'native_manifest_sha256':NC.MANIFEST_SHA,'native_cpu_inventory_sha256':receipt['native_cpu_inventory_sha256'],'native_cpu_evidence_sha256':receipt['native_cpu_evidence_sha256'],'source_variant':NC.VARIANT}
 if field:gate[field]=None
 NC.write(out/'receipt.json',receipt);NC.write(base/'launch.json',gate);calls=[];real=remote.run_step
 def fixture(out,label,argv,deadline,limit,**kw):
  calls.append(label);assert label=='12-native-parent' and argv[argv.index('--compiler-dumps')+1]=='off' and argv[argv.index('--manifest-sha256')+1]==NC.MANIFEST_SHA
  r=real(out,label,[sys.executable,'-B',str(HERE/'fixtures/ordinary_native_parent.py'),'--output',str(out/'native')],time.time()+15,10,cwd=out);groups.append(NC.read(out/'steps/12-native-parent/ownership.json'));return r
 with patch.object(remote,'run_step',fixture):code=remote.execute(base,NC.sha(base/'payload.zip'),epoch,'native')
 final=NC.read(out/'receipt.json')
 if field:assert code==1 and calls==[] and final['error']['message']=='COMPILER_GATE_NOT_REQUESTED'
 else:
  assert code==0 and calls==['12-native-parent'] and final['status']=='NATIVE_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED';assert not any(k in gate for k in fields)
 rows.append({'case':'ordinary_native_correct_gate_launch'if field is None else 'ordinary_native_reject_compiler_gate_'+field,'status':'PASS','calls':calls})
for field in fields:
 bad=copy.deepcopy(m);bad[field]=None
 try:remote.verify_experiment(bad)
 except ValueError as e:assert str(e)=='COMPILER_NOT_REQUESTED',(field,str(e))
 else:raise AssertionError('FALSE_ACCEPT:'+field)
 rows.append({'case':'ordinary_native_reject_compiler_manifest_'+field,'status':'PASS'})
NC.write(a.output/'results.json',{'status':'PASS','count':len(rows),'controls':rows,'registered_groups':[{'pid':g['pid'],'pgid':g['pgid'],'cleanup':g['cleanup']}for g in groups],'provider_calls':0,'actual_TPU':'NOT_RUN','scope':'OWNED_STDLIB_ADMISSION_FIXTURE_ONLY'});print(json.dumps({'status':'PASS','count':len(rows)}))
