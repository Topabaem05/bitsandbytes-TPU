"""Independent recovered training/launch/source/wheel reader. No milestone acceptance writer."""
import argparse,hashlib,json,sys,zipfile
from pathlib import Path
import contract as C
import remote as D

def wheel(path,package,expected):
 with zipfile.ZipFile(path)as z:
  names=[n for n in z.namelist()if n.startswith(package+'/')and n.endswith('.py')];C.require(len(names)==len(set(names)),'WHEEL_DUPLICATE_SOURCE')
  actual={n.removeprefix(package+'/'):hashlib.sha256(z.read(n)).hexdigest()for n in names}
  C.require(actual==expected,'BUILT_COMPLETE_WHEEL_SOURCE:'+package)

def source_proof(packet,records):
 a=C.source(packet);admission=a.source();C.require(C.inventory(records/'source')==C.inventory(packet),'FULL_RECOVERED_SOURCE_PACKET')
 got=C.read(records/'installed-source.json');C.require(got=={'kind':'M5_INSTALLED_SOURCE_READBACK','status':'SOURCE_BYTES_PASS','source_admission_sha256':C.sha(packet/'integration/source-admission.json'),'upstream_python_files':admission['upstream_python_files'],'plugin_python_files':admission['plugin_python_files'],'py_typed_sha256':admission['upstream_package_files']['py.typed']},'ACTUAL_INSTALLED_SOURCE_RECORD')
 wheel(records/'wheels/upstream.whl','bitsandbytes',admission['upstream_python_files']);wheel(records/'wheels/plugin.whl','bitsandbytes_tpu',admission['plugin_python_files'])
 with zipfile.ZipFile(records/'wheels/upstream.whl')as z:C.require(hashlib.sha256(z.read('bitsandbytes/py.typed')).hexdigest()==admission['upstream_package_files']['py.typed'],'WHEEL_PY_TYPED')
 return a

def launch(records,phase,parent,packet,remote_base,birth,deadline):
 ownership=C.read(records/'steps'/phase/'ownership.json');result=C.read(records/'steps'/phase/'result.json');base=Path(remote_base)
 argv=[str(base/'venv/bin/python'),'-B',str(base/'payload/source/training.py'),*D.science_arguments(base/'payload',base/'records',phase,parent['process_token'],birth,deadline,m4_sha=C.sha(packet/'accepted-m4.json'),oracle_sha=C.sha(records/'cpu/oracle-seal.json'),gate_sha=C.sha(records/'cpu-gate.json')if phase=='device'else None)]
 # Paths and hashes in the argument builder refer to recovered records, not the root filesystem.
 for flag,relative in [('--m4-acceptance-sha256','accepted-m4.json')]:argv[argv.index(flag)+1]=C.sha(packet/relative)
 if phase=='device':
  argv[argv.index('--oracle-sha256')+1]=C.sha(records/'cpu/oracle-seal.json');argv[argv.index('--cpu-gate-sha256')+1]=C.sha(records/'cpu-gate.json')
 C.require(ownership.get('argv')==result.get('argv')==argv and parent['pid']==parent['pgid']==ownership.get('pid')==ownership.get('pgid')and parent['parent_pid']==ownership.get('owner_pid'),'ACTUAL_OWNED_TRAINING_LAUNCH')
 close=ownership.get('cleanup',{});summary=result.get('cleanup',{});C.require(result.get('status')in('PASS','CHILD_FAILED')and result.get('exit_code')in(0,2)and not result.get('error')and close.get('exit_status')==ownership.get('exit_status')==result['exit_code']and close.get('status')=='CLEANUP_VERIFIED'and close.get('leader_reaped')is True and close.get('group_absence')=='OBSERVED_NO_SUCH_GROUP'and close.get('errors')==[]and summary.get('errors')==[]and summary.get('groups')==[close]and summary.get('restoration',{}).get('handlers_restored')is True and summary.get('restoration',{}).get('timer_restored')is True,'ACTUAL_TRAINING_OUTER_CLOSURE')
 C.require(result['environment']=={'CPU':D.CPU_ENV,'PJRT_DEVICE':'TPU'if phase=='device'else None},'ACTUAL_TRAINING_ENVIRONMENT')
 C.require(parent['deadline_epoch']==deadline and parent['allocation_epoch']==birth and birth<=parent['started_epoch']<=parent['finished_epoch']<=deadline,'ORIGINAL_TRAINING_ALLOCATION_TIMELINE')
 return ownership,result

def audit(packet,records,remote_base,phase,packet_sha):
 packet=Path(packet);records=Path(records);m=C.admit(packet);a=source_proof(packet,records);allocation=C.read(records/'allocation.json');birth=allocation['birth_epoch'];deadline=birth+3600-C.CLOSURE_RESERVE
 C.require(allocation=={'birth_epoch':birth,'original_deadline_epoch':birth+3600,'science_deadline_epoch':deadline,'packet_sha256':packet_sha},'ALLOCATION_NO_RESET')
 sys.path.insert(0,str(packet/'source'));import reader as V
 kwargs={'m4_acceptance':packet/'accepted-m4.json','m4_acceptance_sha256':m['m4_acceptance_sha256'],'qualified':True}
 context=V.R.setup(local=True);context.refs=V.R.references(context);kwargs['_context']=context
 parent=C.read(records/'cpu/oracle-seal.json');owner,result=launch(records,'cpu',parent,packet,remote_base,birth,deadline);report=V.verify(records/'cpu',**kwargs)
 C.require(report['record_validation']==report['numerical_status']=='PASS','FRESH_QUALIFIED_CPU_NUMERICS')
 gate={'kind':'ROOT_M5_FRESH_QUALIFIED_CPU_GATE','status':'QUALIFIED_LINUX20_RESUME10_VERIFIED','record_validation':'PASS','numerical_status':'PASS','packet_sha256':packet_sha,'source_bundle_sha256':report['source_bundle_sha256'],'m4_acceptance_sha256':m['m4_acceptance_sha256'],'oracle_sha256':C.sha(records/'cpu/oracle-seal.json'),'ownership_sha256':C.sha(records/'steps/cpu/ownership.json'),'step_result_sha256':C.sha(records/'steps/cpu/result.json'),'installed_source_sha256':C.sha(records/'installed-source.json'),'upstream_wheel_sha256':C.sha(records/'wheels/upstream.whl'),'plugin_wheel_sha256':C.sha(records/'wheels/plugin.whl'),'allocation_epoch':birth,'original_deadline_epoch':birth+3600,'m5_status':'NOT_QUALIFIED'}
 if phase=='cpu':return gate
 C.require(C.read(records/'cpu-gate.json')==gate,'RECOVERED_FRESH_CPU_GATE_EXACT')
 actual=C.read(records/'device/receipt.json');launch(records,'device',actual,packet,remote_base,birth,deadline);C.require(parent['finished_epoch']<=actual['started_epoch'],'DEVICE_AFTER_FRESH_QUALIFIED_CPU')
 report=V.verify(records/'device',device=True,oracle=records/'cpu',oracle_sha256=gate['oracle_sha256'],**kwargs)
 return {**report,'packet_sha256':packet_sha,'m4_acceptance_sha256':m['m4_acceptance_sha256'],'actual_receipt_sha256':C.sha(records/'device/receipt.json'),'fresh_cpu_gate_sha256':C.sha(records/'cpu-gate.json')}

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['cpu','device']);p.add_argument('--packet',type=Path,required=True);p.add_argument('--records',type=Path,required=True);p.add_argument('--remote-base',required=True);p.add_argument('--packet-sha256',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 try:r=audit(args.packet,args.records,args.remote_base,args.phase,args.packet_sha256)
 except Exception as e:r={'record_validation':'FAIL','numerical_status':'NOT_EVALUATED','m5_status':'NOT_QUALIFIED','error':{'type':type(e).__name__,'message':str(e)}}
 C.write(args.output,r);print(json.dumps({k:v for k,v in r.items()if k not in ['gates','independent_dense_gates']}));return 0 if r.get('record_validation')==r.get('numerical_status')=='PASS'else 2
if __name__=='__main__':raise SystemExit(main())
