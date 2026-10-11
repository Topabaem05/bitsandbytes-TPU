"""Synthetic saved PUBLIC15 records. No package installation or actual device."""
import argparse,copy,sys,time
from pathlib import Path
from types import SimpleNamespace
p=argparse.ArgumentParser()
for n in ('public-source','base-packet','cpu-fixture','jax071-python','fixture-source','output'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False);public=a.public_source.resolve();base=a.base_packet.resolve();cpu=a.cpu_fixture.resolve()
sys.path.insert(0,str(public));sys.path.insert(0,str(a.fixture_source.resolve()))
import contract as C,verify_public as V,record_fixture as F
ad=C.read(public/'package-admission.json');args=SimpleNamespace(admission=str(public/'package-admission.json'),protocol=str(public/'public-protocol.json'),package_root=str(public/'plugin/src/bitsandbytes_tpu_pallas'),upstream_source=str(cpu/'upstream/bitsandbytes'),repo_root=str(base/'public-baseline'),plugin_wheel=str(cpu/'plugin-synthetic-source-only.whl'),upstream_wheel=str(cpu/'upstream-synthetic-source-only.whl'),runtime_wheels=None,oracle=str(cpu/'oracle'),oracle_sha256=C.sha(cpu/'oracle/oracle-seal.json'),native_acceptance=None,native_acceptance_sha256=None,mosaic_audit_python=str(a.jax071_python.absolute()),audit_deadline_epoch=time.time()+180)
args.actual=str(F.make_fixture(out/'actual',args));args.receipt_sha256=C.sha(Path(args.actual)/'receipt.json');args.ownership=str(out/'outer.json');args.ownership_sha256=C.sha(args.ownership)
results=[]
def check():args.audit_deadline_epoch=time.time()+120;return V.verify(args,synthetic=True)
def good(name,**d):
 results.append({'name':name,'status':'PASS',**d});C.write(out/'progress.json',{'completed_controls':results,'terminal':'PARTIAL'})
def reject(name,fn,label):
 try:fn()
 except ValueError as e:assert label in str(e),(label,str(e));good(name,rejected=str(e));return
 raise AssertionError('incorrect fixture accepted:'+name)
r=check();assert r['record_validation']=='PASS'and r['numerical_status']=='PASS'and r['native_case_count']==11 and r['reference_case_count']==4;good('new_oracle_full15_saved_public_record_consumer',conversion_replays=len(r['independent_conversion_replay']))
# Restore exact fixture bytes after each isolated mutation.
root=Path(args.actual);receipt=root/'receipt.json';primary=root/'rows/direct-float32.json';old={q:q.read_bytes()for q in(receipt,primary)}
try:
 raw=C.read(primary);raw['outputs']['y']['values'][0]+=1.;raw['forward']['output']=copy.deepcopy(raw['outputs']['y']);target=C.read(Path(args.oracle)/'rows/direct-float32.json');B=C.load(public/'helpers/probe_backend.py','bounded_device_flow_gate');tol=C.read(args.protocol)['tolerances']['fp32_forward_gradient'];raw['gates']['y']=B.numeric(raw['outputs']['y']['values'],target['outputs']['y']['values'],tol);raw['status']='FAIL';C.write(primary,raw);r=C.read(receipt);r['cases'][0]['status']='FAIL';r['numerical_status']='FAIL';C.write(receipt,r);F.reseal(root);args.receipt_sha256=C.sha(receipt);r=check();assert r['record_validation']=='PASS'and r['numerical_status']=='FAIL';good('unchanged_device_gate_preserves_material_numerical_fail')
finally:
 for q,b in old.items():q.write_bytes(b)
 args.receipt_sha256=C.sha(receipt)
old=receipt.read_bytes()
try:
 r=C.read(receipt);r['scientific_variant']='public-pallas-device-mosaic7-gather-bf16-fp32-v1';C.write(receipt,r);args.receipt_sha256=C.sha(receipt);reject('old_device_generation_rejected',check,'DEVICE_EXACT_SOURCE_GENERATION')
finally:receipt.write_bytes(old);args.receipt_sha256=C.sha(receipt)
# The old failed seal cannot enter a device verifier through a new outer receipt.
seal=Path(args.oracle)/'oracle-seal.json';old=seal.read_bytes();oldreceipt=receipt.read_bytes();oldsha=args.oracle_sha256
try:
 s=C.read(seal);s['kind']='PUBLIC_PALLAS_CPU_ORACLE';C.write(seal,s);args.oracle_sha256=C.sha(seal);r=C.read(receipt);r['oracle_sha256']=args.oracle_sha256;C.write(receipt,r);args.receipt_sha256=C.sha(receipt);reject('old_oracle_kind_rejected_through_device_consumer',check,'CPU_TERMINAL')
finally:seal.write_bytes(old);receipt.write_bytes(oldreceipt);args.oracle_sha256=oldsha;args.receipt_sha256=C.sha(receipt)
# First-error handling remains the unchanged source contract.
oldreceipt=receipt.read_bytes();oldrows={q:q.read_bytes()for q in(root/'rows').glob('*.json')}
try:
 r=C.read(receipt);r['numerical_status']='ERROR'
 for i,item in enumerate(r['cases']):
  name=item['case'];raw={'case':name,'status':'ERROR'if i==0 else'NOT_RUN','pid':r['pid'],'process_token':r['process_token']}
  if i==0:
   raw.update(error_type='RuntimeError',error_message='SYNTHETIC_FIRST_DEVICE_FAILURE');(root/'errors').mkdir();(root/'errors'/(name+'.log')).write_text('SYNTHETIC_FIRST_DEVICE_FAILURE\n')
  else:raw.update(first_failure=r['cases'][0]['case'],reason='SHARED_XLA_STATE_AFTER_FIRST_FAILURE')
  item['status']=raw['status'];C.write(root/'rows'/(name+'.json'),raw)
 C.write(receipt,r);F.reseal(root);args.receipt_sha256=C.sha(receipt);r=check();assert r['numerical_status']=='ERROR'and r['rows'][0]['status']=='ERROR'and all(x['status']=='NOT_RUN'for x in r['rows'][1:]);good('first_error_retained_and_remaining14_notrun')
finally:
 for q,b in oldrows.items():q.write_bytes(b)
 (root/'errors/direct-float32.log').unlink(missing_ok=True);(root/'errors').rmdir();receipt.write_bytes(oldreceipt);args.receipt_sha256=C.sha(receipt)
C.write(out/'results.json',{'status':'PASS','count':len(results),'controls':results,'scope':'SYNTHETIC_SAVED_PUBLIC_RECORD_CONSUMER','actual_device':'NOT_RUN','qualified_linux_cpu':'NOT_RUN','provider_calls':0});print({'status':'PASS','count':len(results)})
