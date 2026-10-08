"""Isolated simulated cloud controls. All device/qualified Linux records are explicit fixtures."""
import copy,hashlib,importlib.util,json,os,shutil,signal,sys,tarfile
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];CLOUD=HERE.parent;sys.path.insert(0,str(CLOUD))
import build_packet as B,remote,owner,transport,native_contract as NC
import build_native_packet as N
FROZEN=HERE.parents[1]/'native';FIXTURES=HERE/'fixtures/native';OUT=None;ARCHIVE=None;ORACLE=None

def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def write(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);NC.write(p,v)
def prepare():
 OUT.mkdir(exist_ok=False)
 unpack_cpu_fixture(OUT/'cpu-fixture')
 admission_sha='8b2b3e26e2e7c98c0fde52f51663bed84999ac733dcbc39573ef8b3aeca4f9a1'
 dep={'kind':'ROOT_M2_DEPENDENCY_ACCEPTANCE','status':'ACCEPTED_ACTUAL_M2','source_variant':NC.VARIANT,'runtime_lock_sha256':B.RUNTIME_SHA,'patch_manifest_sha256':B.TRANSFER_PATCH_MANIFEST_SHA,'source_admission_sha256':admission_sha,'accepted_result_sha256':'c'*64}
 write(OUT/'m2-dependency.json',dep)
 original_run=B.subprocess.run;original_check=B.subprocess.check_output
 def run(argv,**kwargs):
  if argv[:1]==['git'] and 'archive'in argv:kwargs['stdout'].write(ARCHIVE.read_bytes());return
  return original_run(argv,**kwargs)
 def check(argv,**kwargs):return B.COMMIT+'\n' if argv[:1]==['git'] and 'rev-parse'in argv else original_check(argv,**kwargs)
 packet=OUT/'packet-fixture'
 with patch.object(B.subprocess,'run',run),patch.object(B.subprocess,'check_output',check):result=N.build(OUT/'SYNTHETIC_NO_CHECKOUT',packet,candidate=FROZEN,candidate_sha=NC.MANIFEST_SHA,m2_dependency=OUT/'m2-dependency.json',m2_dependency_sha=NC.sha(OUT/'m2-dependency.json'))
 write(OUT/'packet-result.json',{**result,'upstream_archive_method':'FIXTURE_SUBSTITUTES_GIT_ARCHIVE_WITH_EXACT_REVIEWED_BYTES','provider_calls':0})
 manifest=NC.read(packet/'manifest.json');owner.preflight(packet,result['payload_sha256'])
 return packet,manifest,result

def make_records(packet,manifest,base):
 base.mkdir();records=base/'records';records.mkdir();shutil.copytree(OUT/'cpu-fixture',records/'cpu-oracle')
 S=NC.module(packet);seal=S.read(records/'cpu-oracle/oracle-seal.json');seal.update(sources=S.sources(),cpu_runtime={'python':'3.12','torch':'2.9.0+cpu','platform':'linux','machine':'x86_64'},qualified_runtime=True,pid=34000,pgid=34000,parent_pid=32000,process_token='3'*32);S.write(records/'cpu-oracle/oracle-seal.json',seal)
 # This is a synthetic current-generation source seal over retained CPU values.
 # The immutable original corpus/seal is unchanged and is not current device evidence.
 # Source controls and wheel proofs are synthetic, declared only in this isolated driver.
 fixtures=load(ROOT/'experiments/2026-10-04-bitsandbytes-tpu/cloud/tests/test_transfer_cloud.py','native_controls_legacy_fixture')
 write(records/'source-controls.json',fixtures.controls_fixture());admission=NC.read(packet/'source-admission.json')
 write(records/'installed-source.json',{'status':'POST_PATCH_PYTHON_SOURCE_PASS','patch_manifest_sha256':B.TRANSFER_PATCH_MANIFEST_SHA,'source_admission_sha256':manifest['source_admission_sha256'],'installed_python_files':{n:admission[n]['files']for n in ('bitsandbytes','bitsandbytes_tpu')}})
 built={'status':'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS','patch_manifest_sha256':B.TRANSFER_PATCH_MANIFEST_SHA,'python_files':admission['bitsandbytes']['files'],'wheel_name':'SYNTHETIC_NO_WHEEL.whl','wheel_sha256':'f'*64};write(records/'built-source.json',built)
 argv=['python','-B','/content/bnb-tpu-first/payload/native/prepare_oracle.py','--upstream-source','/content/bnb-tpu-first/upstream/bitsandbytes','--patch-manifest','/content/bnb-tpu-first/payload/patches/params4bit-xla-v1.json','--output','/content/bnb-tpu-first/records/cpu-oracle','--process-token','3'*32]
 cleanup={'status':'CLEANUP_VERIFIED','leader_reaped':True,'group_absence':'OBSERVED_NO_SUCH_GROUP','errors':[],'exit_status':0}
 write(records/'steps/11-cpu-oracle/ownership.json',{'pid':34000,'pgid':34000,'owner_pid':32000,'argv':argv,'cleanup':cleanup})
 write(records/'steps/11-cpu-oracle/result.json',{'status':'PASS','exit_code':0,'cleanup':{'errors':[]}})
 F=load(HERE/'native_fixture.py','native_shape_fixture');data=F.make_fixture(records/'native',records/'cpu-oracle',admission=manifest['source_admission_sha256'])
 for p in (records/'native/device/raw').glob('*.boundary-record.json'):
  raw=S.read(p);raw['scope']='ACTUAL_NATIVE_BOUNDARY';S.write(p,raw)
 F.reseal(records/'native');write(records/'steps/12-native-parent/ownership.json',data['outer'])
 receipt={'status':'NATIVE_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED','experiment':NC.MODE,'precision':'highest','packet_sha256':NC.sha(packet/'payload.zip'),'manifest_sha256':NC.sha(packet/'manifest.json'),'source_admission_sha256':manifest['source_admission_sha256'],'runtime_lock_sha256':manifest['runtime_lock_sha256'],'runtime_status':'PASS_TPU_RUNTIME_PROBE_ONLY','cpu_status':'PASS','tpu_status':'NATIVE_RECORDS_TERMINAL','built_source_status':'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS','installed_source_status':'POST_PATCH_PYTHON_SOURCE_PASS','source_controls_status':'PASS_QUALIFIED_LINUX_SOURCE_CONTROLS','oracle_sha256':S.sha(records/'cpu-oracle/oracle-seal.json'),'built_wheels':[{'name':built['wheel_name'],'sha256':built['wheel_sha256']}],'steps':[{'scope':'SYNTHETIC_NO_EXECUTION','cleanup':{'errors':[]}}],**{k:manifest[k]for k in remote.TRANSFER_BINDINGS},'native_manifest_sha256':NC.MANIFEST_SHA,'native_source_variant':NC.VARIANT,'native_expected_sha256':S.sha(records/'native/expected.json'),'native_parent_sha256':S.sha(records/'native/parent.json'),'native_outer_owner_sha256':S.sha(records/'steps/12-native-parent/ownership.json')}
 receipt.update(NC.cpu_archive(records));transport.split(records/'native-cpu-evidence.zip',base/'native-cpu-parts',expected_sha256=receipt['native_cpu_evidence_sha256'],expected_bytes=receipt['native_cpu_evidence_bytes']);write(records/'receipt.json',receipt)
 return records,receipt,F

def chain(packet,manifest,fault):
 base=OUT/('correct' if fault is None else fault);records,receipt,F=make_records(packet,manifest,base)
 S=NC.module(packet)
 if fault in {'cpu-pid','cpu-open-group','cpu-runtime','cpu-source'}:
  if fault=='cpu-pid':p=records/'steps/11-cpu-oracle/ownership.json';r=NC.read(p);r['pid']=r['pgid']=99999
  elif fault=='cpu-open-group':p=records/'steps/11-cpu-oracle/ownership.json';r=NC.read(p);r['cleanup']['group_absence']='UNVERIFIED'
  else:p=records/'cpu-oracle/oracle-seal.json';r=NC.read(p);r['cpu_runtime']['machine']='arm64' if fault=='cpu-runtime' else r['cpu_runtime']['machine'];r['source_post']='0'*64 if fault=='cpu-source' else r['source_post']
  write(p,r);receipt['oracle_sha256']=NC.sha(records/'cpu-oracle/oracle-seal.json');receipt.update(NC.cpu_archive(records));shutil.rmtree(base/'native-cpu-parts');transport.split(records/'native-cpu-evidence.zip',base/'native-cpu-parts',expected_sha256=receipt['native_cpu_evidence_sha256'],expected_bytes=receipt['native_cpu_evidence_bytes'])
 if fault in {'native-number','native-open-group','native-config','native-proto','native-source'}:
  if fault=='native-open-group':p=records/'steps/12-native-parent/ownership.json';r=NC.read(p);r['cleanup']['group_absence']='UNVERIFIED';write(p,r)
  elif fault=='native-source':p=records/'native/parent.json';r=NC.read(p);r['source_post']='0'*64;write(p,r)
  elif fault=='native-proto':p=records/'native/device/raw/direct-fp32.printer.hlo.pb';p.write_bytes(b'wrongproto')
  elif fault=='native-config':p=records/'native/device/raw/direct-fp32.native-config.json';p.write_text('{}')
  else:
   for p in [records/'native/device/raw/direct-fp32.json',records/'native/device/raw/direct-fp32.boundary-record.json']:
    r=NC.read(p);r['output']['values'][0]+=1000;write(p,r)
  F.reseal(records/'native');receipt.update(native_parent_sha256=NC.sha(records/'native/parent.json'),native_outer_owner_sha256=NC.sha(records/'steps/12-native-parent/ownership.json'))
 out=base/'host';identity=base/'cli-identity.json';write(identity,{})
 gate={'status':'ACTUAL_DISPATCH_AUTHORIZED','packet_sha256':NC.sha(packet/'payload.zip'),'driver_sha256':NC.sha(CLOUD/'owner.py'),'output':str(out.resolve()),'plugin_source_manifest_sha256':manifest['plugin_source_manifest_sha256'],'runtime_lock_sha256':manifest['runtime_lock_sha256'],'budget':manifest['budget'],'one_allocation_only':True,'provider_or_solver':'FORBIDDEN','cli_identity_sha256':NC.sha(identity),'experiment':NC.MODE,'precision':'highest',**{k:manifest[k]for k in remote.TRANSFER_BINDINGS},'native_manifest_sha256':NC.MANIFEST_SHA,'m2_dependency_sha256':manifest['m2_dependency_sha256'],'native_source_variant':NC.VARIANT,'native_scope':manifest['native_scope']};write(base/'root-gate.json',gate)
 calls=[];session=[]
 def api(label,argv,timeout):
  calls.append(label)
  if label=='03-one-allocation':session.append(argv[argv.index('-s')+1]);receipt['allocation_epoch']=NC.read(out/'owner.json')['allocation_epoch'];write(records/'receipt.json',receipt)
  if label in {'04-sessions-after','90-before-stop'}:return {'status':'PASS'},'['+session[0]+']\nHardware: V6E1'
  if label=='13-tpu':
   assert "'native'"in Path(argv[-1]).read_text();assert NC.read(out/'launch.json')['oracle_sha256']==receipt['oracle_sha256'];receipt['native_oracle_gate_sha256']=NC.sha(out/'launch.json');write(records/'receipt.json',receipt)
  if label=='21-export':write(records/'receipt.json',{**receipt,**remote.package(records)})
  if label=='24-export-parts':transport.split(records/'evidence.zip',base/'result-parts',expected_sha256=NC.sha(records/'evidence.zip'),expected_bytes=(records/'evidence.zip').stat().st_size)
  if argv[0]=='download':
   destination=Path(argv[-1])
   if label in {'09-install-receipt','10-cpu-receipt','11-native-cpu-receipt'}:write(destination,{**receipt,'status':'INSTALLED_NOT_QUALIFIED'if label=='09-install-receipt' else 'CPU_ORACLE_READY_TPU_NOT_RUN'})
   else:
    rel=argv[-2].removeprefix('content/bnb-tpu-first/');source=records/rel.removeprefix('records/')if rel.startswith('records/')else base/rel;shutil.copyfile(source,destination)
    if fault=='cpu-part-bytes' and label.startswith('11c-native-part-'):destination.write_bytes(destination.read_bytes()+b'bad')
    if fault=='archive-part-bytes' and label.startswith('26-part-'):destination.write_bytes(destination.read_bytes()+b'bad')
  if fault=='stop-failure' and label=='91-stop-exact':return {'status':'CLI_FAILED'},'SYNTHETIC_STOP_REFUSED'
  return {'status':'PASS'},'Active assignments: 0\nUsage rate: 0.00/hr'if argv==['usage']else'No active sessions found on server.'
 with patch.object(owner,'verify_cli_identity',lambda *a:None):result=owner.drive(packet,out,NC.sha(packet/'payload.zip'),base/'root-gate.json',NC.sha(base/'root-gate.json'),Path('SYNTHETIC_PYTHON'),identity,api=api,simulated=True)
 assert result['mode']=='SIMULATED_NO_CLOUD';assert calls.count('03-one-allocation')==1;assert calls[-4:]==['90-before-stop','91-stop-exact','92-after-stop','93-usage-after']
 success=result['status']=='PASS_BOUNDED_NATIVE_DIAGNOSTIC'
 if success!=(fault is None):raise AssertionError((fault,result.get('status'),result.get('original_error'),result.get('local_readback_error')))
 if fault and (fault.startswith('cpu-')):assert '13-tpu'not in calls
 return {'case':fault or'correct-shaped-record','status':'PASS','owner_status':result['status'],'device_phase_requests':calls.count('13-tpu'),'scope':'SIMULATED_NO_CLOUD_NO_DEVICE_SCIENCE'}

def unpack_cpu_fixture(target):
 target.mkdir();inventory=NC.read(FIXTURES/'cpu-inventory.json');archive=FIXTURES/'cpu-unqualified.tar.gz'
 assert archive.stat().st_size==inventory['archive_bytes'] and NC.sha(archive)==inventory['archive_sha256']
 with tarfile.open(archive,'r:gz')as tar:
  assert len(tar.getmembers())==len(inventory['members']) and {m.name for m in tar.getmembers()}==set(inventory['members'])
  for member in tar.getmembers():
   assert member.isfile() and Path(member.name).name==member.name;raw=tar.extractfile(member).read();rec=inventory['members'][member.name];assert len(raw)==rec['bytes'] and hashlib.sha256(raw).hexdigest()==rec['sha256'];(target/member.name).write_bytes(raw)

def compact_closed_case(root):
 # Cases are isolated test output. Preserve their exact bytes before removing
 # expanded copies, so the fixture driver uses bounded disk space.
 import gzip,io
 inventory={p.relative_to(root).as_posix():{'sha256':NC.sha(p),'bytes':p.stat().st_size}for p in sorted(root.rglob('*'))if p.is_file()};archive=root.with_suffix('.tar.gz')
 with archive.open('wb')as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0)as gz,tarfile.open(fileobj=gz,mode='w')as tar:
  for name,r in inventory.items():
   body=(root/name).read_bytes();assert len(body)==r['bytes'] and hashlib.sha256(body).hexdigest()==r['sha256'];info=tarfile.TarInfo(name);info.size=len(body);info.mtime=0;info.mode=0o644;tar.addfile(info,io.BytesIO(body))
 with tarfile.open(archive,'r:gz')as tar:
  assert len(tar.getmembers())==len(inventory)
  for member in tar.getmembers():
   raw=tar.extractfile(member).read();r=inventory[member.name];assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
 NC.write(root.with_suffix('.inventory.json'),{'scope':'CLOSED_SYNTHETIC_TEST_OUTPUT','archive_sha256':NC.sha(archive),'archive_bytes':archive.stat().st_size,'members':inventory});shutil.rmtree(root)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--upstream-archive',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();ARCHIVE=args.upstream_archive.resolve()
 packet,m,r=prepare();rows=[]
 for fault in [None,'cpu-pid','cpu-open-group','cpu-runtime','cpu-source','cpu-part-bytes','native-number','native-open-group','native-config','native-proto','native-source','archive-part-bytes','stop-failure']:
  rows.append(chain(packet,m,fault));print(rows[-1],flush=True)
  compact_closed_case(OUT/(fault or 'correct'))
 write(OUT/'control-results.json',{'status':'PASS','controls':rows,'provider_calls':0,'actual_tpu':'NOT_RUN','qualified_cpu':'SYNTHETIC_RECORDS_ONLY','m6':'NOT_QUALIFIED'})
