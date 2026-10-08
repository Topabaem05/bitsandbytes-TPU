"""Current-source CPU gate fixtures. Runtime qualification fields are synthetic, not measurements."""
import argparse,copy,hashlib,json,shutil,sys,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent));import compiler_cloud_contract as CC,remote
from compiler_source_fixture import controls_fixture

def run(packet,output):
 packet=Path(packet).resolve();out=Path(output).resolve();out.mkdir(exist_ok=False);m=CC.read(packet/'manifest.json');oracle=out/'cpu-oracle';oracle.mkdir();inventory=CC.read(HERE/'fixtures/native/compiler-mosaic-cpu-inventory.json');archive=HERE/'fixtures/native/compiler-mosaic-cpu-unqualified.tar.gz'
 assert archive.stat().st_size==inventory['archive_bytes'] and CC.sha(archive)==inventory['archive_sha256']
 with tarfile.open(archive,'r:gz') as t:
  assert len(t.getmembers())==len(inventory['members'])
  for member in t.getmembers():
   assert member.isfile() and Path(member.name).name==member.name;b=t.extractfile(member).read();r=inventory['members'][member.name];assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'];(oracle/member.name).write_bytes(b)
 S=CC.module(packet);original=CC.read(oracle/'oracle-seal.json');seal=copy.deepcopy(original);seal.update(sources=S.sources(),cpu_runtime={'python':'3.12','torch':'2.9.0+cpu','platform':'linux','machine':'x86_64'},qualified_runtime=True,pid=34000,pgid=34000,parent_pid=32000,process_token='3'*32);CC.write(oracle/'oracle-seal.json',seal)
 admission=CC.read(packet/'source-admission.json');CC.write(out/'source-controls.json',controls_fixture());CC.write(out/'installed-source.json',{'status':'POST_PATCH_PYTHON_SOURCE_PASS','patch_manifest_sha256':m['patch_manifest_sha256'],'source_admission_sha256':m['source_admission_sha256'],'installed_python_files':{n:admission[n]['files']for n in ('bitsandbytes','bitsandbytes_tpu')}})
 built={'status':'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS','patch_manifest_sha256':m['patch_manifest_sha256'],'python_files':admission['bitsandbytes']['files'],'wheel_name':'SYNTHETIC_NO_WHEEL.whl','wheel_sha256':'f'*64};CC.write(out/'built-source.json',built)
 argv=['python','-B','/content/bnb-tpu-first/payload/native/prepare_oracle.py','--upstream-source','/content/bnb-tpu-first/upstream/bitsandbytes','--patch-manifest','/content/bnb-tpu-first/payload/patches/params4bit-xla-v1.json','--output','/content/bnb-tpu-first/records/cpu-oracle','--process-token','3'*32]
 cleanup={'status':'CLEANUP_VERIFIED','leader_reaped':True,'group_absence':'OBSERVED_NO_SUCH_GROUP','errors':[],'exit_status':0}
 (out/'steps/11-cpu-oracle').mkdir(parents=True);ownership={'pid':34000,'pgid':34000,'owner_pid':32000,'argv':argv,'cleanup':cleanup};CC.write(out/'steps/11-cpu-oracle/ownership.json',ownership);CC.write(out/'steps/11-cpu-oracle/result.json',{'status':'PASS','exit_code':0,'cleanup':{'errors':[]}})
 receipt={**CC.NATIVE_DEPENDENCY_BINDINGS,'compiler_manifest_sha256':CC.MANIFEST_SHA,'compiler_generation':CC.GENERATION,'runtime_status':'PASS_TPU_RUNTIME_PROBE_ONLY','cpu_status':'PASS','built_source_status':'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS','installed_source_status':'POST_PATCH_PYTHON_SOURCE_PASS','source_controls_status':'PASS_QUALIFIED_LINUX_SOURCE_CONTROLS','oracle_sha256':CC.sha(oracle/'oracle-seal.json'),'built_wheels':[{'name':built['wheel_name'],'sha256':built['wheel_sha256']}],**CC.cpu_archive(out)}
 rows=[]
 gate=CC.cpu_gate(packet,out,receipt);assert gate['kind']=='ROOT_COMPILER_CPU_ORACLE_GATE' and gate['compiler_manifest_sha256']==CC.MANIFEST_SHA and gate['compiler_policy_sha256']==CC.POLICY_SHA;rows.append({'case':'current-generation-cpu-gate-synthetic-correct','status':'PASS'})
 for key in CC.NATIVE_DEPENDENCY_BINDINGS:
  bad_receipt=copy.deepcopy(receipt);bad_receipt[key]=None
  try:CC.cpu_gate(packet,out,bad_receipt)
  except ValueError as e:assert str(e)=='COMPILER_ACCEPTED_NATIVE_BINDING';rows.append({'case':'cpu-receipt-missing-'+key,'status':'PASS','rejected':str(e)})
  else:raise AssertionError('FALSE_ACCEPT:CPU_DEPENDENCY:'+key)
 bad_receipt=copy.deepcopy(receipt);bad_receipt['compiler_manifest_sha256']='9abfc68d38fd742418e6ec529ee42a1c2b9a0b328dd65431f973b4384a11634d'
 try:CC.cpu_gate(packet,out,bad_receipt)
 except ValueError as e:assert str(e)=='COMPILER_CPU_RECEIPT_GENERATION';rows.append({'case':'cpu-receipt-old-pending-manifest','status':'PASS','rejected':str(e)})
 else:raise AssertionError('FALSE_ACCEPT:OLD_CPU_RECEIPT')
 def wrong(name,edit,reason):
  value=copy.deepcopy(seal);edit(value);CC.write(oracle/'oracle-seal.json',value);r={**receipt,'oracle_sha256':CC.sha(oracle/'oracle-seal.json')}
  try:CC.cpu_gate(packet,out,r)
  except ValueError as e:assert str(e)==reason,(name,str(e),reason);rows.append({'case':name,'status':'PASS','rejected':str(e)})
  else:raise AssertionError('FALSE_ACCEPT:'+name)
 previous=CC.read(HERE.parents[1]/'previous-compiler-manifest.json');previous_by_name={Path(n).name:r['sha256']for n,r in previous['sources'].items()};old_sources={n:previous_by_name[n]for n in S.sources()}
 wrong('old-compiler-source-seal',lambda r:r.__setitem__('sources',old_sources),'ORACLE_CANDIDATE_SOURCE')
 wrong('old-native-v1-source-seal',lambda r:r.__setitem__('sources',original['sources']),'ORACLE_CANDIDATE_SOURCE')
 wrong('unqualified-local-runtime',lambda r:r.__setitem__('qualified_runtime',False),'ORACLE_QUALIFIED_CPU_REQUIRED')
 wrong('wrong-machine',lambda r:r['cpu_runtime'].__setitem__('machine','arm64'),'ORACLE_QUALIFIED_CPU_REQUIRED')
 wrong('source-pre-altered',lambda r:r.__setitem__('source_pre','0'*64),'ORACLE_UPSTREAM_FULL_SOURCE_MAP')
 wrong('source-post-altered',lambda r:r.__setitem__('source_post','0'*64),'ORACLE_UPSTREAM_FULL_SOURCE_MAP')
 wrong('source-python-map-altered',lambda r:r.__setitem__('upstream_python_map_sha256','0'*64),'ORACLE_UPSTREAM_FULL_SOURCE_MAP')
 CC.write(oracle/'oracle-seal.json',seal);bad=copy.deepcopy(ownership);bad['cleanup']['group_absence']='UNVERIFIED';CC.write(out/'steps/11-cpu-oracle/ownership.json',bad)
 try:CC.cpu_gate(packet,out,receipt)
 except ValueError as e:assert str(e)=='NATIVE_CPU_GROUP_CLOSED';rows.append({'case':'unclosed-cpu-group','status':'PASS','rejected':str(e)})
 else:raise AssertionError('FALSE_ACCEPT:CPU_GROUP')
 CC.write(out/'results.json',{'status':'PASS','controls':rows,'scope':'SYNTHETIC_GATE_FIXTURE_ONLY','local_torch':__import__('torch').__version__,'qualified_CPU':'NOT_RUN','provider_calls':0});print(json.dumps({'status':'PASS','controls':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--packet',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.packet,a.output)
