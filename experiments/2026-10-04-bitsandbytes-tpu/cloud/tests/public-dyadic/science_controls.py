"""Portable local CPU controls. Synthetic wheels are not installed or qualified."""
import argparse,copy,os,secrets,struct,sys,tarfile,time,zipfile
from pathlib import Path
from types import SimpleNamespace
p=argparse.ArgumentParser()
for n in ('public-source','base-packet','ownership-source','output'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False);public=a.public_source.resolve();base=a.base_packet.resolve();sys.path.insert(0,str(public));sys.path.insert(0,str(a.ownership_source.resolve()))
import contract as C,oracle_math as M,verify_public_oracle as O,reference_math as R,dyadic as D
from cleanup_lifecycle import Ownership
results=[]
def good(name,**details):results.append({'name':name,'status':'PASS',**details})
def reject(name,fn,match=None):
 try:fn()
 except ValueError as e:
  if match is not None:assert match in str(e),(name,str(e))
  good(name,rejected=str(e));return
 raise AssertionError('incorrect fixture accepted:'+name)
up=out/'upstream';up.mkdir()
with tarfile.open(base/'patched-upstream.tar')as t:
 for member in t.getmembers():
  assert(member.isfile()or member.isdir())and not Path(member.name).is_absolute()and '..'not in Path(member.name).parts
  if member.isdir():continue
  target=up/member.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(t.extractfile(member).read())
up=up/'bitsandbytes';plugin=public/'plugin/src/bitsandbytes_tpu_pallas';ad=C.read(public/'package-admission.json')
def wheel(kind,namespace,root,mapping,entry=None):
 path=out/(kind+'-synthetic-source-only.whl')
 with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED)as z:
  for n in sorted(mapping):z.writestr(namespace+'/'+n,(root/n).read_bytes())
  if entry:z.writestr(namespace+'-0.0.0.dist-info/entry_points.txt',entry+'\n')
 return str(path)
args=SimpleNamespace(admission=str(public/'package-admission.json'),protocol=str(public/'public-protocol.json'),package_root=str(plugin),upstream_source=str(up),repo_root=str(base/'public-baseline'),plugin_wheel=wheel('plugin',ad['import_name'],plugin,{**ad['plugin_python_files'],'source_manifest.json':ad['plugin_manifest_sha256']},'[bitsandbytes.backends]\ntpu_pallas_forward = bitsandbytes_tpu_pallas:register'),upstream_wheel=wheel('upstream','bitsandbytes',up,ad['upstream_python_files']),runtime_wheels=None,oracle=str(out/'oracle'),cpu_ownership=None,cpu_ownership_sha256=None)
owner=Ownership(out/'cpu-owner');(out/'cpu-owner').mkdir();deadline=time.time()+180;argv=[sys.executable,'-B',str(public/'prepare_public_oracle.py')]
for n in ('admission','protocol','package_root','upstream_source','repo_root','plugin_wheel','upstream_wheel'):argv+=['--'+n.replace('_','-'),getattr(args,n)]
argv+=['--output',args.oracle,'--process-token',secrets.token_hex(16),'--deadline-epoch',str(deadline),'--unqualified-local']
with owner.guard(180):
 with(out/'cpu.stdout').open('wb')as log,(out/'cpu.stderr').open('wb')as err:
  child=owner.launch(argv,record=out/'cpu-launch.json',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=err);code=child.wait(timeout=175)
 owner.stop(child)
assert code==0 and not owner.summary()['errors'],(code,(out/'cpu.stderr').read_text());args.oracle_sha256=C.sha(Path(args.oracle)/'oracle-seal.json')
try:os.killpg(child.pid,0);raise AssertionError('owned CPU group remains')
except ProcessLookupError:pass
good('fresh_unqualified_cpu_generator_owned_and_closed',pid=child.pid,pgid=child.pid)
report,rows=O.verify(args,qualified=False);assert report['reference_semantics']==R.SEMANTICS and len(rows)==15;good('independent_full15_primary_and_supplement_replay')
sealpath=Path(args.oracle)/'oracle-seal.json';s=C.read(sealpath);assert len(s['artifacts'])==30 and s['kind']=='PUBLIC_PALLAS_CPU_ORACLE_V2';good('complete30_artifact_seal')
assert C.sha(public/'oracle_math.py')==C.sha(base/'public/oracle_math.py');good('original_primary_generator_byte_identity')
original=C.read(base/'public/public-protocol.json');current=C.read(public/'public-protocol.json');assert all(original[n]==current[n]for n in('cases','recipes','tolerances','runtime','first_error_policy'));good('unchanged_cases_inputs_runtime_tolerances_firsterror')
assert C.inventory(plugin)==C.inventory(base/'public-plugin/src/bitsandbytes_tpu_pallas');good('unchanged_public_plugin_science_source')
assert all(s[n]==C.ADMISSION_SHA for n in('admission_sha256','source_pre','source_post'));good('source_pre_post_and_new_admission')
oldkind=s['kind'];assert C.read(base/'public/package-admission.json')['public_protocol_sha256']!=C.PROTOCOL_SHA;good('new_explicit_scientific_generation')
# Mutations touch only this generated local fixture and restore its exact bytes.
def mutation(name,edit,*,row=None,match=None):
 target=sealpath if row is None else Path(args.oracle)/row;before={q:q.read_bytes()for q in {target,sealpath}};oldsha=args.oracle_sha256
 try:
  obj=C.read(target);edit(obj);C.write(target,obj)
  if row is not None:
   seal=C.read(sealpath);seal['artifacts']=C.inventory(args.oracle,('oracle-seal.json',));C.write(sealpath,seal)
  args.oracle_sha256=C.sha(sealpath);reject(name,lambda:O.verify(args,qualified=False),match)
 finally:
  for q,b in before.items():q.write_bytes(b)
  args.oracle_sha256=oldsha
mutation('old_kind_rejected_after_outer_reseal',lambda x:x.update(kind='PUBLIC_PALLAS_CPU_ORACLE'),match='CPU_TERMINAL')
mutation('missing_semantics_rejected',lambda x:x.pop('reference_semantics'),match='CPU_EXACT_REFERENCE_SEMANTICS')
mutation('wrong_reference_rounding_semantics_rejected',lambda x:x['reference_semantics'].update(supplement='FLOAT64_BLAS'),match='CPU_EXACT_REFERENCE_SEMANTICS')
mutation('old_protocol_link_rejected',lambda x:x.update(protocol_sha256=original and C.sha(base/'public/public-protocol.json')),match='CPU_SOURCE_PROTOCOL')
mutation('wrong_scientific_source_map_rejected',lambda x:x['scientific_sources'].pop('dyadic.py'),match='CPU_FULL_SCIENTIFIC_SOURCE')
mutation('wrong_post_source_rejected',lambda x:x.update(source_post='0'*64),match='CPU_SOURCE_PROTOCOL')
mutation('wrong_upstream_source_inventory_rejected',lambda x:x.update(upstream_python_map_sha256='0'*64),match='CPU_FULL_SCIENTIFIC_SOURCE')
mutation('wrong_case_matrix_rejected',lambda x:x['case_ids'].reverse(),match='CPU_CASE_MATRIX')
mutation('wrong_deadline_rejected',lambda x:x.update(finished_epoch=x['deadline_epoch']+1),match='CPU_FINITE_TIME')
mutation('false_import_guard_rejected',lambda x:x.update(no_package_device_import=False),match='CPU_IMPORT_THREADS')
mutation('wrong_thread_count_rejected',lambda x:x.update(thread_count=2),match='CPU_IMPORT_THREADS')
mutation('wrong_built_wheel_rejected',lambda x:x['wheel_records']['plugin'].update(sha256='0'*64),match='CPU_BUILT_WHEEL_BINDING')
reject('unqualified_cpu_never_accepted_as_qualified',lambda:O.verify(args,qualified=True),'CPU_QUALIFIED_RUNTIME')
mutation('material_primary_value_rejected_under_old_gate',lambda x:x['outputs']['y']['values'].__setitem__(0,struct.unpack('<f',struct.pack('<f',x['outputs']['y']['values'][0]+.01))[0]),row='rows/direct-float32.json',match='ORIGINAL_DEVICE_GATE')
mutation('wrong_primary_input_digest_rejected',lambda x:x.update(inputs_sha256='0'*64),row='rows/direct-float32.json',match='PRIMARY_INPUT_IDENTITY')
mutation('wrong_primary_shape_rejected',lambda x:x['outputs']['y'].update(shape=[1,32768]),row='rows/direct-float32.json',match='PRIMARY_METADATA')
mutation('supplement_one_ulp_rejected_after_complete_reseal',lambda x:x['outputs']['y']['words'].__setitem__(0,x['outputs']['y']['words'][0]^1),row='supplements/direct-float32.json',match='INDEPENDENT_EXACT_SUPPLEMENT')
mutation('supplement_altered_proof_rejected',lambda x:x['precision_proof']['y'].update(bound_bits=1),row='supplements/direct-float32.json',match='INDEPENDENT_EXACT_SUPPLEMENT')
mutation('supplement_wrong_input_rejected',lambda x:x.update(inputs_sha256='0'*64),row='supplements/direct-float32.json',match='INDEPENDENT_EXACT_SUPPLEMENT')
mutation('supplement_wrong_nibble_boundary_rejected',lambda x:x.update(decoded_dense_fp32_words_sha256='0'*64),row='supplements/direct-float32.json',match='INDEPENDENT_EXACT_SUPPLEMENT')
mutation('supplement_wrong_bf16_rounding_rejected',lambda x:x['outputs']['y']['words'].__setitem__(0,x['outputs']['y']['words'][0]^1),row='supplements/direct-bfloat16.json',match='INDEPENDENT_EXACT_SUPPLEMENT')
mutation('non_bf16_primary_value_rejected',lambda x:x['outputs']['y']['values'].__setitem__(0,1.0000001192092896),row='rows/direct-bfloat16.json',match='EXACT_BF16_VALUE_REQUIRED')
oldseal=sealpath.read_bytes();supp=Path(args.oracle)/'supplements/direct-float32.json';old=supp.read_bytes();oldsha=args.oracle_sha256
try:
 supp.unlink();reject('missing_supplement_rejected',lambda:O.verify(args,qualified=False),'CPU_COMPLETE_ARTIFACT_INVENTORY')
finally:supp.write_bytes(old);sealpath.write_bytes(oldseal);args.oracle_sha256=oldsha
# A one-ULP platform difference is accepted through the unchanged mathematical gate.
path=Path(args.oracle)/'rows/direct-float32.json';before=path.read_bytes();oldseal=sealpath.read_bytes();oldsha=args.oracle_sha256
try:
 obj=C.read(path);w=D.float_word(obj['outputs']['y']['values'][0]);obj['outputs']['y']['values'][0]=D.values({'shape':[1],'dtype':'float32','words':[w^1]})[0];C.write(path,obj);seal=C.read(sealpath);seal['artifacts']=C.inventory(args.oracle,('oracle-seal.json',));C.write(sealpath,seal);args.oracle_sha256=C.sha(sealpath);O.verify(args,qualified=False);good('alternative_one_ulp_primary_allowed_by_original_gate')
finally:path.write_bytes(before);sealpath.write_bytes(oldseal);args.oracle_sha256=oldsha
assert C.sha(sealpath)==args.oracle_sha256;good('generated_fixture_restored_exactly')
C.write(out/'results.json',{'status':'PASS','count':len(results),'controls':results,'scope':'UNQUALIFIED_LOCAL_CPU_AND_ISOLATED_RECORDS','qualified_linux_primary_and_supplement':'NOT_RUN','actual_device':'NOT_RUN','provider_calls':0,'owned_cpu_cleanup':owner.summary()})
print({'status':'PASS','count':len(results)})
