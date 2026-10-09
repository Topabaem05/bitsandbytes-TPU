"""Seal this private preparation. This tool does not write canonical files or dispatch."""
import argparse,datetime,hashlib,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=next(p for p in HERE.parents if(p/'packages/bitsandbytes-tpu/pyproject.toml').is_file());OLD=ROOT/'.work/r6-compiler-bf16-accepted-fix';SCI='experiments/2026-10-04-bitsandbytes-tpu';sys.path.insert(0,str(HERE/'cloud'));import compiler_cloud_contract as C

def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n')
def seal(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}
p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args();assert not(HERE/'review.json').exists(),'FROZEN_GENERATION_IMMUTABLE'
names=set(read(OLD/'source-map.json'))|{'compiler-native/compiler_flags.py','compiler-native/compiler_flag_probe.py','cloud/compiler_flag_preflight.py','cloud/build_public_packet.py','cloud/public_contract.py','cloud/public_gate.py','cloud/public_transport.py','flag_controls.py','public_branch_controls.py','prepare_flag_pins.py','compare_actual_flags.py','compose_cloud_base.py','compose_cloud_resume.py','fetch_wheel.py','inspect_wheel.py','fetch_sources.py','fetch_torch_source.py'}
names|={q.relative_to(HERE).as_posix()for q in (HERE/'evidence').rglob('*')if q.is_file()and q.suffix in('.py','.json','.cc','.bzl','.patch')}
source={n:seal(HERE/n)for n in sorted(names)};write(HERE/'source-map.json',source)
old=read(OLD/'adoption-map.json');prior={r['source']:r for r in old['files']+old['exact_prerequisites']};rows=[];prereqs=[]
for n,r in source.items():
 if n.startswith(('cloud/','compiler-native/'))and '/tests/'not in n:
  target=SCI+'/'+n;role='COMPILER_FLAG_MODE_PRODUCTION'
 elif n in prior:target=prior[n]['target'];role=prior[n]['role']
 else:target=SCI+'/cloud/tests/compiler-accepted/'+n;role='PORTABLE_CONTROL_OR_RESEARCH_RECORD'
 row={'source':n,'target':target,'role':role,**r};q=ROOT/target
 if q.is_file()and not q.is_symlink()and seal(q)==r:row['role']='EXACT_PREREQUISITE_NOT_IMPLICIT_ADOPTION';prereqs.append(row)
 else:rows.append(row)
adoption={'status':'PROPOSED_LIBTPU021_COMPILER_FLAG_CORRECTION_REQUIRES_ROOT_REVIEW','accepted_cloud_revision':'5ada90eeef3f8a9ec3913a2a1a83d877be709f65','generation':C.GENERATION,'compiler_manifest_sha256':C.MANIFEST_SHA,'compiler_policy_sha256':C.POLICY_SHA,**C.NATIVE_DEPENDENCY_BINDINGS,'native_dependency_acceptance':C.NATIVE_DEPENDENCY_BINDINGS['actual_native_acceptance'],'actual_compiler_acceptance':None,'adoption_revision':None,'files':rows,'exact_prerequisites':prereqs,'map_metadata_targets':{'source-map.json':SCI+'/results/compiler-libtpu021-preparation/source-map.json','adoption-map.json':SCI+'/results/compiler-libtpu021-preparation/adoption-map.json'},'portable_control_layout':'Materialize the full relative source tree from exact canonical destinations with materialize_controls.py; no private fixture is required.'};write(HERE/'adoption-map.json',adoption)
prior_sources=read(OLD/'source-map.json');write(HERE/'flag-composition.json',{'generation':C.GENERATION,'compiler_manifest_sha256':C.MANIFEST_SHA,'changed_from_frozen':{n:{'before':prior_sources.get(n),'after':r}for n,r in source.items()if prior_sources.get(n)!=r},'adopted_public_cloud_revision':adoption['accepted_cloud_revision'],'actual_libtpu_parse':'NOT_RUN','actual_compiler_TPU':'NOT_RUN'})
if not a.final:print(json.dumps({'status':'MAPS_READY','source_files':len(source),'changed_adoption_files':len(rows),'exact_prerequisites':len(prereqs)}));raise SystemExit
checks={n:read(HERE/n)for n in ('final-affected/results.json','final-accepted/results.json','final-plain/results.json','flag-controls-v3/results.json','final-public/results.json','final-portable-v4/results.json')};counts=[67,41,23,35,3,2];assert all(v['status']=='PASS'for v in checks.values());assert [v['count']for v in checks.values()]==counts;assert(HERE/'final-packet1/payload.zip').read_bytes()==(HERE/'final-packet2/payload.zip').read_bytes()
oldmanifest=read(OLD/'compiler-native/manifest.json');current=read(HERE/'compiler-native/manifest.json');changed=[n for n,r in oldmanifest['sources'].items()if current['sources'][n]!=r];assert set(changed)=={'compiler_contract.py','compiler_verify.py','coordinator.py','compiler-policy.json'}
for n,r in read(HERE/'native/manifest.json')['sources'].items():assert seal(HERE/'native'/n)==r
write(HERE/'unchanged-science-readback.json',{'status':'EXACT_BYTE_PROOF_PASS','ordinary_native_sources':27,'unchanged_prior_compiler_sources':30,'changed_prior_compiler_sources':changed,'added_compiler_sources':sorted(set(current['sources'])-set(oldmanifest['sources'])),'kernel_sha256':seal(HERE/'compiler-native/kernel.py')['sha256'],'runtime_lock_sha256':current['runtime_lock_sha256'],'five_cpu_oracle_recipe_and_native_case_body':'UNCHANGED','actual_new_native_cases':'NOT_RUN'})
groups={}
for q in HERE.rglob('*.json'):
 if not q.is_file()or q.is_symlink():continue
 try:v=read(q)
 except(ValueError,UnicodeError):continue
 if isinstance(v,dict)and v.get('ownership')=='Popen child in dedicated start_new_session group'and isinstance(v.get('pid'),int):
  closure=v.get('cleanup',{});assert closure.get('status')=='CLEANUP_VERIFIED'and closure.get('leader_reaped')is True and closure.get('group_absence')=='OBSERVED_NO_SUCH_GROUP'and closure.get('errors')==[],q
  try:os.killpg(v['pgid'],0)
  except ProcessLookupError:pass
  else:raise AssertionError('LIVE_GROUP:'+str(q))
  groups[v['pgid']]={'pid':v['pid'],'pgid':v['pgid'],'cleanup':closure}
write(HERE/'closure-review.json',{'status':'ALL_REGISTERED_ACTUAL_GROUPS_REAPED_ABSENT','count':len(groups),'groups':list(groups.values())})
dep=read(OLD/'dependency-map.json');external=dep['external_files']
for n,r in read(OLD/'source-map.json').items():external[str(OLD.relative_to(ROOT)/n)]=r
for n,r in read(OLD/'review.json')['seals'].items():external[str(OLD.relative_to(ROOT)/n)]=r
external[str(OLD.relative_to(ROOT)/'review.json')]=seal(OLD/'review.json')
for q in (ROOT/SCI/'public').rglob('*'):
 if q.is_file()and not q.is_symlink():external[q.relative_to(ROOT).as_posix()]=seal(q)
for q in (ROOT/'.work/root-compiler-bf16-colab-review/actual-failure-readback.json',ROOT/SCI/'results/compiler-bf16-colab.json'):
 if q.is_file():external[q.relative_to(ROOT).as_posix()]=seal(q)
for r in read(HERE/'evidence/actual-flag-comparison.json')['actual_sources']:
 q=Path(r['path']);expected_source={k:r[k]for k in ('bytes','sha256')};assert seal(q)==expected_source;external[q.relative_to(ROOT).as_posix()]=expected_source
write(HERE/'dependency-map.json',{'kind':'R6_LIBTPU021_FLAG_PRIVATE_DEPENDENCIES_V1','external_files':external,'adopted_cloud_revision':adoption['accepted_cloud_revision'],'native_dependency_bindings':C.NATIVE_DEPENDENCY_BINDINGS,'prior_frozen_candidate':'PRESERVED_UNCHANGED','actual_failed_compiler_run':'PRESERVED_UNCHANGED'})
# Seal every retained local artifact, including failed invocation/fixture records and official binary evidence.
seals={q.relative_to(HERE).as_posix():seal(q)for q in HERE.rglob('*')if q.is_file()and not q.is_symlink()and q.relative_to(HERE).as_posix()not in source and q.name!='review.json'}
review={'status':'PRIVATE_LIBTPU021_COMPILER_FLAG_PREPARATION_PASS','candidate_frozen':True,'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'generation':C.GENERATION,'compiler_manifest_sha256':C.MANIFEST_SHA,'compiler_policy_sha256':C.POLICY_SHA,**C.NATIVE_DEPENDENCY_BINDINGS,'native_dependency_acceptance':C.NATIVE_DEPENDENCY_BINDINGS['actual_native_acceptance'],'actual_compiler_acceptance':None,'adoption_revision':None,'actual_corrected_libtpu_parse':'NOT_RUN','actual_compiler_TPU':'NOT_RUN','fresh_qualified_linux_cpu_oracle':'REQUIRED_NOT_RUN_FOR_THIS_GENERATION','source_inventory_files':len(source),'source_inventory_bytes':sum(v['bytes']for v in source.values()),'current_distinct_controls':sum(counts),'controls':dict(zip(checks,counts)),'staged_replay_not_double_counted':True,'packet':read(HERE/'final-packet1.json'),'deterministic_packet_second_build':'BYTE_IDENTICAL','ordinary_native_packet':read(HERE/'final-plain-packet.json'),'public_packet':checks['final-public/results.json']['packet'],'actual_registered_groups_closed':len(groups),'preserved_new_failures':['flag-control-owner-v1','build-owner-v1','affected-owner-v1','affected-owner-v2','affected-owner-v3','evidence/adopted-public-cloud/compose-first-failure.json','evidence/adopted-public-cloud/compose-second-failure.json','evidence/freeze-first-failure.json','evidence/freeze-second-failure.json'],'failure_interpretation':'Invocation, isolated fixture adaptation, and metadata-sealing errors remain sealed. No actual provider retry or scientific acceptance occurred.','complete_libtpu_registry':'NOT_OBTAINED','libtpu_internal_source_revision':'UNKNOWN','provider_calls':0,'installation':'NOT_RUN','canonical_writes':'NONE','compiler_dispatch':'REQUIRES_EXACT_NEW_ROOT_ADOPTION_AND_DISPATCH_GATE','M6':'NOT_QUALIFIED','limits':{'file_bytes':16777216,'monitored_total_bytes':134217728,'entries':1024,'poll_seconds':.01,'minimum_free_bytes':268435456,'scientific_record_budget_bytes':8388608,'hard_aggregate_filesystem_quota':'UNAVAILABLE','polling_overshoot':'POSSIBLE_UNBOUNDED_BY_WRITER_RATE','physical_memory':'UNKNOWN','allocator_memory':'UNKNOWN','executable_memory':'UNKNOWN'},'seals':seals};write(HERE/'review.json',review);print(json.dumps({'status':review['status'],'source_files':len(source),'controls':sum(counts),'groups':len(groups),'review':seal(HERE/'review.json')}))
