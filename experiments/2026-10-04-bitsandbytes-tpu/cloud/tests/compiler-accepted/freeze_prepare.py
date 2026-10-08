"""Prepare maps and seal local evidence for this private repair. No canonical writes."""
import argparse,datetime,hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OLD=ROOT/'.work/r6-compiler-bf16-accepted'
def read(p):return json.loads(p.read_text())
def write(p,m):p.write_text(json.dumps(m,sort_keys=True,indent=2)+'\n')
def seal(p):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args()
import sys;sys.path.insert(0,str(HERE/'cloud'));import compiler_cloud_contract as C
names=set(read(OLD/'source-map.json'))|{'fix-basis.json','build_plain_native.py','fixtures/ordinary_native_parent.py','plain_native_controls.py','materialize_controls.py','portable_layout_controls.py','freeze_prepare.py'}
source={n:seal(HERE/n)for n in sorted(names)};write(HERE/'source-map.json',source)
adoption=read(OLD/'adoption-map.json');adoption.update(**C.NATIVE_DEPENDENCY_BINDINGS,compiler_manifest_sha256=C.MANIFEST_SHA,generation=C.GENERATION,status='PROPOSED_REPAIRED_ACCEPTED_NATIVE_COMPILER_MAP_REQUIRES_ROOT_REVIEW',portable_control_layout='Run materialize_controls.py with root-adopted maps. Copy full relative source tree from exact canonical destinations into fresh generated test output; never use private prepared fixtures.',map_metadata_targets={'source-map.json':'experiments/2026-10-04-bitsandbytes-tpu/results/compiler-accepted-preparation/source-map.json','adoption-map.json':'experiments/2026-10-04-bitsandbytes-tpu/results/compiler-accepted-preparation/adoption-map.json'})
oldrows={r['source']:r for r in adoption['files']+adoption['exact_prerequisites']};files=[];prereqs=[]
for n,r in source.items():
 row=dict(oldrows.get(n,{'source':n,'target':'experiments/2026-10-04-bitsandbytes-tpu/cloud/tests/compiler-accepted/'+n,'role':'PORTABLE_CONTROL_FIXTURE_OR_REVIEW_DOC'}));row.update(r)
 (prereqs if row['role']=='EXACT_PREREQUISITE_NOT_IMPLICIT_ADOPTION'else files).append(row)
adoption['files']=files;adoption['exact_prerequisites']=prereqs;write(HERE/'adoption-map.json',adoption)
dep=read(OLD/'dependency-map.json');dep['current_source_map']=seal(HERE/'source-map.json');dep['rejected_candidate_review']=seal(OLD/'review.json');dep['dependency_bindings']=C.NATIVE_DEPENDENCY_BINDINGS
# Preserve the full frozen rejected preparation identities, including its evidence seals.
for n,r in read(OLD/'source-map.json').items():dep['external_files']['.work/r6-compiler-bf16-accepted/'+n]=r
for n,r in read(OLD/'review.json')['seals'].items():dep['external_files']['.work/r6-compiler-bf16-accepted/'+n]=r
dep['external_files']['.work/r6-compiler-bf16-accepted/review.json']=seal(OLD/'review.json');write(HERE/'dependency-map.json',dep)
write(HERE/'fix-composition.json',{'generation':C.GENERATION,'compiler_manifest_sha256':C.MANIFEST_SHA,'changed_from_rejected':{n:{'before':read(OLD/'source-map.json').get(n),'after':r}for n,r in source.items()if read(OLD/'source-map.json').get(n)!=r},'actual_compiler':'NOT_RUN'})
if not a.final:print(json.dumps({'status':'MAPS_PREPARED','source_files':len(source)}));raise SystemExit
assert(HERE/'packet-v1/payload.zip').read_bytes()==(HERE/'packet-v2/payload.zip').read_bytes()
groups={};ownershipfiles=[]
for q in HERE.rglob('*.json'):
 if not q.is_file()or q.is_symlink()or q.name in ('review.json','closure-review.json'):continue
 try:r=read(q)
 except (ValueError,UnicodeError):continue
 if isinstance(r,dict)and r.get('ownership')=='Popen child in dedicated start_new_session group'and isinstance(r.get('pid'),int):
  c=r.get('cleanup',{});assert c.get('status')=='CLEANUP_VERIFIED'and c.get('leader_reaped')is True and c.get('group_absence')=='OBSERVED_NO_SUCH_GROUP'and c.get('errors')==[],q
  try:os.killpg(r['pgid'],0)
  except ProcessLookupError:pass
  else:raise AssertionError('LIVE_GROUP:'+str(q))
  groups[r['pgid']]={'pid':r['pid'],'pgid':r['pgid'],'cleanup':c};ownershipfiles.append(q.relative_to(HERE).as_posix())
write(HERE/'closure-review.json',{'status':'ALL_ACTUAL_REGISTERED_GROUPS_CLOSED','count':len(groups),'groups':list(groups.values())})
checks={n:read(HERE/n)for n in ('affected-v1/results.json','accepted-v1/results.json','plain-v1/results.json','portable-v1/results.json')}
assert all(r['status']=='PASS'for r in checks.values());assert[checks[n]['count']for n in checks]==[65,41,23,2]
paths=['source-map.json','adoption-map.json','dependency-map.json','fix-basis.json','fix-composition.json','closure-review.json','native-adoption-readback.json','packet-v1-result.json','packet-v2-result.json','plain-packet-result.json','packet-v1/payload.zip','packet-v2/payload.zip','plain-packet/payload.zip',*checks,*ownershipfiles]
review={'status':'PRIVATE_ACCEPTED_NATIVE_COMPILER_ORDINARY_NATIVE_REPAIR_PASS','candidate_frozen':True,'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'generation':C.GENERATION,'compiler_manifest_sha256':C.MANIFEST_SHA,**C.NATIVE_DEPENDENCY_BINDINGS,'native_dependency_acceptance':C.NATIVE_DEPENDENCY_BINDINGS['actual_native_acceptance'],'source_inventory_files':len(source),'source_inventory_bytes':sum(r['bytes']for r in source.values()),'current_distinct_controls':131,'affected_controls':65,'accepted_dependency_controls':41,'ordinary_native_controls':23,'portable_layout_controls':2,'ordinary_native_packet':read(HERE/'plain-packet-result.json'),'packet':read(HERE/'packet-v1-result.json'),'deterministic_packet_second_build':'BYTE_IDENTICAL','actual_registered_groups_closed':len(groups),'staged_replay':{n:read(HERE/n).get('count')for n in ('staged-affected/results.json','staged-accepted/results.json','staged-plain/results.json')if(HERE/n).is_file()},'staged_replay_not_double_counted':True,'new_preparation_errors':[],'rejected_candidate':'PRESERVED_UNCHANGED_FULL_SEALS','prior_failures':'PRESERVED_IN_FROZEN_DEPENDENCY_MAP','actual_compiler_TPU':'NOT_RUN','fresh_qualified_linux_cpu_oracle':'REQUIRED_NOT_RUN_FOR_THIS_COMPILER_GENERATION','provider_calls':0,'installation':'NOT_RUN','compiler_dispatch':'EXACT_NEW_ROOT_ADOPTION_AND_DISPATCH_GATE_REQUIRED','M6':'NOT_QUALIFIED','limits':read(OLD/'review.json')['limits'],'seals':{n:seal(HERE/n)for n in sorted(set(paths))}}
for n in ('staged-affected/results.json','staged-accepted/results.json','staged-plain/results.json'):
 if(HERE/n).is_file():review['seals'][n]=seal(HERE/n)
write(HERE/'review.json',review);print(json.dumps({'status':review['status'],'controls':131,'groups':len(groups),'source_files':len(source),'review':seal(HERE/'review.json')}))
