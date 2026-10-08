"""Read-only actual admission plus isolated incorrect records and owned fake broker."""
import argparse,copy,hashlib,io,json,os,sys,tarfile,time
from pathlib import Path
from save_reference import admit
from continue_owner import validate_saved,sha,read
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--original',type=Path,required=True);p.add_argument('--canonical-host',type=Path,required=True);p.add_argument('--contract',type=Path,required=True);p.add_argument('--contract-sha256',required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
sys.path.insert(0,str(a.canonical_host));from supervisor import supervise
rows=[];groups=[];before={str(q):sha(q)for d in [a.original/'owner',a.original/'supervision/provider/001-submit']for q in d.rglob('*')if q.is_file()}
def yes(name,fn):fn();rows.append({'name':name,'status':'PASS'})
def no(name,fn,reason):
 try:fn()
 except ValueError as e:assert reason in str(e);rows.append({'name':name,'status':'REJECT','error':str(e)})
 else:raise AssertionError('FALSE_ACCEPT:'+name)
contract=a.contract;plan,prior,known,c=validate_saved(a.candidate,a.original,contract,a.contract_sha256,a.canonical_host)
rows.append({'name':'actual_original_sealed_success_admitted_read_only','status':'PASS'})
raw=known['raw_response'];ident=known['identity']
for name,value in [('canonical',known['normalized_ref']),('exact_code_path','/code/'+known['normalized_ref'])]:
 r=copy.deepcopy(raw);r['ref']=value;snapshot=copy.deepcopy(r);v=admit(r,ident['owner'],ident['slug'],kernel_id=ident['kernel_id']);assert v['identity']==ident and r==snapshot and v['raw_response']==r;rows.append({'name':'correct_'+name,'status':'PASS'})
faults={'wrong_owner':('ref','/code/other/'+ident['slug']),'wrong_slug':('ref','/code/'+ident['owner']+'/other'),'url_as_ref':('ref',raw['url']),'extra_prefix':('ref','/code/'+raw['ref']),'trailing_slash':('ref',raw['ref']+'/'),'ref_query':('ref',raw['ref']+'?x=1'),'ref_fragment':('ref',raw['ref']+'#x'),'wrong_url_owner':('url','https://www.kaggle.com/code/other/'+ident['slug']),'wrong_url_slug':('url','https://www.kaggle.com/code/'+ident['owner']+'/other'),'url_query':('url',raw['url']+'?x=1'),'url_fragment':('url',raw['url']+'#x'),'url_http':('url',raw['url'].replace('https:','http:')),'wrong_id':('kernel_id',ident['kernel_id']+1),'boolean_id':('kernel_id',True),'string_id':('kernel_id',str(ident['kernel_id'])),'wrong_version':('version_number',2),'boolean_version':('version_number',True),'missing_url':('url',None),'save_error':('error','retained error'),'invalid_source':('invalid_kernel_sources',['wrong'])}
for name,(field,value)in faults.items():
 bad=copy.deepcopy(raw);bad[field]=value;no(name,lambda b=bad:admit(b,ident['owner'],ident['slug'],kernel_id=ident['kernel_id']),'SAVE_')
# A caller cannot extend the original absolute deadline even with a new contract hash.
bad=copy.deepcopy(c);bad['deadline_epoch']+=1;path=a.output/'wrong-deadline-contract.json';path.write_text(json.dumps(bad));no('deadline_extension',lambda:validate_saved(a.candidate,a.original,path,sha(path),a.canonical_host),'CONTINUATION_ORIGINAL_PLAN')
bad=copy.deepcopy(c);bad['identity']['kernel_id']+=1;path2=a.output/'wrong-id-contract.json';path2.write_text(json.dumps(bad));no('accepted_version_collision',lambda:validate_saved(a.candidate,a.original,path2,sha(path2),a.canonical_host),'SAVE_EXACT_VERSION_ID')
# Small synthetic archive; the scientific callback is explicitly not recovery acceptance.
fixture_plan=copy.deepcopy(plan);fixture_plan['expected_member_paths']=['records/result.json'];data=io.BytesIO();body=b'SYNTHETIC_SCIENCE_CALLBACK_NOT_DEVICE'
with tarfile.open(fileobj=data,mode='w:')as t:i=tarfile.TarInfo('records/result.json');i.size=len(body);t.addfile(i,io.BytesIO(body))
archive=data.getvalue();manifest={'format':'m8-evidence-v1','binding':plan['binding'],'archive':{'file':'m8-evidence.tar','bytes':len(archive),'sha256':hashlib.sha256(archive).hexdigest()},'members':{'records/result.json':{'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()}}};(a.output/'m8-evidence.tar').write_bytes(archive);(a.output/'m8-evidence-manifest.json').write_text(json.dumps(manifest))
for fault in ['correct','source','submit']:
 out=a.output/fault;out.mkdir();cfg={'fault':fault,'plan':fixture_plan,'admitted':known,'prior':prior,'prior_sha':c['original_owner_sha256'],'artifacts':{n:str((a.output/n).resolve())for n in ['m8-evidence.tar','m8-evidence-manifest.json']}};cfgpath=out/'fixture.json';cfgpath.write_text(json.dumps(cfg));os.environ['M8_CONTINUATION_FIXTURE']=str(cfgpath.resolve());os.environ['M8_CANONICAL_HOST']=str(a.canonical_host.resolve())
 report=supervise([sys.executable,'-B',str(HERE/'fake_owner.py'),'--output',str(out.resolve())],[sys.executable,'-B',str(HERE/'fake_worker.py')],out/'supervision',plan['deadline_epoch']);assert report['status']=='OWNER_FINISHED',report
 assert report['owner_cleanup']['status']=='CLOSED'and all(r['status']=='CLOSED'for r in report['provider_cleanup']);groups +=[report['owner_cleanup'],*report['provider_cleanup']]
 result=read(out/'result.json');registry=read(out/'supervision/provider-registry.json')if(out/'supervision/provider-registry.json').exists()else{'rows':[]};operations=[r['operation']for r in registry['rows']];assert 'submit'not in operations
 if fault=='correct':assert result['record_validation']=='PASS'and result['state']=='VALIDATED_AWAITING_CLOSURE'and result['m8_status']=='NOT_QUALIFIED'and operations==['source','status','outputs','download','download']
 elif fault=='source':assert result['state']=='INVALID_RECORDS'and result['error_code']=='VERSION_SOURCE_READBACK'and operations==['source']
 else:assert result['status']=='REJECT'and result['error']=='CONTINUATION_SUBMIT_FORBIDDEN'and operations==[]
 rows.append({'name':'real_owned_broker_'+fault,'status':'PASS','operations':operations,'scope':'SYNTHETIC_SDK_AND_SCIENCE_CALLBACK_NO_DEVICE'})
for pg in {r['pgid']for r in groups}:
 try:os.killpg(pg,0)
 except ProcessLookupError:pass
 else:raise AssertionError(('LIVE_GROUP',pg))
assert {q:sha(q)for q in before}==before
result={'status':'PASS','count':len(rows),'controls':rows,'groups':groups,'groups_closed':len({r['pgid']for r in groups}),'actual_M8':'NOT_QUALIFIED','provider_calls':0,'original_records_readback':'UNCHANGED','scope':'READ_ONLY_ADMISSION_AND_SYNTHETIC_OWNED_CONTROLS'};(a.output/'results.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps({'status':'PASS','count':len(rows),'groups_closed':result['groups_closed']}))
