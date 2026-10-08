"""Small synthetic closure controls. No provider or actual acceptance."""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--canonical-host',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
sys.path.insert(0,str(a.canonical_host.resolve()))
from continued_runner import ContinuedRunner
now=time.time();identity={'owner':'fixture-owner','slug':'fixture-script','kernel_id':7,'version':1}
raw={'ref':'/code/fixture-owner/fixture-script','url':'https://www.kaggle.com/code/fixture-owner/fixture-script','kernel_id':7,'version_number':1,'error':None}
proof=a.output/'synthetic-closure.txt';proof.write_text('SYNTHETIC exact-version closure fixture; no provider observation.\n')
plan={'owner':identity['owner'],'slug':identity['slug'],'binding':{'nonce':'f'*32}}
base={'identity':identity,'submit_response':raw,'submission_observed_epoch':now-20,'terminal_observed_epoch':now-10,'state':'VALIDATED_AWAITING_CLOSURE','record_validation':'PASS','numerical_status':'PASS','closure':'UNCONFIRMED','m8_status':'NOT_QUALIFIED','error_type':None,'error_code':None}
observation={'kind':'ROOT_KAGGLE_CLOSURE_OBSERVATION','observer':'root','identity':identity,'nonce':'f'*32,'observed_closed':True,'observed_at_epoch':now,'method':'exact_version_provider_ui','evidence_path':str(proof.resolve()),'evidence_sha256':hashlib.sha256(proof.read_bytes()).hexdigest()}
rows=[]
def owner(name,change=None):
    o=ContinuedRunner.__new__(ContinuedRunner);o.plan=copy.deepcopy(plan);o.root=a.output/name;o.root.mkdir();o.report=copy.deepcopy(base)
    if change:change(o.report)
    o.save();return o
def reject(name,mutate,expected,*,report_change=None):
    o=owner(name,report_change);before=copy.deepcopy(o.report);c=copy.deepcopy(observation);mutate(c)
    try:o.record_closure(c)
    except ValueError as e:
        assert expected in str(e),(name,str(e));assert o.report==before;rows.append({'name':name,'status':'REJECT','error':str(e)})
    else:raise AssertionError('FALSE_ACCEPT:'+name)
o=owner('correct-normalized');r=o.record_closure(copy.deepcopy(observation));assert r['closure']=='ROOT_OBSERVED_CLOSED' and r['submit_response']==raw and r['state']==base['state'] and r['m8_status']=='NOT_QUALIFIED';rows.append({'name':'correct_normalized_save_closure','status':'PASS'})
for name,state,record,numerical in [('numeric-fail','VALIDATED_AWAITING_CLOSURE','PASS','FAIL'),('invalid-records','INVALID_RECORDS','FAIL','ERROR')]:
    o=owner(name,lambda r,s=state,v=record,n=numerical:r.update(state=s,record_validation=v,numerical_status=n,error_type='ValueError',error_code='RETAINED_PRIMARY_ERROR'))
    before={k:o.report[k]for k in ['record_validation','numerical_status','error_type','error_code']};r=o.complete_closure(copy.deepcopy(observation))
    assert before=={k:r[k]for k in before} and r['state']==('SCIENTIFIC_FAIL_CLOSED'if numerical=='FAIL'else'INVALID_RECORDS') and r['m8_status']=='NOT_QUALIFIED';rows.append({'name':name+'_closure_never_passes_science','status':'PASS'})
for field,value in [('owner','other'),('slug','other'),('kernel_id',8),('version',2)]:reject('wrong-'+field,lambda c,f=field,v=value:c['identity'].__setitem__(f,v),'CLOSURE_IDENTITY')
reject('wrong-nonce',lambda c:c.update(nonce='e'*32),'CLOSURE_IDENTITY')
reject('past-time',lambda c:c.update(observed_at_epoch=now-30),'CLOSURE_TIME')
reject('future-time',lambda c:c.update(observed_at_epoch=now+100),'CLOSURE_TIME')
reject('wrong-evidence',lambda c:c.update(evidence_sha256='e'*64),'CLOSURE_EVIDENCE')
reject('replayed-closure',lambda c:None,'CLOSURE_REPLAY_FORBIDDEN',report_change=lambda r:r.update(root_closure_observation=copy.deepcopy(observation)))
reject('invalid-submission-time',lambda c:None,'SUBMISSION_OBSERVATION_TIME',report_change=lambda r:r.update(submission_observed_epoch='invalid'))
result={'status':'PASS','count':len(rows),'controls':rows,'provider_calls':0,'actual_M8':'NOT_QUALIFIED','scope':'SYNTHETIC_CLOSURE_ONLY'};(a.output/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'PASS','count':len(rows)}))
