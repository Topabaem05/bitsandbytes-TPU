"""Explicit root-reviewed retrieval of one saved version. No submission branch."""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
from save_reference import admit,need
HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def check_sources(map_path,map_sha,host):
    need(sha(map_path)==map_sha,'CONTINUATION_SOURCE_MAP_SHA')
    m=read(map_path)
    for n,pin in m['production'].items():
        p=HERE/n;need(not p.is_symlink()and p.is_file()and sha(p)==pin,'CONTINUATION_SOURCE:'+n)
    for n,pin in m['canonical'].items():
        p=Path(host)/n;need(not p.is_symlink()and p.is_file()and sha(p)==pin,'CANONICAL_SOURCE:'+n)
    return m

def validate_saved(candidate,original,contract,contract_sha,host,*,check_deadline=True):
    need(sha(contract)==contract_sha,'CONTINUATION_CONTRACT_SHA');c=read(contract);candidate=Path(candidate);original=Path(original);host=Path(host)
    need(c['format']=='M8_EXACT_SAVE_CONTINUATION_V1'and c['scope']=='READ_RETRIEVE_ONLY_NO_SUBMIT','CONTINUATION_CONTRACT_SCOPE')
    for n,pin in c['canonical_host'].items():need(sha(host/n)==pin,'CONTINUATION_CANONICAL_SOURCE:'+n)
    sys.path.insert(0,str(host))
    from build_download_candidate import validate_host
    validate_host(host)
    from recover_download import validate_candidate
    from runner import validate_plan,check_response,digest
    from process_provider import Outcome
    for n,pin in c['candidate_files'].items():need(sha(candidate/n)==pin,'CONTINUATION_CANDIDATE_SOURCE:'+n)
    plan=read(candidate/'plan.json');validate_plan(plan);validate_candidate(candidate,plan['binding'])
    need(plan['binding']==c['binding']and plan['deadline_epoch']==c['deadline_epoch'],'CONTINUATION_ORIGINAL_PLAN')
    if check_deadline:need(time.time()+3<plan['deadline_epoch'],'ORIGINAL_DEADLINE_EXPIRED')
    owner_file=original/'owner/owner.json';need(sha(owner_file)==c['original_owner_sha256'],'CONTINUATION_PRIOR_OWNER_SHA');prior=read(owner_file)
    need(prior['state']=='INVALID_RECORDS'and prior['error_code']=='SAVE_IDENTITY_OR_COLLISION'and prior['error_type']=='ValueError','CONTINUATION_ORIGINAL_FAILURE')
    need(prior['binding']==plan['binding']and prior['deadline_epoch']==plan['deadline_epoch'],'CONTINUATION_PRIOR_BINDING')
    calls=prior['calls'];need(len(calls)==1 and calls[0]['operation']=='submit'and calls[0]['status']=='SUCCESS','CONTINUATION_EXACT_ONE_SUCCESS_SUBMIT')
    directory=original/'supervision/provider/001-submit';need(Path(calls[0]['directory']).resolve()==directory.resolve(),'CONTINUATION_SUBMIT_DIRECTORY')
    for name,key in [('launch.json','original_submit_launch_sha256'),('response.json','original_submit_response_sha256'),('request.json','original_submit_request_sha256')]:need(sha(directory/name)==c[key],'CONTINUATION_ORIGINAL_SUBMIT_FILE:'+name)
    launch=read(directory/'launch.json');response=read(directory/'response.json');saved=read(directory/'request.json')
    need(launch==calls[0]['launch']and launch['operation']=='submit'and launch['deadline_epoch']==plan['deadline_epoch']and launch.get('cleanup_errors')==[]and launch['exit_code']==0,'CONTINUATION_SUBMIT_LAUNCH')
    need(launch['worker_argv'][2]==str((host/'sdk_worker.py').resolve())and launch['worker_file_sha256'].get(str((host/'sdk_worker.py').resolve()))==c['canonical_host']['sdk_worker.py'],'CONTINUATION_ORIGINAL_SDK_WORKER')
    request={'folder':plan['folder'],'timeout':'3600','acc':plan['machine_shape'],'wrapper_sha256':plan['wrapper_sha256'],'metadata_sha256':sha(Path(plan['folder'])/'kernel-metadata.json')}
    need(saved['operation']=='submit'and saved['request']==request and saved['deadline_epoch']==plan['deadline_epoch']and saved['process_token']==launch['process_token']and digest(json.dumps(request,sort_keys=True,separators=(',',':')).encode())==launch['request_sha256'],'CONTINUATION_SUBMIT_REQUEST')
    result=check_response(Outcome('SUCCESS',response,launch,directory),'submit',request)
    need(result==prior['submit_response'],'CONTINUATION_RAW_RESPONSE_BINDING')
    identity=c['identity'];need(identity['owner']==plan['owner']and identity['slug']==plan['slug'],'CONTINUATION_KNOWN_OWNER_SLUG')
    admitted=admit(result,plan['owner'],plan['slug'],kernel_id=identity['kernel_id'],version=identity['version'])
    observed=prior['submission_observed_epoch'];need(type(observed)in(int,float)and math.isfinite(observed)and observed<plan['deadline_epoch'],'CONTINUATION_SUBMISSION_TIME')
    return plan,prior,admitted,c

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for cmd in ('check','continue','_worker','close'):
        q=sub.add_parser(cmd)
        for n in ('candidate','original','contract','canonical-host','source-map'):q.add_argument('--'+n,type=Path,required=True)
        for n in ('contract-sha256','source-map-sha256'):q.add_argument('--'+n,required=True)
        if cmd in ('continue','_worker','close'):q.add_argument('--output',type=Path,required=True)
        if cmd=='continue':q.add_argument('--live-provider',action='store_true',required=True);q.add_argument('--kaggle-python',type=Path,required=True)
        if cmd=='_worker':q.add_argument('--broker-fd',type=int,required=True);q.add_argument('--broker-token',required=True);q.add_argument('--deadline-epoch',type=float,required=True)
        if cmd=='close':q.add_argument('--root-observation',type=Path,required=True)
    a=p.parse_args();check_sources(a.source_map,a.source_map_sha256,a.canonical_host)
    plan,prior,admitted,c=validate_saved(a.candidate,a.original,a.contract,a.contract_sha256,a.canonical_host,check_deadline=a.command!='close')
    if a.command=='check':print(json.dumps({'status':'EXACT_SAVED_VERSION_ADMITTED_READ_ONLY','identity':admitted['identity'],'normalization':admitted['normalization'],'deadline_epoch':plan['deadline_epoch'],'submit_calls':0,'actual_M8':'NOT_QUALIFIED'}));return 0
    from continued_runner import ContinuedRunner
    from supervisor import BrokerProvider,supervise
    from recover_download import recover
    from recovery import recover as original_recover
    if a.command=='close':
        obj=ContinuedRunner.__new__(ContinuedRunner);obj.plan=plan;obj.root=a.output/'owner';obj.report=read(obj.root/'owner.json')
        need(obj.report['continuation']['prior_owner_sha256']==c['original_owner_sha256']and obj.report['binding']==plan['binding'],'CONTINUATION_CLOSE_BINDING');result=obj.complete_closure(read(a.root_observation))
    elif a.command=='_worker':
        need(a.deadline_epoch==plan['deadline_epoch'],'BROKER_ORIGINAL_DEADLINE')
        provider=BrokerProvider(a.broker_fd,a.broker_token,plan['deadline_epoch'])
        callback=lambda members,binding:recover(members,binding,a.candidate,a.output/'recovered',original_recover)
        obj=ContinuedRunner(plan,provider,a.output/'owner',callback);result=obj.run_saved(admitted,prior,c['original_owner_sha256'])
    else:
        common=[]
        for n in ('candidate','original','contract','canonical-host','source-map','contract-sha256','source-map-sha256'):common+=['--'+n,str(getattr(a,n.replace('-','_')))]
        common+=['--output',str(a.output.resolve())]
        report=supervise([sys.executable,'-B',str(HERE/'continue_owner.py'),'_worker',*common],[str(a.kaggle_python),'-B',str(a.canonical_host/'sdk_worker.py'),'--live-provider'],a.output/'supervision',plan['deadline_epoch'])
        path=a.output/'owner/owner.json'
        if not path.exists():print(json.dumps({'state':report['status'],'submit_calls':0,'actual_M8':'NOT_QUALIFIED'}));return 2
        result=read(path)
        if report['status']!='OWNER_FINISHED':print(json.dumps({'state':report['status'],'scientific_state':result['state'],'actual_M8':'NOT_QUALIFIED'}));return 2
    need(sha(a.original/'owner/owner.json')==c['original_owner_sha256'],'ORIGINAL_OWNER_POST_READBACK')
    print(json.dumps({'state':result['state'],'record_validation':result['record_validation'],'numerical_status':result['numerical_status'],'closure':result['closure'],'m8_status':'NOT_QUALIFIED','submit_calls':0}));return 0 if result['record_validation']=='PASS'else 2
if __name__=='__main__':raise SystemExit(main())
