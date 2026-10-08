"""Private M8 owner prototype. Provider calls require an explicit interface."""
import hashlib
import io
import json
import math
from pathlib import Path
import re
import tarfile
import time
import uuid
from process_provider import write_json

SHA=re.compile(r'[0-9a-f]{64}')
BINDING_KEYS={'nonce','packet_sha256','runtime_lock_sha256','profile_sha256','input_sha256','source_manifest_sha256'}


def digest(data):return hashlib.sha256(data).hexdigest()
def safe_path(name):return type(name) is str and name and not name.startswith('/') and '\\' not in name and '\x00' not in name and all(p not in ('','.','..') for p in name.split('/'))


def validate_plan(plan):
    if set(plan)!={'owner','slug','title','folder','wrapper_sha256','binding','machine_shape','session_timeout_seconds','deadline_epoch','max_archive_bytes','max_members','expected_member_paths'}:raise ValueError('PLAN_SCHEMA')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',plan['owner']) or not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',plan['slug']):raise ValueError('PLAN_REF')
    if plan['title'].lower().replace(' ','-')!=plan['slug']:raise ValueError('TITLE_SLUG')
    if not SHA.fullmatch(plan['wrapper_sha256']):raise ValueError('WRAPPER_SHA')
    binding=plan['binding']
    if set(binding)!=BINDING_KEYS or not re.fullmatch(r'[0-9a-f]{32}',binding['nonce']):raise ValueError('BINDING_SCHEMA')
    if not all(SHA.fullmatch(binding[k]) for k in BINDING_KEYS-{'nonce'}):raise ValueError('BINDING_SHA')
    if plan['session_timeout_seconds']!=3600 or plan['machine_shape']!='TpuV6E8':raise ValueError('UNREVIEWED_RUNTIME_REQUEST')
    if type(plan['deadline_epoch']) not in (float,int) or not math.isfinite(plan['deadline_epoch']):raise ValueError('DEADLINE')
    if type(plan['max_archive_bytes']) is not int or not 1<=plan['max_archive_bytes']<=512*1024*1024:raise ValueError('ARCHIVE_BOUND')
    if type(plan['max_members']) is not int or not 1<=plan['max_members']<=10000:raise ValueError('MEMBER_BOUND')
    paths=plan['expected_member_paths']
    if type(paths) is not list or not paths or len(paths)!=len(set(paths)) or not all(safe_path(p) for p in paths):raise ValueError('MEMBER_PATHS')
    folder=Path(plan['folder']).resolve();metadata=json.loads((folder/'kernel-metadata.json').read_text())
    expected={'id':f"{plan['owner']}/{plan['slug']}",'title':plan['title'],'code_file':'wrapper.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':False,'enable_tpu':True,'enable_internet':True,'dataset_sources':[],'kernel_sources':[],'competition_sources':[],'model_sources':[]}
    if metadata!=expected:raise ValueError('METADATA_DRIFT')
    if digest((folder/'wrapper.py').read_bytes())!=plan['wrapper_sha256']:raise ValueError('WRAPPER_CHANGED')


def wire(identity,operation,**extra):
    if operation=='download':return {'ownerSlug':identity['owner'],'kernelSlug':identity['slug'],'versionNumber':identity['version'],**extra}
    return {'userName':identity['owner'],'kernelSlug':identity['slug'],'versionLabel':f"v{identity['version']}",**extra}


def check_response(outcome,operation,request):
    if outcome.status!='SUCCESS':raise ValueError(f'PROVIDER_{operation.upper()}_{outcome.status}')
    if not outcome.launch.get('reaped') or not outcome.launch.get('group_absent'):raise ValueError('PROVIDER_PROCESS_UNCLOSED')
    response=outcome.response
    if type(response) is not dict or response.get('operation')!=operation or response.get('request')!=request:raise ValueError('STALE_OR_LATEST_REQUEST')
    if response.get('process_token')!=outcome.launch['process_token']:raise ValueError('CALL_TOKEN')
    if response.get('pid')!=outcome.launch['launched_pid'] or response.get('pgid')!=outcome.launch['pgid'] or response.get('parent_pid')!=outcome.launch['parent_pid']:raise ValueError('CALL_IDENTITY')
    if response.get('source_pins_match') is not True:raise ValueError('SDK_SOURCE_CHANGED')
    return response['result']


def download_bytes(outcome,result,limit):
    if set(result)!={'artifact','bytes','sha256'} or result['artifact']!='download.bin':raise ValueError('DOWNLOAD_DESCRIPTOR')
    if type(result['bytes']) is not int or not 0<=result['bytes']<=limit or not SHA.fullmatch(result['sha256']):raise ValueError('DOWNLOAD_BOUND')
    path=outcome.directory/result['artifact']
    if path.is_symlink() or not path.is_file():raise ValueError('DOWNLOAD_PATH')
    data=path.read_bytes()
    if len(data)!=result['bytes'] or digest(data)!=result['sha256']:raise ValueError('DOWNLOAD_CHANGED')
    return data


def audit_archive(data,manifest,plan):
    if set(manifest)!={'format','binding','archive','members'} or manifest['format']!='m8-evidence-v1' or manifest['binding']!=plan['binding']:raise ValueError('OUTPUT_BINDING')
    if manifest['archive']!={'file':'m8-evidence.tar','bytes':len(data),'sha256':digest(data)}:raise ValueError('ARCHIVE_HASH')
    if set(manifest['members'])!=set(plan['expected_member_paths']):raise ValueError('SCIENTIFIC_INVENTORY')
    members={};total=0
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:') as tar:
        for item in tar:
            if not item.isfile() or not safe_path(item.name) or item.name in members:raise ValueError('ARCHIVE_MEMBER')
            if len(members)>=plan['max_members']:raise ValueError('MEMBER_COUNT')
            total+=item.size
            if total>plan['max_archive_bytes']:raise ValueError('UNPACKED_BOUND')
            value=tar.extractfile(item).read();spec=manifest['members'].get(item.name)
            if spec!={'bytes':len(value),'sha256':digest(value)}:raise ValueError('MEMBER_HASH')
            members[item.name]=value
    if set(members)!=set(manifest['members']):raise ValueError('MISSING_MEMBER')
    return members


def closure_check(observation,identity,binding,terminal_time):
    if observation is None:return False
    required={'kind','observer','identity','nonce','observed_closed','observed_at_epoch','method','evidence_path','evidence_sha256'}
    if set(observation)!=required or observation['kind']!='ROOT_KAGGLE_CLOSURE_OBSERVATION' or observation['observer']!='root':raise ValueError('CLOSURE_SCHEMA')
    if observation['identity']!=identity or observation['nonce']!=binding['nonce'] or observation['observed_closed'] is not True:raise ValueError('CLOSURE_IDENTITY')
    if observation['method']!='exact_version_provider_ui':raise ValueError('CLOSURE_METHOD')
    when=observation['observed_at_epoch']
    if type(when) not in (int,float) or not math.isfinite(when) or when<terminal_time or when>time.time()+1:raise ValueError('CLOSURE_TIME')
    p=Path(observation['evidence_path'])
    if p.is_symlink() or not p.is_file() or not SHA.fullmatch(observation['evidence_sha256']) or digest(p.read_bytes())!=observation['evidence_sha256']:raise ValueError('CLOSURE_EVIDENCE')
    return True


class Runner:
    def __init__(self,plan,provider,output,science_verifier,*,sleep=time.sleep):
        validate_plan(plan)
        if getattr(provider,'deadline_epoch',None)!=plan['deadline_epoch']:raise ValueError('PROVIDER_DEADLINE_DRIFT')
        self.plan=json.loads(json.dumps(plan,allow_nan=False));self.provider=provider;self.verifier=science_verifier;self.sleep=sleep
        self.metadata_sha256=digest((Path(plan['folder'])/'kernel-metadata.json').read_bytes())
        self.root=Path(output).resolve();self.root.mkdir(parents=True,exist_ok=False)
        self.report={'kind':'PRIVATE_M8_RUNNER_PROTOTYPE','nonce':plan['binding']['nonce'],'binding':plan['binding'],'deadline_epoch':plan['deadline_epoch'],'state':'PREPARED','record_validation':'NOT_RUN','numerical_status':'NOT_RUN','closure':'UNCONFIRMED','m8_status':'NOT_QUALIFIED','calls':[]}
        self.save()
    def save(self):write_json(self.root/'owner.json',self.report)
    def call(self,op,request,seconds=30):
        if time.time()>=self.plan['deadline_epoch']:raise ValueError('OVERALL_DEADLINE')
        outcome=self.provider.invoke(op,request,call_seconds=seconds)
        self.report['calls'].append({'operation':op,'status':outcome.status,'launch':outcome.launch,'directory':str(outcome.directory)})
        self.save();return outcome
    def run(self,closure_observation=None):
        # This object never launches submission twice, including after failure.
        if self.report['state']!='PREPARED':raise ValueError('OWNER_REENTRY_FORBIDDEN')
        identity=None
        try:
            validate_plan(self.plan)
            if digest((Path(self.plan['folder'])/'kernel-metadata.json').read_bytes())!=self.metadata_sha256:raise ValueError('METADATA_CHANGED_AFTER_ADMISSION')
            self.report['state']='SUBMITTING';self.save()
            request={'folder':self.plan['folder'],'timeout':'3600','acc':self.plan['machine_shape'],'wrapper_sha256':self.plan['wrapper_sha256'],'metadata_sha256':self.metadata_sha256}
            outcome=self.call('submit',request,120)
            if outcome.status!='SUCCESS':
                self.report['state']='NOT_RUN_INSUFFICIENT_BUDGET' if outcome.status=='NOT_RUN' else 'SUBMISSION_UNKNOWN';self.report['submission_outcome']=outcome.status;self.save();return self.report
            result=check_response(outcome,'submit',request)
            self.report['submit_response']=result;self.save()
            expected_ref=f"{self.plan['owner']}/{self.plan['slug']}"
            if result.get('error') or any(result.get(k) for k in ('invalid_tags','invalid_dataset_sources','invalid_kernel_sources','invalid_competition_sources','invalid_model_sources')):raise ValueError('SAVE_ERROR')
            if result.get('ref')!=expected_ref or type(result.get('kernel_id')) is not int or result['kernel_id']<=0 or type(result.get('version_number')) is not int or result['version_number']!=1:raise ValueError('SAVE_IDENTITY_OR_COLLISION')
            identity={'owner':self.plan['owner'],'slug':self.plan['slug'],'kernel_id':result['kernel_id'],'version':1}
            self.report['identity']=identity;self.report['state']='SUBMITTED';self.save()
            request=wire(identity,'source');outcome=self.call('source',request);source=check_response(outcome,'source',request)
            if source.get('ref')!=expected_ref or source.get('kernel_id')!=identity['kernel_id'] or source.get('current_version_number')!=1 or source.get('is_private') is not True or source.get('language')!='python' or source.get('kernel_type')!='script' or source.get('source_sha256')!=self.plan['wrapper_sha256']:raise ValueError('VERSION_SOURCE_READBACK')
            self.report['source_readback']=source
            for index in range(120):
                request=wire(identity,'status');outcome=self.call('status',request);status=check_response(outcome,'status',request)
                if status.get('status') in ('COMPLETE','ERROR','CANCEL_ACKNOWLEDGED'):
                    self.report['terminal_status']=status['status'];self.report['terminal_observed_epoch']=time.time();break
                if status.get('status') not in ('QUEUED','RUNNING','CANCEL_REQUESTED'):raise ValueError('STATUS_SCHEMA')
                if self.plan['deadline_epoch']-time.time()<=20:raise ValueError('PENDING_AT_DEADLINE')
                self.sleep(20)
            else:raise ValueError('POLL_CEILING')
            names=[];token=None;seen=set()
            for page in range(20):
                request=wire(identity,'outputs',pageSize=20,**({'pageToken':token} if token else {}));outcome=self.call('outputs',request);result=check_response(outcome,'outputs',request)
                page_names=result.get('files');token=result.get('next_page_token')
                if type(page_names) is not list or not all(safe_path(n) for n in page_names):raise ValueError('OUTPUT_LIST')
                names.extend(page_names)
                if len(names)!=len(set(names)):raise ValueError('DUPLICATE_OUTPUT')
                if not token:break
                if type(token) is not str or token in seen:raise ValueError('PAGE_TOKEN');
                seen.add(token)
            else:raise ValueError('PAGE_CEILING')
            if set(names)!={'m8-evidence-manifest.json','m8-evidence.tar'}:raise ValueError('OUTPUT_LIST_INVENTORY')
            downloaded={}
            for name in ('m8-evidence-manifest.json','m8-evidence.tar'):
                request=wire(identity,'download',filePath=name);outcome=self.call('download',request,60);result=check_response(outcome,'download',request)
                downloaded[name]=download_bytes(outcome,result,1024*1024 if name.endswith('.json') else self.plan['max_archive_bytes'])
                (self.root/name).write_bytes(downloaded[name])
            manifest=json.loads(downloaded['m8-evidence-manifest.json']);members=audit_archive(downloaded['m8-evidence.tar'],manifest,self.plan)
            science=self.verifier(members,self.plan['binding'])
            if set(science)!={'record_validation','numerical_status','fixed_runtime','source_and_input_match','all_required_cases'} or science['record_validation']!='PASS':raise ValueError('SCIENTIFIC_RECORDS')
            if science['numerical_status'] not in ('PASS','FAIL','ERROR'):raise ValueError('NUMERICAL_STATUS')
            if not all(science[k] is True for k in ('fixed_runtime','source_and_input_match','all_required_cases')):raise ValueError('SCIENCE_SOURCE_RUNTIME_OR_MATRIX')
            self.report.update(record_validation='PASS',numerical_status=science['numerical_status'],science= science,state='VALIDATED_AWAITING_CLOSURE')
            self.save()
            self.complete_closure(closure_observation)
        except Exception as error:
            code=str(error) if isinstance(error,ValueError) else 'EXCEPTION_RETAINED'
            incomplete=code in ('OVERALL_DEADLINE','PENDING_AT_DEADLINE','POLL_CEILING') or code.endswith(('_TIMEOUT','_NOT_RUN'))
            self.report.update(state='INCOMPLETE_REMOTE_UNCONFIRMED' if incomplete else 'INVALID_RECORDS',record_validation='NOT_RUN' if incomplete else 'FAIL',error_type=type(error).__name__,error_code=code)
            self.save()
        return self.report
    def complete_closure(self,observation):
        if self.report['record_validation']!='PASS':raise ValueError('NO_VALID_SCIENCE')
        try:closed=closure_check(observation,self.report['identity'],self.plan['binding'],self.report['terminal_observed_epoch'])
        except Exception:
            self.report.update(state='INVALID_CLOSURE_OBSERVATION',closure='UNCONFIRMED',m8_status='NOT_QUALIFIED');self.save();raise
        self.report['closure']='ROOT_OBSERVED_CLOSED' if closed else 'UNCONFIRMED'
        science=self.report['science']
        if closed and science['numerical_status']=='PASS' and all(science[k] is True for k in ('fixed_runtime','source_and_input_match','all_required_cases')):
            self.report['state']='READY_FOR_ROOT_REVIEW';self.report['m8_status']='READY_FOR_ROOT_REVIEW'
        elif closed:self.report['state']='SCIENTIFIC_FAIL_CLOSED'
        self.save();return self.report
