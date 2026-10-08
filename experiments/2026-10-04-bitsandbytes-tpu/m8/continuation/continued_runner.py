"""Exact-version continuation; canonical runner methods remain unchanged."""
import copy,json,math,time
from pathlib import Path
from save_reference import admit,need
from runner import validate_plan,wire,check_response,download_bytes,audit_archive,digest,safe_path
from recovery import PartialM2Runner
import runner

class NoSubmitProvider:
    def __init__(self,provider):self.provider=provider;self.deadline_epoch=provider.deadline_epoch
    def invoke(self,operation,request,*,call_seconds):
        need(operation in ('source','status','outputs','download'),'CONTINUATION_SUBMIT_FORBIDDEN')
        return self.provider.invoke(operation,request,call_seconds=call_seconds)

class ContinuedRunner(PartialM2Runner):
    def __init__(self,plan,provider,output,science_verifier,**kwargs):
        super().__init__(plan,NoSubmitProvider(provider),output,science_verifier,**kwargs)
    def run_saved(self,admitted,prior,prior_sha,closure_observation=None):
        need(self.report['state']=='PREPARED','OWNER_REENTRY_FORBIDDEN')
        identity=admitted['identity'];expected_ref=admitted['normalized_ref']
        need(self.provider.deadline_epoch==self.plan['deadline_epoch'],'PROVIDER_DEADLINE_DRIFT')
        self.report.update(identity=identity,state='SUBMITTED',submit_response=admitted['raw_response'],save_ref_normalization=admitted,submission_observed_epoch=prior['submission_observed_epoch'],continuation={'prior_owner_sha256':prior_sha,'prior_state':prior['state'],'prior_error_type':prior['error_type'],'prior_error_code':prior['error_code'],'original_deadline_epoch':prior['deadline_epoch'],'submit_calls':0})
        self.save()
        try:
            validate_plan(self.plan)
            request=wire(identity,'source');outcome=self.call('source',request);source=check_response(outcome,'source',request)
            if source.get('ref')!=expected_ref or source.get('kernel_id')!=identity['kernel_id'] or source.get('current_version_number')!=1 or source.get('is_private') is not True or source.get('language')!='python' or source.get('kernel_type')!='script' or source.get('source_sha256')!=self.plan['wrapper_sha256'] or source.get('machine_shape')!=self.plan['machine_shape']:raise ValueError('VERSION_SOURCE_READBACK')
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

    def record_closure(self,observation):
        # Closure can be known while science is invalid. Do not change that result.
        if observation is None:return self.report
        if self.report.get('root_closure_observation') is not None:raise ValueError('CLOSURE_REPLAY_FORBIDDEN')
        identity=self.report.get('identity');submitted=self.report.get('submit_response')
        if type(identity) is not dict or type(submitted) is not dict:raise ValueError('NO_KNOWN_SUBMITTED_IDENTITY')
        expected={'owner':self.plan['owner'],'slug':self.plan['slug'],'kernel_id':submitted.get('kernel_id'),'version':1}
        if identity!=expected or type(identity['kernel_id']) is not int or identity['kernel_id']<=0 or admit(submitted,self.plan['owner'],self.plan['slug'],kernel_id=identity['kernel_id'],version=identity['version'])['normalized_ref']!=f"{identity['owner']}/{identity['slug']}" or submitted.get('version_number')!=1 or submitted.get('error'):raise ValueError('NO_KNOWN_SUBMITTED_IDENTITY')
        when=self.report.get('submission_observed_epoch')
        if type(when) not in (int,float) or not math.isfinite(when):raise ValueError('SUBMISSION_OBSERVATION_TIME')
        after=max(when,self.report.get('terminal_observed_epoch',when))
        runner.closure_check(observation,identity,self.plan['binding'],after)
        self.report.update(closure='ROOT_OBSERVED_CLOSED',m8_status='NOT_QUALIFIED',
            root_closure_observation=dict(observation),root_closure_recorded_epoch=time.time(),
            root_closure_scientific_state=self.report['state'],partial_m8_scope='M2_API42_TRANSFER4_ONLY')
        self.save();return self.report
