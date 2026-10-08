"""Read-only scientific recovery. All acceptance inputs are explicit."""
import argparse,hashlib,io,json,math,os,sys,tempfile,zipfile,time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import batch
import runner


def owned_steps(records,receipt,outer):
    rows=receipt['steps'];batch.need([r['label'] for r in rows]==batch.LABELS,'BATCH_STEP_ORDER')
    seen=set();previous=0
    for row in rows:
        label=row['label'];directory=records/'steps'/label
        launch=batch.read(directory/'ownership.json');cleanup=batch.read(directory/'cleanup.json');result=batch.read(directory/'result.json')
        batch.need(row=={'label':label,**result},'STEP_RESULT_BINDING')
        batch.need(not result.get('error') and result['cleanup']==cleanup and cleanup['errors']==[],'STEP_ERROR_OR_CLEANUP')
        allowed={('PASS',0),('CHILD_FAILED',2)} if label=='12-transfer' else {('PASS',0)}
        batch.need((result['status'],result['exit_code']) in allowed,'STEP_EXIT')
        pid=launch['pid'];batch.need(type(pid)is int and pid>0 and pid not in seen and launch['pgid']==pid,'STEP_FRESH_PID')
        seen.add(pid);batch.need(launch['owner_pid']==outer['pid'],'STEP_OWNER_PID')
        groups=cleanup['groups'];batch.need(len(groups)==1 and groups[0]['pid']==groups[0]['pgid']==pid,'STEP_GROUP')
        c=groups[0];batch.need(c['status']=='CLEANUP_VERIFIED' and c['errors']==[] and c['leader_reaped'] is True and c['group_absence']=='OBSERVED_NO_SUCH_GROUP','STEP_OPEN_GROUP')
        batch.need(launch.get('cleanup')==c and launch['exit_status']==result['exit_code'] and c['exit_status']==result['exit_code'],'STEP_CLOSURE_BINDING')
        started=datetime.fromisoformat(launch['started_at']).timestamp();ended=datetime.fromisoformat(launch['ended_at']).timestamp()
        batch.need(previous<=started<=ended<=outer['work_deadline_epoch'],'STEP_TIME_ORDER');previous=ended
        batch.need(math.isfinite(result['timeout_seconds']) and 0<result['timeout_seconds']<=outer['work_deadline_epoch']-started+1,'STEP_BUDGET')
        batch.need(launch['argv']==result['argv'] and type(launch['argv']) is list,'STEP_ARGV_BINDING')
        argv=launch['argv'];base=Path(outer['work_root']);p=base/'payload';python=base/'venv/bin/python'
        if label=='11-cpu-oracle':
            expected=[str(python),'-B',str(p/'probe_transfer.py'),'prepare','--admission',str(p/'source-admission.json'),'--admission-sha256',receipt['source_admission_sha256'],'--output',str(base/'records/cpu-oracle'),'--backend-probe',str(p/'probe_backend.py'),'--route-probe',str(p/'probe_routes.py'),'--precision-probe',str(p/'probe_precision.py'),'--patch-manifest',str(p/'patches/params4bit-xla-v1.json')]
            batch.need(argv==expected,'CPU_EXACT_ARGV')
        elif label in ('11b-local-cpu-gate','13-local-science-verify'):
            command='cpu-audit' if label=='11b-local-cpu-gate' else 'science-audit'
            batch.need(argv==[str(python),'-B',str(p/'batch.py'),command,'--base',str(base)],'AUDIT_EXACT_ARGV')
        elif label=='12-transfer':
            # Original fixed remote creates this exact argument set.
            expected=[str(python),'-B',str(p/'probe_transfer.py'),'execute','--backend-probe',str(p/'probe_backend.py'),'--admission',str(p/'source-admission.json'),'--admission-sha256',receipt['source_admission_sha256'],'--oracle',str(base/'records/cpu-oracle'),'--oracle-sha256',receipt['oracle_sha256'],'--deadline-epoch']
            batch.need(argv[:len(expected)]==expected,'TRANSFER_EXACT_ARGV_PREFIX')
            deadline=float(argv[len(expected)]);batch.need(math.isfinite(deadline) and started<deadline<=min(outer['work_deadline_epoch'],receipt['science_deadline_epoch'],started+1501),'TRANSFER_DEADLINE')
            suffix=['--output',str(base/'records/transfer'),'--route-probe',str(p/'probe_routes.py'),'--precision-probe',str(p/'probe_precision.py'),'--patch-manifest',str(p/'patches/params4bit-xla-v1.json')]
            batch.need(argv[len(expected)+1:]==suffix,'TRANSFER_EXACT_ARGV_SUFFIX')
            rec=batch.read(records/'transfer/receipt.json');batch.need(rec['pid']==rec['pgid']==pid,'TRANSFER_LAUNCH_PID')
    return rows


def recover(members,binding,candidate,output,*,allow_synthetic=False):
    candidate=Path(candidate);payload=candidate/'packet';manifest=batch.read(payload/'manifest.json')
    plan=batch.read(candidate/'plan.json');runner.validate_plan(plan);batch.need(binding==plan['binding'],'ROOT_CANDIDATE_BINDING')
    batch.need(batch.sha(candidate/'payload.zip')==binding['packet_sha256'],'ROOT_PACKET_SHA')
    with zipfile.ZipFile(candidate/'payload.zip') as sealed:
        batch.need(set(sealed.namelist())==set(manifest['files'])|{'manifest.json'},'ROOT_PACKET_MEMBER_SET')
        for name in sealed.namelist():
            batch.need(sealed.read(name)==(payload/name).read_bytes(),'ROOT_EXTRACTED_PACKET_CHANGED:'+name)
    remote=batch.validate_packet(payload,manifest)
    records=Path(output)/'records';records.mkdir(parents=True,exist_ok=False)
    for name,data in members.items():
        batch.need(runner.safe_path(name) and name.startswith('records/'),'RECOVERED_PATH')
        target=Path(output)/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    outer=batch.read(records/'batch.json');receipt=batch.read(records/'receipt.json')
    batch.need(outer['kind']=='M8_M2_BATCH' and outer['status']=='COMPLETE' and outer['error'] is None,'BATCH_TERMINAL')
    batch.need(outer['binding']==binding and outer['deadline_epoch']==plan['deadline_epoch'] and outer['work_deadline_epoch']==plan['deadline_epoch']-660,'BATCH_DEADLINE_BINDING')
    batch.need(outer['orchestration']==batch.DELTA and outer['host_pre_device_review'] is False and outer['m8_status']=='NOT_QUALIFIED' and outer['partial_m8_scope']=='M2_API42_TRANSFER4_ONLY','BATCH_SCOPE')
    batch.need(outer['execution_mode']=='ACTUAL_REQUIRED' or (allow_synthetic and outer['execution_mode']=='SYNTHETIC_FIXTURE'),'SYNTHETIC_NOT_ACTUAL')
    batch.need([r['phase'] for r in outer['phases']]==['install','cpu','runtime','transfer','verify'] and all(r['status']=='COMPLETE' for r in outer['phases']),'PHASE_MATRIX')
    batch.need(type(outer['pid'])is int and outer['pid']>0 and Path(outer['work_root']).is_absolute(),'BATCH_IDENTITY')
    for k in ['packet_sha256','runtime_lock_sha256']:
        batch.need(receipt[k]==binding[k],'RECEIPT_'+k)
    batch.need(receipt['manifest_sha256']==batch.sha(payload/'manifest.json') and receipt['source_admission_sha256']==manifest['source_admission_sha256'],'RECEIPT_SOURCE')
    batch.need(receipt['allocation_epoch']==plan['deadline_epoch']-3600 and receipt['error'] is None,'RECEIPT_EPOCH_ERROR')
    batch.need(receipt['status']=='TRANSFER_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED' and receipt['tpu_status']=='TRANSFER_RECORDS_COMPLETE','TRANSFER_TERMINAL')
    batch.need(all(receipt[k]=='PASS' for k in ['installation_status','cpu_status']) and receipt['runtime_status']=='PASS_TPU_RUNTIME_PROBE_ONLY','PHASE_QUALIFICATION')
    remote.validate_source_controls(batch.read(records/'source-controls.json'))
    owned_steps(records,receipt,outer)
    admission=batch.read(payload/'source-admission.json')
    installed=batch.read(records/'installed-source.json')
    batch.need(installed['status']=='POST_PATCH_PYTHON_SOURCE_PASS' and installed['source_admission_sha256']==manifest['source_admission_sha256'] and installed['patch_manifest_sha256']==manifest['patch_manifest_sha256'] and installed['installed_python_files']=={n:admission[n]['files'] for n in ['bitsandbytes','bitsandbytes_tpu']},'INSTALLED_SOURCE_MAP')
    wheels={r['name']:r for r in batch.read(payload/'runtime/requirements.lock.json')['wheels']}
    for r in batch.read(payload/'build-requirements.lock.json')['wheels']:
        if r['name'] in wheels:batch.need(wheels[r['name']]['sha256']==r['sha256'],'LOCK_CONFLICT')
        wheels[r['name']]=r
    proof=batch.read(records/'installed-metadata.json')
    batch.need(proof['python']=='3.12.14' and proof['packages']=={n:{'version':r['version'],'metadata_sha256':r['metadata_sha256'],'wheel_sha256':r['sha256']} for n,r in wheels.items()},'INSTALLED_RUNTIME_MAP')
    runtime=batch.read(records/'runtime-probe.json');core={'torch':'2.9.0+cpu','torch-xla':'2.9.0','libtpu':'0.0.21','jax':'0.7.1','jaxlib':'0.7.1'}
    batch.need(runtime['status']=='PASS_TPU_RUNTIME_PROBE_ONLY' and runtime['runtime_executed'] is True and runtime['python']=='3.12.14' and runtime['system']=='Linux' and runtime['machine']=='x86_64' and runtime['versions']==core,'TPU_RUNTIME')
    batch.need(runtime['observed_backend']=='TPU' and runtime['devices'] and all(d.startswith('TPU:') for d in runtime['devices']) and runtime['loss']==5.0 and runtime['gradient']==[2.0,4.0],'RUNTIME_OUTPUT')
    bootstrap=batch.read(records/'bootstrap/bootstrap-result.json');PB=batch.load(payload/'cloud/python_bootstrap.py','m8_pinned_python')
    batch.need(bootstrap['status']=='PINNED_ARCHIVE_EXTRACTED_NOT_EXECUTED' and bootstrap['archive_sha256']==PB.SHA and bootstrap['archive_bytes']==PB.SIZE and bootstrap['python_sha256']==PB.PYTHON_SHA and bootstrap['inventory_sha256']==batch.sha(records/'bootstrap/archive-inventory.json'),'PYTHON_BOOTSTRAP')
    built=receipt['built_wheels'];batch.need(len(built)==2,'BUILT_WHEELS')
    for r in built:
        p=records/'built-wheels'/r['name'];batch.need(p.is_file() and p.stat().st_size==r['bytes'] and batch.sha(p)==r['sha256'],'BUILT_WHEEL_BYTES')
        if r['name'].startswith('bitsandbytes-'):remote.verify_transfer_wheel(p,admission['bitsandbytes']['files'])
        elif r['name'].startswith('bitsandbytes_tpu-'):
            with zipfile.ZipFile(p) as z:
                names=z.namelist();batch.need(len(names)==len(set(names)),'PLUGIN_WHEEL_DUPLICATE')
                observed={n.removeprefix('bitsandbytes_tpu/'):hashlib.sha256(z.read(n)).hexdigest() for n in names if n.startswith('bitsandbytes_tpu/') and n.endswith('.py')}
            batch.need(observed==admission['bitsandbytes_tpu']['files'],'PLUGIN_WHEEL_MAP')
            plugin=batch.read(records/'built-plugin-source.json');batch.need(plugin=={'status':'EXACT_PLUGIN_WHEEL_PASS','python_files':observed,'wheel_sha256':r['sha256']},'BUILT_PLUGIN_PROOF')
        else:raise ValueError('UNEXPECTED_BUILT_WHEEL')
    gate=batch.read(records/'batch-cpu-gate.json');expected=batch.cpu_audit(payload,records,manifest)
    for k in expected:
        if k not in ('pid','pgid'):batch.need(gate.get(k)==expected[k],'RECOVERED_CPU_GATE_'+k)
    launch=batch.read(records/'steps/11b-local-cpu-gate/ownership.json');batch.need(gate['pid']==gate['pgid']==launch['pid'],'CPU_GATE_PID')
    result=batch.scientific_audit(payload,records,manifest)
    batch.need(result==batch.read(records/'batch-science-verify.json'),'INDEPENDENT_SCIENTIFIC_RECOMPUTE')
    rows=result['matrix']+result['transfers'];batch.need(len(result['matrix'])==42 and len(result['transfers'])==4,'CASE_MATRIX')
    numerical='ERROR' if any(r['status']=='ERROR' for r in rows) else result['numerical_status']
    comparisons=sum(len(r['gates']) for r in rows)
    if numerical!='ERROR':batch.need(comparisons==196,'COMPARISON_MATRIX')
    report={'kind':'M8_M2_ROOT_RECOVERY','record_validation':'PASS','numerical_status':numerical,
        'fixed_runtime':True,'source_and_input_match':True,'all_required_cases':True,
        'case_count':46,'comparison_count':comparisons,'orchestration':batch.DELTA,
        'm8_status':'NOT_QUALIFIED','actual_execution':outer['execution_mode']=='ACTUAL_REQUIRED',
        'resource_closure':'ROOT_OBSERVATION_REQUIRED','scientific_result':result}
    batch.write(Path(output)/'recovery.json',report)
    return {k:report[k] for k in ['record_validation','numerical_status','fixed_runtime','source_and_input_match','all_required_cases']}

class PartialM2Runner(runner.Runner):
    def call(self,op,request,seconds=30):
        outcome=super().call(op,request,seconds)
        if op=='submit' and outcome.status=='SUCCESS':
            self.report['submission_observed_epoch']=time.time();self.save()
        return outcome
    def record_closure(self,observation):
        # Closure can be known while science is invalid. Do not change that result.
        if observation is None:return self.report
        if self.report.get('root_closure_observation') is not None:raise ValueError('CLOSURE_REPLAY_FORBIDDEN')
        identity=self.report.get('identity');submitted=self.report.get('submit_response')
        if type(identity) is not dict or type(submitted) is not dict:raise ValueError('NO_KNOWN_SUBMITTED_IDENTITY')
        expected={'owner':self.plan['owner'],'slug':self.plan['slug'],'kernel_id':submitted.get('kernel_id'),'version':1}
        if identity!=expected or type(identity['kernel_id']) is not int or identity['kernel_id']<=0 or submitted.get('ref')!=f"{identity['owner']}/{identity['slug']}" or submitted.get('version_number')!=1 or submitted.get('error'):raise ValueError('NO_KNOWN_SUBMITTED_IDENTITY')
        when=self.report.get('submission_observed_epoch')
        if type(when) not in (int,float) or not math.isfinite(when):raise ValueError('SUBMISSION_OBSERVATION_TIME')
        after=max(when,self.report.get('terminal_observed_epoch',when))
        runner.closure_check(observation,identity,self.plan['binding'],after)
        self.report.update(closure='ROOT_OBSERVED_CLOSED',m8_status='NOT_QUALIFIED',
            root_closure_observation=dict(observation),root_closure_recorded_epoch=time.time(),
            root_closure_scientific_state=self.report['state'],partial_m8_scope='M2_API42_TRANSFER4_ONLY')
        self.save();return self.report
    def complete_closure(self,observation):
        result=self.record_closure(observation)
        if result['record_validation']=='PASS' and result.get('closure')=='ROOT_OBSERVED_CLOSED':
            result['state']='PARTIAL_M2_READY_FOR_ROOT_REVIEW' if result['numerical_status']=='PASS' else 'SCIENTIFIC_FAIL_CLOSED'
        # INVALID_RECORDS and all original errors stay unchanged.
        result['m8_status']='NOT_QUALIFIED';result['partial_m8_scope']='M2_API42_TRANSFER4_ONLY';self.save();return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    plan=batch.read(a.candidate/'plan.json');data=(a.evidence/'m8-evidence.tar').read_bytes();manifest=batch.read(a.evidence/'m8-evidence-manifest.json')
    members=runner.audit_archive(data,manifest,plan);print(json.dumps(recover(members,plan['binding'],a.candidate,a.output),sort_keys=True))
