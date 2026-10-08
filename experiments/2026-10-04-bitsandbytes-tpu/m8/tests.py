"""Offline end-to-end controls. Provider and device records are synthetic."""
import argparse,ast,copy,hashlib,io,json,os,shutil,socket,sys,time,unittest,zipfile,signal,threading,subprocess
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
P=argparse.ArgumentParser();P.add_argument('--output',type=Path,required=True);P.add_argument('--project-source',type=Path,required=True);P.add_argument('--base-archive',type=Path,required=True);P.add_argument('--kaggle-python',type=Path,required=True);OPTIONS=P.parse_args();OUT=OPTIONS.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
HERE=Path(__file__).resolve().parent;ROOT=OPTIONS.project_source.resolve();sys.path.insert(0,str(HERE));attempts={'network':0,'credentials':0}
def guard(event,args):
    if event in ('socket.connect','socket.getaddrinfo'):attempts['network']+=1;raise RuntimeError('NO_NETWORK')
    if event=='open' and isinstance(args[0],(str,bytes)) and '/.kaggle/' in os.fsdecode(args[0]):attempts['credentials']+=1;raise RuntimeError('NO_CREDENTIAL_READ')
sys.addaudithook(guard)
import batch as B
import build_candidate as C
import recovery as V
import runner as R
from process_provider import Outcome
from supervisor import supervise

# Use original portable scientific fixture functions, not production records.
fixture_path=ROOT/'tests/test_transfer_probe.py'
namespace={'__file__':str(fixture_path),'__name__':'m8_original_science_fixture'}
exec(compile(fixture_path.read_text().split('class TransferControls')[0],str(fixture_path),'exec'),namespace)
make_science=namespace['make_fixture'];reseal_science=namespace['reseal_fixture']
node=next(n for n in ast.parse((ROOT/'experiments/2026-10-04-bitsandbytes-tpu/cloud/tests/test_transfer_cloud.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='controls_fixture')
ns={'B':SimpleNamespace(TRANSFER_PATCH_MANIFEST_SHA='e745fbf21aac10ed9118167a131d6505bf6dbe03e1fab0a669c663b26c5a5732')}
exec(compile(ast.Module(body=[node],type_ignores=[]),'original_source_control_fixture','exec'),ns);source_controls=ns['controls_fixture']

class BatchFixture:
    synthetic=True
    def __init__(self,base,manifest,fault=None):self.base=base;self.p=base/'payload';self.o=base/'records';self.m=manifest;self.fault=fault;self.phases=[]
    def receipt(self):return B.read(self.o/'receipt.json')
    def step(self,label,argv=None,exit_code=0):
        rec=self.receipt();pid=20000+len(rec['steps']);now=datetime.now(timezone.utc).isoformat();directory=self.o/'steps'/label;directory.mkdir(parents=True)
        if argv is None:argv=[sys.executable,'-B','SYNTHETIC_ONLY:'+label]
        c={'pid':pid,'pgid':pid,'status':'CLEANUP_VERIFIED','errors':[],'leader_reaped':True,'group_absence':'OBSERVED_NO_SUCH_GROUP','exit_status':exit_code,'started_at':now,'ended_at':now}
        clean={'groups':[c],'errors':[]}
        launch={'pid':pid,'pgid':pid,'owner_pid':os.getpid(),'argv':argv,'started_at':now,'ended_at':now,'exit_status':exit_code,'cleanup':c}
        result={'status':'PASS' if exit_code==0 else 'CHILD_FAILED','exit_code':exit_code,'argv':argv,'timeout_seconds':60,'cleanup':clean}
        for name,value in [('cleanup.json',clean),('ownership.json',launch),('result.json',result)]:B.write(directory/name,value)
        (directory/'lifecycle.jsonl').write_text(json.dumps({'synthetic_fixture':True})+'\n')
        for n in ['stdout.raw','stderr.raw']:(directory/n).write_text('SYNTHETIC FIXTURE LOG\n')
        rec['steps'].append({'label':label,**result});B.write(self.o/'receipt.json',rec);return pid
    def phase(self,label):
        self.phases.append(label);p=self.p;o=self.o;base=self.base;m=self.m;python=base/'venv/bin/python'
        if label=='install':
            rec={'packet_sha256':B.sha(base/'payload.zip'),'manifest_sha256':B.sha(p/'manifest.json'),'source_admission_sha256':m['source_admission_sha256'],'runtime_lock_sha256':m['runtime_lock_sha256'],
                 'allocation_epoch':PLAN_DEADLINE-3600,'steps':[],'error':None,'installation_status':'PASS','cpu_status':'NOT_RUN','runtime_status':'NOT_RUN','tpu_status':'NOT_RUN'}
            B.write(o/'receipt.json',rec)
            for name in B.LABELS[:10]:self.step(name)
            adm=B.read(p/'source-admission.json')
            B.write(o/'installed-source.json',{'status':'POST_PATCH_PYTHON_SOURCE_PASS','source_admission_sha256':m['source_admission_sha256'],'patch_manifest_sha256':m['patch_manifest_sha256'],'installed_python_files':{n:adm[n]['files'] for n in ['bitsandbytes','bitsandbytes_tpu']}})
            wheels={r['name']:r for r in B.read(p/'runtime/requirements.lock.json')['wheels']}
            wheels.update({r['name']:r for r in B.read(p/'build-requirements.lock.json')['wheels']})
            B.write(o/'installed-metadata.json',{'python':'3.12.14','packages':{n:{'version':r['version'],'metadata_sha256':r['metadata_sha256'],'wheel_sha256':r['sha256']} for n,r in wheels.items()}})
            PB=B.load(p/'cloud/python_bootstrap.py','fixture_python');B.write(o/'bootstrap/archive-inventory.json',{'synthetic_fixture':True,'files':[]})
            B.write(o/'bootstrap/bootstrap-result.json',{'status':'PINNED_ARCHIVE_EXTRACTED_NOT_EXECUTED','archive_sha256':PB.SHA,'archive_bytes':PB.SIZE,'python_sha256':PB.PYTHON_SHA,'inventory_sha256':B.sha(o/'bootstrap/archive-inventory.json')})
            Rm=B.validate_packet(p,m);Rm.unpack_source(p/'patched-upstream.tar',base/'fixture-upstream')
            built=[];target=o/'built-wheels';target.mkdir()
            for src,name,prefix in [(base/'fixture-upstream/bitsandbytes','bitsandbytes-0.50.3.dev0-cp312-cp312-linux_x86_64.whl','bitsandbytes'),(p/'plugin/src/bitsandbytes_tpu','bitsandbytes_tpu-0.1.0-py3-none-any.whl','bitsandbytes_tpu')]:
                wheel=target/name
                with zipfile.ZipFile(wheel,'w') as z:
                    for f in sorted(src.rglob('*.py')):z.write(f,prefix+'/'+f.relative_to(src).as_posix())
                built.append({'name':name,'bytes':wheel.stat().st_size,'sha256':B.sha(wheel)})
            rec=self.receipt();rec['built_wheels']=built;B.write(o/'receipt.json',rec)
            B.write(o/'built-source.json',{'status':'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS','python_files':adm['bitsandbytes']['files']})
            B.write(o/'built-plugin-source.json',{'status':'EXACT_PLUGIN_WHEEL_PASS','python_files':adm['bitsandbytes_tpu']['files'],'wheel_sha256':built[1]['sha256']});return
        if label=='cpu':
            B.write(o/'source-controls.json',source_controls());self.step('10-transfer-source-controls')
            self.f=make_science(base/'fixture-science',m['source_admission_sha256'])
            shutil.copytree(self.f.oracle,o/'cpu-oracle')
            argv=[str(python),'-B',str(p/'probe_transfer.py'),'prepare','--admission',str(p/'source-admission.json'),'--admission-sha256',m['source_admission_sha256'],'--output',str(o/'cpu-oracle'),'--backend-probe',str(p/'probe_backend.py'),'--route-probe',str(p/'probe_routes.py'),'--precision-probe',str(p/'probe_precision.py'),'--patch-manifest',str(p/'patches/params4bit-xla-v1.json')]
            self.step('11-cpu-oracle',argv)
            if self.fault=='cpu-source':
                seal=B.read(o/'cpu-oracle/oracle-seal.json');seal['source_pre']='b'*64;B.write(o/'cpu-oracle/oracle-seal.json',seal)
            gate=B.cpu_audit(p,o,m);pid=self.step('11b-local-cpu-gate',[str(python),'-B',str(p/'batch.py'),'cpu-audit','--base',str(base)])
            gate.update(pid=pid,pgid=pid);B.write(o/'batch-cpu-gate.json',gate)
            rec=self.receipt();rec.update(cpu_status='PASS',oracle_sha256=B.sha(o/'cpu-oracle/oracle-seal.json'),science_deadline_epoch=min(PLAN_DEADLINE-660,time.time()+1800));B.write(o/'receipt.json',rec);return
        if label=='runtime':
            self.step('10-runtime-probe');B.write(o/'runtime-probe.json',{'status':'PASS_TPU_RUNTIME_PROBE_ONLY','runtime_executed':True,'python':'3.12.14','system':'Linux','machine':'x86_64','versions':{'torch':'2.9.0+cpu','torch-xla':'2.9.0','libtpu':'0.0.21','jax':'0.7.1','jaxlib':'0.7.1'},'observed_backend':'TPU','devices':['TPU:0'],'loss':5.0,'gradient':[2.0,4.0]})
            rec=self.receipt();rec['runtime_status']='PASS_TPU_RUNTIME_PROBE_ONLY';B.write(o/'receipt.json',rec);return
        if label=='transfer':
            shutil.copytree(self.f.actual,o/'transfer');rec=self.receipt();deadline=min(rec['science_deadline_epoch'],time.time()+1500)
            argv=[str(python),'-B',str(p/'probe_transfer.py'),'execute','--backend-probe',str(p/'probe_backend.py'),'--admission',str(p/'source-admission.json'),'--admission-sha256',m['source_admission_sha256'],'--oracle',str(o/'cpu-oracle'),'--oracle-sha256',rec['oracle_sha256'],'--deadline-epoch',str(deadline),'--output',str(o/'transfer'),'--route-probe',str(p/'probe_routes.py'),'--precision-probe',str(p/'probe_precision.py'),'--patch-manifest',str(p/'patches/params4bit-xla-v1.json')]
            pid=self.step('12-transfer',argv,2 if self.fault in ['numeric','case-error'] else 0)
            actual=o/'transfer';r=B.read(actual/'receipt.json');r.update(pid=pid,pgid=pid)
            for path in [*sorted((actual/'matrix').glob('*.json')),*sorted((actual/'transfer').glob('*.json'))]:
                v=B.read(path);v['pid']=pid;B.write(path,v)
            path=actual/'transfer/module_to_xla-linear-float32-rank2-bias1.json';raw=B.read(path)
            if self.fault=='numeric':raw['outputs']['y']['values'][0]=10.0
            if self.fault=='case-error':raw.update(status='ERROR',error_type='SyntheticCaseError',error_message='SYNTHETIC_ONLY',traceback='SYNTHETIC_ONLY')
            B.write(path,raw)
            r['artifacts']=self.f.B.inventory(actual,'receipt.json');B.write(actual/'receipt.json',r)
            rec=self.receipt();rec.update(status='TRANSFER_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED',tpu_status='TRANSFER_RECORDS_COMPLETE');B.write(o/'receipt.json',rec);return
        if label=='verify':
            result=B.scientific_audit(p,o,m);B.write(o/'batch-science-verify.json',result)
            self.step('13-local-science-verify',[str(python),'-B',str(p/'batch.py'),'science-audit','--base',str(base)]);return
        raise AssertionError(label)

class ProviderFixture:
    def __init__(self,root,plan,files,fault=None):self.root=root;root.mkdir();self.plan=plan;self.files=files;self.deadline_epoch=plan['deadline_epoch'];self.calls=[];self.fault=fault
    def invoke(self,op,request,call_seconds):
        self.calls.append(op);d=self.root/str(len(self.calls));d.mkdir();pid=30000+len(self.calls)
        launch={'reaped':True,'group_absent':True,'process_token':'2'*32,'parent_pid':os.getpid(),'pgid':pid,'launched_pid':pid}
        response={'operation':op,'request':dict(request),'source_pins_match':True,'process_token':'2'*32,'parent_pid':os.getpid(),'pid':pid,'pgid':pid}
        identity=f"{self.plan['owner']}/{self.plan['slug']}"
        if op=='submit':
            if self.fault=='ambiguous':return Outcome('TIMEOUT',None,launch,d)
            result={'ref':identity,'kernel_id':101,'version_number':1,'error':''}
            if self.fault=='wrong-version':result['version_number']=2
        elif op=='source':result={'ref':identity,'kernel_id':101,'current_version_number':1,'is_private':True,'language':'python','kernel_type':'script','source_sha256':self.plan['wrapper_sha256']}
        elif op=='status':
            result={'status':'COMPLETE'}
            if self.fault=='latest':response['request'].pop('versionLabel')
        elif op=='outputs':result={'files':list(self.files),'next_page_token':''}
        elif op=='download':
            data=self.files[request['filePath']];(d/'download.bin').write_bytes(data);result={'artifact':'download.bin','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        else:raise AssertionError(op)
        response['result']=result;return Outcome('SUCCESS',response,launch,d)

PLAN_DEADLINE=time.time()+3600
class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidate=OUT/'candidate';cls.plan=C.build(cls.candidate,'fixture-owner','m8-m2-fixture','M8 M2 Fixture','1'*32,PLAN_DEADLINE,project_source=ROOT,base_archive=OPTIONS.base_archive)
        cls.wrapper=B.load(cls.candidate/'submission/wrapper.py','offline_embedded_wrapper')
    def setUp(self):self.base=OUT/self._testMethodName;self.base.mkdir()
    def bundle(self,fault=None,mutate=None):
        providers=[]
        def factory(base,m):p=BatchFixture(base,m,fault);providers.append(p);return p
        record=self.wrapper.entry(provider_factory=factory,workspace=self.base/'batch',output=self.base/'output')
        if mutate:
            mutate(self.base/'batch/records');
            for p in (self.base/'output').iterdir():p.unlink()
            B.export(self.base/'batch/records',self.base/'output',self.plan['binding'])
        files={p.name:p.read_bytes() for p in (self.base/'output').iterdir()};return record,files,providers[0]
    def owner(self,files,provider_fault=None,synthetic=True):
        p=ProviderFixture(self.base/'provider',self.plan,files,provider_fault)
        callback=lambda members,binding:V.recover(members,binding,self.candidate,self.base/'recovered',allow_synthetic=synthetic)
        o=V.PartialM2Runner(self.plan,p,self.base/'owner',callback,sleep=lambda _:None);return o,p,o.run()
    def test_full_packet_wrapper_archive_owner_original_verifier(self):
        record,files,b=self.bundle();self.assertEqual(record['status'],'COMPLETE');self.assertEqual(b.phases,['install','cpu','runtime','transfer','verify'])
        o,p,result=self.owner(files);self.assertEqual(result['record_validation'],'PASS');self.assertEqual(result['numerical_status'],'PASS');self.assertEqual(result['m8_status'],'NOT_QUALIFIED');self.assertEqual(p.calls.count('submit'),1)
        review=B.read(self.base/'recovered/recovery.json');self.assertEqual((review['case_count'],review['comparison_count']),(46,196));self.assertFalse(review['actual_execution'])
        proof=self.base/'root-closure.txt';proof.write_text('SYNTHETIC root closure observation\n')
        closure={'kind':'ROOT_KAGGLE_CLOSURE_OBSERVATION','observer':'root','identity':result['identity'],'nonce':self.plan['binding']['nonce'],'observed_closed':True,'observed_at_epoch':time.time(),'method':'exact_version_provider_ui','evidence_path':str(proof),'evidence_sha256':B.sha(proof)}
        o.complete_closure(closure);self.assertEqual(result['state'],'PARTIAL_M2_READY_FOR_ROOT_REVIEW');self.assertEqual(result['m8_status'],'NOT_QUALIFIED')
    def test_valid_numerical_failure_retained(self):
        _,files,_=self.bundle('numeric');_,_,r=self.owner(files);self.assertEqual((r['record_validation'],r['numerical_status']),('PASS','FAIL'))
    def test_valid_case_error_retained(self):
        _,files,_=self.bundle('case-error');_,_,r=self.owner(files);self.assertEqual((r['record_validation'],r['numerical_status']),('PASS','ERROR'))
    def test_cpu_source_failure_prevents_device_phase(self):
        r,files,p=self.bundle('cpu-source');self.assertEqual(r['status'],'INVALID_OR_BLOCKED');self.assertEqual(p.phases,['install','cpu']);self.assertNotIn('runtime',p.phases)
    def bad(self,mutate):
        _,files,_=self.bundle(mutate=mutate);_,_,r=self.owner(files);self.assertEqual(r['record_validation'],'FAIL');return r
    def change(self,path,fn):
        def f(records):v=B.read(records/path);fn(v);B.write(records/path,v)
        return f
    def test_source_admission_change(self):self.bad(self.change('installed-source.json',lambda v:v['installed_python_files']['bitsandbytes'].pop('functional.py')))
    def test_runtime_change(self):self.bad(self.change('installed-metadata.json',lambda v:v['packages']['torch'].update(version='2.14.1')))
    def test_wrong_precision(self):self.bad(self.change('transfer/receipt.json',lambda v:v['precision'].update(readback='default')))
    def test_missing_case(self):self.bad(lambda r:(r/'transfer/matrix/api42-quant-float32-1-codes.json').unlink())
    def test_changed_input_binding(self):self.bad(self.change('transfer/matrix/api42-quant-float32-1-codes.json',lambda v:v.update(input_sha256='b'*64)))
    def test_latest_only_status(self):
        _,files,_=self.bundle();_,_,r=self.owner(files,'latest');self.assertEqual(r['error_code'],'STALE_OR_LATEST_REQUEST')
    def test_wrong_saved_version(self):
        _,files,_=self.bundle();_,p,r=self.owner(files,'wrong-version');self.assertEqual(p.calls,['submit']);self.assertEqual(r['record_validation'],'FAIL')
    def test_ambiguous_submission_no_replay(self):
        _,files,_=self.bundle();o,p,r=self.owner(files,'ambiguous');self.assertEqual(r['state'],'SUBMISSION_UNKNOWN');self.assertEqual(p.calls,['submit'])
        with self.assertRaises(ValueError):o.run()
    def test_archive_mismatch(self):
        _,files,_=self.bundle();files['m8-evidence.tar']+=b'changed';_,_,r=self.owner(files);self.assertEqual(r['error_code'],'ARCHIVE_HASH')
    def test_open_child_group(self):self.bad(self.change('steps/12-transfer/cleanup.json',lambda v:v['groups'][0].update(group_absence='UNVERIFIED')))
    def test_cpu_gate_cannot_claim_host_interleaving(self):self.bad(self.change('batch-cpu-gate.json',lambda v:v.update(host_pre_device_review=True)))
    def test_synthetic_records_cannot_pass_actual_recovery(self):
        _,files,_=self.bundle();_,_,r=self.owner(files,synthetic=False);self.assertEqual(r['error_code'],'SYNTHETIC_NOT_ACTUAL')
    def test_changed_sealed_science_packet(self):
        original=self.wrapper.PAYLOAD_B64;self.wrapper.PAYLOAD_B64='AAAA'
        try:
            with self.assertRaisesRegex(ValueError,'EMBEDDED_PACKET_SHA'):self.wrapper.entry(workspace=self.base/'unused')
        finally:self.wrapper.PAYLOAD_B64=original
    def test_post_audit_exception_never_passes(self):
        _,files,_=self.bundle();p=ProviderFixture(self.base/'provider',self.plan,files)
        def callback(m,b):V.recover(m,b,self.candidate,self.base/'recovered',allow_synthetic=True);raise RuntimeError('SYNTHETIC post audit')
        o=V.PartialM2Runner(self.plan,p,self.base/'owner',callback);r=o.run();self.assertEqual(r['record_validation'],'FAIL')
    def test_default_cpu_adapter_orders_original_commands(self):
        calls=[]
        def factory(base,m):
            fixture=BatchFixture(base,m)
            class Composite:
                synthetic=True
                def phase(inner,label):
                    if label!='cpu':return fixture.phase(label)
                    fixed=B.FixedBatch(base,m,B.sha(base/'payload.zip'),PLAN_DEADLINE)
                    def run_step(out,name,argv,deadline,limit,**kwargs):
                        calls.append((name,argv,deadline,limit))
                        if name=='10-transfer-source-controls':B.write(out/'source-controls.json',source_controls())
                        elif name=='11-cpu-oracle':
                            fixture.f=make_science(base/'fixture-science',m['source_admission_sha256']);shutil.copytree(fixture.f.oracle,out/'cpu-oracle')
                        elif name=='11b-local-cpu-gate':
                            gate=B.cpu_audit(base/'payload',out,m)
                        pid=fixture.step(name,argv)
                        if name=='11b-local-cpu-gate':gate.update(pid=pid,pgid=pid);B.write(out/'batch-cpu-gate.json',gate)
                        return B.read(out/'steps'/name/'result.json')
                    with patch.object(fixed.remote,'run_step',run_step):fixed.phase('cpu')
            return Composite()
        r=self.wrapper.entry(provider_factory=factory,workspace=self.base/'batch',output=self.base/'output')
        self.assertEqual(r['status'],'COMPLETE');self.assertEqual([c[0] for c in calls],['10-transfer-source-controls','11-cpu-oracle','11b-local-cpu-gate'])
        self.assertEqual([c[3] for c in calls],[300,300,60]);self.assertTrue(all(c[2]<=PLAN_DEADLINE-660 for c in calls))
        files={p.name:p.read_bytes() for p in (self.base/'output').iterdir()};_,_,report=self.owner(files);self.assertEqual(report['record_validation'],'PASS')
    def test_real_owned_success_and_timeout_cleanup(self):
        remote=B.validate_packet(self.candidate/'packet',B.read(self.candidate/'packet/manifest.json'))
        for label,code,budget in [('success','print("offline owned child")',3),('timeout','import time;time.sleep(10)',.15)]:
            rec=remote.run_step(self.base,label,[sys.executable,'-B','-c',code],time.time()+budget,budget,cwd=self.base)
            self.assertEqual(rec['cleanup']['errors'],[]);self.assertTrue(rec['cleanup']['groups'][0]['leader_reaped']);self.assertEqual(rec['cleanup']['groups'][0]['group_absence'],'OBSERVED_NO_SUCH_GROUP')
            self.assertEqual(rec['status'],'PASS' if label=='success' else 'BLOCKED')
    def test_expired_original_deadline_never_runs_install(self):
        previous=self.wrapper.DEADLINE;self.wrapper.DEADLINE=time.time()+659
        class Never:
            synthetic=True
            def phase(self,label):raise AssertionError('NO_PHASE_MAY_START')
        try:
            with self.assertRaisesRegex(ValueError,'INSUFFICIENT_WORK_BUDGET'):self.wrapper.entry(provider_factory=lambda b,m:Never(),workspace=self.base/'batch',output=self.base/'output')
        finally:self.wrapper.DEADLINE=previous
    def test_wrong_cpu_owned_argv_rejected(self):
        self.bad(self.change('steps/11-cpu-oracle/ownership.json',lambda v:v['argv'].__setitem__(3,'verify')))
    def test_root_extracted_source_changed(self):
        _,files,_=self.bundle()
        copied=self.base/'tampered-candidate';shutil.copytree(self.candidate,copied)
        source=copied/'packet/probe_transfer.py';source.write_text(source.read_text()+'\n# changed source\n')
        with self.assertRaisesRegex(ValueError,'ROOT_EXTRACTED_PACKET_CHANGED'):
            members=R.audit_archive(files['m8-evidence.tar'],json.loads(files['m8-evidence-manifest.json']),self.plan)
            V.recover(members,self.plan['binding'],copied,self.base/'recovered',allow_synthetic=True)
    def closure(self,owner):
        proof=self.base/'closure.txt';proof.write_text('SYNTHETIC exact failed-run closure observation\n')
        return {'kind':'ROOT_KAGGLE_CLOSURE_OBSERVATION','observer':'root','identity':owner.report['identity'],'nonce':self.plan['binding']['nonce'],'observed_closed':True,'observed_at_epoch':time.time(),'method':'exact_version_provider_ui','evidence_path':str(proof),'evidence_sha256':B.sha(proof)}
    def test_failed_scientific_records_can_record_root_closure(self):
        _,files,_=self.bundle(mutate=self.change('installed-metadata.json',lambda v:v['packages']['torch'].update(version='wrong')))
        o,_,r=self.owner(files);before={k:r.get(k) for k in ['state','record_validation','numerical_status','error_code','error_type']}
        self.assertEqual(r['state'],'INVALID_RECORDS');o.complete_closure(self.closure(o))
        self.assertEqual(r['closure'],'ROOT_OBSERVED_CLOSED');self.assertEqual(before,{k:r.get(k) for k in before});self.assertEqual(r['m8_status'],'NOT_QUALIFIED')
    def test_failed_run_closure_wrong_identity_replay_and_time(self):
        _,files,_=self.bundle(mutate=self.change('runtime-probe.json',lambda v:v.update(system='wrong')))
        o,_,r=self.owner(files);c=self.closure(o)
        for fault in ['identity','past','future']:
            bad=copy.deepcopy(c)
            if fault=='identity':bad['identity']['version']=2
            if fault=='past':bad['observed_at_epoch']=r['submission_observed_epoch']-1
            if fault=='future':bad['observed_at_epoch']=time.time()+5
            with self.assertRaises(ValueError):o.complete_closure(bad)
            self.assertEqual(r['closure'],'UNCONFIRMED');self.assertEqual(r['state'],'INVALID_RECORDS')
        o.complete_closure(c)
        with self.assertRaisesRegex(ValueError,'CLOSURE_REPLAY'):o.complete_closure(c)
    def test_ambiguous_submission_has_no_admitted_closure_identity(self):
        _,files,_=self.bundle();o,_,r=self.owner(files,'ambiguous')
        with self.assertRaisesRegex(ValueError,'NO_KNOWN_SUBMITTED_IDENTITY'):o.complete_closure({'untrusted':True})
        self.assertEqual(r['state'],'SUBMISSION_UNKNOWN');self.assertEqual(r['closure'],'UNCONFIRMED')
    def test_final_tar_byte_bound_before_manifest_success(self):
        records=self.base/'small';records.mkdir();(records/'one').write_bytes(b'x')
        with patch.object(B,'MAX_OUTPUT',1024):
            with self.assertRaisesRegex(ValueError,'FINAL_TAR_BYTE_BOUND'):B.export(records,self.base/'export',self.plan['binding'])
        self.assertFalse((self.base/'export/m8-evidence-manifest.json').exists())
    def supervise_fixture(self,mode,kill_owner=False):
        worker=self.base/'worker.py';owner=self.base/'broker_owner.py'
        worker.write_text('''import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--output');a=p.parse_args();r=json.load(open(a.request))
MODE='''+repr(mode)+'''
if MODE=='descendant':
    child=subprocess.Popen([sys.executable,'-B','-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)'])
    Path(a.output,'descendant.pid').write_text(str(child.pid));time.sleep(.1)
if MODE!='success':
    Path(a.output,'ready').write_text('ready');time.sleep(30)
response={'operation':r['operation'],'request':r['request'],'source_pins_match':True,'process_token':r['process_token'],'parent_pid':os.getppid(),'pid':os.getpid(),'pgid':os.getpgid(0),'result':{'status':'COMPLETE'}}
Path(a.output,'response.json').write_text(json.dumps(response))
''')
        owner.write_text('''import argparse,sys
from pathlib import Path
sys.path.insert(0,'''+repr(str(HERE))+''')
from supervisor import BrokerProvider
p=argparse.ArgumentParser();p.add_argument('--broker-fd',type=int);p.add_argument('--broker-token');p.add_argument('--deadline-epoch',type=float);a=p.parse_args()
b=BrokerProvider(a.broker_fd,a.broker_token,a.deadline_epoch)
o=b.invoke('status',{'versionLabel':'v1'},call_seconds=20)
raise SystemExit(0 if o.status=='SUCCESS' else 2)
''')
        watcher=[]
        def launched(proc,root):
            if not kill_owner:return
            def stop():
                until=time.monotonic()+5
                while time.monotonic()<until:
                    ready=root/'provider/001-status/ready';registration=root/'provider-registry.json'
                    if ready.exists() and registration.exists():
                        os.killpg(proc.pid,signal.SIGKILL);return
                    time.sleep(.01)
                raise AssertionError('FIXTURE_LAUNCH_NOT_READY')
            t=threading.Thread(target=stop);watcher.append(t);t.start()
        deadline=time.time()+(4 if mode=='deadline' else 10)
        report=supervise([sys.executable,'-B',str(owner)],[sys.executable,'-B',str(worker)],self.base/'supervision',deadline,on_owner_launch=launched)
        for t in watcher:t.join(timeout=6);self.assertFalse(t.is_alive())
        self.assertEqual(report['owner_cleanup']['status'],'CLOSED');self.assertTrue(report['owner_cleanup']['reaped']);self.assertTrue(report['owner_cleanup']['group_absent'])
        self.assertEqual(len(report['provider_cleanup']),1)
        row=report['provider_cleanup'][0];self.assertTrue(row['reaped']);self.assertTrue(row['group_absent']);self.assertEqual(row['status'],'CLOSED')
        registry=B.read(self.base/'supervision/provider-registry.json')['rows'][0]
        self.assertEqual(registry['owner_pid'],report['owner_launch']['pid']);self.assertEqual(registry['supervisor_pid'],os.getpid())
        self.assertEqual(registry['pid'],registry['pgid']);self.assertEqual(report['remote_resource_closure'],'UNCONFIRMED')
        with self.assertRaises(ProcessLookupError):os.killpg(registry['pgid'],0)
        return report
    def test_supervisor_success_direct_child_reap(self):
        r=self.supervise_fixture('success');self.assertEqual(r['status'],'OWNER_FINISHED')
    def test_supervisor_owner_sigkill_closes_sdk_worker(self):
        r=self.supervise_fixture('sleep',True);self.assertEqual(r['owner_exit_before_cleanup'],-9)
    def test_supervisor_owner_sigkill_closes_ignored_term_descendant(self):
        r=self.supervise_fixture('descendant',True);self.assertEqual(r['owner_exit_before_cleanup'],-9)
        self.assertIn(signal.SIGKILL,r['provider_cleanup'][0]['signals'])
    def test_supervisor_original_deadline_closes_all_children(self):
        r=self.supervise_fixture('deadline');self.assertEqual(r['status'],'OWNER_FAILED_OR_INTERRUPTED')
    def test_supervisor_leaves_unrelated_owned_fixture_group_alive(self):
        sentinel=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(20)'],start_new_session=True)
        try:
            self.supervise_fixture('sleep',True)
            self.assertIsNone(sentinel.poll());os.killpg(sentinel.pid,0)
        finally:
            try:os.killpg(sentinel.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            sentinel.wait(timeout=2)
        with self.assertRaises(ProcessLookupError):os.killpg(sentinel.pid,0)
    def copied_project(self):
        root=self.base/'relocated-project';root.mkdir()
        for row in C.source_pins()['files'].values():
            target=root/row['project_path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/row['project_path'],target)
        return root
    def test_explicit_relocated_project_reproduces_source_and_output_contract(self):
        source=self.copied_project()
        plan=C.build(self.base/'relocated-candidate','fixture-owner','m8-m2-fixture','M8 M2 Fixture','1'*32,PLAN_DEADLINE,project_source=source,base_archive=OPTIONS.base_archive)
        self.assertEqual(plan['expected_member_paths'],self.plan['expected_member_paths'])
        self.assertEqual(plan['binding'],self.plan['binding'])
        self.assertEqual(plan['wrapper_sha256'],self.plan['wrapper_sha256'])
    def test_wrong_genuine_base_archive_rejected_before_output(self):
        bad=self.base/'wrong.tar';bad.write_bytes(b'not the reviewed archive')
        with self.assertRaisesRegex(ValueError,'GENUINE_BASE_ARCHIVE'):
            C.build(self.base/'invalid','fixture-owner','m8-m2-fixture','M8 M2 Fixture','1'*32,PLAN_DEADLINE,project_source=ROOT,base_archive=bad)
        self.assertFalse((self.base/'invalid').exists())
    def test_changed_explicit_scientific_source_rejected(self):
        source=self.copied_project();path=source/'experiments/2026-10-04-bitsandbytes-tpu/probe-inputs.json';path.write_bytes(path.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'REVIEWED_PROJECT_SOURCE'):
            C.build(self.base/'invalid','fixture-owner','m8-m2-fixture','M8 M2 Fixture','1'*32,PLAN_DEADLINE,project_source=source,base_archive=OPTIONS.base_archive)
        self.assertFalse((self.base/'invalid').exists())
    def test_unreviewed_payload_pin_map_rejected(self):
        directory=self.base/'bad-pin-source';directory.mkdir();(directory/'payload-source-pins.json').write_text('{}')
        with patch.object(C,'HERE',directory):
            with self.assertRaisesRegex(ValueError,'REVIEWED_PAYLOAD_SOURCE_PINS'):C.source_pins()
    def sdk_source_check(self,wrong=False):
        directory=self.base/'sdk-pin-input';directory.mkdir()
        pins=B.read(HERE/'installed-source-pins.json')
        if wrong:pins[next(iter(pins))]='0'*64
        B.write(directory/'installed-source-pins.json',pins)
        code='''import importlib.util,json,os,sys
from pathlib import Path
attempts={'network':0,'credentials':0}
def guard(event,args):
    if event in ('socket.connect','socket.getaddrinfo'):attempts['network']+=1;raise RuntimeError('NO_NETWORK')
    if event=='open' and isinstance(args[0],(str,bytes)) and '/.kaggle/' in os.fsdecode(args[0]):attempts['credentials']+=1;raise RuntimeError('NO_CREDENTIALS')
sys.addaudithook(guard)
spec=importlib.util.spec_from_file_location('sdk_source_only',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.HERE=Path(sys.argv[2])
try:m.assert_sources();status='PASS'
except ValueError as e:status=str(e)
print(json.dumps({'status':status,'attempts':attempts,'kaggle_module_imported':'kaggle' in sys.modules}))
'''
        result=subprocess.run([str(OPTIONS.kaggle_python),'-B','-c',code,str(HERE/'sdk_worker.py'),str(directory)],capture_output=True,text=True,timeout=10,check=True)
        value=json.loads(result.stdout);B.write(self.base/'sdk-source-check.json',value)
        self.assertEqual(value['attempts'],{'network':0,'credentials':0});self.assertFalse(value['kaggle_module_imported'])
        self.assertEqual(value['status'],'INSTALLED_SOURCE_CHANGED' if wrong else 'PASS')
    def test_sdk_source_location_from_supplied_python_without_auth_import(self):self.sdk_source_check()
    def test_wrong_sdk_source_hash_rejected_without_auth_import(self):self.sdk_source_check(True)
    def test_terminal_status_does_not_prove_closure(self):
        _,files,_=self.bundle();_,_,r=self.owner(files);self.assertEqual(r['terminal_status'],'COMPLETE');self.assertEqual(r['closure'],'UNCONFIRMED');self.assertEqual(r['m8_status'],'NOT_QUALIFIED')

suite=unittest.defaultTestLoader.loadTestsFromTestCase(Controls);start=time.time()
with (OUT/'unittest.stderr.raw').open('w') as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
report={'kind':'M8_M2_OFFLINE_INTEGRATION_CONTROLS','status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'seconds':time.time()-start,'network_attempts':attempts['network'],'credential_read_attempts':attempts['credentials'],'provider_calls':'SYNTHETIC_ONLY','actual_cpu':'NOT_RUN','actual_tpu':'NOT_RUN','m8_status':'NOT_QUALIFIED'}
B.write(OUT/'controls.json',report);print(json.dumps(report));raise SystemExit(0 if result.wasSuccessful() and not any(attempts.values()) else 1)
