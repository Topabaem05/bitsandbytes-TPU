"""V5E8-only request controls. No account or device calls."""
import argparse,copy,hashlib,importlib.util,json,os,socket,sys,time,unittest
from pathlib import Path
from types import SimpleNamespace
P=argparse.ArgumentParser()
P.add_argument('--host-source',type=Path,default=Path(__file__).resolve().parent)
P.add_argument('--baseline-host-source',type=Path,required=True)
P.add_argument('--project-source',type=Path,required=True)
P.add_argument('--base-archive',type=Path,required=True)
P.add_argument('--kaggle-python',required=True)
P.add_argument('--output',type=Path,required=True)
A=P.parse_args();A.output.mkdir(parents=True,exist_ok=False)
ROOT=Path(__file__).resolve().parent
HOST=A.host_source.resolve()
BASE=A.baseline_host_source.resolve()
sys.path.insert(0,str(HOST));attempts={'network':0,'credentials':0}
def guard(event,args):
    if event in ('socket.connect','socket.getaddrinfo'):
        attempts['network']+=1;raise AssertionError('NETWORK_FORBIDDEN')
    if event=='open' and isinstance(args[0],(str,bytes)) and any(s in os.fsdecode(args[0]) for s in ('/.kaggle/','access_token','credentials.json')):
        attempts['credentials']+=1;raise AssertionError('CREDENTIAL_READ_FORBIDDEN')
sys.addaudithook(guard)
def load(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

BASELINE_HASHES={'build_candidate.py': '14130e56ddd7f4874ef54a1918f2f0ca3ec01e66a7755b12cd87b27afe3c844d', 'runner.py': '503c3ebc3691c14d34fb151b0944617122fc658997d5ed317ae932312ab075ad', 'batch.py': '30af975218b7283d76a9129d76336671c7100f3bf12a78fd7a2256501db28305', 'wrapper_template.py': '3b70c23597cd83f9f7016d1336c098d03343753ef7d195ca51b871cac785e3cd', 'payload-source-pins.json': 'f7b39c3b1647d05a7d48f10c166c723a021fd7780c0fcc14366a5e98cd32f617'}
for name,digest in BASELINE_HASHES.items():
    path=BASE/name
    if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('BASELINE_HOST_SOURCE_CHANGED:'+name)

TESTED_HOST_HASHES={'build_candidate.py': '7fc7fc5aa5ff0fb5893d5a9f59c1bbc8b58b774e15d0a3e33db5147e039fa1d3', 'runner.py': 'b77029fa8b6cf191e3e0cecf70a30d431abce93d05e28bd7cddb3570953e6e45', 'sdk_worker.py': 'c33fc21a6b2405edb56486cfcc8881e683432bb0b2bd686ed66ab2c4eb49541b', 'batch.py': '30af975218b7283d76a9129d76336671c7100f3bf12a78fd7a2256501db28305', 'wrapper_template.py': '3b70c23597cd83f9f7016d1336c098d03343753ef7d195ca51b871cac785e3cd', 'payload-source-pins.json': 'f7b39c3b1647d05a7d48f10c166c723a021fd7780c0fcc14366a5e98cd32f617', 'http_error_retention.py': '5d57d9bf2ed6538b629c46a67996572c6caf28ca9019b9b1a9716cc8491845e0', 'process_provider.py': 'b53cf34ab127fa17c250e89ff966226a1c0087e3830372dcbf57d1d4bf608c84', 'installed-source-pins.json': '2a4b414f644bd8920deb6ab89f03b86aa9bb81e381d258b1de7cb5b4a92304b2'}
for name,digest in TESTED_HOST_HASHES.items():
    path=HOST/name
    if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('TESTED_HOST_SOURCE_CHANGED:'+name)

import runner as R
from process_provider import ProcessProvider,Outcome
new=load(HOST/'build_candidate.py','new_hardware_builder')
old=load(BASE/'build_candidate.py','old_hardware_builder')
# Bind the baseline builder to its original reviewed validator, not the V5 module cache.
old.runner=load(BASE/'runner.py','old_hardware_runner')
args=('fixture-owner','offline-v5-review','Offline V5 Review','d'*32,time.time()+3600)
kw={'project_source':A.project_source,'base_archive':A.base_archive}
original=old.build(A.output/'baseline-candidate',*args,**kw)
plan=new.build(A.output/'candidate',*args,**kw)

class Provider:
    def __init__(self,root,shape):
        self.root=root;root.mkdir();self.shape=shape;self.deadline_epoch=plan['deadline_epoch'];self.calls=[]
    def invoke(self,op,request,call_seconds):
        self.calls.append(op);d=self.root/str(len(self.calls));d.mkdir();pid=20000+len(self.calls)
        launch={'reaped':True,'group_absent':True,'process_token':'a'*32,'parent_pid':os.getpid(),'pgid':pid,'launched_pid':pid}
        if op=='status':return Outcome('TIMEOUT',None,launch,d)
        result={'ref':'fixture-owner/offline-v5-review','kernel_id':123,'version_number':1,'error':''} if op=='submit' else {'ref':'fixture-owner/offline-v5-review','kernel_id':123,'current_version_number':1,'is_private':True,'language':'python','kernel_type':'script','source_sha256':plan['wrapper_sha256']}
        if op=='source' and self.shape is not None:result['machine_shape']=self.shape
        response={'operation':op,'request':dict(request),'result':result,'source_pins_match':True,'process_token':'a'*32,'parent_pid':os.getpid(),'pid':pid,'pgid':pid}
        return Outcome('SUCCESS',response,launch,d)

class Controls(unittest.TestCase):
    def directory(self):
        d=A.output/self._testMethodName;d.mkdir();return d
    def test_exact_science_and_wrapper_unchanged(self):
        for name in ['payload.zip','submission/wrapper.py']:
            self.assertEqual((A.output/'baseline-candidate'/name).read_bytes(),(A.output/'candidate'/name).read_bytes())
        for key in original:
            if key not in ('machine_shape','folder'):self.assertEqual(original[key],plan[key])
        self.assertEqual(plan['machine_shape'],'TpuV5E8')
        self.assertEqual(plan['binding']['packet_sha256'],'654641d70a43bb1154097301b8ba41dfe5d2f95d16a2dd0e01c181e69eead80a')
        self.assertEqual(len(plan['expected_member_paths']),298)
        self.assertEqual((BASE/'wrapper_template.py').read_bytes(),(HOST/'wrapper_template.py').read_bytes())
    def test_v6_plan_rejected(self):
        p=copy.deepcopy(plan);p['machine_shape']='TpuV6E8'
        with self.assertRaisesRegex(ValueError,'UNREVIEWED_RUNTIME_REQUEST'):R.validate_plan(p)
    def test_unknown_plan_rejected(self):
        p=copy.deepcopy(plan);p['machine_shape']='TPU v5e-8'
        with self.assertRaisesRegex(ValueError,'UNREVIEWED_RUNTIME_REQUEST'):R.validate_plan(p)
    def observe(self,shape):
        d=self.directory();provider=Provider(d/'provider',shape)
        owner=R.Runner(plan,provider,d/'owner',lambda *_:self.fail('science must not run'),sleep=lambda *_:None)
        return owner.run(),provider.calls
    def test_exact_planned_shape_readback_reaches_status(self):
        result,calls=self.observe('TpuV5E8')
        self.assertEqual(calls,['submit','source','status'])
        self.assertEqual(result['source_readback']['machine_shape'],'TpuV5E8')
        self.assertEqual(result['state'],'INCOMPLETE_REMOTE_UNCONFIRMED')
        self.assertEqual(result['m8_status'],'NOT_QUALIFIED')
    def test_wrong_shape_readback_rejected(self):
        result,calls=self.observe('TpuV6E8')
        self.assertEqual(calls,['submit','source']);self.assertEqual(result['state'],'INVALID_RECORDS')
        self.assertEqual(result['error_code'],'VERSION_SOURCE_READBACK')
    def test_missing_shape_readback_rejected(self):
        result,calls=self.observe(None)
        self.assertEqual(calls,['submit','source']);self.assertEqual(result['state'],'INVALID_RECORDS')
    def test_ui_label_readback_rejected(self):
        result,calls=self.observe('TPU v5e-8')
        self.assertEqual(calls,['submit','source']);self.assertEqual(result['state'],'INVALID_RECORDS')
    def test_v6_worker_request_rejected_before_auth(self):
        w=load(HOST/'sdk_worker.py','wrong_acc_worker');w.assert_sources=lambda:None
        with self.assertRaisesRegex(ValueError,'SAVE_REQUEST'):
            w.run({'operation':'submit','deadline_epoch':time.time()+60,'request':{'folder':plan['folder'],'timeout':'3600','acc':'TpuV6E8','wrapper_sha256':plan['wrapper_sha256'],'metadata_sha256':R.digest((Path(plan['folder'])/'kernel-metadata.json').read_bytes())}},self.directory())
    def test_original_cli_sdk_wire_requests_exact_v5_once(self):
        d=self.directory();folder=Path(plan['folder']);(folder/'fixture-mode.json').write_text(json.dumps({'mode':'good','host_source':str(HOST)})+'\n')
        p=ProcessProvider([A.kaggle_python,'-B',str(ROOT/'hardware_fixture_worker.py')],d/'provider',plan['deadline_epoch'])
        request={'folder':plan['folder'],'timeout':'3600','acc':'TpuV5E8','wrapper_sha256':plan['wrapper_sha256'],'metadata_sha256':R.digest((folder/'kernel-metadata.json').read_bytes())}
        outcome=p.invoke('submit',request,call_seconds=10)
        self.assertEqual(outcome.status,'SUCCESS')
        self.assertTrue(outcome.launch['reaped']);self.assertTrue(outcome.launch['group_absent'])
        wire=json.loads((outcome.directory/'wire-request.sanitized.json').read_text())
        self.assertEqual(wire,{'machineShape':'TpuV5E8','sessionTimeoutSeconds':3600,'isPrivate':True,'text_sha256':plan['wrapper_sha256']})
        with self.assertRaisesRegex(ValueError,'SUBMIT_REPLAY_FORBIDDEN'):p.invoke('submit',request,call_seconds=10)

result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
report={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'network_attempts':attempts['network'],'credential_read_attempts':attempts['credentials'],'provider_calls':'FAKE_ONLY','packet_sha256':plan['binding']['packet_sha256'],'wrapper_sha256':plan['wrapper_sha256'],'machine_shape':'TpuV5E8_ONLY','m8_status':'NOT_QUALIFIED'}
(A.output/'controls.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report))
raise SystemExit(0 if result.wasSuccessful() and not any(attempts.values()) else 1)
