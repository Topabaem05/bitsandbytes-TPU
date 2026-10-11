"""Execute real driver and remote stage branches with an isolated fake OfficialCLI.

All receipts, source reviews, acceptances and audit results here are synthetic protocol
fixtures. No science verifier is replaced in a real run. No provider call is made.
"""
import argparse,copy,json,shutil,sys,tempfile,time,types,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import contract as C,driver as D,remote as R,build_packet as B,transport
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);reports=[]

def acceptance(packet_sha,records):
 return {'kind':'ROOT_NESTED_EXACT79_DEVICE_ACCEPTANCE','status':'ACCEPTED_NESTED_EXACT79_DEVICE_RECORDS','record_validation':'PASS','numerical_status':'PASS','case_count':79,'source_variant':'nested-exact-v1','source_admission_sha256':C.ADMISSION_SHA,'integration_map_sha256':C.MAP_SHA,'science_map_sha256':C.SCIENCE_SHA,'inputs_sha256':'b7490f4c622aef793d14bd9a7a38727754cb1fb8e6e1dec7ff91ef8c45fbafbc','profile_sha256':'ad53f6416bdabf6440f08b313963c947a468cd39afd413bfe41bb381c4a7a166','runtime_lock_sha256':'323371ff61c5fbcc4f79fd6a358cf2ba17cb72b907382a5ceaac07f91dc66ed6','saved_state_status':'NOT_RUN','m4_status':'NOT_QUALIFIED','packet_sha256':packet_sha,'actual_receipt_sha256':C.sha(records/'device79/receipt.json'),'fresh_cpu_oracle_sha256':C.sha(records/'cpu79/oracle-seal.json'),'fixture_only':True}

def run_case(scenario,packet,packet_zip,packet_sha):
 with tempfile.TemporaryDirectory(prefix='driver-'+scenario+'-',dir=a.output)as temp:
  root=Path(temp);out=root/'host';server=root/'server';server.mkdir();identity=root/'fixture-cli-identity.json';C.write(identity,{'fixture_only':True});dispatch=root/'fixture-dispatch.json';C.write(dispatch,{'kind':'ROOT_M4_EXACT_NESTED_DISPATCH','status':'ACTUAL_DISPATCH_AUTHORIZED','packet_sha256':packet_sha,'driver_sha256':C.sha(HERE/'driver.py'),'cli_identity_sha256':C.sha(identity),'allocation_seconds':3600,'allocation_attempts':1,'science_manifest_sha256':C.SCIENCE_SHA,'fixture_only':True});accepted=root/'external-root-acceptance.json';events=[];science=[];pause=[];deadlines=[];clock=types.SimpleNamespace(value=time.time())
  def now():return clock.value
  def sleep(seconds):
   clock.value+=seconds
   if(out/'root-review-request.json').exists()and not pause:
    pause.append({'before_state_cpu':not any(e=='science:cpustate'for e in events),'at_original_birth':C.read(out/'driver.json')['allocation_epoch']})
    if scenario=='missing_acceptance':clock.value+=3000;return
    records=server/'records';r=acceptance(packet_sha,records)
    if scenario in('wrong_packet','old_packet'):r['packet_sha256']='e'*64
    if scenario=='wrong_numerics':r['numerical_status']='FAIL'
    if scenario=='reset_deadline':clock.value+=3000;r['original_deadline_epoch']=clock.value+3600;r['allocation_epoch']=clock.value
    C.write(accepted,r);events.append('external-root-writer:accepted79')
  fakeclock=types.SimpleNamespace(time=now,sleep=sleep)
  class FakeOfficialCLI:
   def __init__(self,python,output):self.stopped=False;self.allocated=False
   def __call__(self,label,argv,timeout):
    rec={'status':'PASS','exit_code':0,'cleanup':{'errors':[],'groups':[]}};text='';events.append('cli:'+argv[0])
    if argv[0]=='sessions':text='No active sessions found on server.'if not self.allocated or self.stopped else'[fixture-session] fixture-endpoint | Hardware: V6E1 | Variant: TPU'
    elif argv[0]=='usage':text='Active assignments: 0\nUsage rate: 0.00/hr\n'if not self.allocated or self.stopped else'Active assignments: 1\nUsage rate: 2.00/hr\n'
    elif argv[0]=='--allocation-transport-v1':self.allocated=True
    elif argv[0]=='stop':self.stopped=True
    elif argv[0]=='upload':
     src=Path(argv[-2]);relative=argv[-1].split('content/m4-exact-fixture-session/',1)[1];dest=server/relative;dest.parent.mkdir(parents=True,exist_ok=True)
     if not(scenario=='missing_state_audit'and relative=='records/state-cpu-launch-audit.json'):shutil.copyfile(src,dest)
     else:events.append('fixture-dropped-state-cpu-audit')
    elif argv[0]=='download':
     relative=argv[-2].split('content/m4-exact-fixture-session/',1)[1];shutil.copyfile(server/relative,argv[-1])
    elif argv[0]=='exec':
     source=Path(argv[-1]);stem=source.stem
     if stem=='readiness':(server/'control/ownership').mkdir(parents=True)
     elif stem.startswith('remote-'):
      phase=stem.removeprefix('remote-');events.append('remote:'+phase)
      if phase=='install':pass
      else:
       args=types.SimpleNamespace(command=phase,base=server,allocation_epoch=C.read(out/'driver.json')['allocation_epoch'],packet_sha256=packet_sha)
       with patch.object(R,'run_step',science_step):R.operation(args)
      if scenario=='operation_failure'and phase=='device79':rec.update(status='CLI_FAILED',exit_code=1)
     elif stem.startswith('export-'):
      phase=stem.removeprefix('export-');label2='recovery-'+phase;events.append('recover:'+phase);summary=C.export(server/'records',server/(label2+'.zip'));C.write(server/(label2+'-export.json'),summary);transport.split(server/(label2+'.zip'),server/(label2+'-parts'),expected_sha256=summary['sha256'],expected_bytes=summary['bytes'])
     elif stem.startswith('dispose-'):
      label2='recovery-'+stem.removeprefix('dispose-');(server/(label2+'.zip')).unlink();shutil.rmtree(server/(label2+'-parts'));(server/(label2+'-export.json')).unlink()
     else:raise ValueError('UNEXPECTED_FAKE_KERNEL:'+stem)
    else:raise ValueError('UNEXPECTED_FAKE_CLI:'+str(argv))
    if scenario=='usage_not_zero'and self.stopped and argv[0]=='usage':text='Active assignments: 0\nUsage rate: 0.10/hr\n'
    return rec,text
  def science_step(records,label,argv,deadline,limit,**kw):
   # Run only the real stage's argument and dependency branches. Never run fake science.
   science.append(label);events.append('science:'+label);deadlines.append(deadline);output=Path(argv[argv.index('--output')+1]);output.mkdir();C.write(output/('oracle-seal.json'if label.startswith('cpu')else'receipt.json'),{'fixture_only':True,'finished_epoch':now(),'deadline_epoch':deadline,'process_token':argv[argv.index('--process-token')+1]});return {'status':'PASS','exit_code':0}
  def host_audit_fixture(records,label,argv,deadline,limit,**kw):
   phase=label.removeprefix('host-');output=Path(argv[argv.index('--output')+1]);r={'fixture_only':True,'status':'QUALIFIED_LINUX_CPU_ORACLE_VERIFIED'if phase=='cpu79'else'ACTUAL_STATE_CPU_LAUNCH_VERIFIED'if phase=='cpustate'else'FIXTURE_ONLY','record_validation':'PASS','numerical_status':'PASS','case_count':79 if phase=='device79'else 8}
   if phase=='cpustate':r['state_oracle_sha256']=C.sha(server/'records/cpustate/oracle-seal.json')
   C.write(output,r);return {'status':'PASS','exit_code':0}
  args=types.SimpleNamespace(packet=packet,packet_zip=packet_zip,packet_sha256=packet_sha,cli_python=Path(sys.executable),cli_identity=identity,dispatch=dispatch,session='fixture-session',browser_adoption=None,browser_adoption_sha256=None,output=out,accept79_file=accepted,stop_after_79=False)
  error=None
  with patch.object(D.T,'verify_cli_identity',lambda *a:None),patch.object(D.T,'OfficialCLI',FakeOfficialCLI),patch.object(D,'time',fakeclock),patch.object(C,'time',fakeclock),patch.object(D,'run_step',host_audit_fixture):
   try:D.run(args)
   except BaseException as e:error={'type':type(e).__name__,'message':str(e)}
  record=C.read(out/'driver.json');result={'scenario':scenario,'driver_status':record['status'],'error':error,'science':science,'events':events,'pause':pause,'deadlines':deadlines,'birth':record['allocation_epoch'],'server_empty_observed':record.get('server_empty_observed'),'usage_zero_observed':record.get('usage_zero_observed'),'last_recovered_envelope':record.get('last_recovered_envelope'),'scope':'ACTUAL_DRIVER_REMOTE_BRANCHES_WITH_SYNTHETIC_PROTOCOL_AND_AUDIT_FIXTURES_NO_SCIENCE','provider_calls':0}
  assert all(d==result['birth']+3600-C.CLOSURE_RESERVE for d in deadlines)
  assert record['server_empty_observed']is True
  assert 'cli:stop'in events
  discarded=C.inventory(root);result['disposable_bytes_before']=sum(v['bytes']for v in discarded.values());result['disposable_files_before']=len(discarded);result['fixture_root']=str(root)
  return result

class Controls(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.inputs=tempfile.TemporaryDirectory(prefix='driver-shared-packet-',dir=a.output);root=Path(cls.inputs.name);review=root/'synthetic-source-review.json';C.write(review,{'kind':'ROOT_NESTED_EXACT_SOURCE_REVIEW','status':'SOURCE_REVIEW_ACCEPTED_FOR_EXPERIMENT','integration_map_sha256':C.MAP_SHA,'source_admission_sha256':C.ADMISSION_SHA,'science_map_sha256':C.SCIENCE_SHA,'device_acceptance':None,'fixture_only':True});cls.packet=root/'packet';r=B.build(HERE.parents[1],review,C.sha(review),cls.packet);cls.archive=Path(r['archive']);cls.sha=r['packet_sha256']
 @classmethod
 def tearDownClass(cls):
  shared=C.inventory(Path(cls.inputs.name));cls.shared_disposed={'bytes':sum(v['bytes']for v in shared.values()),'files':len(shared)};cls.inputs.cleanup()
 def case(self,name):r=run_case(name,self.packet,self.archive,self.sha);r['fixture_root_absent']=not Path(r['fixture_root']).exists();assert r['fixture_root_absent'];reports.append(r);return r
 def test_01_correct_flow_waits_external_root_then_independent_state_cpu_audit(self):
  r=self.case('correct');self.assertIsNone(r['error']);self.assertEqual(r['science'],['cpu79','device79','cpustate','devicestate']);self.assertTrue(r['pause'][0]['before_state_cpu']);self.assertLess(r['events'].index('external-root-writer:accepted79'),r['events'].index('science:cpustate'));self.assertTrue(r['usage_zero_observed']);self.assertEqual(r['driver_status'],'STATE8_RECOVERED_ROOT_REVIEW_REQUIRED')
 def test_02_missing_wrong_old_packet_wrong_numerics_and_reset_deadline_reject_before_state(self):
  for name in('missing_acceptance','wrong_packet','old_packet','wrong_numerics','reset_deadline'):
   with self.subTest(name=name):
    r=self.case(name);self.assertIsNotNone(r['error']);self.assertEqual(r['science'],['cpu79','device79']);self.assertTrue(r['pause'][0]['before_state_cpu']);self.assertTrue(r['usage_zero_observed'])
 def test_03_independent_state_cpu_audit_required_before_device_child(self):
  r=self.case('missing_state_audit');self.assertIsNotNone(r['error']);self.assertEqual(r['science'],['cpu79','device79','cpustate']);self.assertIn('recover:devicestate',r['events']);self.assertTrue(r['usage_zero_observed'])
 def test_04_operation_failure_recovers_before_exact_closure(self):
  r=self.case('operation_failure');self.assertIsNotNone(r['error']);self.assertIn('recover:device79',r['events']);self.assertLess(r['events'].index('recover:device79'),r['events'].index('cli:stop'));self.assertTrue(r['usage_zero_observed'])
 def test_05_cli_success_is_not_zero_usage_proof(self):
  r=self.case('usage_not_zero');self.assertIsNone(r['error']);self.assertFalse(r['usage_zero_observed']);self.assertEqual(r['driver_status'],'BLOCKED_CLEANUP')
  for text in('Active assignments: 01\nUsage rate: 0.00/hr','Active assignments: 0\nUsage rate: 0.10/hr','Active assignments: 0\nUsage rate: 0.00/hr\nUsage rate: 2.00/hr','command exit: 0'):
   self.assertFalse(D.usage_zero(text)['zero_observed'])
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));C.write(a.output/'review.json',{'status':'PASS'if result.wasSuccessful()else'FAIL','tests_run':result.testsRun,'scenario_count':len(reports),'shared_packet_disposed':Controls.shared_disposed,'disposable_bytes_before':sum(r['disposable_bytes_before']for r in reports)+Controls.shared_disposed['bytes'],'disposable_files_before':sum(r['disposable_files_before']for r in reports)+Controls.shared_disposed['files'],'failures':len(result.failures),'errors':len(result.errors),'controls':reports,'disposable_fixture_roots_absent':not any(p.is_dir()for p in a.output.iterdir()),'TPU':'NOT_RUN','provider_calls':0,'m4_status':'NOT_QUALIFIED'});raise SystemExit(0 if result.wasSuccessful()else 1)
