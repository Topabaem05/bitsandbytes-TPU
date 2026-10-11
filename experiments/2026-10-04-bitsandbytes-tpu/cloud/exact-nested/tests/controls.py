"""Isolated integration controls. Synthetic identity fixtures do not qualify science."""
import argparse,copy,hashlib,json,os,shutil,sys,tempfile,time,unittest,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import contract as C,build_packet as B,remote as R,driver as D
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False);summaries=[]
class Controls(unittest.TestCase):
 def test_01_packet_complete_frozen_science_source_and_wrong_member(self):
  source=HERE.parents[1];review=a.output/'fixture-source-review.json';C.write(review,{'kind':'ROOT_NESTED_EXACT_SOURCE_REVIEW','status':'SOURCE_REVIEW_ACCEPTED_FOR_EXPERIMENT','integration_map_sha256':C.MAP_SHA,'source_admission_sha256':C.ADMISSION_SHA,'science_map_sha256':C.SCIENCE_SHA,'device_acceptance':None,'fixture_only':True})
  packet=a.output/'packet';result=B.build(source,review,C.sha(review),packet);C.recover(result['archive'],result['packet_sha256'],a.output/'recovered',packet=True);C.packet_preflight(packet,result['archive'],result['packet_sha256'])
  sourcefile=packet/'nested-exact/integration/upstream/bitsandbytes/functional.py';original=sourcefile.read_bytes();sourcefile.write_bytes(original+b'\n')
  with self.assertRaisesRegex(ValueError,'EXACT_PHYSICAL_INVENTORY'):C.admit(packet)
  sourcefile.write_bytes(original);C.admit(packet);manifest=C.read(packet/'manifest.json');changed=copy.deepcopy(manifest);changed['diagnostic_fixture_only']='changed loose manifest';C.write(packet/'manifest.json',changed)
  with self.assertRaisesRegex(ValueError,'LOOSE_PACKET_DIFFERS_FROM_ADMITTED_ARCHIVE'):C.packet_preflight(packet,result['archive'],result['packet_sha256'])
  C.write(packet/'manifest.json',manifest);C.packet_preflight(packet,result['archive'],result['packet_sha256']);C.require(C.sha(packet/'nested-exact/source/science-manifest.json')==C.SCIENCE_SHA,'SCIENCE_STABLE')
  summaries.append({'control':'COMPLETE_PACKET_FROZEN_19_SOURCES_PLUS48_AND11_CORRECT_WRONG','status':'PASS','packet_sha256':result['packet_sha256'],'packet_bytes':result['packet_bytes'],'scope':'OFFLINE_SOURCE_ONLY'})
 def test_02_original_deadline_no_reset_and_state_reserve(self):
  self.assertEqual(C.budget(100,500,120,now=3200),380)
  for birth,now in[(100,3581),(float('nan'),100),(100,99)]:
   with self.assertRaises((ValueError,TimeoutError)):C.budget(birth,500,120,now=now)
  self.assertEqual(C.budget(100,1,C.STATE_RESERVE,now=2919),1)
  with self.assertRaises(TimeoutError):C.budget(100,1,C.STATE_RESERVE,now=2920)
  summaries.append({'control':'ORIGINAL_BIRTH_DEADLINE_CLOSURE_STATE_RESERVE_FINITE_CORRECT_WRONG','status':'PASS'})
 def test_03_full_physical_envelope_archive_recovery_corruption_and_traversal(self):
  root=a.output/'records';root.mkdir();(root/'receipt.json').write_text('{"fixture_only":true}\n');(root/'exact.bin').write_bytes(bytes(range(256))*1000);export=C.export(root,a.output/'evidence.zip');C.recover(a.output/'evidence.zip',export['sha256'],a.output/'envelope-recovered');self.assertEqual((root/'exact.bin').read_bytes(),(a.output/'envelope-recovered/exact.bin').read_bytes())
  zipbad=a.output/'traversal.zip'
  with zipfile.ZipFile(zipbad,'w')as z:z.writestr('../escape','x')
  with self.assertRaisesRegex(ValueError,'CONFINED_NAME'):C.recover(zipbad,C.sha(zipbad),a.output/'bad-recovery')
  zipwrong=a.output/'wrong.zip'
  with zipfile.ZipFile(zipwrong,'w')as z:z.writestr('archive-inventory.json',json.dumps({'files':{'receipt.json':{'bytes':1,'sha256':'0'*64}}}));z.writestr('receipt.json','y')
  with self.assertRaisesRegex(ValueError,'EXACT_PHYSICAL_INVENTORY'):C.recover(zipwrong,C.sha(zipwrong),a.output/'wrong-recovery')
  self.assertFalse((a.output/'wrong-recovery').exists());summaries.append({'control':'WHOLE_ENVELOPE_SHA_MEMBERS_RECOVERY_HASH_PATH_FAILURE_DISPOSAL','status':'PASS',**export})
 def test_04_original_physical_member_and_byte_cap_boundaries(self):
  correct={f'f{i}':{'bytes':0,'sha256':'0'*64}for i in range(2000)};C.limits(correct);wrong=dict(correct,f2000={'bytes':0,'sha256':'0'*64})
  with self.assertRaisesRegex(ValueError,'MEMBER_LIMIT'):C.limits(wrong)
  C.limits({'f':{'bytes':96*1024*1024,'sha256':'0'*64}})
  with self.assertRaisesRegex(ValueError,'PHYSICAL_BYTES_LIMIT'):C.limits({'f':{'bytes':96*1024*1024+1,'sha256':'0'*64}})
  summaries.append({'control':'EXACT_96MIB_INNER_100MIB_OUTER_2000_MEMBERS_CORRECT_WRONG','status':'PASS'})
 def test_05_owned_correct_and_incorrect_child_closure(self):
  root=a.output/'owned';root.mkdir();good=R.run_step(root,'good',[sys.executable,'-B','-c','print("owned fixture")'],time.time()+20,10,cwd=root);bad=R.run_step(root,'wrong',[sys.executable,'-B','-c','raise SystemExit(3)'],time.time()+20,10,cwd=root)
  self.assertEqual(good['status'],'PASS');self.assertEqual(bad['status'],'CHILD_FAILED');self.assertEqual(bad['exit_code'],3)
  for phase in('good','wrong'):
   ownership=C.read(root/'steps'/phase/'ownership.json');self.assertTrue(ownership['cleanup']['leader_reaped']);self.assertEqual(ownership['cleanup']['group_absence'],'OBSERVED_NO_SUCH_GROUP');self.assertEqual(ownership['cleanup']['errors'],[])
  self.assertEqual(good['environment']['CPU'],R.CPU_ENV);summaries.append({'control':'OWNED_REAL_CHILD_CORRECT_WRONG_EXIT_REAP_GROUP_ABSENCE_AVX2_SINGLETHREAD_ENV','status':'PASS','ownership_sha256':C.sha(root/'steps/good/ownership.json')})
 def test_06_dispatch_admission_correct_wrong_before_provider(self):
  identity=a.output/'fixture-cli-identity.json';C.write(identity,{'fixture_only':True});r={'kind':'ROOT_M4_EXACT_NESTED_DISPATCH','status':'ACTUAL_DISPATCH_AUTHORIZED','packet_sha256':'a'*64,'driver_sha256':C.sha(HERE/'driver.py'),'cli_identity_sha256':C.sha(identity),'allocation_seconds':3600,'allocation_attempts':1,'science_manifest_sha256':C.SCIENCE_SHA};D.gate('a'*64,identity,r)
  for k,v in [('packet_sha256','b'*64),('allocation_attempts',2),('allocation_seconds',3601),('science_manifest_sha256','0'*64)]:
   with self.assertRaisesRegex(ValueError,'ROOT_DISPATCH_GATE'):D.gate('a'*64,identity,dict(r,**{k:v}))
  summaries.append({'control':'ROOT_DISPATCH_GATE_CORRECT_WRONG_NO_PROVIDER_CALLS','status':'PASS'})
 def test_07_frozen_accepted79_schema_correct_wrong_real_boundary_limits(self):
  sys.path.insert(0,str(HERE.parents[1]/'nested-exact/source'));import exact_admission as A
  rec={'kind':'ROOT_NESTED_EXACT79_DEVICE_ACCEPTANCE','status':'ACCEPTED_NESTED_EXACT79_DEVICE_RECORDS','record_validation':'PASS','numerical_status':'PASS','case_count':79,'source_variant':A.VARIANT,'source_admission_sha256':C.ADMISSION_SHA,'integration_map_sha256':C.MAP_SHA,'science_map_sha256':C.SCIENCE_SHA,'inputs_sha256':'b7490f4c622aef793d14bd9a7a38727754cb1fb8e6e1dec7ff91ef8c45fbafbc','profile_sha256':'ad53f6416bdabf6440f08b313963c947a468cd39afd413bfe41bb381c4a7a166','runtime_lock_sha256':'323371ff61c5fbcc4f79fd6a358cf2ba17cb72b907382a5ceaac07f91dc66ed6','saved_state_status':'NOT_RUN','m4_status':'NOT_QUALIFIED','packet_sha256':'a'*64,'actual_receipt_sha256':'b'*64,'fresh_cpu_oracle_sha256':'c'*64,'fixture_only':True};p=a.output/'synthetic-accepted79-never-dispatch.json';C.write(p,rec)
  C.accepted79(p,C.sha(p),packet_sha='a'*64,receipt_sha='b'*64,oracle_sha='c'*64,completed_epoch=200,birth=100,now=1000)
  for changed in[dict(rec,numerical_status='FAIL'),dict(rec,case_count=78),dict(rec,packet_sha256='d'*64)]:
   C.write(p,changed)
   with self.assertRaises(ValueError):C.accepted79(p,C.sha(p),packet_sha='a'*64,receipt_sha='b'*64,oracle_sha='c'*64,completed_epoch=200,birth=100,now=1000)
  C.write(p,rec)
  with self.assertRaises(TimeoutError):C.accepted79(p,C.sha(p),packet_sha='a'*64,receipt_sha='b'*64,oracle_sha='c'*64,completed_epoch=200,birth=100,now=2920)
  summaries.append({'control':'FROZEN79_ACCEPTANCE_SCHEMA_PROVENANCE_COMPLETION_STATE_RESERVE_CORRECT_WRONG','status':'PASS','scope':'SYNTHETIC_SCHEMA_FIXTURE_NOT_SCIENCE_OR_AUTHORIZATION'})
 def test_08_archive_metadata_counts_inside_cap_and_failure_cleans(self):
  from unittest.mock import patch
  root=a.output/'overhead-cap';root.mkdir()
  for name in('a','b','c'):(root/name).write_text(name)
  with patch.object(C,'MAX_FILES',3):
   with self.assertRaisesRegex(ValueError,'MEMBER_LIMIT'):C.export(root,a.output/'should-not-exist.zip')
  self.assertFalse((root/'archive-inventory.json').exists());self.assertFalse((a.output/'should-not-exist.zip').exists())
  summaries.append({'control':'COMPLETE_ENVELOPE_METADATA_COUNTS_AND_FAILED_EXPORT_CLEANS','status':'PASS','scope':'ISOLATED_REDUCED_MEMBER_CAP_FIXTURE'})
 def test_09_materializer_additive_idempotent_conflict_rejection(self):
  import importlib.util
  source=HERE.parents[3];path=HERE.parents[1]/'nested-exact/materialize.py';spec=importlib.util.spec_from_file_location('m4_fixture_materializer',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  manifest=a.output/'adoption-manifest.json';C.write(manifest,{'files':C.inventory(source),'science_manifest_sha256':C.SCIENCE_SHA});destination=a.output/'adopted';destination.mkdir();check=m.materialize(source,manifest,C.sha(manifest),destination,True);self.assertEqual(check['created_files'],0);first=m.materialize(source,manifest,C.sha(manifest),destination);second=m.materialize(source,manifest,C.sha(manifest),destination);self.assertGreater(first['created_files'],0);self.assertEqual(second['created_files'],0)
  wrong=destination/'experiments/2026-10-04-bitsandbytes-tpu/nested-exact/integration/upstream/bitsandbytes/functional.py';wrong.write_bytes(wrong.read_bytes()+b'\n')
  with self.assertRaisesRegex(ValueError,'DIFFERENT_CANONICAL_BYTES'):m.materialize(source,manifest,C.sha(manifest),destination)
  summaries.append({'control':'ADDITIVE_MATERIALIZER_READBACK_IDEMPOTENCE_DIFFERENT_BYTES_REJECT','status':'PASS','members':first['members'],'canonical_replacements':0})
 def test_10_browser_intent_record_correct_wrong_without_account_calls(self):
  import browser_record as BR,browser_adoption as BA
  identity=a.output/'fixture-cli-identity.json';directory=a.output/'browser-fixture';intent=BR.prepare('a'*64,HERE/'driver.py',identity,'fixture-browser',directory);cell=(directory/'first-cell.py').read_text();self.assertIn(intent['marker_sha256'],cell);self.assertIn('assert not p.exists()',cell);out=a.output/'synthetic-browser-adoption-never-dispatch.json';r=BR.finalize(directory/'intent.json','fixture-endpoint','V6E1',out);BA.record(out,r['sha256'])
  wrong=C.read(out);wrong['allocation_epoch']=time.time()-3601;C.write(out,wrong)
  with self.assertRaisesRegex(ValueError,'BROWSER_ADOPTION_STALE_OR_EXPIRED'):BA.record(out,C.sha(out))
  summaries.append({'control':'BROWSER_INTENT_MARKER_BINDINGS_FROZEN_ADOPTION_SCHEMA_OLD_BIRTH_REJECT','status':'PASS','scope':'SYNTHETIC_LOCAL_SCHEMA_FIXTURE_NO_RUNTIME_CREATED'})

 def test_11_host_failed_observer_closure_keeps_returned_numeric_rows(self):
  import host_audit as H
  def bad():raise ValueError('SYNTHETIC_POST_RETURN_OBSERVER_OR_CLOSURE_FAILURE')
  for status in('PASS','FAIL'):
   rows=[{'case_id':'synthetic-returned-row','status':status}];r=H.preserve_observations(bad,lambda:{'numerical_status':status,'rows':rows,'record_validation':'PASS'})
   self.assertEqual(r['record_validation'],'FAIL');self.assertEqual(r['numerical_status'],status);self.assertEqual(r['rows'],rows);self.assertEqual(r['m4_status'],'NOT_QUALIFIED');self.assertIn('POST_RETURN',r['record_validation_reason'])
  r=H.preserve_observations(bad,bad);self.assertEqual(r['record_validation'],'FAIL');self.assertEqual(r['numerical_status'],'ERROR');self.assertIn('POST_RETURN',r['record_validation_reason']);self.assertIn('diagnostic_error',r)
  summaries.append({'control':'HOST_FAILED_OBSERVER_OR_CLOSURE_PRESERVES_CORRECT_WRONG_NUMERICS_NEVER_ACCEPTS','status':'PASS','scope':'SYNTHETIC_ROW_DIAGNOSTICS_NOT_SCIENCE'})

result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));files=C.inventory(a.output);before=sum(r['bytes']for r in files.values());hashes={'packet_zip':C.sha(a.output/'packet.zip')if(a.output/'packet.zip').exists()else None}
for p in list(a.output.iterdir()):
 if p.is_dir():shutil.rmtree(p)
 else:p.unlink()
C.write(a.output/'review.json',{'status':'PASS'if result.wasSuccessful()else'FAIL','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'controls':summaries,'disposable_bytes_before':before,'disposable_files_before':len(files),'disposable_files_after':0,'cleanup_verified':True,'hashes':hashes,'provider_calls':0,'TPU':'NOT_RUN','m4_status':'NOT_QUALIFIED'});raise SystemExit(0 if result.wasSuccessful()else 1)
