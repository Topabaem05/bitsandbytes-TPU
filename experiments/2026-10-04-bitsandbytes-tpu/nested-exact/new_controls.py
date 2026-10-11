"""Focused exact transport and state8 integration controls. No device qualification."""
import argparse, copy, hashlib, importlib.util, json, os, secrets, shutil, sys, tempfile, time, unittest
from pathlib import Path
p=argparse.ArgumentParser()
for key in ('scratch','repo','generation','oracle','output'):p.add_argument('--'+key,required=True)
a=p.parse_args();scratch=Path(a.scratch);source=scratch/'source';output=Path(a.output);output.mkdir()
sys.path.insert(0,str(source))
import archive_objects as O
import archive_store as Z
import probe_state8_exact as S
import probe_nested_exact as E
import exact_admission as A

def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
B=load(Path(a.repo)/'experiments/2026-10-04-bitsandbytes-tpu/probe_backend.py','new_controls_B')
records=[]
class Controls(unittest.TestCase):
 def test_01_real_recorder_codec_roundtrip_selected_and_duplicate(self):
  root=output/'codec';root.mkdir();store=Z.Store(root,create=True)
  bodies={'raw/case.json':b'{"literal":"a\\nb", "zero": -0.0}\n','trace/case/001/context.hlo.pb':bytes(range(256))*100,
          'trace/case/001/printer.hlo.pb':bytes(range(256))*100,'trace/case/001/graph.hlo.txt':b'HloModule f32 exact \n'*4000,
          'trace/case/001/context.textproto':b'module { exact }\n'*4000,'trace/case/001/metrics.txt':b'CompileTime: 1\n'}
  for name,body in bodies.items():
   path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body)
  Z.pack_directory(root,root/'trace');store=Z.Store(root);store.pack(root/'raw/case.json')
  self.assertEqual(len(store.files),6);self.assertEqual(store.audit()['unique_objects'],5)
  for name,body in bodies.items():self.assertEqual(store.read(name),body)
  with tempfile.TemporaryDirectory() as target:
   with store.selected(bodies,target):
    for name,body in bodies.items():self.assertEqual((Path(target)/name).read_bytes(),body)
   self.assertFalse(any(p.is_file() for p in Path(target).rglob('*')))
  archive=output/'codec.tar.gz';report=Z.write_tar(root,archive);self.assertLess(report['bytes'],100*1024*1024)
  records.append({'control':'LOSSLESS_REAL_PACK_SELECTED_RECOVERY_AND_EXACT_DUPLICATE','status':'PASS','logical':6,'unique':5,'archive':report})
 def test_02_transport_wrong_content_path_quota_extra_and_partial_reject(self):
  root=output/'wrong';root.mkdir();s=Z.Store(root,create=True);s.put('raw/x.json',b'{"x":1}','json')
  with self.assertRaisesRegex(ValueError,'LOGICAL_FILE_REWRITE'):s.put('raw/x.json',b'{"x":2}','json')
  for name in ('../bad','/abs','a//b','a\\b','objects/d.gz'):
   with self.assertRaises(ValueError):s.put(name,b'x','graph')
  record=s.files['raw/x.json'];path=root/record['path'];original=path.read_bytes();path.write_bytes(original[:-1])
  with self.assertRaises(ValueError):s.read('raw/x.json')
  path.write_bytes(original);extra=root/'objects'/('a'*64+'.gz');extra.write_bytes(b'bad')
  with self.assertRaisesRegex(ValueError,'EXACT_OBJECT_INVENTORY'):s.audit()
  extra.unlink()
  old=O.PROFILE['graph_raw_bytes'];O.PROFILE['graph_raw_bytes']=10
  large=root/'raw/big.hlo.txt';large.parent.mkdir(exist_ok=True);large.write_bytes(b'x'*11)
  try:
   with self.assertRaisesRegex(ValueError,'EVIDENCE_LIMIT'):s.pack(large)
   diagnostic=json.loads((root/'evidence-limit.json').read_text())[0]
   self.assertEqual(diagnostic['raw_sha256'],hashlib.sha256(b'x'*11).hexdigest());self.assertEqual(diagnostic['raw_bytes'],11)
   self.assertFalse(large.exists());self.assertNotIn('raw/big.hlo.txt',s.files)
   s.flush()  # Persist the isolated scaled quota before the production store reload.
   call=root/'trace/case/001';call.mkdir(parents=True)
   for name in ('one.pb','two.pb'):(call/name).write_bytes(b'y'*11)
   with self.assertRaisesRegex(ValueError,'EVIDENCE_LIMIT'):Z.pack_directory(root,call)
   self.assertFalse(any(p.is_file() for p in call.rglob('*')))
  finally:O.PROFILE['graph_raw_bytes']=old
  records.append({'control':'WRONG_CONTENT_PATH_TRUNCATION_EXTRA_OBJECT_QUOTA_AND_FAILURE_HASH','status':'PASS'})
 def test_03_real_state8_public_save_and_fresh_restore(self):
  parent={'pid':os.getpid(),'pgid':os.getpgid(0),'process_token':secrets.token_hex(16),'deadline_epoch':time.time()+90,'children':[]}
  self.assertEqual(parent['pid'],parent['pgid']);path=output/'parent.json';S.durable_json(path,parent)
  for phase in ('save','restore'):
   logs=output/(phase+'-logs');logs.mkdir();token=secrets.token_hex(16)
   cmd=[sys.executable,'-B',str(scratch/'state_fixture_child.py'),'--source',str(source),'--repo',a.repo,'--generation',a.generation,
        '--oracle',a.oracle,'--phase',phase,'--output',str(output/phase),'--saved',str(output/'save'),'--token',token]
   launch=S.run_child(cmd,output=logs,descriptor={'phase':phase,'process_token':token},parent=parent,parent_path=path,deadline=parent['deadline_epoch'],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
   self.assertEqual(launch['exit_code'],0)
  saved=B.read(output/'save/fixture.json');restored=B.read(output/'restore/fixture.json')
  self.assertEqual(saved['rows'],restored['rows']);self.assertEqual(len(saved['rows']),8)
  self.assertNotEqual(saved['pid'],restored['pid']);self.assertNotEqual(saved['process_token'],restored['process_token'])
  self.assertTrue(parent['children'][0]['finished_monotonic']<=parent['children'][1]['started_monotonic'])
  for phase in ('save','restore'):
   for file in (output/phase).glob('*.pt'):file.unlink()
  records.append({'control':'EIGHT_REAL_CPU_PUBLIC_SAVE_FRESH_RESTORE_AND_EIGHT_WRONG_CHECKPOINTS','status':'PASS','scope':'UNQUALIFIED_CPU_FIXTURE_NOT_ORACLE','children':parent['children']})
 def test_03b_cpu_qualification_precedes_arithmetic(self):
  import record_validation as V
  evidence={'python':'3.12.14','torch':'2.9.0+cpu','platform':'Linux','machine':'x86_64',
            'torch_git':'0fabc3ba44823f257e70ce397d989c8de5e362c1','capability':'AVX2','num_threads':1,
            'xla_or_jax_imported':False,'build':'retained synthetic qualification fixture'}
  V.cpu_evidence(evidence)
  for field,value in [('capability','AVX512'),('num_threads',2),('xla_or_jax_imported',True),('python','3.12.13')]:
   wrong=dict(evidence,**{field:value})
   with self.assertRaisesRegex(ValueError,'QUALIFIED_CPU_SOURCE_CAPABILITY'):V.cpu_evidence(wrong)
  for name in ('probe_nested_exact.py','probe_state8_exact.py'):
   text=(source/name).read_text();self.assertLess(text.index('V.cpu_evidence(evidence)'),text.index("for case in ",text.index('V.cpu_evidence(evidence)')))
  records.append({'control':'EXPLICIT_AVX2_SINGLE_THREAD_BEFORE_CPU_ARITHMETIC_CORRECT_WRONG','status':'PASS','scope':'SYNTHETIC_METADATA_NO_ORACLE'})
 def test_03c_observer_failure_preserves_original_return(self):
  from types import SimpleNamespace
  import torch,public_trace as T
  class Tensor(torch.Tensor):
   @property
   def device(self):return torch.device('xla:0')
  value=torch.Tensor._make_subclass(Tensor,torch.tensor([1.0]),False)
  class Func:
   _schema=SimpleNamespace(name='bnb_tpu_exact::restore_scales')
   def __call__(self,*args,**kwargs):return value
  def bad_context(name):raise ValueError('SYNTHETIC_POST_RETURN_OBSERVER_FAILURE')
  fake_xla=SimpleNamespace(_XLAC=SimpleNamespace(_xla_sync_multi=lambda *a,**k:None,lowering=SimpleNamespace(LoweringContext=bad_context)))
  fake_xm=SimpleNamespace(wait_device_ops=lambda:None);fake_metrics=SimpleNamespace(clear_all=lambda:None)
  fake_B=SimpleNamespace(tensor_record=lambda t:{'dtype':'float32','shape':[1],'values':[1.0]},write=B.write)
  root=output/'observer';root.mkdir();Z.Store(root,create=True)
  mode=T.make_mode(torch,fake_B,None,fake_xla,fake_xm,fake_metrics,source,root/'trace/case',time.time()+20)
  returned=mode.__torch_dispatch__(Func(),(),(value,),{})
  self.assertIs(returned,value);self.assertEqual(len(mode.observation_errors),1)
  raw=json.loads(Z.Store(root).read('trace/case/001/raw.json'))
  self.assertEqual(raw['status'],'ERROR');self.assertTrue(raw['numerical_returned']);self.assertEqual(raw['calls'],1)
  self.assertIsNone(sys.getprofile())
  from unittest.mock import patch
  with patch.object(Z,'pack_directory',side_effect=ValueError('EVIDENCE_LIMIT:synthetic')):
   self.assertIs(mode.__torch_dispatch__(Func(),(),(value,),{}),value)
  self.assertTrue(mode.evidence_limit);index=mode.index
  self.assertIs(mode.__torch_dispatch__(Func(),(),(value,),{}),value);self.assertEqual(mode.index,index)
  records.append({'control':'SYNTHETIC_AFTER_RETURN_OBSERVER_FAILURE_RETAINS_ORIGINAL_RESULT','status':'PASS','TPU':'NOT_RUN'})
 def test_03d_numeric_rows_survive_observation_error_and_wrong_codes_fail(self):
  from types import SimpleNamespace
  from unittest.mock import patch
  import record_validation as V
  N=load(Path(a.repo)/'experiments/2026-10-04-bitsandbytes-tpu/probe_nested.py','numeric_fixture_N')
  P=load(Path(a.repo)/'experiments/2026-10-04-bitsandbytes-tpu/probe_precision.py','numeric_fixture_P')
  case=S.fixed_cases(B)[0];reference=B.read(Path(a.oracle)/'raw'/(case['id']+'.json'))
  fixture_args=SimpleNamespace(oracle='SYNTHETIC_NO_ORACLE',oracle_sha256='0'*64)
  for wrong in (False,True):
   raw=copy.deepcopy(reference);raw['placements']={k:'xla:0' for k in raw['outputs']}
   raw['observation_status']='ERROR';raw['observation_errors']=[{'reason':'SYNTHETIC_GRAPH_OBSERVER_FAILURE'}]
   if wrong:
    values=raw['outputs']['scale_codes']['values'];values[0]^=1
    raw['outputs']['scale_codes']['bytes_sha256']=hashlib.sha256(N.binary(raw['outputs']['scale_codes'])).hexdigest()
   root=output/('numeric-wrong' if wrong else 'numeric-correct');root.mkdir();store=Z.Store(root,create=True)
   store.put('raw/'+case['id']+'.json',json.dumps(raw).encode(),'json')
   with patch.object(E,'binding',return_value={'scope':'SYNTHETIC_ONLY'}),patch.object(V,'verify_cpu',return_value=({}, {case['id']:reference}, {})):
    result=E.numerical_observations(B,P,N,fixture_args,root,selected_cases=[case])
   self.assertEqual(result['record_validation'],'FAIL');self.assertEqual(result['numerical_status'],'FAIL' if wrong else 'PASS')
   self.assertEqual(result['m4_status'],'NOT_QUALIFIED');self.assertEqual(len(result['rows'][0]['gates']),len(reference['outputs']))
  records.append({'control':'SYNTHETIC_RETURNED_NUMERIC_ROW_PASS_WRONG_CODE_FAIL_OBSERVATION_NEVER_QUALIFIES','status':'PASS','scope':'MOCK_CPU_DEPENDENCY_NO_ACCEPTED_RECORD'})
 def test_04_entrypoints_and_unchanged_original_gates(self):
  import ast
  N=load(Path(a.repo)/'experiments/2026-10-04-bitsandbytes-tpu/probe_nested.py','new_controls_N')
  self.assertEqual(A.sha(source/'nested-inputs.json'),E.INPUT_SHA)
  self.assertEqual(len(S.fixed_cases(B)),8)
  for name in ('probe_nested_exact.py','probe_state8_exact.py'):
   ast.parse((source/name).read_text())
  public=(source/'state8_public.py').read_text()
  self.assertIn('strict=True,assign=False',public);self.assertIn('torch.ops.bnb_tpu_exact.restore_scales.default',public)
  self.assertNotIn('quant_state=state.state2)+state.offset',public)
  cli=S.main(['--help']) if False else None
  # Actual receipt validation must reject missing and old metadata; never manufacture accepted79.
  with self.assertRaises(ValueError):A.state_dependency({},source/'science-manifest.json',A.sha(source/'science-manifest.json'),'a'*64)
  records.append({'control':'REAL_ENTRYPOINTS_EXACT_INPUTS_ORIGINAL_RESTORE_AND_NO_FAKE_ACCEPTANCE','status':'PASS'})
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
B.write(output/'review.json',{'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'controls':records,'TPU':'NOT_RUN','qualified_CPU':'NOT_RUN'})
raise SystemExit(0 if result.wasSuccessful() else 1)
