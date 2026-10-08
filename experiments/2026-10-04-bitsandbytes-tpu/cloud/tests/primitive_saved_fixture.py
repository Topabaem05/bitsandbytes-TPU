"""Portable synthetic audit controls. All fixtures are generated in private temporary dirs."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[2];SCIENCE=HERE
ROOT=next(p for p in HERE.parents if (p/'packages/bitsandbytes-tpu/pyproject.toml').is_file())
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
V=load(HERE/'probe_primitives.py','tested_primitives')
ARGS=SimpleNamespace(backend_probe=SCIENCE/'probe_backend.py',route_probe=SCIENCE/'probe_routes.py',precision_probe=SCIENCE/'probe_precision.py',transfer_admission=SCIENCE/'transfer_admission.py',
 nested_probe=SCIENCE/'probe_nested.py',nested_state_probe=SCIENCE/'probe_nested_state.py',patch_manifest=ROOT/'patches/params4bit-xla-v1.json',admission_sha256='a'*64)
B,R,P,A,N,S=V.helpers(ARGS);DATA=V.spec(B,N);REF,H=V.pure();PROFILE,_=B.load_spec();SCHEMAS=N.spec(B)[2];EXPECTED=REF.expected(DATA)

def admission_document():
 manifest=A.load_manifest(B,ARGS.patch_manifest)
 return {'format':'bnb-tpu.probe-source-admission.v1','runtime_lock_sha256':PROFILE['runtime_lock_sha256'],
 'bitsandbytes':{'commit':PROFILE['source_commit'],'patch_manifest_sha256':A.PATCH_MANIFEST_SHA,'files':A.python_files(manifest['post_patch_package_files'])},
 'bitsandbytes_tpu':{'source_variant':'nested-v1','nested_source_sha256':N.SOURCE_SHA,'files':N.spec(B)[1]['files']}}

def shape(dtype,dimensions):return dtype+'['+','.join(map(str,dimensions))+']'

def builder_hlo(source,target,dims):
 return f'HloModule SYNTHETIC_BUILDER_ONLY\nENTRY bitcast (p: {shape(source,dims)}) -> {shape(target,dims)} {{\n p = {shape(source,dims)} parameter(0)\n ROOT result = {shape(target,dims)} bitcast-convert(p)\n}}\n'

def row_hlo(name,source,ids,expected,call=True):
 """Typed direct or call/GTE bitcasts and explicit unique input lineages."""
 lines=[];functions=[];outputs=[]
 for key,value in source.items():lines.append(f' p{ids[key]} = {shape("s32" if value["dtype"]=="int32" else "f32",value["shape"])} parameter({ids[key]})')
 for i,(key,value) in enumerate(expected.items()):
  target='s32' if value['dtype']=='int32' else 'f32';source_dtype='f32' if target=='s32' else 's32';dims=value['shape'];arg='p'+str(ids['int_input' if (key=='float' and name!='float-arithmetic') else 'float_input'])
  if name.startswith('mean-'):
   c=next(c for c in DATA['mean_cases'] if name=='mean-'+c['case_id']);n=c['count']
   if key=='count_bits':
    lines.extend([f' s{i} = f32[1] slice(p{ids["count_input"]}), slice={{[0:1]}}',f' a{i} = f32[] reshape(s{i})']);arg='a'+str(i)
   elif key=='input_bits':lines.append(f' a{i} = f32[{n}] slice(p{ids["float_input"]}), slice={{[0:{n}]}}');arg='a'+str(i)
   else:
    lines.append(f' reduce{i} = f32[] reduce(p{ids["float_input"]})');arg='reduce'+str(i)
    if key in ('sum_div','sum_reciprocal','ordered_div8'):
     lines.extend([f' s{i} = f32[1] slice(p{ids["count_input"]}), slice={{[0:1]}}',f' denom{i} = f32[] reshape(s{i})',f' a{i} = f32[] divide({arg}, denom{i})']);arg='a'+str(i)
  elif name=='float-arithmetic':
   if key=='maximum':lines.append(f' a{i} = f32[] reduce({arg})')
   else:lines.append(f' a{i} = f32[20] abs({arg})')
   arg='a'+str(i)
  if call:
   functions.append(f'cast{i} (p: {shape(source_dtype,dims)}) -> {shape(target,dims)} {{\n cp{i} = {shape(source_dtype,dims)} parameter(0)\n ROOT bit{i} = {shape(target,dims)} bitcast-convert(cp{i})\n}}')
   lines.append(f' out{i} = {shape(target,dims)} call({arg}), to_apply=cast{i}')
  else:lines.append(f' out{i} = {shape(target,dims)} bitcast-convert({arg})')
  outputs.append('out'+str(i))
 lines.append(' ROOT result = ('+', '.join(shape('s32' if v['dtype']=='int32' else 'f32',v['shape']) for v in expected.values())+') tuple('+', '.join(outputs)+')')
 return 'HloModule SYNTHETIC_RECORD_ONLY\n'+'\n'.join(functions)+'\nENTRY main {\n'+'\n'.join(lines)+'\n}\n'

def reseal(fixture,oracle=False):
 if oracle:
  rec=B.read(fixture.oracle/'oracle-seal.json');rec['artifacts']=B.inventory(fixture.oracle,'oracle-seal.json');B.write(fixture.oracle/'oracle-seal.json',rec);fixture.oracle_sha256=B.sha(fixture.oracle/'oracle-seal.json')
 parent=B.read(fixture.actual/'receipt.json');parent['oracle_sha256']=fixture.oracle_sha256
 for launch in parent['children']:
  root=fixture.actual/launch['phase'];rec=B.read(root/'receipt.json');rec['oracle_sha256']=fixture.oracle_sha256;rec['artifacts']=B.inventory(root,'receipt.json');B.write(root/'receipt.json',rec);launch['receipt_sha256']=B.sha(root/'receipt.json')
  argv=launch['argv'];argv[argv.index('--oracle-sha256')+1]=fixture.oracle_sha256
 parent['artifacts']=B.inventory(fixture.actual,'receipt.json');B.write(fixture.actual/'receipt.json',parent)

def make_fixture(base,admission=None,admission_document_value=None,admission_path=None):
 """Return CLI args. Supply exact admission bytes/path for an external owner seal."""
 base=Path(base);base.mkdir(parents=True,exist_ok=True);fixture=SimpleNamespace(**vars(ARGS),oracle=base/'oracle',actual=base/'actual');fixture.oracle.mkdir();fixture.actual.mkdir()
 if admission_path:content=Path(admission_path).read_bytes()
 else:content=(json.dumps(admission_document_value or admission_document(),indent=2)+'\n').encode()
 source_sha=hashlib.sha256(content).hexdigest()
 if admission is not None and admission!=source_sha:raise ValueError('FIXTURE_ADMISSION_BYTES_REQUIRED')
 fixture.admission_sha256=source_sha;fixture.admission=base/'source-admission.json';fixture.admission.write_bytes(content)
 gold=dict(V.binding(B,N,fixture),status='COMPLETE',pid=1001,pgid=1001,parent_pid=999,process_token='1'*32,source_pre=source_sha,source_post=source_sha,public_api=N.PUBLIC.copy(),schemas=SCHEMAS,cpu_dispatch_observed={k:True for k in SCHEMAS},
 runtime={'python':'3.12','torch':'2.9.0+cpu','platform':'linux','machine':'x86_64'},oracle_method='FRESH_DEFAULT_PYTHON_SCALES_PLUS_SOURCE_FIXED_FP32_BITS')
 B.write(fixture.oracle/'inputs.json',DATA);B.write(fixture.oracle/'outputs.json',EXPECTED);(fixture.oracle/'source-admission.json').write_bytes(content)
 build='SYNTHETIC_BUILD_TEXT_NOT_DEVICE_EVIDENCE';evidence={'capability':'AVX512','capability_api':'torch.backends.cpu.get_cpu_capability','sum_dispatch':'PINNED_SUM_STUB_AVX2_EIGHT_LANE_FALLBACK','torch_git_version':V.TORCH_COMMIT,'loaded_torch':'2.9.0+cpu','build_api':'torch.__config__.show','build':build,'build_sha256':hashlib.sha256(build.encode()).hexdigest(),'threads':4,'interop_threads':4,'aten_cpu_capability_override':None,'default_body_sha256':N.DEFAULT_SHA,'first_scales':{c['case_id']:REF.array(c['first_scales'],[c['count']],'float32') for c in DATA['mean_cases']},'runtime_inventory':PROFILE['runtime'],**{k:gold[k] for k in ('pid','pgid','parent_pid','process_token')}}
 B.write(fixture.oracle/'cpu-evidence.json',evidence);gold['artifacts']=B.inventory(fixture.oracle,'oracle-seal.json');B.write(fixture.oracle/'oracle-seal.json',gold);fixture.oracle_sha256=B.sha(fixture.oracle/'oracle-seal.json')
 now=time.time();parent=dict(V.binding(B,N,fixture),status='COMPLETE',pid=1002,pgid=1002,parent_pid=999,process_token='2'*32,source_pre=source_sha,source_post=source_sha,oracle_sha256=fixture.oracle_sha256,deadline_epoch=now+420,started_epoch=now,started_monotonic=9,children=[],runtime=PROFILE['runtime'],precision_environment={k:None for k in P.ENV_KEYS})
 matrix,_=V.row_matrix(DATA)
 for phase,index in zip(V.PHASES,(0,1)):
  root=fixture.actual/phase;root.mkdir();logs=fixture.actual/(phase+'-logs');logs.mkdir();(logs/'stdout.raw').write_text('SYNTHETIC_ONLY\n');(logs/'stderr.raw').write_text('SYNTHETIC_ONLY\n')
  rec=dict(gold,pid=1003+index,pgid=1002,parent_pid=1002,process_token=('3' if index==0 else '4')*32,parent_process_token='2'*32,phase=phase,case_ids=matrix[phase],deadline_epoch=now+180*(index+1),runtime=PROFILE['runtime'],oracle_sha256=fixture.oracle_sha256,device={'type':'xla','hardware':'TPU','pjrt':'TPU'},dispatch={k:True for k in SCHEMAS},precision_environment={k:None for k in P.ENV_KEYS},precision={'requested':'highest','readback':'highest','set_calls':1,'before_graph':True},materialized_before_arithmetic=True)
  for name in matrix[phase]:
   expected=EXPECTED[name];source=V.source_inputs(REF,DATA,name,EXPECTED['device-int-float']['float']);ids={k:i for i,k in enumerate(source)};hlo=row_hlo(name,source,ids,expected);stem=root/'raw'/name;stem.parent.mkdir(exist_ok=True);stem.with_suffix('.hlo.txt').write_text(hlo);stem.with_suffix('.metrics.txt').write_text('SYNTHETIC_ONLY\n')
   raw={'case_id':name,'pid':rec['pid'],'process_token':rec['process_token'],'status':'OBSERVED','source_inputs':source,'source_inputs_sha256':V.digest(source),'input_manifest_sha256':V.INPUT_SHA,'input_parameter_ids':ids,'parameter_values':{str(ids[k]):v for k,v in source.items()},'output_order':list(expected),'outputs':copy.deepcopy(expected),'placements':{k:'xla:0' for k in expected},'counters':{},'execution_metrics':{'ExecuteTime':[1,1.,[[1.,1.]]]},'synchronized':True,'builder_computations':{},'builder_source_sha256':V.BUILDER_SHA,'parameter_mapping_api':'LoweringContext.device_parameter_id_tensor_mapping+tensor_parameter_id','materialized_from':'device-int-float' if name=='float-arithmetic' else None}
   raw['hlo_witnesses']=V.row_graph(B,H,name,raw,hlo,expected)
   if phase=='builder':
    for key,value in expected.items():
     target='s32' if value['dtype']=='int32' else 'f32';source_type='f32' if target=='s32' else 's32';text=builder_hlo(source_type,target,value['shape']);path=stem.parent/(name+'.'+key+'.builder.hlo.txt');path.write_text(text);raw['builder_computations'][key]={'artifact':path.relative_to(root).as_posix(),'sha256':B.sha(path),'witness':H.computation(text,source_type,target,value['shape'])}
   B.write(stem.with_suffix('.json'),raw)
  rec['artifacts']=B.inventory(root,'receipt.json');B.write(root/'receipt.json',rec)
  cmd=[sys.executable,'-B',str(HERE/'probe_primitives.py'),'device-child']
  for key in (*V.COMMON,'oracle','oracle_sha256','admission'):cmd+=['--'+key.replace('_','-'),str(getattr(fixture,key))]
  cmd+=['--phase',phase,'--output',str(root),'--parent-pid','1002','--parent-process-token','2'*32,'--process-token',rec['process_token'],'--deadline-epoch',str(rec['deadline_epoch'])]
  parent['children'].append({'phase':phase,'pid':rec['pid'],'launched_pid':rec['pid'],'pgid':1002,'process_token':rec['process_token'],'parent_pid':1002,'parent_process_token':'2'*32,'reaped':True,'child_absent':True,'cleanup_errors':[],'timeout':False,'exit_code':0,'receipt':phase+'/receipt.json','started_epoch':now+index*2,'finished_epoch':now+index*2+1,'started_monotonic':10+index*2,'finished_monotonic':11+index*2,'deadline_epoch':rec['deadline_epoch'],'argv':cmd})
 B.write(fixture.actual/'receipt.json',parent);reseal(fixture);return fixture

class Controls(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.f=make_fixture(self.temp.name);self.guard=patch.object(socket.socket,'connect',side_effect=AssertionError('NETWORK_FORBIDDEN'));self.guard.start()
 def tearDown(self):self.guard.stop();self.temp.cleanup()
 def verify(self):return V.verify(B,R,P,A,N,S,self.f)
 def mutate(self,path,change,oracle=False):
  value=B.read(path);change(value);B.write(path,value);reseal(self.f,oracle=oracle)
 def row(self,name='float-arithmetic',phase='builder'):return self.f.actual/phase/'raw'/(name+'.json')
 def test_complete(self):
  result=self.verify();self.assertEqual((result['record_validation'],result['numerical_status'],result['row_count']),('PASS','PASS',32));self.assertEqual(result['cpu_oracle']['output_count'],206);self.assertGreater(result['gate_count'],206);self.assertEqual(result['m4_status'],'NOT_QUALIFIED')
 def test_host_cpu_gate_import_free(self):
  with patch.dict(sys.modules,{'torch':None,'torch_xla':None}):self.assertEqual(V.verify_cpu_oracle(B,R,P,A,N,S,self.f)['oracle_status'],'VERIFIED')
 def test_numerical_fail_retained(self):
  self.mutate(self.row(),lambda r:r['outputs'].update(add_zero=REF.array([0]*20,[20],'int32')));result=self.verify();self.assertEqual((result['record_validation'],result['builder_bit_status']),('PASS','FAIL'))
 def test_native_unsupported_valid(self):
  self.mutate(self.row('native-bitcast','native-view'),lambda r:r.update(status='OBSERVED_UNSUPPORTED',reason='NATIVE_VIEW_EXCEPTION',error_type='RuntimeError',error_message='isolated native dtype-view unsupported',traceback='synthetic traceback',outputs={}))
  self.mutate(self.f.actual/'receipt.json',lambda r:r['children'][0].update(exit_code=2));self.assertEqual(self.verify()['native_view_status'],'UNSUPPORTED')
 def test_native_unsupported_malformed_metrics_rejected(self):
  self.mutate(self.row('native-bitcast','native-view'),lambda r:r.update(status='OBSERVED_UNSUPPORTED',reason='NATIVE_VIEW_EXCEPTION',error_type='RuntimeError',error_message='isolated unsupported',traceback='synthetic traceback',outputs={},execution_metrics={'ExecuteTime':[1,1.,'bad samples']}))
  self.mutate(self.f.actual/'receipt.json',lambda r:r['children'][0].update(exit_code=2))
  with self.assertRaisesRegex(ValueError,'METRIC_OBSERVATION_SAMPLES'):self.verify()
 def test_native_fallback_valid_builder_fallback_rejected(self):
  self.mutate(self.row('native-bitcast','native-view'),lambda r:r.update(status='OBSERVED_UNSUPPORTED',reason='NATIVE_VIEW_ATEN_FALLBACK',outputs={},counters={'aten::view':1}))
  self.mutate(self.f.actual/'receipt.json',lambda r:r['children'][0].update(exit_code=2));self.assertEqual(self.verify()['native_view_status'],'UNSUPPORTED')
  self.mutate(self.row(),lambda r:r.update(counters={'aten::view':1}))
  with self.assertRaisesRegex(ValueError,'CPU_FALLBACK'):self.verify()
 def test_wrong_oracle_and_agreeing_output(self):
  self.mutate(self.f.oracle/'outputs.json',lambda r:r['host-float-bits'].update(bits=REF.array([0]*20,[20],'int32')),oracle=True)
  self.mutate(self.row('host-float-bits'),lambda r:r['outputs'].update(bits=REF.array([0]*20,[20],'int32')))
  with self.assertRaisesRegex(ValueError,'CPU_SOURCE_FIXED_BITS'):self.verify()
 def test_signed_zero_oracle_rejected(self):
  self.mutate(self.f.oracle/'outputs.json',lambda r:r['device-int-float'].update(float=REF.array([0. if x==0 else x for x in r['device-int-float']['float']['values']],[20],'float32')),oracle=True)
  with self.assertRaisesRegex(ValueError,'CPU_SOURCE_FIXED_BITS'):self.verify()
 def test_wrong_first_scales_cpu(self):
  self.mutate(self.f.oracle/'cpu-evidence.json',lambda r:r.update(first_scales={}),oracle=True)
  with self.assertRaisesRegex(ValueError,'CPU_DEFAULT_BODY_SCALES'):self.verify()
 def test_unqualified_cpu_rejected(self):
  self.mutate(self.f.oracle/'oracle-seal.json',lambda r:r.update(runtime={'python':'3.12','torch':'2.14.1','platform':'darwin','machine':'arm64'}),oracle=True)
  with self.assertRaisesRegex(ValueError,'CPU_ORACLE_RUNTIME'):self.verify()
 def test_wrong_cpu_capability_build_thread_override(self):
  p=self.f.oracle/'cpu-evidence.json';old=B.read(p)
  for change in ({'capability':'DEFAULT'},{'build_sha256':'0'*64},{'torch_git_version':'0'*40},{'threads':0},{'aten_cpu_capability_override':'AVX2'},{'process_token':'e'*32}):
   self.mutate(p,lambda r:r.update(change),oracle=True)
   with self.assertRaises(ValueError):self.verify()
   B.write(p,old);reseal(self.f,oracle=True)
 def test_correctly_resealed_wrong_overlay_and_variant_rejected(self):
  for kind in ('overlay','variant'):
   doc=admission_document()
   if kind=='overlay':doc['bitsandbytes']['files']['nn/modules.py']='0'*64
   else:doc['bitsandbytes_tpu']['source_variant']='old-nested'
   f=make_fixture(Path(self.temp.name)/kind,admission_document_value=doc)
   with self.assertRaisesRegex(ValueError,'SOURCE_EXACT_OVERLAY|NESTED_VARIANT_LINK'):V.verify_cpu_oracle(B,R,P,A,N,S,f)
 def test_reused_child_pid_and_token_rejected(self):
  path=self.f.actual/'builder/receipt.json';original=B.read(path)
  for change in ({'pid':1003},{'process_token':'3'*32}):
   self.mutate(path,lambda r:r.update(change))
   with self.assertRaisesRegex(ValueError,'FRESH_CHILD'):self.verify()
   B.write(path,original);reseal(self.f)
 def test_wrong_source_admission_overlay(self):
  doc=B.read(self.f.oracle/'source-admission.json');doc['bitsandbytes']['files']['nn/modules.py']='0'*64;B.write(self.f.oracle/'source-admission.json',doc);reseal(self.f,oracle=True)
  with self.assertRaisesRegex(ValueError,'CPU_SOURCE_ADMISSION_BYTES'):self.verify()
 def test_source_schema_public_precision(self):
  p=self.f.actual/'builder/receipt.json';old=B.read(p)
  for change in ({'source_post':'0'*64},{'schemas':{}},{'public_api':{}},{'precision':{}},{'primitive_variant':'old'},{'runtime':{}}):
   self.mutate(p,lambda r:r.update(change))
   with self.assertRaises(ValueError):self.verify()
   B.write(p,old);reseal(self.f)
 def test_missing_and_empty_rows(self):
  self.row().unlink();reseal(self.f)
  with self.assertRaises(FileNotFoundError):self.verify()
 def test_empty_outputs_and_metrics_samples(self):
  p=self.row();old=B.read(p)
  for change in ({'outputs':{}},{'execution_metrics':{}},{'execution_metrics':{'ExecuteTime':[1,1.,'not metric samples']}},{'synchronized':False},{'status':'ERROR'},{'error_type':'fake error'}):
   self.mutate(p,lambda r:r.update(change))
   with self.assertRaises(ValueError):self.verify()
   B.write(p,old);reseal(self.f)
 def test_launch_pid_token_group_reap_order_deadline(self):
  p=self.f.actual/'receipt.json';old=B.read(p)
  for change in (lambda r:r['children'][0].update(pid=1005),lambda r:r['children'][0].update(launched_pid=1005),lambda r:r['children'][0].update(reaped=False),lambda r:r['children'][0].update(receipt='wrong'),lambda r:r['children'][1].update(started_monotonic=0),lambda r:r.update(deadline_epoch=float('inf')),lambda r:r.update(process_token='z'*32),lambda r:r.update(pgid=99)):
   r=copy.deepcopy(old);change(r)
   # B.write refuses nonfinite as required; malformed JSON is still rejected.
   if not V.number(r['deadline_epoch']):(p).write_text(json.dumps(r))
   else:B.write(p,r)
   reseal(self.f) if V.number(r['deadline_epoch']) else None
   with self.assertRaises(ValueError):self.verify()
   B.write(p,old);reseal(self.f)
 def test_hlo_unused_convert_and_constant_lineage(self):
  p=self.row('host-float-bits').with_suffix('.hlo.txt');old=p.read_text()
  for text in (old.replace('ROOT result = (s32[20]) tuple(out0)','ROOT result = s32[20] constant(0)'),old.replace('bitcast-convert(cp0)','convert(cp0)'),old.replace('p0 = f32[20] parameter(0)','p0 = f32[20] constant(0)')):
   p.write_text(text);reseal(self.f)
   with self.assertRaises(ValueError):self.verify()
  p.write_text(old);reseal(self.f)
 def test_wrong_parameter_values_are_numeric_observation(self):
  p=self.row('host-float-bits');self.mutate(p,lambda r:r['parameter_values'].update({'0':REF.array([0.]*20,[20],'float32')}));self.assertEqual(self.verify()['builder_bit_status'],'FAIL')
 def test_hlo_wrong_type_shape_duplicate_parameter(self):
  p=self.row('host-float-bits').with_suffix('.hlo.txt');old=p.read_text()
  for text in (old.replace('cp0 = f32[20]','cp0 = s32[20]'),old.replace('cp0 = f32[20]','cp0 = f32[19]'),old.replace(' p0 = f32[20] parameter(0)',' p0 = f32[20] parameter(0)\n dup = f32[20] parameter(0)')):
   p.write_text(text);reseal(self.f)
   with self.assertRaises(ValueError):self.verify()
  p.write_text(old);reseal(self.f)
 def test_direct_and_call_tuple_projection_valid(self):
  name='host-float-bits';p=self.row(name);raw=B.read(p);text=row_hlo(name,raw['source_inputs'],raw['input_parameter_ids'],EXPECTED[name],call=False);raw['hlo_witnesses']=V.row_graph(B,H,name,raw,text,EXPECTED[name]);B.write(p,raw);p.with_suffix('.hlo.txt').write_text(text);reseal(self.f);self.assertEqual(self.verify()['record_validation'],'PASS')
  text=row_hlo(name,raw['source_inputs'],raw['input_parameter_ids'],EXPECTED[name]);text=text.replace('ROOT bit0 = s32[20] bitcast-convert(cp0)','bit0 = s32[20] bitcast-convert(cp0)\n ROOT wrapper = (s32[20]) tuple(bit0)').replace('out0 = s32[20] call(p0)','call0 = (s32[20]) call(p0)').replace('ROOT result = (s32[20]) tuple(out0)','out0 = s32[20] get-tuple-element(call0), index=0\n ROOT result = (s32[20]) tuple(out0)');raw['hlo_witnesses']=V.row_graph(B,H,name,raw,text,EXPECTED[name]);B.write(p,raw);p.with_suffix('.hlo.txt').write_text(text);reseal(self.f);self.assertEqual(self.verify()['record_validation'],'PASS')
 def test_post_validation_failure_marks_failed(self):
  parent=B.read(self.f.actual/'receipt.json')
  with patch.object(V,'verify',side_effect=ValueError('isolated post-validation failure')):
   with self.assertRaises(ValueError):V.audit_after_execution(B,R,P,A,N,S,self.f,self.f.actual,parent)
  self.assertEqual(B.read(self.f.actual/'receipt.json')['status'],'FAILED')
  with self.assertRaises(ValueError):self.verify()

class LifecycleControls(unittest.TestCase):
 def test_remaining_clamp_and_insufficient_budget(self):
  self.assertEqual(V.phase_deadline(B,2000.,0,1000.),1180.)
  self.assertEqual(V.phase_deadline(B,2000.,1,1000.),1900.)
  self.assertEqual(V.phase_deadline(B,1100.,1,1000.),1090.)
  self.assertEqual(V.phase_deadline(B,1100.,0,1000.),1045.)
  with self.assertRaises(V.PhaseBudgetUnavailable):V.phase_deadline(B,1025.,1,1000.)
 def test_real_direct_success_and_timeout_reaped(self):
  import subprocess,os
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);script=root/'coordinator.py'
   script.write_text('''import importlib.util,json,os,sys,time
spec=importlib.util.spec_from_file_location('state',sys.argv[1]);S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
root=__import__('pathlib').Path(sys.argv[2]);parent={'pid':os.getpid(),'pgid':os.getpgid(0),'process_token':'a'*32,'deadline_epoch':time.time()+20,'children':[]}
assert parent['pid']==parent['pgid']
results=[]
for mode,wait in [('success',0),('timeout',5)]:
 logs=root/mode;logs.mkdir();deadline=time.time()+(5 if not wait else .15);descriptor={'phase':mode,'process_token':'b'*32}
 try:S.run_child([sys.executable,'-B','-c','import time;time.sleep('+str(wait)+')'],output=logs,descriptor=descriptor,parent=parent,parent_path=root/'parent.json',deadline=deadline);error=None
 except TimeoutError as e:error=type(e).__name__
 assert descriptor['reaped'] and descriptor['child_absent'] and descriptor['cleanup_errors']==[]
 assert descriptor['pgid']==parent['pid'] and descriptor['pid']==descriptor['launched_pid']
 assert (mode=='timeout')==descriptor['timeout']
 results.append({'mode':mode,'error':error,**descriptor})
(root/'result.json').write_text(json.dumps(results))
''')
   child=subprocess.Popen([sys.executable,'-B',str(script),str(ARGS.nested_state_probe),str(root)],start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   try:out,error=child.communicate(timeout=12)
   except BaseException:
    import signal
    os.killpg(child.pid,signal.SIGKILL);child.wait();raise
   self.assertEqual(child.returncode,0,error.decode());results=B.read(root/'result.json');self.assertEqual([r['error'] for r in results],[None,'TimeoutError']);self.assertNotEqual(results[0]['pid'],results[1]['pid'])

if __name__=='__main__':unittest.main(verbosity=2)
