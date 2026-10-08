"""Full recovered graph/ownership/numerical verification plus independent real frontend replay."""
import argparse,base64,copy,gzip,json,sys,time
from pathlib import Path

import protocol as S,verify_run as R,mosaic_fixture as F

def run(out,oracle,python,ambient,fixtures):
 F.FIXTURES=fixtures
 out.mkdir(exist_ok=False);root=out/'record';data=F.make_fixture(root,oracle);rows=[]
 def verify(interpreter=python):return R.verify(root,oracle,data['oracle_sha'],data['admission_sha'],data['expected_sha'],data['outer'],actual_device=False,mosaic_audit_python=interpreter,audit_deadline_epoch=time.time()+30)
 result=verify();assert result['status']=='OFFLINE_SHAPED_FIXTURE_PASS';S.write(out/'correct-report.json',result);rows.append({'case':'full-correct-converted-executed-graph-shaped-fixture','status':'PASS'})
 original={p:p.read_bytes() for p in root.rglob('*')if p.is_file()};prefix=root/'device/raw/direct-fp32'
 def jsonedit(path,fn):v=S.read(path);fn(v);S.write(path,v)
 def both(fn):
  for suffix in('.json','.boundary-record.json'):jsonedit(prefix.with_suffix(suffix),fn)
 def wrong_ir():
  def fn(raw):raw['native_boundary']['payload_conversion']['current_ir_sha256']='0'*64
  both(fn);jsonedit(prefix.with_suffix('.conversion-audit.json'),lambda v:v.update(current_ir_sha256='0'*64))
 def original_changed():
  text=prefix.with_suffix('.original-native-config.json').read_text().replace('true','false',1);prefix.with_suffix('.original-native-config.json').write_text(text)
 def converted_v8():
  old=S.read(prefix.with_suffix('.json'))['native_boundary']['original_payload']
  both(lambda raw:raw['native_boundary'].update(payload=old,payload_sha256=S.V.digest(old)))
  prefix.with_suffix('.native-config.json').write_text(old);prefix.with_suffix('.mosaic-body.bin').write_bytes(base64.b64decode(json.loads(old)['custom_call_config']['body']))
 def body_only():prefix.with_suffix('.original-mosaic-body.bin').write_bytes(b'truncated')
 def trace_only():
  both(lambda raw:raw['native_boundary'].update(calls=0,status='TRACE_RETURNED'))
 def numerical():both(lambda raw:raw['output']['values'].__setitem__(0,raw['output']['values'][0]+1000))
 def execution():both(lambda raw:raw.update(execution_metrics={}))
 def graphwrong():prefix.with_suffix('.printer.hlo.pb').write_bytes(b'wrong actual graph')
 def original_graph():
  for suffix in('.hlo.txt','.hlo.pb','.printer.hlo.pb'):
   src=fixtures/'original-printer'/('f32'+('.hlo.pb'if suffix=='.printer.hlo.pb'else suffix));prefix.with_suffix(suffix).write_bytes(src.read_bytes())
  both(lambda raw:raw.update(hlo_sha256=S.sha(prefix.with_suffix('.hlo.txt'))))
 def source():jsonedit(root/'parent.json',lambda raw:raw.update(source_post='0'*64))
 def failmatrix():
  jsonedit(root/'device/receipt.json',lambda raw:raw.update(cases=[{'name':name,'status':'ERROR'if i==0 else'NOT_RUN'}for i,name in enumerate(S.spec()['case_ids'])],numerical_status='FAIL'))
 def audit_missing():prefix.with_suffix('.conversion-audit.json').unlink()
 tests=[('forged-recorded-ir-audit',wrong_ir),('altered-original-config-artifact',original_changed),('unconverted-v8-body-used-at-native-boundary',converted_v8),('truncated-original-body-artifact',body_only),('preflight-trace-without-native-dispatch',trace_only),('wrong-finite-output',numerical),('missing-positive-execution',execution),('wrong-actual-graph-proto',graphwrong),('same-shaped-original-v8-graph-with-converted-preflight-record',original_graph),('post-source-change',source),('first-error-subsequent-notrun-matrix',failmatrix),('missing-independent-audit-artifact',audit_missing)]
 for name,mutation in tests:
  mutation();F.reseal(root)
  try:verify();raise AssertionError('incorrect fixture accepted: '+name)
  except (ValueError,KeyError,FileNotFoundError)as error:
   record={'case':name,'status':'PASS','rejected':str(error),'type':type(error).__name__};S.write(out/(name+'.json'),record);rows.append(record)
   # Retain exact changed records once in compressed form; the correct full corpus remains unchanged.
   changed={str(p.relative_to(root)):base64.b64encode(p.read_bytes()).decode() for p in root.rglob('*')if p.is_file() and original.get(p)!=p.read_bytes()}
   with (out/(name+'.changes.json.gz')).open('wb')as f:
    with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0)as z:z.write(json.dumps(changed,sort_keys=True).encode())
  for p in root.rglob('*'):
   if p.is_file()and p not in original:p.unlink()
  for p,b in original.items():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
 try:verify(ambient);raise AssertionError('ambient runtime accepted')
 except ValueError as error:rows.append({'case':'full-recovery-explicit-wrong-runtime','status':'PASS','rejected':str(error)})
 S.write(out/'results.json',{'status':'PASS','count':len(rows),'controls':rows,'scope':'SYNTHETIC_EXECUTION_OWNER_NUMERICAL_RECORDS_WITH_GENUINE_PINNED_JAX071_CONVERSION_REPLAY','M6':'NOT_QUALIFIED','actual_TPU':'NOT_RUN'});print(json.dumps({'status':'PASS','count':len(rows)}))
