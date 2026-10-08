"""Real pinned-frontend replay and bounded inherited-child controls. No TPU/provider."""
import argparse,copy,json,os,sys,time
from pathlib import Path
import conversion_recovery as C
import verifier as V

def run(out,python,ambient,fixtures):
 out.mkdir(exist_ok=False);rows=[]
 metadata=json.loads((fixtures/'printer/metadata.json').read_text())['cases']['f32'];original=metadata['original'];converted=metadata['converted'];audit=metadata['audit'];row={'original_payload':original,'payload':converted,'original_payload_sha256':V.digest(original),'payload_sha256':V.digest(converted),'payload_conversion':audit}
 def check(name,data,accept=True,interpreter=python):
  try:
   report=C.replay(data,interpreter,time.time()+20)
  except Exception as error:
   if accept:raise
   value={'case':name,'status':'PASS','outcome':'REJECTED_INCORRECT','error':str(error),'type':type(error).__name__}
  else:
   if not accept:raise AssertionError('incorrect record accepted')
   value={'case':name,'status':'PASS','outcome':'ACCEPTED_CORRECT','replay':report}
  (out/(name+'.json')).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');rows.append({'case':name,'status':'PASS','outcome':value['outcome']})
 check('genuine-independent-8-to-7',{'direct-fp32':row});check('explicit-runtime-only-preflight',{})
 for name,change in [
 ('forged-roundtrip-boolean',lambda x:x['payload_conversion'].update(exact_current_ir_equal=False)),
 ('forged-ir-hash',lambda x:x['payload_conversion'].update(current_ir_sha256='0'*64)),
 ('wrong-serializer-source-claim',lambda x:x['payload_conversion'].update(serializer_python_sha256='0'*64)),
 ('wrong-converter-source',lambda x:x['payload_conversion'].update(converter_sha256='0'*64)),
 ('changed-other-config',lambda x:x.update(payload=json.dumps({**json.loads(x['payload']),'unadmitted':True},separators=(',',':')))),
 ('truncated-original-body',lambda x:x.update(original_payload=x['original_payload'][:-24])),
 ('converted-still-version8',lambda x:x.update(payload=x['original_payload'],payload_sha256=x['original_payload_sha256'])),
 ('original-already-version7',lambda x:x.update(original_payload=x['payload'],original_payload_sha256=x['payload_sha256']))]:
  bad=copy.deepcopy(row);change(bad)
  # Keep trivial payload digests consistent, so parsed IR/audit gates must do the work.
  bad['original_payload_sha256']=V.digest(bad['original_payload']);bad['payload_sha256']=V.digest(bad['payload']);check(name,{'direct-fp32':bad},False)
 check('ambient-jax0112-rejected',{'direct-fp32':row},False,ambient)
 check('missing-explicit-interpreter',{'direct-fp32':row},False,None)
 try:C.replay({'direct-fp32':row},python,time.time()-1);raise AssertionError('expired deadline accepted')
 except ValueError as error:rows.append({'case':'expired-original-deadline','status':'PASS','error':str(error)})
 result={'status':'PASS','controls':rows,'count':len(rows),'actual_TPU':'NOT_RUN','M6':'NOT_QUALIFIED','scope':'GENUINE_JAX071_CPU_FRONTEND_REPLAY_WITH_INHERITED_SUBPROCESSES'};(out/'results.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps({'status':'PASS','count':len(rows)}))
