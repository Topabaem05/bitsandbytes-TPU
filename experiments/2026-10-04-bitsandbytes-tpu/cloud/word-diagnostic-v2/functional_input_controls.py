"""Exercise native functional CPU inputs at the diagnostic harness boundary."""
import argparse,copy,hashlib,json,sys,tempfile,time
from pathlib import Path
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'packet/source'));import probes as P,run_diagnostic as R
rows=[]
def test(name,fn):fn();rows.append({'name':name,'status':'PASS'})
def reject(fn,text):
 try:fn()
 except RuntimeError as e:assert text in str(e),(text,str(e));return
 raise AssertionError('INCORRECT_FUNCTIONAL_INPUT_ACCEPTED')
def main(evidence=None):
 evidence=ROOT if evidence is None else evidence.resolve()
 if evidence!=ROOT:evidence.mkdir(parents=True,exist_ok=False)
 F=P.load('original')[1];key=torch._C.DispatchKey.Functionalize;before=torch._C._dispatch_tls_is_dispatch_key_excluded(key)
 raw=torch.tensor([0x3f800000,0],dtype=torch.int32);functional=torch._to_functional_tensor(raw.clone());functional.add_(1)
 def preserve():
  values,meta=R.backend_inputs({'raw':raw,'functional':functional,'repeated':functional});assert values['raw']is raw and values['functional']is values['repeated'];assert not any(torch._is_functional_tensor(v)for v in values.values());assert values['functional'].tolist()==[0x3f800001,1];assert meta['functional']['source_native_functional']is True;assert values['functional'].dtype==functional.dtype and values['functional'].shape==functional.shape and values['functional'].stride()==functional.stride()and values['functional'].device==functional.device
 test('raw_identity_pending_update_metadata_bits_and_repeated_alias',preserve)
 def guard():
  called=[]
  def body(value):called.append(True);return value
  reject(lambda:F.functionalized_kernel(body)(functional),'Backend kernel inputs must be unwrapped tensors');assert not called
 test('unchanged_backend_guard_rejects_functional_input_before_body',guard)
 test('actual_torch_double_wrap_rejected',lambda:reject(lambda:torch._to_functional_tensor(functional),'isFunctionalTensor(base_)'))
 def once():
  values,_=R.backend_inputs({'x':functional,'alias':functional});seen=[]
  def body(a,b):seen.append((torch._is_functional_tensor(a),a is b));return a+0
  result=F.functionalized_kernel(body)(values['x'],values['alias']);assert seen==[(True,True)]and result.tolist()==[0x3f800001,1]
 test('normalized_inputs_wrapped_once_inside_frozen_backend',once)
 with tempfile.TemporaryDirectory(prefix='m4-diagnostic-v2-native-functional-',dir='/tmp')as temporary:
  root=Path(temporary);original_inputs=P.inputs
  def run(label,inputs):
   with patch.object(P,'inputs',side_effect=inputs):
    args=argparse.Namespace(mode='cpu',output=root/label,deadline_epoch=time.time()+60,oracle=None,oracle_sha256=None,unqualified_cpu=True);assert R.run(args)==0
   return json.loads((root/label/'report.json').read_text())
  plain=run('raw',original_inputs)
  wrapped=run('functional',lambda device:{k:torch._to_functional_tensor(v)for k,v in original_inputs(device).items()})
  def compare():
   assert len(plain['rows'])==len(wrapped['rows'])==24 and plain['inputs']==wrapped['inputs'];assert [r['outputs']for r in plain['rows']]==[r['outputs']for r in wrapped['rows']];assert all(r['status']=='RETURNED'for r in wrapped['rows']);assert all(r['source_native_functional']and not r['backend_native_functional']for r in wrapped['input_boundary'].values());assert all(not r['source_native_functional']for r in plain['input_boundary'].values());assert R.check_cpu(wrapped['rows'])['status']=='PASS'
   for row in wrapped['rows'][:4]:assert row['metadata']['source_native_functional']==row['metadata']['requested_functional']
  test('all24_simulated_native_functional_inputs_equal_raw_cpu_oracle',compare)
  (evidence/'functional-input-cpu-report.json').write_text(json.dumps(wrapped,indent=2,sort_keys=True)+'\n')
 assert not Path(temporary).exists();assert torch._C._dispatch_tls_is_dispatch_key_excluded(key)==before
 (evidence/'functional-input-controls.json').write_text(json.dumps({'kind':'M4_DIAGNOSTIC_V2_NATIVE_FUNCTIONAL_INPUT_CONTROLS','status':'PASS','controls':rows,'control_count':len(rows),'all24_returned':True,'fixed_input_arrays_preserved':True,'TLS_unchanged':True,'fixtures':'DELETED_AFTER_TEST','torch_version':torch.__version__,'qualification':'UNQUALIFIED_HOST_CPU_CONTROL','provider_calls':0,'actual_tpu':False,'m4_status':'NOT_QUALIFIED'},indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'PASS','controls':len(rows),'native_functional_cpu_rows':24}))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--evidence-output',type=Path);args=parser.parse_args();main(args.evidence_output)
