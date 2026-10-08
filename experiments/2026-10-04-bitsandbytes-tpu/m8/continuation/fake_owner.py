"""Real owned broker topology with a synthetic verifier and SDK worker."""
import argparse,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,os.environ['M8_CANONICAL_HOST'])
from supervisor import BrokerProvider
from continued_runner import ContinuedRunner,NoSubmitProvider
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--broker-fd',type=int,required=True);p.add_argument('--broker-token',required=True);p.add_argument('--deadline-epoch',type=float,required=True);a=p.parse_args();c=json.loads(Path(os.environ['M8_CONTINUATION_FIXTURE']).read_text());provider=BrokerProvider(a.broker_fd,a.broker_token,a.deadline_epoch)
if c['fault']=='submit':
 try:NoSubmitProvider(provider).invoke('submit',{},call_seconds=1)
 except ValueError as e:result={'status':'REJECT','error':str(e),'scope':'SYNTHETIC_CONTROL'}
 else:raise AssertionError('FALSE_ACCEPT')
else:
 def verifier(members,binding):
  assert members=={'records/result.json':b'SYNTHETIC_SCIENCE_CALLBACK_NOT_DEVICE'}and binding==c['plan']['binding']
  return {'record_validation':'PASS','numerical_status':'PASS','fixed_runtime':True,'source_and_input_match':True,'all_required_cases':True}
 obj=ContinuedRunner(c['plan'],provider,a.output/'owner',verifier);result=obj.run_saved(c['admitted'],c['prior'],c['prior_sha']);result['scope']='SYNTHETIC_PROVIDER_AND_SCIENCE_CONTROL_NO_DEVICE';obj.save()
(a.output/'result.json').write_text(json.dumps(result));print(json.dumps({'state':result.get('state',result.get('status')),'m8_status':'NOT_QUALIFIED'}))
