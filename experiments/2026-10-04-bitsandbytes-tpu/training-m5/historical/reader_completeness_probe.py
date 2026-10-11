"""Independent negative probes against real driver-generated synthetic protocol fixtures."""
import copy,hashlib,json,sys
from pathlib import Path
source=Path('/tmp/bnb-tpu-m5-v2-20261011-jcc5zoqq');out=Path(__file__).resolve().parent;sys.path[:0]=[str(source/'control'),str(source/'tests')]
import inspect_actual as I,driver_controls as D
original=I.kernel_readbacks;rows=[];done=False
def intercepted(packet,actual,driver):
 global done
 result=original(packet,actual,driver)
 if not done:
  done=True
  for name,change in [('remove-all-kernel-proofs-and-command-rows',lambda d:d.update(commands=[],kernel_proofs=[])),('remove-install-proof-and-matching-command-row',lambda d:d.update(commands=[r for r in d['commands']if r['label']!='remote-install'],kernel_proofs=[r for r in d['kernel_proofs']if r['label']!='remote-install']))]:
   corrupted=copy.deepcopy(driver);change(corrupted)
   try:value=original(packet,actual,corrupted);rows.append({'case':name,'actual_outcome':'INCORRECTLY_ACCEPTED','result':value,'remaining_script_files':len(list(actual.glob('*.py')))})
   except ValueError as e:rows.append({'case':name,'actual_outcome':'REJECTED','error':str(e)})
 return result
I.kernel_readbacks=intercepted
D.main(out/'driver_probe_replay.json')
(out/'reader_completeness_probe.json').write_text(json.dumps({'status':'REVIEW_FINDING'if any(r['actual_outcome']=='INCORRECTLY_ACCEPTED'for r in rows)else'PASS','controls':rows,'source_file':str(source/'control/inspect_actual.py'),'source_sha256':hashlib.sha256((source/'control/inspect_actual.py').read_bytes()).hexdigest(),'provider_calls':0,'fixture_only':True,'scientific_acceptance':'NOT_EVALUATED'},indent=2,sort_keys=True)+'\n')
print(json.dumps(rows))
