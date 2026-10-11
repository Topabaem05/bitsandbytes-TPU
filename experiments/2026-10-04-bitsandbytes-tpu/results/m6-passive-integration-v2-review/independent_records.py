import sys,json,tempfile,importlib.util,copy,shutil
from pathlib import Path
kit=Path(sys.argv[1]).resolve();packets=Path(sys.argv[2]).resolve();phase=sys.argv[3];output=Path(sys.argv[4]);report=[]
sys.argv=['fixture','--output','unused','--packets',str(packets),'--phase',phase]
spec=importlib.util.spec_from_file_location('fixture',kit/'tests/readback_controls.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
C=f.C;V=f.V
with tempfile.TemporaryDirectory(prefix='m6-independent-retained-',dir='/tmp')as tmp:
 base=Path(tmp).resolve()/'base';packet=packets/phase;records,compiler,launch,receipt=f.fixture(base,packet,phase)
 report.append({'case':'baseline','status':V.verify(packet,records,compiler if phase=='compiler5'else None)['status']})
 original=receipt.read_bytes();v=C.read(receipt);v['original_error']={'type':'LookupError','message':'independent fixture exception','traceback':'independent fixture'};f.put(receipt,v)
 try:r=V.verify(packet,records,compiler if phase=='compiler5'else None);report.append({'case':'nonempty_original_error_with_zero_exit','observed':'ACCEPT','verifier_status':r['status']})
 except Exception as e:report.append({'case':'nonempty_original_error_with_zero_exit','observed':'REJECT','error':str(e)})
 receipt.write_bytes(original)
 v=C.read(receipt);v.pop('original_error');f.put(receipt,v)
 try:r=V.verify(packet,records,compiler if phase=='compiler5'else None);report.append({'case':'missing_original_error_field','observed':'ACCEPT','verifier_status':r['status']})
 except Exception as e:report.append({'case':'missing_original_error_field','observed':'REJECT','error':str(e)})
 receipt.write_bytes(original)
 if phase=='compiler5':
  f.put(records/'native/independent-tamper.json',{'unknown':True})
  try:V.verify(packet,records,compiler)
  except Exception as e:assert str(e)=='FULL_CROSS_ARCHIVE_SCIENCE_READBACK';report.append({'case':'cross_archive_unlisted_science','observed':'REJECT','error':str(e)})
  else:raise AssertionError('unlisted cross archive science accepted')
 else:
  p=records/'steps/12-public/lifecycle.jsonl';p.write_text(p.read_text()+json.dumps({'kind':'child_started','pid':999,'pgid':999})+'\n')
  try:V.verify(packet,records,None)
  except Exception as e:assert str(e)=='OWNED_STEP_COMPLETE_LEDGER:12-public';report.append({'case':'public_extra_live_child','observed':'REJECT','error':str(e)})
  else:raise AssertionError('extra public child accepted')
output.write_text(json.dumps({'phase':phase,'controls':report,'provider_calls':0,'actual_linux':'NOT_RUN','TPU':'NOT_RUN','M6':'NOT_QUALIFIED','fixture_bulk':'DELETED'},sort_keys=True,indent=2)+'\n');print(json.dumps(report))
