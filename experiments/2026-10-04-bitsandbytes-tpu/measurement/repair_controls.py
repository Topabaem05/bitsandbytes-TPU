"""Consequential retained-record counterexamples. Isolated fake clock/allocator only."""
import copy,json
from pathlib import Path
import collector as C
from controls import Fake
HERE=Path(__file__).resolve().parent

def main():
 out=HERE/'repair-controls';out.mkdir(exist_ok=False);rows=[];a,b=Fake().collect(),Fake('SAME_DEVICE_REFERENCE').collect();C.compare(a,b)
 rows.append({'case':'correct-complete-record','status':'PASS'})
 def reject(name,mutate,reason):
  x,y=copy.deepcopy(a),copy.deepcopy(b);mutate(x,y)
  try:C.compare(x,y)
  except ValueError as e:
   assert str(e)==reason,(name,str(e),reason);rows.append({'case':name,'status':'PASS','rejected':str(e)})
  else:raise AssertionError('FALSE_ACCEPT:'+name)
 def phase_bypass(x,y):
  x['samples'][0]['phase']='WARMUP';x['samples'][0]['metrics']['CompileTime']=C.metric([1,10,[[0,10]]])
 reject('root-measured-phase-with-compile-bypass',phase_bypass,'RAW_ROW_PHASE_INDEX_GROUP')
 reject('root-forged-reported-peak-minus-one',lambda x,y:x['memory_snapshots'][0]['snapshot'].__setitem__('reported_high_water_bytes',-1),'RAW_MEMORY_RECOMPUTATION')
 reject('root-forged-cold-compile-minus-one',lambda x,y:x.__setitem__('backend_compile_seconds',-1),'RAW_COLD_COMPILE_RECOMPUTATION')
 reject('root-setup-sync-wait-false',lambda x,y:x['setup']['sync'].__setitem__('wait',False),'RAW_SETUP_CONTRACT')
 for name,phase in [('warmup-as-measured','MEASURED'),('warmup-as-cold','COLD_FIRST_CALL')]:reject(name,lambda x,y,p=phase:x['warmups'][0].__setitem__('phase',p),'RAW_ROW_PHASE_INDEX_GROUP')
 reject('cold-as-warmup',lambda x,y:x['compile_first_call'].__setitem__('phase','WARMUP'),'RAW_ROW_PHASE_INDEX_GROUP')
 reject('cold-index-not-zero',lambda x,y:x['compile_first_call'].__setitem__('index',1),'RAW_ROW_PHASE_INDEX_GROUP')
 reject('warmup-group-not-none',lambda x,y:x['warmups'][0].__setitem__('group',0),'RAW_ROW_PHASE_INDEX_GROUP')
 reject('bool-sample-index',lambda x,y:x['samples'][0].__setitem__('index',False),'RAW_ROW_PHASE_INDEX_GROUP')
 reject('bool-sample-group',lambda x,y:x['samples'][0].__setitem__('group',False),'RAW_ROW_PHASE_INDEX_GROUP')
 reject('setup-wrong-target',lambda x,y:x['setup']['sync'].__setitem__('targets',['actual']),'RAW_SETUP_CONTRACT')
 reject('setup-wrong-scope',lambda x,y:x['setup'].__setitem__('scope','HOST_SUBMIT_ONLY'),'RAW_SETUP_CONTRACT')
 reject('setup-extra-field',lambda x,y:x['setup'].__setitem__('forged',True),'RAW_SETUP_CONTRACT')
 reject('setup-duration-forged',lambda x,y:x['setup'].__setitem__('elapsed_ns',1),'RAW_SETUP_DURATION')
 reject('setup-negative-time',lambda x,y:x['setup'].__setitem__('started_ns',-1),'RAW_ORDERED_TIME_INTERVAL')
 reject('cold-compile-wrong-status',lambda x,y:x.__setitem__('backend_compile_status','NOT_OBSERVED'),'RAW_COLD_COMPILE_RECOMPUTATION')
 reject('cold-compile-wrong-scope',lambda x,y:x.__setitem__('backend_compile_scope','WHOLE_COMPILER'),'RAW_COLD_COMPILE_RECOMPUTATION')
 reject('cold-wall-wrong-scope',lambda x,y:x.__setitem__('cold_call_wall_scope','PURE_COMPILE'),'RAW_COLD_COMPILE_RECOMPUTATION')
 reject('memory-missing-stage',lambda x,y:x['memory_snapshots'].pop(),'RAW_MEMORY_STAGE_MATRIX')
 reject('memory-duplicate-stage',lambda x,y:x['memory_snapshots'][1].__setitem__('phase','BEFORE_SETUP'),'RAW_MEMORY_STAGE_MATRIX')
 reject('memory-reordered-stage',lambda x,y:x['memory_snapshots'].reverse(),'RAW_MEMORY_STAGE_MATRIX')
 reject('memory-extra-stage-field',lambda x,y:x['memory_snapshots'][0].__setitem__('fake',True),'RAW_MEMORY_STAGE_SCHEMA')
 reject('memory-derived-current-forged',lambda x,y:x['memory_snapshots'][0]['snapshot'].__setitem__('current_bytes',99),'RAW_MEMORY_RECOMPUTATION')
 reject('memory-derived-limit-forged',lambda x,y:x['memory_snapshots'][0]['snapshot'].__setitem__('limit_bytes',999),'RAW_MEMORY_RECOMPUTATION')
 reject('memory-claimed-interval-peak',lambda x,y:x['memory_snapshots'][0]['snapshot'].__setitem__('target_interval_peak',200),'RAW_MEMORY_RECOMPUTATION')
 reject('memory-claimed-scope',lambda x,y:x['memory_snapshots'][0]['snapshot'].__setitem__('reported_high_water_scope','TARGET_INTERVAL'),'RAW_MEMORY_RECOMPUTATION')
 reject('memory-raw-peak-invalid',lambda x,y:x['memory_snapshots'][0]['snapshot']['raw'].__setitem__('peak_bytes_used',-1),'MEMORY_PEAK_VALIDITY')
 reject('memory-raw-current-invalid',lambda x,y:x['memory_snapshots'][0]['snapshot']['raw'].__setitem__('bytes_used',-1),'MEMORY_CURRENT_LIMIT')
 reject('memory-negative-clock',lambda x,y:x['memory_snapshots'][0].__setitem__('started_ns',-1),'RAW_ORDERED_TIME_INTERVAL')
 reject('memory-clock-backwards',lambda x,y:x['memory_snapshots'][0].__setitem__('finished_ns',-1),'RAW_ORDERED_TIME_INTERVAL')
 reject('memory-in-wrong-time-slot',lambda x,y:x['memory_snapshots'][0].update(started_ns=100,finished_ns=100),'RAW_CHRONOLOGICAL_SEQUENCE')
 def overlap(x,y):
  r=x['samples'][1];r['started_ns']=x['samples'][0]['started_ns'];r['finished_ns']=r['started_ns']+r['elapsed_ns']
 reject('sample-overlap-valid-individual-durations',overlap,'RAW_CHRONOLOGICAL_SEQUENCE')
 reject('complete-row-with-error',lambda x,y:x.__setitem__('error',{'message':'failed'}),'RAW_RECORD_SCOPE')
 for fault in ('peak-absent','memory-api-absent'):
  x,y=Fake(fault=fault).collect(),Fake('SAME_DEVICE_REFERENCE',fault=fault).collect();C.compare(x,y);rows.append({'case':fault+'-valid-unknown','status':'PASS'})
  if fault=='memory-api-absent':
   x['memory_snapshots'][0]['snapshot']['reported_high_water_bytes']=200
   try:C.compare(x,y)
   except ValueError as e:assert str(e)=='RAW_MEMORY_UNKNOWN_SCHEMA';rows.append({'case':'unknown-memory-claimed-number','status':'PASS','rejected':str(e)})
   else:raise AssertionError('UNKNOWN_FALSE_ACCEPT')
 # An absent cold compile metric may be valid. It must retain UNKNOWN duration.
 x,y=copy.deepcopy(a),copy.deepcopy(b)
 for r in (x,y):
  for key in C.COMPILE_KEYS:r['compile_first_call']['metrics'][key]=C.metric(None)
  r['backend_compile_status']='NOT_OBSERVED';r['backend_compile_seconds']='UNKNOWN'
 C.compare(x,y);rows.append({'case':'cold-compilation-not-observed-valid-unknown','status':'PASS'})
 x['backend_compile_seconds']=0
 try:C.compare(x,y)
 except ValueError as e:assert str(e)=='RAW_COLD_COMPILE_RECOMPUTATION';rows.append({'case':'cold-not-observed-forged-zero-duration','status':'PASS','rejected':str(e)})
 else:raise AssertionError('COLD_UNKNOWN_FALSE_ACCEPT')
 C.compare(a,b)
 for name,r in [('native',a),('reference',b)]: (out/(name+'.json')).write_text(json.dumps(r,indent=2)+'\n')
 (out/'results.json').write_text(json.dumps({'status':'PASS','count':len(rows),'controls':rows,'scope':'ISOLATED_FAKE_RECORDS_NO_TPU_NO_PROVIDER','m7':'NOT_QUALIFIED'},indent=2)+'\n');print({'status':'PASS','count':len(rows)})
if __name__=='__main__':main()
