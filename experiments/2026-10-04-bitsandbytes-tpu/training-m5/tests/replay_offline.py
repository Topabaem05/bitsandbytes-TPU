"""Bounded offline replay. Outputs stay outside the frozen source."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
output=Path(sys.argv[1]).resolve()
if output.is_relative_to(ROOT):raise ValueError('REPLAY_OUTPUT_OUTSIDE_SOURCE')
output.mkdir(exist_ok=False);started=time.time();checks={}
for name in ('generation_delta','frozen_science','scope_controls','cloud_controls','kernel_controls','driver_controls','bundle_controls'):
 result=output/(name+'.json');r=subprocess.run([sys.executable,'-B',str(ROOT/'tests'/(name+'.py')),str(result)],capture_output=True,text=True,timeout=180)
 if r.returncode!=0:raise RuntimeError(name+':'+r.stdout+r.stderr)
 checks[name]=json.loads(result.read_text());assert checks[name]['status']=='PASS'
summary={'status':'PASS','checks':{n:{k:(len(v)if isinstance(v,list)else v)for k,v in r.items()if k in('status','count','controls','independent_reader_controls','fixtures_deleted','provider_calls','members')}for n,r in checks.items()},'elapsed_seconds':time.time()-started,'provider_calls':0,'local20_repeat':'NOT_RUN_UNCHANGED_SCIENCE','qualified_linux':'NOT_RUN','TPU':'NOT_RUN','m4_acceptance':'NOT_SUPPLIED','m5_status':'NOT_QUALIFIED'}
(output/'replay-summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n');print(json.dumps(summary))
