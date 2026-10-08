"""Use the unchanged cloud owner for real local process-group closure."""
import argparse,json,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'cloud'));import remote
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=int,default=120);p.add_argument('argv',nargs=argparse.REMAINDER);a=p.parse_args();a.output.mkdir(exist_ok=False)
r=remote.run_step(a.output,'control',a.argv,time.time()+a.seconds,a.seconds-5,cwd=HERE,tpu=False)
(a.output/'summary.json').write_text(json.dumps(r,sort_keys=True,indent=2)+'\n');print(json.dumps({'status':r['status'],'exit_code':r['exit_code'],'cleanup':r['cleanup']['status']}));raise SystemExit(0 if r['status']=='PASS' and not r['cleanup']['errors'] else 2)
