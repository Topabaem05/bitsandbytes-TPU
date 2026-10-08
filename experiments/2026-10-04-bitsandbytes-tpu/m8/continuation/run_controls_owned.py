"""Finite local control owner; use the unchanged canonical cleanup guard."""
import argparse,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--ownership',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cwd',type=Path,required=True);p.add_argument('--seconds',type=int,default=90);p.add_argument('argv',nargs=argparse.REMAINDER);a=p.parse_args()
sys.path.insert(0,str(a.ownership.resolve()));from cleanup_lifecycle import Ownership
a.output.mkdir(exist_ok=False);o=Ownership(a.output);code=None
try:
 with o.guard(a.seconds):
  with(a.output/'stdout.raw').open('wb')as out,(a.output/'stderr.raw').open('wb')as err:
   proc=o.launch(a.argv,record=a.output/'launch.json',cwd=a.cwd.resolve(),env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=out,stderr=err);code=proc.wait(timeout=a.seconds-5)
  o.stop(proc)
finally:
 summary=o.summary();result={'status':'PASS'if code==0 and not summary['errors']else'FAIL','exit_code':code,'cleanup':summary,'provider_calls':0};(a.output/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
raise SystemExit(0 if result['status']=='PASS'else 2)
