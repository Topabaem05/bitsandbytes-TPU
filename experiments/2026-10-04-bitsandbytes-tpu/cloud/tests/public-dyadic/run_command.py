"""Run one bounded, owned offline command and retain closure evidence."""
import argparse,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--ownership-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=int,default=240);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False);sys.path.insert(0,str(a.ownership_source.resolve()));from cleanup_lifecycle import Ownership
owner=Ownership(out);code=None;error=None
try:
 with owner.guard(a.seconds):
  with(out/'stdout').open('wb')as log,(out/'stderr').open('wb')as err:
   child=owner.launch(a.command,record=out/'launch.json',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=err);code=child.wait(timeout=a.seconds-5)
  owner.stop(child)
except BaseException as e:error={'type':type(e).__name__,'message':str(e)}
finally:
 r={'status':'PASS'if code==0 and not error and not owner.summary()['errors']else'FAIL','exit_code':code,'error':error,'cleanup':owner.summary(),'provider_calls':0,'actual_device':'NOT_RUN','qualified_linux_cpu':'NOT_RUN'};(out/'review.json').write_text(json.dumps(r,sort_keys=True,indent=2)+'\n');print(json.dumps({'status':r['status'],'exit_code':code}))
raise SystemExit(0 if r['status']=='PASS'else 2)
