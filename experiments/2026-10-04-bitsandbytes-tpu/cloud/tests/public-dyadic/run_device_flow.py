"""Own the complete local saved-record consumer control group."""
import argparse,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser()
for n in ('public-source','base-packet','cpu-fixture','jax071-python','fixture-source','ownership-source','output'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False);sys.path.insert(0,str(a.ownership_source.resolve()));from cleanup_lifecycle import Ownership
owner=Ownership(out);code=None;error=None
try:
 with owner.guard(240):
  argv=[sys.executable,'-B',str(Path(__file__).with_name('device_flow_controls.py'))]
  for n in ('public_source','base_packet','cpu_fixture','fixture_source'):argv+=['--'+n.replace('_','-'),str(getattr(a,n).resolve())]
  argv+=['--jax071-python',str(a.jax071_python.absolute()),'--output',str(out/'controls')]
  with(out/'stdout').open('wb')as log,(out/'stderr').open('wb')as err:
   child=owner.launch(argv,record=out/'launch.json',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=err);code=child.wait(timeout=235)
  owner.stop(child)
except BaseException as e:error={'type':type(e).__name__,'message':str(e)}
finally:
 closure=owner.summary();result={'status':'PASS'if code==0 and not error and not closure['errors']else'FAIL','exit_code':code,'error':error,'cleanup':closure,'provider_calls':0,'actual_device':'NOT_RUN','qualified_linux_cpu':'NOT_RUN'};(out/'review.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps({'status':result['status'],'exit_code':code}))
raise SystemExit(0 if result['status']=='PASS'else 2)
