"""Run portable host controls in one finite owned group. No provider or installation."""
import argparse,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for name in ('cloud','packet','runtime-wheels','jax071-python','cli-python','output'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--seconds',type=int,default=180);a=p.parse_args()
sys.path.insert(0,str(a.cloud/'ownership'));from cleanup_lifecycle import Ownership
out=a.output.resolve();out.mkdir(exist_ok=False);owner=Ownership(out);code=None;error=None
try:
 with owner.guard(a.seconds):
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PUBLIC15_TEST_CLOUD=str(a.cloud.resolve()),PUBLIC15_TEST_PACKET=str(a.packet.resolve()),PUBLIC15_RUNTIME_WHEELS=str(a.runtime_wheels.resolve()),PUBLIC15_JAX071_PYTHON=str(a.jax071_python.absolute()),PUBLIC15_CLI_PYTHON=str(a.cli_python.absolute()))
  tests=Path(__file__).resolve().parent;argv=[sys.executable,'-B','-m','pytest',*[str(tests/n)for n in ('test_public_cloud.py','test_public_integrated.py','test_public_limits_modes.py')],'-q','--basetemp='+str(out/'fixtures')]
  with(out/'stdout').open('wb')as log,(out/'stderr').open('wb')as err:
   child=owner.launch(argv,record=out/'launch.json',cwd=tests,env=env,stdout=log,stderr=err);code=child.wait(timeout=a.seconds-5)
  owner.stop(child)
except BaseException as exc:error={'type':type(exc).__name__,'message':str(exc)}
finally:
 closure=owner.summary();groups={};observations=[]
 for path in out.rglob('*.json'):
  if path.name not in ('launch.json','ownership.json','audit-preflight-launch.json','local-verifier-ownership.json','gate-owned.json'):continue
  try:r=json.loads(path.read_text())
  except (ValueError,OSError):continue
  if r.get('ownership')!='Popen child in dedicated start_new_session group':continue
  pid=r.get('pid');cleanup=r.get('cleanup',{})
  if type(pid)is not int or r.get('pgid')!=pid:continue
  groups.setdefault(pid,[]).append(path.relative_to(out).as_posix())
 for pid,paths in groups.items():
  try:os.killpg(pid,0);absent=False
  except ProcessLookupError:absent=True
  except PermissionError:absent=False
  observations.append({'pid':pid,'pgid':pid,'source_records':paths,'group_absent':absent})
 result={'status':'PASS'if code==0 and error is None and not closure['errors']and all(r['group_absent']for r in observations)else'FAIL','exit_code':code,'error':error,'cleanup':closure,'actual_groups':observations,'provider_calls':0,'actual_device':'NOT_RUN','qualified_linux_cpu_oracle':'NOT_RUN','scope':'OFFLINE_HOST_SOURCE_ARITHMETIC_AND_SYNTHETIC_RECORD_CONTROLS'}
 (out/'review.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps({'status':result['status'],'exit_code':code,'actual_groups':len(observations),'cleanup':closure['status']}))
raise SystemExit(0 if result['status']=='PASS'else 2)
