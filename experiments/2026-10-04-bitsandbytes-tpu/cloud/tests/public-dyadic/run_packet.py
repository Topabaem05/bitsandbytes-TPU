"""Build one deterministic source packet under finite local ownership."""
import argparse,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for n in ('cloud','repo-root','public-source','upstream','ownership-source','output'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False);sys.path.insert(0,str(a.ownership_source.resolve()));from cleanup_lifecycle import Ownership
owner=Ownership(out);code=None;error=None
try:
 with owner.guard(120):
  import hashlib
  source=a.public_source.resolve();digest=hashlib.sha256((source/'public-device-manifest.json').read_bytes()).hexdigest();argv=[sys.executable,'-B',str(a.cloud.resolve()/'build_public_packet.py'),'--upstream',str(a.upstream.resolve()),'--out',str(out/'packet'),'--repo-root',str(a.repo_root.resolve()),'--public-source',str(source),'--public-manifest-sha256',digest]
  with(out/'stdout').open('wb')as log,(out/'stderr').open('wb')as err:
   child=owner.launch(argv,record=out/'launch.json',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=err);code=child.wait(timeout=115)
  owner.stop(child)
except BaseException as e:error={'type':type(e).__name__,'message':str(e)}
finally:
 result={'status':'PASS'if code==0 and not error and not owner.summary()['errors']else'FAIL','exit_code':code,'error':error,'cleanup':owner.summary(),'provider_calls':0,'qualified_linux_cpu':'NOT_RUN','actual_device':'NOT_RUN'};(out/'review.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps({'status':result['status'],'exit_code':code}))
raise SystemExit(0 if result['status']=='PASS'else 2)
