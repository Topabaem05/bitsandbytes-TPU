"""Use unchanged canonical compiler/plain fixtures against explicit composed cloud."""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['compiler','native'],required=True)
for name in ('cloud','repo-root','control-tree','packet','output'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();sys.path.insert(0,str(a.cloud.resolve()));import remote,owner,native_contract,compiler_cloud_contract
assert remote.HERE.resolve()==a.cloud.resolve()
base=a.control_tree.resolve();source=base/('cloud/tests/compiler_phase_controls.py'if a.mode=='compiler'else'plain_native_controls.py')
sys.argv=[str(source),'--packet',str(a.packet.resolve()),'--output',str(a.output.resolve())]
runpy.run_path(str(source),run_name='__main__')
result=json.loads((a.output/'results.json').read_text());result['composed_remote_sha256']=hashlib.sha256((a.cloud/'remote.py').read_bytes()).hexdigest();result['unchanged_canonical_fixture_source']={'path':str(source.relative_to(base)),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()};(a.output/'results.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
