"""Build and preflight the ordinary native packet with this cloud source tree."""
import argparse,json,sys,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=next(p for p in HERE.parents if(p/'packages/bitsandbytes-tpu/pyproject.toml').is_file());sys.path.insert(0,str(HERE/'cloud'))
import build_native_packet as N,build_packet as B,native_contract as C,owner
N.ROOT=B.PROJECT=ROOT
p=argparse.ArgumentParser();p.add_argument('--upstream',required=True);p.add_argument('--out',required=True);p.add_argument('--result',required=True);a=p.parse_args()
r=N.build(Path(a.upstream),Path(a.out),candidate=HERE/'native',candidate_sha=C.MANIFEST_SHA,m2_dependency=HERE/'m2-dependency.json',m2_dependency_sha=C.sha(HERE/'m2-dependency.json'));m=owner.preflight(Path(a.out),r['payload_sha256'])
with zipfile.ZipFile(Path(a.out)/'payload.zip')as z:
 assert len(z.infolist())==len(m['files'])+1 and set(z.namelist())=={*m['files'],'manifest.json'}
 for i in z.infolist():assert i.create_system==3 and i.external_attr==0o100644<<16 and z.read(i)==(Path(a.out)/i.filename).read_bytes()
r.update(status='ORDINARY_NATIVE_GENUINE_PACKET_PREFLIGHT_PASS',generation=m['native_generation'],native_manifest_sha256=C.MANIFEST_SHA,actual_TPU='NOT_RUN',provider_calls=0);Path(a.result).write_text(json.dumps(r,sort_keys=True,indent=2)+'\n');print(json.dumps(r))
