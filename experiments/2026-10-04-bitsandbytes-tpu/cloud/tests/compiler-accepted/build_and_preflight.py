"""Private genuine packet builder and complete byte readback. No provider/installation."""
import argparse,hashlib,json,sys,zipfile
import pin_readback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=next(p for p in HERE.parents if(p/'packages/bitsandbytes-tpu/pyproject.toml').is_file());sys.path.insert(0,str(HERE/'cloud'))
import build_compiler_packet as N,build_packet as B,compiler_cloud_contract as NC,owner
N.ROOT=B.PROJECT=ROOT
p=argparse.ArgumentParser();p.add_argument('--upstream',required=True);p.add_argument('--out',required=True);p.add_argument('--result',required=True);a=p.parse_args()
pin_readback.validate(HERE/'compiler-native')
r=N.build(Path(a.upstream),Path(a.out),candidate=HERE/'compiler-native',candidate_sha=NC.MANIFEST_SHA,m2_dependency=HERE/'m2-dependency.json',m2_dependency_sha=NC.sha(HERE/'m2-dependency.json'));packet=Path(a.out);pin_readback.validate(packet/'native');m=owner.preflight(packet,r['payload_sha256'])
with zipfile.ZipFile(packet/'payload.zip')as z:
 assert set(z.namelist())=={*m['files'],'manifest.json'} and len(z.infolist())==len(m['files'])+1
 for i in z.infolist():
  assert i.create_system==3 and i.external_attr==0o100644<<16;b=z.read(i);assert b==(packet/i.filename).read_bytes()
  if i.filename!='manifest.json':assert hashlib.sha256(b).hexdigest()==m['files'][i.filename]['sha256'] and len(b)==m['files'][i.filename]['bytes']
assert m['base_cloud_revision']=='2c64223350ced1512fb7f6e78e1dde8cb0fc0ee9';assert m['files']['cloud/primitive_contract.py']['sha256']=='35888719248865816d4a76d175cc74d8b401f0a3c26b7f9af20149fc8a38a0a1'
assert m['base_native_manifest_sha256']=='30b820b90a4404a89da6e6c7b6fc91dbec47afbe06ae322eb5835b9eb0e70901';NC.dependency_fields(m)
r.update(status='GENUINE_PACKET_PREFLIGHT_READBACK_PASS',compiler_manifest_sha256=NC.MANIFEST_SHA,source_admission_sha256=m['source_admission_sha256'],generation=m['compiler_generation'],upstream_method='ACTUAL_GIT_ARCHIVE_FIXED_833649043474794b8fe7a4136e0c40faf077b2e0',fresh_qualified_linux_cpu_oracle='REQUIRED_NOT_RUN',actual_TPU='NOT_RUN',M6='NOT_QUALIFIED',provider_calls=0);Path(a.result).write_text(json.dumps(r,sort_keys=True,indent=2)+'\n');print(json.dumps(r))
