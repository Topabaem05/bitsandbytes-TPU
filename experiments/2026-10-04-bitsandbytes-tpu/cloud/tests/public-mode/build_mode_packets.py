"""Explicit private builder configuration for three admitted source modes. No science."""
import argparse,importlib.util,json,shutil,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for name in ('cloud','repo-root','upstream','public-source','m2-dependency','output'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();sys.path.insert(0,str(a.cloud.resolve()));import build_packet as B,build_native_packet as N,build_compiler_packet as K,build_public_packet as P,owner as O,public_contract as U,compiler_cloud_contract as CC,native_contract as NC
repo=a.repo_root.resolve();B.PROJECT=repo;N.ROOT=repo;K.ROOT=repo;root=a.output.resolve();root.mkdir(exist_ok=False)
for name,src in [('native-dependency.json',repo/'experiments/2026-10-04-bitsandbytes-tpu/results/native-bf16-acceptance.json'),('accepted-native-result.json',repo/'experiments/2026-10-04-bitsandbytes-tpu/results/native-bf16-colab.json')]:
 target=a.cloud.parent/name
 if target.exists():assert target.read_bytes()==src.read_bytes()
 else:shutil.copyfile(src,target)
records={}
records['public']=P.build(a.upstream,root/'public',repo,a.public_source,U.MANIFEST_SHA)
records['compiler']=K.build(a.upstream,root/'compiler',candidate=repo/'experiments/2026-10-04-bitsandbytes-tpu/compiler-native',candidate_sha=CC.MANIFEST_SHA,m2_dependency=a.m2_dependency,m2_dependency_sha=U.sha(a.m2_dependency))
# Existing native builder accepts an explicit corrected candidate and M2 dependency.
records['native']=N.build(a.upstream,root/'native',candidate=repo/'experiments/2026-10-04-bitsandbytes-tpu/native',candidate_sha=NC.MANIFEST_SHA,m2_dependency=a.m2_dependency,m2_dependency_sha=U.sha(a.m2_dependency))
for name in records:
 m=O.preflight(root/name,records[name]['payload_sha256']);assert m['experiment']=={'public':U.MODE,'compiler':CC.MODE,'native':NC.MODE}[name]
 if name!='public':assert not any(k in m for k in U.FIELDS)
records.update(scope='SOURCE_PACKET_PREFLIGHT_ONLY',actual_device='NOT_RUN',qualified_cpu='NOT_RUN',provider_calls=0)
(root/'review.json').write_text(json.dumps(records,sort_keys=True,indent=2)+'\n');print(json.dumps({'status':'PASS','mode_packets':3}))
