"""Compose the private compiler-only delta with the root-adopted public cloud base."""
import difflib,hashlib,json,shutil,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=next(p for p in HERE.parents if(p/'packages/bitsandbytes-tpu/pyproject.toml').is_file());SCI='experiments/2026-10-04-bitsandbytes-tpu';REV='5ada90eeef3f8a9ec3913a2a1a83d877be709f65'
EXPECTED={'build_public_packet.py':'0e9f0eb944d6cfd105df91c4c14611d4d2deab50fcfa7580e5cbb89385c1eda7','owner.py':'dcfa490f94de91bb1dc6734f2b89c564d37dc27886ff122e64c7ed64ba99736b','public_contract.py':'7c1669d611e3425c5f2fb540980526eb9a44a326309e8eeb36090b55ec9c3421','public_gate.py':'e7d9f842e303b33e36db2b5993c10c474ab79f30c84666c08a5e68bebc903f14','public_transport.py':'8ff569c37d805474e773210f36a541eb59750aac3b9e23f69f152ea26f8fbabd','remote.py':'5d03b80630cfd88fae91056b9a59825c27839c15f2e6c5a4f64de64e1fe0f616'}
assert subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==REV
assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True)
base=subprocess.check_output(['git','-C',str(ROOT),'show','0b9f79a5a94e9342817582b4fb786ec0c73fc104:'+SCI+'/cloud/remote.py'],text=True)
private=(HERE/'cloud/remote.py').read_text();patch=''.join(difflib.unified_diff(base.splitlines(True),private.splitlines(True),fromfile='a/cloud/remote.py',tofile='b/cloud/remote.py'))
evidence=HERE/'evidence/adopted-public-cloud';evidence.mkdir(exist_ok=False)
for n,h in EXPECTED.items():
 p=ROOT/SCI/'cloud'/n;assert hashlib.sha256(p.read_bytes()).hexdigest()==h;shutil.copyfile(p,evidence/n)
(evidence/'compiler-only-delta.patch').write_text(patch)
for n in EXPECTED:shutil.copyfile(evidence/n,HERE/'cloud'/n)
subprocess.run(['git','-C',str(ROOT),'apply','--check','--directory='+str(HERE.relative_to(ROOT)),str(evidence/'compiler-only-delta.patch')],check=True)
subprocess.run(['git','-C',str(ROOT),'apply','--directory='+str(HERE.relative_to(ROOT)),str(evidence/'compiler-only-delta.patch')],check=True)
(evidence/'composition.json').write_text(json.dumps({'status':'ROOT_ADOPTED_PUBLIC_CLOUD_EXACT_BASE_WITH_PRIVATE_COMPILER_ONLY_DELTA','revision':REV,'base_six_sha256':EXPECTED,'compiler_only_delta_sha256':hashlib.sha256(patch.encode()).hexdigest(),'PUBLIC15':'UNCHANGED_CANONICAL_SCIENTIFIC_SOURCES','ordinary_native27':'UNCHANGED','provider_calls':0},sort_keys=True,indent=2)+'\n');print('PRIVATE_COMPOSITION_PASS')
