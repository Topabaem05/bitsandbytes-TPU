"""Build only the explicit compiler capture diagnostic. No installation or provider call."""
import argparse,hashlib,importlib.util,json,shutil,sys,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];CLOUD=HERE;sys.path.insert(0,str(CLOUD))
import build_packet as B
import compiler_cloud_contract as NC
FROZEN=HERE.parent/'compiler-native'
APPROVED_COMMIT='2c64223350ced1512fb7f6e78e1dde8cb0fc0ee9'
def build(upstream,out,*,candidate,candidate_sha,m2_dependency,m2_dependency_sha):
    NC.require(candidate_sha==NC.MANIFEST_SHA and NC.sha(Path(candidate)/'manifest.json')==candidate_sha,'REVIEWED_COMPILER34_HASH')
    NC.require(NC.sha(m2_dependency)==m2_dependency_sha,'M2_DEPENDENCY_HASH')
    scientific=ROOT/'experiments/2026-10-04-bitsandbytes-tpu'
    paths={'route_probe':scientific/'probe_routes.py','precision_probe':scientific/'probe_precision.py','transfer_probe':scientific/'probe_transfer.py','transfer_admission':scientific/'transfer_admission.py','patch_manifest':ROOT/'patches/params4bit-xla-v1.json','source_controls':ROOT/'tests/test_transfer_source.py'}
    B.build(Path(upstream),Path(out),NC.sha(ROOT/'packages/bitsandbytes-tpu/source-manifest.json'),experiment='transfer-api42',**paths,**{n+'_sha256':NC.sha(p)for n,p in paths.items()})
    out=Path(out);manifest=B.sha(out/'manifest.json');m=NC.read(out/'manifest.json')
    for name,record in NC.SOURCES.items():
        source=Path(candidate)/name;NC.require(not source.is_symlink() and NC.sha(source)==record['sha256'] and source.stat().st_size==record['bytes'],'COMPILER34_FILE:'+name);target=out/'native'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    shutil.copyfile(Path(candidate)/'manifest.json',out/'native/manifest.json');shutil.copyfile(m2_dependency,out/'m2-dependency.json')
    for name in ('native-dependency.json','accepted-native-result.json'):shutil.copyfile(HERE.parent/name,out/name)
    m['files']={p.relative_to(out).as_posix():{'sha256':NC.sha(p),'bytes':p.stat().st_size}for p in sorted(out.rglob('*'))if p.is_file() and p.name not in {'manifest.json','payload.zip'}}
    # The nested native manifest is a payload file.
    m['files']['native/manifest.json']={'sha256':NC.MANIFEST_SHA,'bytes':(out/'native/manifest.json').stat().st_size}
    m.update(experiment=NC.MODE,compiler_manifest_sha256=NC.MANIFEST_SHA,compiler_policy_sha256=NC.POLICY_SHA,m2_dependency_sha256=m2_dependency_sha,compiler_source_variant=NC.VARIANT,compiler_scope='OWNED_COMPILER_CAPTURE_ONLY',compiler_generation='compiler-mosaic-serde7-gather-bf16-fp32-accepted-native-fix-v1',base_cloud_revision=APPROVED_COMMIT,**NC.NATIVE_DEPENDENCY_BINDINGS,m6_status='NOT_QUALIFIED',optimized_executable_link='UNKNOWN')
    B.write(out/'manifest.json',m);NC.payload(out,m)
    with zipfile.ZipFile(out/'payload.zip','w',zipfile.ZIP_DEFLATED)as z:
        for n in sorted([*m['files'],'manifest.json']):
            i=zipfile.ZipInfo(n,date_time=(2026,10,8,0,0,0));i.create_system=3;i.external_attr=0o100644<<16;i.compress_type=zipfile.ZIP_DEFLATED;z.writestr(i,(out/n).read_bytes())
    return {'manifest_sha256':NC.sha(out/'manifest.json'),'payload_sha256':NC.sha(out/'payload.zip'),'payload_bytes':(out/'payload.zip').stat().st_size,'members':len(m['files'])+1,'scope':'PRIVATE_NO_DISPATCH','m6':'NOT_QUALIFIED'}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('upstream','out','m2-dependency','m2-dependency-sha256'):p.add_argument('--'+n,required=True)
    p.add_argument('--candidate',default=str(FROZEN));p.add_argument('--candidate-sha256',default=NC.MANIFEST_SHA)
    a=p.parse_args();print(json.dumps(build(a.upstream,a.out,candidate=a.candidate,candidate_sha=a.candidate_sha256,m2_dependency=a.m2_dependency,m2_dependency_sha=a.m2_dependency_sha256)))
