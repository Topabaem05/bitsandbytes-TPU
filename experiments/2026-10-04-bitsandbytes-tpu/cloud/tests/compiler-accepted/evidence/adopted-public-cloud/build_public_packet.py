"""Build the explicit public15 cloud candidate; no provider or installation."""
import argparse,hashlib,json,shutil,sys,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import build_packet as B;import public_contract as U

def build(upstream,out,repo_root,public_source,public_manifest_sha):
 repo=Path(repo_root);public=Path(public_source);out=Path(out);U.require(public_manifest_sha==U.MANIFEST_SHA and U.sha(public/'public-device-manifest.json')==U.MANIFEST_SHA,'FROZEN_PUBLIC_GENERATION');B.PROJECT=repo
 scientific=repo/'experiments/2026-10-04-bitsandbytes-tpu';paths={'route_probe':scientific/'probe_routes.py','precision_probe':scientific/'probe_precision.py','transfer_probe':scientific/'probe_transfer.py','transfer_admission':scientific/'transfer_admission.py','patch_manifest':repo/'patches/params4bit-xla-v1.json','source_controls':repo/'tests/test_transfer_source.py'}
 B.build(Path(upstream),out,B.sha(repo/'packages/bitsandbytes-tpu/source-manifest.json'),experiment='transfer-api42',**paths,**{n+'_sha256':B.sha(p)for n,p in paths.items()})
 m=U.read(out/'manifest.json');s=U.read(public/'public-device-manifest.json');a=U.read(public/'package-admission.json')
 def copy(src,name,expected):
  src=Path(src);U.require(not src.is_symlink()and U.row(src)==expected,'PUBLIC_BUILD_SOURCE:'+name);p=out/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,p)
 for n,r in {**s['source_files'],'public-device-manifest.json':U.row(public/'public-device-manifest.json')}.items():copy(public/n,'public/'+n,r)
 for n,r in a['plugin_files'].items():copy(public/'plugin'/n,'public-plugin/'+n,r)
 baseline=dict(a['fixed_files']);prefix='experiments/2026-10-04-bitsandbytes-tpu/native/';native=U.read(repo/(prefix+'manifest.json'))
 for n,r in native['sources'].items():baseline[prefix+n]=r
 baseline['patches/params4bit-xla-v1.json']=U.row(repo/'patches/params4bit-xla-v1.json')
 for n,r in baseline.items():copy(repo/n,'public-baseline/'+n,r)
 m['files']={p.relative_to(out).as_posix():U.row(p)for p in sorted(out.rglob('*'))if p.is_file()and p.name not in('manifest.json','payload.zip')}
 # Nested manifests are actual payload members, not the outer self-reference.
 for n in ['public/public-device-manifest.json',prefix+'manifest.json']:
  key=n if n.startswith('public/')else'public-baseline/'+n;m['files'][key]=U.row(out/key)
 # The source manifest filename differs and already appears above.
 m.update(experiment=U.MODE,public_manifest_sha256=U.MANIFEST_SHA,public_admission_sha256=U.ADMISSION_SHA,public_protocol_sha256=U.PROTOCOL_SHA,public_native_acceptance_sha256=U.NATIVE_ACCEPTANCE_SHA,public_generation=U.GENERATION,public_source_variant=U.VARIANT,public_scope=U.SCOPE,public_result_max_bytes=U.RESULT_MAX_BYTES,public_result_max_members=U.RESULT_MAX_MEMBERS,base_cloud_revision='0b9f79a5a94e9342817582b4fb786ec0c73fc104',m6_status='NOT_QUALIFIED',optimized_executable_link='UNKNOWN')
 U.write(out/'manifest.json',m);U.payload(out,m)
 with zipfile.ZipFile(out/'payload.zip','w',zipfile.ZIP_DEFLATED)as z:
  for n in sorted([*m['files'],'manifest.json']):
   i=zipfile.ZipInfo(n,date_time=(2026,10,9,0,0,0));i.create_system=3;i.external_attr=0o100644<<16;i.compress_type=zipfile.ZIP_DEFLATED;z.writestr(i,(out/n).read_bytes())
 return {'manifest_sha256':U.sha(out/'manifest.json'),'payload_sha256':U.sha(out/'payload.zip'),'payload_bytes':(out/'payload.zip').stat().st_size,'members':len(m['files'])+1,'scope':'PRIVATE_NO_DISPATCH_BEFORE_ROOT_ADOPTION'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('upstream','out','repo-root','public-source','public-manifest-sha256'):p.add_argument('--'+n,required=True)
 a=p.parse_args();print(json.dumps(build(a.upstream,a.out,a.repo_root,a.public_source,a.public_manifest_sha256),sort_keys=True))
