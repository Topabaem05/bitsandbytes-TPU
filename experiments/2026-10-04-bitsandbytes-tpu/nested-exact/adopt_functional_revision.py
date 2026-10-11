"""Check or apply one reviewed replacement map against exact destination preimages."""
import argparse,hashlib,json,os,stat,tempfile
from pathlib import Path,PurePosixPath

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,text):
    if not ok:raise ValueError(text)
def descriptor(p):return {'sha256':sha(p),'bytes':p.stat().st_size}
def name(value):
    p=PurePosixPath(value)
    require(type(value)is str and value and not p.is_absolute() and '..' not in p.parts and '\\' not in value and p.as_posix()==value,'CONFINED_PATH')
    require(value=='CHANGELOG.md' or value=='docs/changes/r4-functional-output-preparation.md' or value.startswith(('experiments/2026-10-04-bitsandbytes-tpu/nested-exact/','experiments/2026-10-04-bitsandbytes-tpu/cloud/exact-nested/')),'ASSIGNED_SCOPE')
    return value

def apply(bundle,manifest,expected,repository,*,write=False):
    bundle=Path(bundle).resolve();repository=Path(repository).resolve();manifest=Path(manifest)
    require(not manifest.is_symlink() and sha(manifest)==expected,'REPLACEMENT_MAP_SHA')
    m=json.loads(manifest.read_text());require(m['kind']=='M4_FUNCTIONAL_OUTPUT_CLOSED_REPLACEMENT_MAP_V1','REPLACEMENT_MAP_KIND')
    files=m['files'];observed={}
    for p in sorted(bundle.rglob('*')):
        require(not p.is_symlink() and (p.is_dir() or stat.S_ISREG(p.lstat().st_mode)),'SOURCE_REGULAR')
        if p.is_file():require(p.suffix!='.pyc','SOURCE_BYTECODE');observed[p.relative_to(bundle).as_posix()]=descriptor(p)
    require(observed=={n:r['after'] for n,r in files.items()},'CLOSED_SOURCE_INVENTORY')
    selected=[]
    for n,r in files.items():
        name(n);p=repository/n
        for parent in [p,*p.parents]:
            if parent==repository.parent:break
            require(not parent.is_symlink(),'DESTINATION_SYMLINK')
        require(r['action'] in ('KEEP','REPLACE','CREATE'),'REPLACEMENT_ACTION')
        if r['before'] is None:require(not p.exists(),'NEW_DESTINATION_EXISTS')
        else:require(p.is_file() and descriptor(p)==r['before'],'DESTINATION_PREIMAGE_'+n)
        require((r['action']=='CREATE')==(r['before'] is None),'CREATE_PREIMAGE')
        require((r['action']=='KEEP')==(r['before']==r['after']),'UNCHANGED_DESCRIPTOR')
        if r['action']!='KEEP':selected.append(n)
    restored={};created_dirs=[]
    if write:
        try:
            for n in selected:
                p=repository/n;restored[n]=p.read_bytes() if p.exists() else None
                missing=[];parent=p.parent
                while not parent.exists():missing.append(parent);parent=parent.parent
                for parent in reversed(missing):parent.mkdir();created_dirs.append(parent)
                with tempfile.NamedTemporaryFile(prefix='.m4-functional-',dir=p.parent,delete=False) as f:
                    temporary=Path(f.name);f.write((bundle/n).read_bytes());f.flush();os.fsync(f.fileno())
                try:os.replace(temporary,p)
                finally:temporary.unlink(missing_ok=True)
            for n,r in files.items():require(descriptor(repository/n)==r['after'],'ADOPTION_READBACK_'+n)
        except BaseException:
            for n,data in restored.items():
                p=repository/n
                if data is None:p.unlink(missing_ok=True)
                else:p.write_bytes(data)
            for p in reversed(created_dirs):
                if p.exists() and not any(p.iterdir()):p.rmdir()
            raise
    return {'kind':'M4_FUNCTIONAL_OUTPUT_REPLACEMENT_READBACK','status':'APPLIED_READBACK_PASS' if write else 'PREIMAGE_CHECK_PASS','source_files':len(files),'changed_files':len(selected),'changed_paths':selected,'science_manifest_sha256':m['science_manifest_sha256'],'source_variant':m['source_variant'],'provider_calls':0,'TPU':'NOT_RUN','m4_status':'NOT_QUALIFIED'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle-root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--repository',type=Path,required=True);g=p.add_mutually_exclusive_group();g.add_argument('--apply',action='store_true');g.add_argument('--check-only',action='store_true');a=p.parse_args();print(json.dumps(apply(a.bundle_root,a.manifest,a.manifest_sha256,a.repository,write=a.apply),indent=2))
