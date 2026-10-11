"""Copy the reviewed additive closure into a repository. Never replace different bytes."""
import argparse,hashlib,json,shutil,stat
from pathlib import Path,PurePosixPath

def materialize(bundle,manifest,expected,repository,check_only=False):
 bundle=Path(bundle).resolve();repository=Path(repository).resolve();manifest=Path(manifest)
 def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 if manifest.is_symlink()or digest(manifest)!=expected:raise ValueError('ADOPTION_MANIFEST_HASH')
 m=json.loads(manifest.read_text());files=m['files'];created=[]
 for name,rec in files.items():
  key=PurePosixPath(name)
  if not name.startswith(('experiments/2026-10-04-bitsandbytes-tpu/nested-exact/','experiments/2026-10-04-bitsandbytes-tpu/cloud/exact-nested/'))or key.is_absolute()or'..'in key.parts or'\\'in name:raise ValueError('ADDITIVE_SCOPE')
  source=bundle/name;target=repository/name
  if source.is_symlink()or not stat.S_ISREG(source.lstat().st_mode)or digest(source)!=rec['sha256']or source.stat().st_size!=rec['bytes']:raise ValueError('SOURCE_CLOSURE_BYTES')
  for parent in [target,*target.parents]:
   if parent==repository.parent:break
   if parent.is_symlink():raise ValueError('DESTINATION_SYMLINK')
  if target.exists()and(not target.is_file()or digest(target)!=rec['sha256']):raise ValueError('DIFFERENT_CANONICAL_BYTES')
 if not check_only:
  try:
   for name in files:
    target=repository/name
    if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(bundle/name,target);created.append(target)
   for name,rec in files.items():
    if digest(repository/name)!=rec['sha256']:raise ValueError('ADOPTION_READBACK')
  except BaseException:
   for target in created:target.unlink(missing_ok=True)
   raise
 return {'kind':'M4_ADDITIVE_MATERIALIZATION','status':'CHECK_PASS'if check_only else'ADOPTION_READBACK_PASS','members':len(files),'created_files':len(created),'science_manifest_sha256':m['science_manifest_sha256'],'canonical_replacements':0,'provider_calls':0,'TPU':'NOT_RUN'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bundle-root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--repository',type=Path,required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args();print(json.dumps(materialize(a.bundle_root,a.manifest,a.manifest_sha256,a.repository,a.check_only)))
