"""Independent complete regular-member source bundle readback. No extraction."""
import argparse,hashlib,json,stat,tarfile
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(root,archive,expected):
 root=Path(root);archive=Path(archive)
 if sha(archive)!=expected:raise ValueError('SOURCE_ARCHIVE_SHA')
 closure=root/'source-closure.json';manifest=json.loads(closure.read_text());files=manifest['files']
 expected_files={**files,'source-closure.json':{'sha256':sha(closure),'bytes':closure.stat().st_size}}
 with tarfile.open(archive,'r:gz')as t:
  members=t.getmembers()
  if len(members)!=len({m.name for m in members})or {m.name for m in members}!=set(expected_files):raise ValueError('SOURCE_MEMBER_SET')
  for m in members:
   if not m.isfile()or m.issym()or m.islnk()or Path(m.name).is_absolute()or'..'in Path(m.name).parts:raise ValueError('SOURCE_MEMBER_REGULAR_CONFINED')
   data=t.extractfile(m).read();row=expected_files[m.name]
   if len(data)!=row['bytes']or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('SOURCE_MEMBER_BYTES:'+m.name)
   path=root/m.name
   if path.is_symlink()or not stat.S_ISREG(path.lstat().st_mode)or path.read_bytes()!=data:raise ValueError('SOURCE_ROOT_BYTES:'+m.name)
 return {'status':'PASS','archive_sha256':expected,'archive_bytes':archive.stat().st_size,'members':len(members),'source_closure_sha256':sha(closure),'every_member_sha256_verified':True,'source_bytes':sum(r['bytes']for r in expected_files.values()),'provider_calls':0}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--expected-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.resolve().is_relative_to(a.source.resolve()):raise ValueError('READBACK_OUTPUT_OUTSIDE_SOURCE')
 result=verify(a.source,a.archive,a.expected_sha256)
 with a.output.open('x')as f:f.write(json.dumps(result,sort_keys=True,indent=2)+'\n')
 print(json.dumps(result))
