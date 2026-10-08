"""Reconstruct the complete control tree from exact adopted canonical files."""
import argparse,hashlib,json,shutil
from pathlib import Path,PurePosixPath

def read(p):return json.loads(Path(p).read_text())
def seal(p):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def safe(root,name):
 q=PurePosixPath(name)
 if q.is_absolute()or '..'in q.parts or '\\'in name or not q.parts:raise ValueError('UNSAFE_MAP_PATH')
 p=root/q
 if any(t.is_symlink()for t in (p,*p.parents)if t==root or root in t.parents):raise ValueError('MAP_SYMLINK')
 return p

def materialize(root,source_map,adoption_map,output):
 root=Path(root).resolve();output=Path(output).absolute();source=read(source_map);adoption=read(adoption_map);rows=adoption['files']+adoption['exact_prerequisites']
 if len(rows)!=len(source)or len({r['source']for r in rows})!=len(rows)or len({r['target']for r in rows})!=len(rows)or {r['source']for r in rows}!=set(source):raise ValueError('MAP_MEMBERS')
 if output.exists():raise ValueError('FRESH_CONTROL_TREE_REQUIRED')
 # Validate all canonical inputs before the first write.
 for r in rows:
  expected={k:r[k]for k in ('bytes','sha256')}
  if source[r['source']]!=expected:raise ValueError('MAP_RECORD:'+r['source'])
  safe(output,r['source']);p=safe(root,r['target'])
  if not p.is_file()or seal(p)!=expected:raise ValueError('CANONICAL_SOURCE_SHA:'+r['source'])
 output.mkdir(parents=True,exist_ok=False)
 for r in rows:
  p=safe(output,r['source']);p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(safe(root,r['target']),p)
  if seal(p)!=source[r['source']]:raise ValueError('MATERIALIZED_SOURCE_SHA:'+r['source'])
 return {'status':'CANONICAL_CONTROL_TREE_EXACT','files':len(rows),'source_map_sha256':seal(Path(source_map))['sha256'],'adoption_map_sha256':seal(Path(adoption_map))['sha256'],'source':'ADOPTED_CANONICAL_FILES_ONLY','output':str(output)}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source-map',required=True);p.add_argument('--adoption-map',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(materialize(a.root,a.source_map,a.adoption_map,a.output),sort_keys=True))
