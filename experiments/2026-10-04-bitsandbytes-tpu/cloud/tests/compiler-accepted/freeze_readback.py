"""Read frozen source, adoption, evidence, and dependency seals without state changes."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def read(p):return json.loads(p.read_text())
def seal(p):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def main():
 source=read(HERE/'source-map.json')
 for n,r in source.items():
  p=Path(n);assert not p.is_absolute()and'..'not in p.parts and not(HERE/p).is_symlink()and seal(HERE/p)==r,n
 adoption=read(HERE/'adoption-map.json');rows=adoption['files']+adoption['exact_prerequisites'];assert len(rows)==len(source)and len({r['source']for r in rows})==len(rows)and len({r['target']for r in rows})==len(rows)
 for r in rows:assert {k:r[k]for k in ('bytes','sha256')}==source[r['source']],r['source']
 dependency=read(HERE/'dependency-map.json')
 for n,r in dependency['external_files'].items():assert seal(ROOT/n)==r,n
 sys.path.insert(0,str(HERE/'cloud'));import compiler_cloud_contract as C
 C.dependency_fields(adoption);C.dependency_payload(HERE,adoption)
 review=read(HERE/'review.json')
 for n,r in review['seals'].items():assert seal(HERE/n)==r,n
 assert (HERE/'packet-v1/payload.zip').read_bytes()==(HERE/'packet-v2/payload.zip').read_bytes()
 print(json.dumps({'status':'FROZEN_ACCEPTED_TRANSITION_READBACK_PASS','source_files':len(source),'external_files':len(dependency['external_files']),'actual_compiler':'NOT_RUN','provider_calls':0}))
if __name__=='__main__':main()
