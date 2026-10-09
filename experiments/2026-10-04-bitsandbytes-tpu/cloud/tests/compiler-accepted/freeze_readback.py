"""Read frozen source, adoption, dependency, and every retained artifact seal."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=next(p for p in HERE.parents if(p/'packages/bitsandbytes-tpu/pyproject.toml').is_file())
def read(p):return json.loads(p.read_text())
def seal(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def main():
 source=read(HERE/'source-map.json');adoption=read(HERE/'adoption-map.json');rows=adoption['files']+adoption['exact_prerequisites'];assert len(rows)==len(source)and len({r['source']for r in rows})==len(rows)and len({r['target']for r in rows})==len(rows)
 for n,r in source.items():assert not(HERE/n).is_symlink()and seal(HERE/n)==r,n
 for r in rows:assert{k:r[k]for k in('bytes','sha256')}==source[r['source']]
 dep=read(HERE/'dependency-map.json')
 for n,r in dep['external_files'].items():assert seal(ROOT/n)==r,n
 review=read(HERE/'review.json')
 for n,r in review['seals'].items():assert seal(HERE/n)==r,n
 sys.path.insert(0,str(HERE/'cloud'));import compiler_cloud_contract as C
 C.dependency_fields(adoption);C.dependency_payload(HERE,adoption);assert review['candidate_frozen']is True and review['actual_corrected_libtpu_parse']=='NOT_RUN'and review['M6']=='NOT_QUALIFIED'
 assert(HERE/'final-packet1/payload.zip').read_bytes()==(HERE/'final-packet2/payload.zip').read_bytes()
 print(json.dumps({'status':'FROZEN_LIBTPU021_FLAG_READBACK_PASS','source_files':len(source),'external_files':len(dep['external_files']),'retained_artifact_seals':len(review['seals']),'actual_compiler_TPU':'NOT_RUN','provider_calls':0}))
if __name__=='__main__':main()
