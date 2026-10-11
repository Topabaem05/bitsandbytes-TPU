"""Build a fresh exact packet only after root source review and genuine M4 acceptance."""
import argparse,json,shutil,zipfile
from pathlib import Path
import contract as C

def build(source,review,review_sha,m4,m4_sha,out):
 source=Path(source);out=Path(out);C.require(not out.exists()and not out.with_suffix('.zip').exists(),'FRESH_PACKET_OUTPUT');a=C.source(source);C.require(C.sha(review)==review_sha,'ROOT_REVIEW_SHA');C.source_review(source,review);a.m4_dependency(m4,m4_sha);out.mkdir(parents=True)
 try:
  for name in ['source','integration','legacy','runtime','control']:shutil.copytree(source/name,out/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
  shutil.copyfile(review,out/'source-review.json');shutil.copyfile(m4,out/'accepted-m4.json');files=C.inventory(out);C.limits(files)
  m={'kind':'M5_EXACT1024_TRAINING_PACKET_V1','files':files,'science_manifest_sha256':C.science(out),'m4_acceptance_sha256':m4_sha,'allocation_seconds':3600,'closure_reserve_seconds':660,'budget':{'total':3600,'install':1200,'science':1800,'retrieval':600,'cleanup':60},'scope':{'steps':20,'checkpoint_after_update':10,'resume_steps':[11,20],'first_scales':1024},'device_acceptance':None,'m5_status':'NOT_QUALIFIED'};C.write(out/'manifest.json',m);C.admit(out,m)
  path=out.with_suffix('.zip')
  with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED)as z:
   for name in sorted(C.inventory(out)):z.write(out/name,name)
  C.require(path.stat().st_size<=C.MAX_BYTES,'PACKET_OUTER_LIMIT');return {'packet':str(out),'archive':str(path),'packet_sha256':C.sha(path),'manifest_sha256':C.sha(out/'manifest.json'),'bytes':path.stat().st_size,'provider_calls':0,'m5_status':'NOT_QUALIFIED'}
 except BaseException:shutil.rmtree(out);out.with_suffix('.zip').unlink(missing_ok=True);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--source-review',type=Path,required=True);p.add_argument('--source-review-sha256',required=True);p.add_argument('--m4-acceptance',type=Path,required=True);p.add_argument('--m4-acceptance-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(build(a.source,a.source_review,a.source_review_sha256,a.m4_acceptance,a.m4_acceptance_sha256,a.output)))
