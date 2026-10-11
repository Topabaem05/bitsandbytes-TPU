"""Materialize one pinned source-only packet. No account calls or installs."""
import argparse,shutil,sys,zipfile
from pathlib import Path
import contract as C

def build(source,review,review_sha,out):
 source=Path(source);out=Path(out);C.require(not out.exists(),'FRESH_PACKET_OUTPUT');C.require(C.sha(review)==review_sha,'REVIEW_HASH');out.mkdir(parents=True)
 try:
  shutil.copytree(source/'nested-exact',out/'nested-exact',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
  shutil.copytree(source/'cloud/exact-nested',out/'control',ignore=shutil.ignore_patterns('__pycache__','*.pyc','tests'))
  shutil.copyfile(review,out/'source-review.json')
  files=C.inventory(out);C.limits(files)
  manifest={'kind':'M4_EXACT_NESTED_PACKET_V1','files':files,'science_manifest_sha256':C.SCIENCE_SHA,'allocation_seconds':3600,'closure_reserve_seconds':C.CLOSURE_RESERVE,'state_start_reserve_seconds':C.STATE_RESERVE,'scope':'FIXED79_THEN_ROOT_ACCEPTED_STATE8','device_acceptance':None,'m4_status':'NOT_QUALIFIED'}
  C.write(out/'manifest.json',manifest);C.admit(out,manifest)
  zpath=out.parent/(out.name+'.zip')
  with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED)as z:
   for name in sorted(C.inventory(out)):z.write(out/name,name)
  C.require(zpath.stat().st_size<=C.MAX_BYTES,'PACKET_OUTER_LIMIT')
  return {'packet':str(out),'archive':str(zpath),'packet_sha256':C.sha(zpath),'packet_bytes':zpath.stat().st_size,'manifest_sha256':C.sha(out/'manifest.json'),'science_manifest_sha256':C.SCIENCE_SHA,'TPU':'NOT_RUN'}
 except BaseException:shutil.rmtree(out);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--source-review',type=Path,required=True);p.add_argument('--source-review-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(__import__('json').dumps(build(a.source,a.source_review,a.source_review_sha256,a.output)))
