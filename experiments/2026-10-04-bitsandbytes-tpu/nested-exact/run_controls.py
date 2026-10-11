"""Portable owned replay of the original16 and new7 controls; dispose bulky fixtures."""
import argparse,json,shutil,sys,tempfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;CLOUD=HERE.parent/'cloud/exact-nested';sys.path.insert(0,str(CLOUD));import contract as C
from remote import run_step

def run(historical,output,mode):
 historical=Path(historical);output=Path(output);output.mkdir(parents=True,exist_ok=False)
 oracle=historical/'cloud-20261008-nested-browser-adopted/recovered/cpu-oracle';lock=historical/'r4-actual-arithmetic-analysis/readback-map.json';fraction=historical/'r4-exact-muldiv-preparation/fraction_reference.py';old=historical/'cloud-20261008-nested-browser-adopted/recovered/nested/receipt.json'
 needed=[oracle/'oracle-seal.json']+([lock,fraction,old]if mode in('all','original')else[])
 for path in needed:C.require(path.is_file()and not path.is_symlink(),'MISSING_HISTORICAL_READONLY_FIXTURE:'+str(path))
 pre={str(p):C.sha(p)for p in needed};reviews={};disposed=[]
 with tempfile.TemporaryDirectory(prefix='bnb-m4-original-controls-')as temporary:
  base=Path(temporary);repo=base/'repo';ex=repo/'experiments/2026-10-04-bitsandbytes-tpu';shutil.copytree(HERE/'legacy',ex);shutil.copytree(CLOUD/'ownership',ex/'cloud/ownership');generation=base/'generation';shutil.copytree(HERE/'integration/upstream',generation/'build-v3/source');shutil.copytree(HERE/'integration/plugin',generation/'plugin')
  for src,dst in [('adoption-map.json','adoption-map.json'),('source-admission.json','source-admission.json'),('patch-manifest.json','build-v3/patch-manifest.json')]:shutil.copyfile(HERE/'integration'/src,generation/dst)
  for phase in(['original','new']if mode=='all'else[mode]):
   if phase=='original':
    command=[sys.executable,'-B',str(HERE/'source/controls.py'),'--repo',str(repo),'--generation',str(generation),'--oracle',str(oracle),'--readback-lock',str(lock),'--mul-reference',str(fraction),'--old-failed-receipt',str(old),'--output',str(base/phase)]
   else:command=[sys.executable,'-B',str(HERE/'new_controls.py'),'--scratch',str(HERE),'--repo',str(repo),'--generation',str(generation),'--oracle',str(oracle),'--output',str(base/phase)]
   result=run_step(output,phase,command,time.time()+240,220,cwd=repo);review=base/phase/'review.json'
   if review.exists():reviews[phase]=C.read(review);C.write(output/(phase+'-review.json'),reviews[phase])
   C.require(result['status']=='PASS','CONTROL_REPLAY_FAILED:'+phase)
  files=C.inventory(base);disposed={'files':len(files),'bytes':sum(r['bytes']for r in files.values())}
 C.require(all(C.sha(p)==h for p,h in pre.items()),'HISTORICAL_FIXTURE_MUTATION');C.write(output/'result.json',{'status':'PASS','controls':{k:v.get('tests_run',v.get('tests'))for k,v in reviews.items()},'historical_readonly_pre_post':pre,'temporary_removed':not Path(temporary).exists(),'disposed':disposed,'TPU':'NOT_RUN','provider_calls':0,'m4_status':'NOT_QUALIFIED'})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--historical-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=('all','original','new'),default='all');a=p.parse_args();run(a.historical_root,a.output,a.mode)
