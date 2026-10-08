"""Exercise the actual remote phase dispatcher with an isolated no-device step double."""
import copy,json,shutil,sys,time
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent));import remote,native_contract as NC
import native_controls as FIX
OUT=None;PACKET=None;SEED=None
def main():
 OUT.mkdir(exist_ok=False);rows=[];m=NC.read(PACKET/'manifest.json')
 for fault in (None,'wrong-root-oracle','wrong-root-manifest','wrong-variant','wrong-cpu-stage','resubmit','expired-deadline','parent-cleanup'):
  d=OUT/(fault or'correct');d.mkdir();shutil.copytree(PACKET,d/'payload');shutil.copyfile(PACKET/'payload.zip',d/'payload.zip');shutil.copytree(SEED/'records',d/'records')
  receipt=NC.read(d/'records/receipt.json');receipt.update(status='CPU_ORACLE_READY_TPU_NOT_RUN',tpu_status='NOT_RUN',science_deadline_epoch=time.time()+800,error=None);receipt['allocation_epoch']=time.time();gate=NC.read(SEED/'host/launch.json')
  if fault=='wrong-root-oracle':gate['oracle_sha256']='0'*64
  if fault=='wrong-root-manifest':gate['native_manifest_sha256']='0'*64
  if fault=='wrong-variant':gate['source_variant']='nested-v1'
  if fault=='wrong-cpu-stage':receipt['cpu_status']='NOT_RUN'
  if fault=='resubmit':receipt['tpu_status']='RUNNING'
  if fault=='expired-deadline':receipt['science_deadline_epoch']=time.time()-100
  NC.write(d/'launch.json',gate);NC.write(d/'records/receipt.json',receipt);calls=[]
  def step(out,label,argv,deadline,limit,**kwargs):
   calls.append(argv);assert label=='12-native-parent' and Path(argv[2]).name=='coordinator.py' and kwargs['tpu']is True and argv[-2:]==['--compiler-dumps','off'] and float(argv[argv.index('--deadline-epoch')+1])<deadline and argv[argv.index('--manifest-sha256')+1]==NC.MANIFEST_SHA
   cleanup={'errors':[]};rec={'status':'PASS','exit_code':0,'cleanup':cleanup}
   if fault=='parent-cleanup':cleanup['errors']=[{'type':'SYNTHETIC_CLEANUP_FAILURE'}]
   return rec
  with patch.object(remote,'run_step',step):code=remote.execute(d,NC.sha(d/'payload.zip'),receipt['allocation_epoch'],'native')
  actual=NC.read(d/'records/receipt.json');assert(code==0)==(fault is None),(fault,actual)
  assert len(calls)==(1 if fault in(None,'parent-cleanup') else 0)
  rows.append({'case':fault or'correct-root-gate','status':'PASS','parent_launch_requests':len(calls),'scope':'NO_DEVICE_DOUBLE_ACTUAL_DISPATCH_FUNCTION'})
  FIX.compact_closed_case(d)
 NC.write(OUT/'results.json',{'status':'PASS','controls':rows,'actual_tpu':'NOT_RUN','provider_calls':0});print(json.dumps({'status':'PASS','controls':len(rows)}))
if __name__=='__main__':
 import argparse,tarfile,hashlib
 p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--seed-archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();PACKET=a.packet.resolve();archive=a.seed_archive.resolve();OUT=a.output.resolve();SEED=OUT.with_name(OUT.name+'-seed');SEED.mkdir(exist_ok=False);inventory=NC.read(archive.with_suffix('').with_suffix('.inventory.json'))['members']
 with tarfile.open(archive,'r:gz')as tar:
  for member in tar.getmembers():
   if not member.name.startswith('records/') and member.name!='host/launch.json':continue
   assert member.isfile() and not Path(member.name).is_absolute() and '..'not in Path(member.name).parts;raw=tar.extractfile(member).read();r=inventory[member.name];assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256'];target=SEED/member.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
 try:main()
 finally:shutil.rmtree(SEED)
