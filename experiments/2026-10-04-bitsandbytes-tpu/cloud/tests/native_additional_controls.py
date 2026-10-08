"""Native manifest rejection and actual generated bootstrap controls; no service calls."""
import copy,json,shutil,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
import remote,owner,native_contract as NC
OUT=None;PACKET=None;SEED=None
def main():
 OUT.mkdir(exist_ok=False);m=NC.read(PACKET/'manifest.json');rows=[]
 mutations={
  'missing-recorder':lambda x:x['files'].pop('native/recorder.py'),
  'wrong-recorder':lambda x:x['files']['native/recorder.py'].update(sha256='0'*64),
  'extra-native-source':lambda x:x['files'].update({'native/extra.py':{'sha256':'0'*64,'bytes':1}}),
  'wrong-source-variant':lambda x:x.update(native_source_variant='nested-v1'),
  'wrong-native-manifest':lambda x:x.update(native_manifest_sha256='0'*64),
  'wrong-precision':lambda x:x.update(precision='default'),
  'unrequested-native':lambda x:x.update(experiment='transfer-api42'),
  'unknown-mode':lambda x:x.update(experiment='unknown-native'),
 }
 remote.verify_experiment(m);rows.append({'case':'correct-native-manifest','status':'PASS'})
 for name,mutate in mutations.items():
  bad=copy.deepcopy(m);mutate(bad)
  try:remote.verify_experiment(bad)
  except ValueError as e:rows.append({'case':name,'status':'PASS','rejection':str(e)})
  else:raise AssertionError(name)
 import tarfile
 with tarfile.open(SEED,'r:gz')as tar:generated=tar.extractfile('host/08-unpack.py').read().decode()
 for name in ('correct','ambient-cached-contract','wrong-control-bytes'):
  base=OUT/name;neutral=base/'neutral';neutral.mkdir(parents=True);control=base/'remote/control';control.mkdir(parents=True)
  for p in (PACKET/'cloud').rglob('*.py'):
   target=control/p.relative_to(PACKET/'cloud');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
  shutil.copyfile(PACKET/'payload.zip',base/'remote/payload.zip')
  code=generated.replace("b=Path('/content/bnb-tpu-first')",f'b=Path({str((base/"remote").resolve())!r})')
  if name=='ambient-cached-contract':code="import types,sys\nsys.modules['native_contract']=types.ModuleType('native_contract')\nsys.modules['native_contract'].MODE='WRONG_AMBIENT_SOURCE'\n"+code
  if name=='wrong-control-bytes':(control/'native_contract.py').write_text('raise RuntimeError("WRONG_CONTROL_BODY")\n')
  script=neutral/'generated.py';script.write_text(code)
  record=remote.run_step(base,'generated-operation',[sys.executable,'-B',str(script.resolve())],time.time()+25,20,cwd=neutral,tpu=False)
  valid=record['status']=='PASS'
  assert valid==(name!='wrong-control-bytes'),record
  assert record['cleanup']['errors']==[] and all(g['status']=='CLEANUP_VERIFIED'for g in record['cleanup']['groups'])
  if valid:assert (base/'remote/payload/manifest.json').read_bytes()==(PACKET/'manifest.json').read_bytes()
  else:assert b'CONTROL_SOURCE_HASH'in(base/'steps/generated-operation/stderr.raw').read_bytes()
  rows.append({'case':'generated-'+name,'status':'PASS','actual_local_subprocess':True,'scope':'UNPACK_ONLY_NO_RUNTIME_DEVICE_OR_PROVIDER'})
 NC.write(OUT/'results.json',{'status':'PASS','controls':rows,'provider_calls':0,'TPU':'NOT_RUN'});print(json.dumps({'status':'PASS','controls':len(rows)}))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--seed-archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();PACKET=a.packet.resolve();SEED=a.seed_archive.resolve();OUT=a.output.resolve();main()
