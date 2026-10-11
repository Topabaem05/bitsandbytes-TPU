"""Owned setup and tiny diagnostic stages under the original 600-second clock."""
import argparse,secrets,sys,time
from pathlib import Path
import contract as C,canonical_remote as R,transport
HERE=Path(__file__).resolve().parent

def operation(a):
 base=a.base;packet=base/'payload';records=base/'records';birth=a.allocation_epoch;deadline=birth+600-120
 C.budget(birth,1,reserve=120)
 if a.command=='unpack':
  C.recover(base/'payload.zip',a.packet_sha256,packet,packet=True);records.mkdir();C.write(records/'allocation.json',{'birth_epoch':birth,'original_deadline_epoch':birth+3600,'diagnostic_deadline_epoch':birth+600,'packet_sha256':a.packet_sha256,'m4_status':'NOT_QUALIFIED'});return
 C.admit(packet);allocation=C.read(records/'allocation.json');C.require(allocation['birth_epoch']==birth and allocation['packet_sha256']==a.packet_sha256,'ORIGINAL_DIAGNOSTIC_ALLOCATION_BINDING')
 def checked(label,argv,limit,tpu=False):
  rec=R.run_step(records,label,[str(x)for x in argv],deadline,limit,cwd=base,tpu=tpu);C.require(rec['status']=='PASS','DIAGNOSTIC_OWNED_STEP:'+label);return rec
 python=base/'venv/bin/python'
 try:
  if a.command=='install':
   checked('bootstrap',[sys.executable,'-B',HERE/'python_bootstrap.py','--output',base/'bootstrap'],120);bootstrap=base/'bootstrap/interpreter/python/bin/python3.12';checked('venv',[bootstrap,'-B','-m','venv',base/'venv'],45)
   wheels=C.read(packet/'runtime/requirements.lock.json')['wheels'];C.write(base/'download-records.json',wheels);checked('download',[bootstrap,'-B',HERE/'remote.py','download','--config',base/'download-records.json','--target',base/'wheel-cache'],300)
   req=base/'install.txt';req.write_text(''.join(f"{r['name']} @ {(base/'wheel-cache'/r['filename']).as_uri()} --hash=sha256:{r['sha256']}\n"for r in wheels));checked('install-runtime',[python,'-B','-m','pip','install','--no-index','--no-deps','--no-compile','--require-hashes','-r',req],120);checked('verify-runtime',[python,'-B',packet/'runtime/resolve.py','--verify-only','--cache',base/'wheel-cache','--lock',packet/'runtime/requirements.lock.json'],45)
  elif a.command in('cpu','tpu'):
   argv=[python,'-B',packet/'source/run_diagnostic.py','--mode',a.command,'--output',records/a.command,'--deadline-epoch',str(deadline)]
   if a.command=='tpu':
    oracle=records/'cpu/report.json';C.require(C.read(oracle).get('status')=='CPU_ORACLE_COMPLETE'and C.read(oracle)['runtime']['status']=='PINNED_LINUX_CPU_DIAGNOSTIC_ORACLE','QUALIFIED_CPU_DIAGNOSTIC_REQUIRED');argv+=['--oracle',oracle,'--oracle-sha256',C.sha(oracle)]
   checked(a.command,argv,60 if a.command=='cpu'else 300,tpu=a.command=='tpu')
  else:raise ValueError('DIAGNOSTIC_REMOTE_COMMAND')
 except BaseException as e:C.write(records/(a.command+'-error.json'),{'status':'ERROR','type':type(e).__name__,'message':str(e),'finished_epoch':time.time()});raise
 finally:C.write(records/'phase.json',{'phase':a.command,'packet_sha256':a.packet_sha256,'allocation_epoch':birth,'diagnostic_deadline_epoch':birth+600,'finished_epoch':time.time(),'m4_status':'NOT_QUALIFIED'})

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=('unpack','install','cpu','tpu','download','export'));p.add_argument('--base',type=Path);p.add_argument('--allocation-epoch',type=float);p.add_argument('--packet-sha256');p.add_argument('--config',type=Path);p.add_argument('--target',type=Path);a=p.parse_args()
 if a.command=='download':R.download(C.read(a.config),a.target)
 elif a.command=='export':
  C.budget(a.allocation_epoch,1,reserve=60);summary=C.export(a.base/'records',a.base/'evidence.zip');C.write(a.base/'export.json',summary);transport.split(a.base/'evidence.zip',a.base/'evidence-parts',expected_sha256=summary['sha256'],expected_bytes=summary['bytes'])
 else:operation(a)
if __name__=='__main__':main()
