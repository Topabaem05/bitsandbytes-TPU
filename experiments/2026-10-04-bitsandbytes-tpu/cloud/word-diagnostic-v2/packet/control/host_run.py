"""Own the one driver group until the original allocation deadline."""
import argparse,json,os,sys,time
from pathlib import Path
import contract as C,browser_adoption as B
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'ownership'))
from cleanup_lifecycle import Ownership

def run(config):
 r=C.read(config);argv=r['argv'];C.require(argv[:3]==[sys.executable,'-B',str(HERE/'driver.py')],'EXACT_OWNED_DRIVER');deadline=r['absolute_deadline_epoch'];adoption_path=Path(argv[argv.index('--browser-adoption')+1]);adoption_sha=argv[argv.index('--browser-adoption-sha256')+1];adoption=B.record(adoption_path,adoption_sha,check_time=False);C.require(type(deadline)in(int,float)and deadline==adoption['allocation_epoch']+600 and 0<deadline-time.time()<=600,'EXACT_DIAGNOSTIC_HOST_DEADLINE');out=Path(r['host_output']);out.mkdir(exist_ok=False);owner=Ownership(out);record={'status':'ERROR','argv':argv,'absolute_deadline_epoch':deadline,'config_sha256':C.sha(config),'started_epoch':time.time(),'packet_sha256':r['packet_sha256']}
 try:
  with(out/'stdout.raw').open('wb')as stdout,(out/'stderr.raw').open('wb')as stderr:
   with owner.guard(deadline-time.time()):
    process=owner.launch(argv,record=out/'ownership.json',stdout=stdout,stderr=stderr,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));process.wait();record.update(status='DRIVER_EXITED',exit_code=process.returncode)
 except BaseException as e:record['error']={'type':type(e).__name__,'message':str(e)}
 finally:record.update(finished_epoch=time.time(),cleanup=owner.summary());C.write(out/'host-finalization.json',record)
 C.closed(record['cleanup']);return 0 if record.get('exit_code')==0 else 2
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();raise SystemExit(run(a.config))
