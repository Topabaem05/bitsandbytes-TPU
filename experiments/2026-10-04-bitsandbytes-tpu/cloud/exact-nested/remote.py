"""Owned exact-nested stages. Entry only through the frozen packet driver."""
import argparse,json,math,os,secrets,shutil,sys,time,urllib.request
from pathlib import Path
import contract as C
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'ownership'))
from cleanup_lifecycle import Ownership
from lifecycle import durable_json
CPU_ENV={'ATEN_CPU_CAPABILITY':'avx2','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}

def environment(tpu=False):
 env={k:v for k,v in os.environ.items()if k not in{'PYTHONPATH','PYTHONHOME','TPU_LIBRARY_PATH','PTXLA_TPU_LIBRARY_PATH','PJRT_DEVICE'}}
 env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TPU_LOG_DIR='disabled',**CPU_ENV)
 if tpu:env['PJRT_DEVICE']='TPU'
 return env

def run_step(records,label,argv,deadline,limit,*,cwd,tpu=False,extra_env=None):
 d=records/'steps'/label;d.mkdir(parents=True,exist_ok=False);owner=Ownership(d);env=environment(tpu);env.update(extra_env or{});streams=[];rec={'status':'BLOCKED','argv':argv,'exit_code':None,'environment':{'CPU':CPU_ENV,'PJRT_DEVICE':env.get('PJRT_DEVICE')}}
 try:
  streams=[(d/n).open('wb',buffering=0)for n in('stdout.raw','stderr.raw')];budget=min(limit,deadline-time.time());C.require(budget>0,'REMOTE_ORIGINAL_DEADLINE');rec['timeout_seconds']=budget
  with owner.guard(budget):
   child=owner.launch(argv,record=d/'ownership.json',cwd=cwd,env=env,stdout=streams[0],stderr=streams[1])
   while child.poll()is None:
    # Enforce the cumulative physical envelope while a graph-producing child runs.
    sizes=[]
    for path in records.rglob('*'):
     try:
      relative=path.relative_to(records).parts
      pending=relative[0]in('device79','devicestate')and ('trace'in relative or path.name.endswith(('.hlo.txt','.hlo.pb','.textproto','.metrics.txt')))
      if path.is_file()and not pending:sizes.append(path.stat().st_size)
     except FileNotFoundError:pass
    C.require(len(sizes)<=C.MAX_FILES and sum(sizes)<=C.INNER_BYTES-1024*1024,'CUMULATIVE_EVIDENCE_RESERVE')
    time.sleep(.1)
   child.wait();rec.update(status='PASS'if child.returncode==0 else'CHILD_FAILED',exit_code=child.returncode)
 except BaseException as e:rec['error']={'type':type(e).__name__,'message':str(e)}
 finally:
  rec['original_outcome']={k:rec.get(k)for k in('status','exit_code','error')}
  for f in streams:
   try:os.fsync(f.fileno())
   except Exception as e:owner.note_error('fsync_stage_stream',e)
   finally:f.close()
  rec['cleanup']=owner.summary()
  if rec['cleanup']['errors']:rec['status']='BLOCKED_CLEANUP'
  durable_json(d/'result.json',rec)
 return rec

def common(packet):
 n=packet/'nested-exact';args=[]
 for key,name in{'backend_probe':'probe_backend.py','route_probe':'probe_routes.py','precision_probe':'probe_precision.py','transfer_admission':'transfer_admission.py','nested_reference':'probe_nested.py'}.items():args+=['--'+key.replace('_','-'),str(n/'legacy'/name)]
 for key,value in{'science_manifest':n/'source/science-manifest.json','science_manifest_sha256':C.SCIENCE_SHA,'admission':n/'integration/source-admission.json','admission_sha256':C.ADMISSION_SHA,'patch_manifest':n/'integration/patch-manifest.json'}.items():args+=['--'+key.replace('_','-'),str(value)]
 return args

def oracle_args(records):return ['--oracle',str(records/'cpu79'),'--oracle-sha256',C.sha(records/'cpu79/oracle-seal.json')]
def review_args(packet):return ['--source-review',str(packet/'source-review.json'),'--source-review-sha256',C.sha(packet/'source-review.json')]
def device_args(packet,records):
 n=packet/'nested-exact';return ['--plugin-source',str(n/'integration/plugin/src/bitsandbytes_tpu'),'--cpu-gate',str(records/'cpu-gate.json'),'--cpu-gate-sha256',C.sha(records/'cpu-gate.json')]

def download(records,target):
 target.mkdir()
 for r in records:
  C.require(Path(r['filename']).name==r['filename']and r['url'].startswith(('https://files.pythonhosted.org/','https://download.pytorch.org/')),'PINNED_WHEEL_URL')
  p=target/r['filename']
  with urllib.request.urlopen(r['url'],timeout=60)as src,p.open('xb')as dst:
   total=0
   for b in iter(lambda:src.read(1024*1024),b''):total+=len(b);C.require(total<=r['size'],'WHEEL_SIZE');dst.write(b)
  C.require(total==r['size']and C.sha(p)==r['sha256'],'WHEEL_HASH')

def operation(args):
 base=args.base;birth=args.allocation_epoch;C.budget(birth,1);deadline=birth+3600-C.CLOSURE_RESERVE;packet=base/'payload';records=base/'records'
 if args.command=='unpack':
  C.recover(base/'payload.zip',args.packet_sha256,packet,packet=True);records.mkdir();C.write(records/'allocation.json',{'birth_epoch':birth,'original_deadline_epoch':birth+3600,'science_deadline_epoch':deadline,'packet_sha256':args.packet_sha256});return
 C.admit(packet);C.require(C.read(records/'allocation.json')['birth_epoch']==birth and C.read(records/'allocation.json')['packet_sha256']==args.packet_sha256,'ALLOCATION_IMMUTABLE')
 n=packet/'nested-exact';python=base/'venv/bin/python';step=lambda label,argv,limit=180,tpu=False,extra=None:run_step(records,label,[str(x)for x in argv],deadline,limit,cwd=base,tpu=tpu,extra_env=extra)
 def checked(label,argv,limit=180,tpu=False,extra=None):
  rec=step(label,argv,limit,tpu,extra);C.require(rec['status']=='PASS','OWNED_STEP_FAILED:'+label);return rec
 if args.command=='install':
  checked('bootstrap',[sys.executable,'-B',HERE/'python_bootstrap.py','--output',base/'bootstrap'],180)
  bootstrap=base/'bootstrap/interpreter/python/bin/python3.12';checked('venv',[bootstrap,'-B','-m','venv',base/'venv'],90)
  runtime=C.read(n/'runtime/requirements.lock.json');build=C.read(packet/'control/build-requirements.lock.json');C.require(runtime['marker_environment']==build['marker_environment'],'LOCK_MARKERS');merged={}
  for r in runtime['wheels']+build['wheels']:
   C.require(r['filename']not in merged or (merged[r['filename']]['sha256'],merged[r['filename']]['size'])==(r['sha256'],r['size']),'WHEEL_LOCK_CONFLICT');merged[r['filename']]=r
  C.write(base/'download-records.json',list(merged.values()));checked('download',[bootstrap,'-B',HERE/'remote.py','download','--config',base/'download-records.json','--target',base/'wheel-cache'],600)
  req=base/'install.txt';req.write_text(''.join(f"{r['name']} @ {(base/'wheel-cache'/r['filename']).as_uri()} --hash=sha256:{r['sha256']}\n"for r in merged.values()))
  checked('install-runtime',[python,'-B','-m','pip','install','--no-index','--no-deps','--no-compile','--require-hashes','-r',req],240)
  checked('verify-runtime',[python,'-B',n/'runtime/resolve.py','--verify-only','--cache',base/'wheel-cache','--lock',n/'runtime/requirements.lock.json'],90)
  for key in('upstream','plugin'):
   source=base/key;shutil.copytree(n/'integration'/key,source);dest=base/('built-'+key);dest.mkdir();checked('build-'+key,[python,'-B','-m','pip','wheel','--no-deps','--no-build-isolation','--no-cache-dir','--wheel-dir',dest,source],180,extra={'BNB_SKIP_CMAKE':'1'})
   wheels=list(dest.glob('*.whl'));C.require(len(wheels)==1,'ONE_BUILT_WHEEL');(records/'wheels').mkdir(exist_ok=True);shutil.copyfile(wheels[0],records/'wheels'/(key+'.whl'))
  checked('install-source',[python,'-B','-m','pip','install','--no-index','--no-deps','--no-compile',*list((base/'built-upstream').glob('*.whl')),*list((base/'built-plugin').glob('*.whl'))],120)
  checked('installed-source',[python,'-B',HERE/'installed_source.py','--packet',packet,'--output',records/'installed-source.json'],45)
 elif args.command in('cpu79','device79','cpustate','devicestate'):
  state=args.command.endswith('state');device=args.command.startswith('device');label=args.command;argv=[str(python),'-B',str(n/'source'/('probe_state8_exact.py'if state else'probe_nested_exact.py')),'execute'if device else'prepare',*common(packet)]
  if state or device:argv+=oracle_args(records)
  if device:argv+=device_args(packet,records)
  if state and device:argv+=['--state-oracle',str(records/'cpustate'),'--state-oracle-sha256',C.sha(records/'cpustate/oracle-seal.json'),'--accepted79',str(records/'accepted79.json'),'--accepted79-sha256',C.sha(records/'accepted79.json')]
  if state:
   actual=C.read(records/'device79/receipt.json');C.accepted79(records/'accepted79.json',C.sha(records/'accepted79.json'),packet_sha=args.packet_sha256,receipt_sha=C.sha(records/'device79/receipt.json'),oracle_sha=C.sha(records/'cpu79/oracle-seal.json'),completed_epoch=actual['finished_epoch'],birth=birth)
   C.require(C.read(records/'cpu-gate.json').get('status')=='QUALIFIED_LINUX_CPU_ORACLE_VERIFIED','CPU79_ROOT_GATE')
   if device:C.require(C.read(records/'state-cpu-launch-audit.json').get('status')=='ACTUAL_STATE_CPU_LAUNCH_VERIFIED'and C.read(records/'state-cpu-launch-audit.json').get('state_oracle_sha256')==C.sha(records/'cpustate/oracle-seal.json'),'SEPARATE_ACTUAL_STATE_CPU_LAUNCH_AUDIT')
  argv+=['--output',str(records/label),*review_args(packet),'--process-token',secrets.token_hex(16),'--deadline-epoch',str(deadline)]
  result=step(label,argv,1200 if not state else 600,tpu=device)
  C.require(result['status']in('PASS','CHILD_FAILED')and result['exit_code']in(0,2),'SCIENTIFIC_STAGE_FAILED:'+label)
 else:raise ValueError('UNKNOWN_OPERATION')
 C.write(records/'phase.json',{'kind':'M4_EXACT_NESTED_STAGE','phase':args.command,'status':'TERMINAL_LOCAL_REVIEW_REQUIRED','packet_sha256':args.packet_sha256,'allocation_epoch':birth,'finished_epoch':time.time(),'m4_status':'NOT_QUALIFIED'})

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=('unpack','install','cpu79','device79','cpustate','devicestate','export','download'));p.add_argument('--base',type=Path);p.add_argument('--allocation-epoch',type=float);p.add_argument('--packet-sha256');p.add_argument('--config',type=Path);p.add_argument('--target',type=Path);p.add_argument('--label');a=p.parse_args()
 if a.command=='download':download(C.read(a.config),a.target)
 elif a.command=='export':
  C.budget(a.allocation_epoch,1,reserve=30);path=a.base/(a.label+'.zip');summary=C.export(a.base/'records',path);C.write(a.base/(a.label+'-export.json'),summary)
  import transport;transport.split(path,a.base/(a.label+'-parts'),expected_sha256=summary['sha256'],expected_bytes=summary['bytes'])
 else:operation(a)
if __name__=='__main__':main()
