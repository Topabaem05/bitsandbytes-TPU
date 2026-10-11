"""One root-created free browser TPU. Recover diagnostic bytes, then close it."""
import argparse,json,math,re,signal,sys,time
from pathlib import Path
import contract as C,cli_transport as T,browser_adoption as B,kernel_protocol as K,transport,cli_display
HERE=Path(__file__).resolve().parent

def run(a):
 packet=a.packet.resolve();out=a.output.resolve();C.require(not out.exists(),'FRESH_DIAGNOSTIC_HOST_OUTPUT');m=C.packet_preflight(packet,a.packet_zip,a.packet_sha256);T.verify_cli_identity(a.cli_python,C.read(a.cli_identity));d=C.read(a.dispatch)
 C.require(d=={'kind':'ROOT_M4_WORD_DIAGNOSTIC_DISPATCH_V2','status':'ACTUAL_DISPATCH_AUTHORIZED','generation':C.GENERATION,'packet_sha256':a.packet_sha256,'driver_sha256':C.sha(__file__),'cli_identity_sha256':C.sha(a.cli_identity),'diagnostic_seconds':600,'allocation_attempts':0,'allocation_seconds':3600,'m4_status':'NOT_QUALIFIED'},'EXPLICIT_ROOT_DIAGNOSTIC_DISPATCH')
 adoption=B.record(a.browser_adoption,a.browser_adoption_sha256);C.require(adoption['packet_sha256']==a.packet_sha256 and adoption['driver_sha256']==C.sha(__file__)and adoption['cli_identity_sha256']==C.sha(a.cli_identity),'DIAGNOSTIC_BROWSER_BINDINGS');birth=adoption['allocation_epoch'];C.budget(birth,1,reserve=120);session=adoption['session'];base='/content/m4-word-diag-v2-'+session
 out.mkdir(parents=True);api=T.OfficialCLI(a.cli_python,out);serial=0;ready=False;owned=False;registered=False
 record={'kind':'M4_WORD_DIAGNOSTIC_DRIVER_V2','generation':C.GENERATION,'status':'BLOCKED','session':session,'remote_base':base,'packet_sha256':a.packet_sha256,'allocation_epoch':birth,'original_deadline_epoch':birth+3600,'diagnostic_deadline_epoch':birth+600,'allocation_attempts':0,'provider_calls':0,'commands':[],'m4_status':'NOT_QUALIFIED'}
 def save():C.write(out/'driver.json',record)
 def call(label,argv,limit=30,reserve=120,cleanup=False):
  nonlocal serial;serial+=1
  remaining=birth+600-time.time();C.require(remaining>0,'DIAGNOSTIC_HARD_DEADLINE');timeout=min(limit,remaining)if cleanup else C.budget(birth,limit,reserve)
  result,text=api(f'{serial:03d}-'+label,argv,timeout);record['commands'].append({'label':label,**result});record['provider_calls']+=1;save();C.closed(result['cleanup']);C.require(result['status']=='PASS','DIAGNOSTIC_CLI:'+label);return text
 def download(label,path,target,reserve=60):target.parent.mkdir(parents=True,exist_ok=True);return call(label,['download','-s',session,str(path).lstrip('/'),str(target)],30,reserve)
 def kernel(label,body,limit=45,reserve=120,artifact=None,expected=None):
  sources={str(Path(base)/n):v['sha256']for n,v in m['files'].items()if n.startswith('control/')and n.endswith('.py')};modules={Path(n).stem:(str(Path(base)/n),v['sha256'])for n,v in m['files'].items()if n.startswith('control/')and n.endswith('.py')and not Path(n).stem.startswith('__')}
  req=K.request(body,label,base,a.packet_sha256,birth,sources,artifact=artifact,artifact_expected=expected);path=out/(label+'.py');path.write_text(K.render(body,req,sources if ready else{},modules if ready else{}));timeout=math.floor(C.budget(birth,limit,reserve));text=call(label,['exec','-s',session,'--timeout',str(timeout),'-f',str(path)],timeout,reserve);K.marker(text,req);receipt=out/'kernel-readbacks'/(label+'.json');download(label+'-completion',req['completion_path'],receipt,reserve)
  local=None
  if artifact:local=out/'kernel-readbacks'/(label+'.artifact.json');download(label+'-artifact',artifact,local,reserve)
  proof=K.verify(text,req,receipt,local);proof['submitted_source_sha256']=C.sha(path);C.write(out/'kernel-readbacks'/(label+'-proof.json'),proof);return text
 def remote(command,limit=60,reserve=120):
  body='import runpy,sys\nfrom pathlib import Path\n'+f'b=Path({base!r});sys.path.insert(0,str(b/"control"));sys.argv=["remote.py",{command!r},"--base",str(b),"--allocation-epoch",{str(birth)!r},"--packet-sha256",{a.packet_sha256!r}];runpy.run_path(str(b/"control/remote.py"),run_name="__main__")\n'
  artifact=Path(base)/('export.json'if command=='export'else'records/allocation.json'if command=='unpack'else'records/phase.json');expected={}if command=='export'else{'packet_sha256':a.packet_sha256,**({'birth_epoch':birth,'diagnostic_deadline_epoch':birth+600}if command=='unpack'else{'phase':command,'allocation_epoch':birth})}
  return kernel('remote-'+command,body,limit,reserve,artifact,expected)
 def recover():
  remote('export',45,reserve=60);dest=out/'recovery';dest.mkdir();download('export-summary',base+'/export.json',dest/'summary.json');s=C.read(dest/'summary.json');C.require(s['bytes']<=C.MAX_BYTES and s['physical_bytes']<=C.INNER_BYTES and s['members']<=C.MAX_FILES,'DIAGNOSTIC_ENVELOPE_LIMITS');download('export-parts',base+'/evidence-parts/manifest.json',dest/'parts-manifest.json');parts=transport.validate_manifest(C.read(dest/'parts-manifest.json'),s['sha256'],s['bytes']);(dest/'parts').mkdir()
  for i,part in enumerate(parts['parts']):download('export-part-'+str(i),base+'/evidence-parts/'+part['name'],dest/'parts'/part['name'])
  transport.assemble(dest/'parts-manifest.json',dest/'parts',dest/'evidence.zip',expected_manifest_sha256=C.sha(dest/'parts-manifest.json'),expected_sha256=s['sha256'],expected_bytes=s['bytes']);C.recover(dest/'evidence.zip',s['sha256'],dest/'records');record['recovered_records']=str(dest/'records');record['recovered_envelope']=s;save()
  # Transport bodies are removed only after exact host recovery.
  kernel('dispose-transfer','from pathlib import Path\nimport shutil\n'+f'b=Path({base!r});(b/"evidence.zip").unlink();shutil.rmtree(b/"evidence-parts");(b/"export.json").unlink()\n',15,reserve=60)
  import shutil;shutil.rmtree(dest/'parts');return dest/'records'
 save()
 try:
  call('sessions-before',['sessions']);call('usage-before',['usage']);registered=True;call('browser-registration',['--adopt-root-browser-v1',str(a.browser_adoption),a.browser_adoption_sha256]);after=call('sessions-after',['sessions']);rows=cli_display.sessions_output(after)['sessions'];C.require(rows==[{'session':session,'endpoint':adoption['endpoint'],'hardware':adoption['hardware'],'shape':'Standard','variant':'TPU'}],'EXACT_ROOT_BROWSER_ASSIGNMENT');owned=True
  kernel('browser-marker','import hashlib\nfrom pathlib import Path\n'+f'p=Path({adoption["marker_path"]!r});assert p.is_file()and not p.is_symlink()and hashlib.sha256(p.read_bytes()).hexdigest()=={adoption["marker_sha256"]!r}\n',20)
  kernel('readiness','from pathlib import Path\n'+f'b=Path({base!r});assert not b.exists()and not b.is_symlink();(b/"control/ownership").mkdir(parents=True)\n',20)
  for name in sorted(m['files']):
   if name.startswith('control/')and name.endswith('.py'):call('upload-'+name.replace('/','-'),['upload','-s',session,str(packet/name),(base+'/'+name).lstrip('/')],30)
  ready=True;call('upload-packet',['upload','-s',session,str(a.packet_zip),(base+'/payload.zip').lstrip('/')],30);remote('unpack',30)
  error=None
  try:remote('install',300);remote('cpu',70);remote('tpu',310)
  except BaseException as e:error=e;record['operation_error']={'type':type(e).__name__,'message':str(e)};save()
  records=recover();record['status']='DIAGNOSTIC_RECOVERED_ROOT_REVIEW_REQUIRED';save()
  if error:raise error
  for mode in('cpu','tpu'):C.require((records/mode/'report.json').is_file(),'DIAGNOSTIC_REPORT_MISSING:'+mode)
  import inspect_diagnostic
  record['host_inspection']=inspect_diagnostic.inspect(packet,records);C.write(out/'host-inspection.json',record['host_inspection']);save()
  cpu=C.read(records/'cpu/report.json');tpu=C.read(records/'tpu/report.json');C.require(cpu['status']=='CPU_ORACLE_COMPLETE'and cpu['runtime']['status']=='PINNED_LINUX_CPU_DIAGNOSTIC_ORACLE'and tpu['status']=='TPU_DIAGNOSTIC_COMPLETE_WITH_EXACT_COMPARISONS'and len(tpu['rows'])==24,'DIAGNOSTIC_COMPLETE_ROWS')
 except BaseException as e:record['error']={'type':type(e).__name__,'message':str(e)};raise
 finally:
  if owned:
   for label,argv,limit in [('sessions-before-stop',['sessions'],10),('stop-exact',['stop','-s',session],20),('sessions-after-stop',['sessions'],10),('usage-after',['usage'],10)]:
    try:record[label]=call(label,argv,limit,cleanup=True)
    except BaseException as e:record.setdefault('closure_errors',[]).append({'step':label,'type':type(e).__name__,'message':str(e)})
   try:record['physical_closure']=cli_display.resource_empty(record.get('sessions-after-stop',''),record.get('usage-after',''))
   except BaseException as e:record.setdefault('closure_errors',[]).append({'step':'physical-closure','message':str(e)})
  elif registered:
   try:call('remove-provisional',['--remove-browser-provisional-v1',str(a.browser_adoption),a.browser_adoption_sha256],10,cleanup=True)
   except BaseException as e:record.setdefault('closure_errors',[]).append({'step':'remove-provisional','message':str(e)})
  record['finished_epoch']=time.time()
  if record.get('closure_errors')or record['finished_epoch']>birth+600:record['status']='BLOCKED_CLEANUP'
  save()

def main():
 p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--packet-zip',type=Path,required=True);p.add_argument('--packet-sha256',required=True);p.add_argument('--cli-python',type=Path,required=True);p.add_argument('--cli-identity',type=Path,required=True);p.add_argument('--dispatch',type=Path,required=True);p.add_argument('--browser-adoption',type=Path,required=True);p.add_argument('--browser-adoption-sha256',required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
if __name__=='__main__':
 previous={sig:signal.getsignal(sig)for sig in(signal.SIGTERM,signal.SIGINT)}
 def interrupt(signum,frame):
  for sig in previous:signal.signal(sig,signal.SIG_IGN)
  raise KeyboardInterrupt('owned diagnostic driver interrupted')
 try:
  for sig in previous:signal.signal(sig,interrupt)
  main()
 finally:
  for sig,handler in previous.items():signal.signal(sig,handler)
