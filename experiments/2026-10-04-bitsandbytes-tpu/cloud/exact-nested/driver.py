"""One root-authorized allocation, recovered gates, explicit root review, exact closure."""
import argparse,json,math,os,re,shutil,signal,sys,time
from pathlib import Path
import contract as C
import cli_transport as T
import browser_adoption as BA
import transport
import kernel_protocol as K
from remote import run_step
HERE=Path(__file__).resolve().parent

def gate(packet_sha,identity,dispatch):
 C.require(dispatch.get('kind')=='ROOT_M4_EXACT_NESTED_DISPATCH'and dispatch.get('status')=='ACTUAL_DISPATCH_AUTHORIZED'and dispatch.get('packet_sha256')==packet_sha and dispatch.get('driver_sha256')==C.sha(__file__)and dispatch.get('cli_identity_sha256')==C.sha(identity)and dispatch.get('allocation_seconds')==3600 and dispatch.get('allocation_attempts')==1 and dispatch.get('science_manifest_sha256')==C.SCIENCE_SHA,'ROOT_DISPATCH_GATE')

def usage_zero(text):
 clean=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',text);assignments=[];rates=[]
 for line in clean.splitlines():
  m=re.fullmatch(r'Active assignments:\s*(\d+)',line.strip())
  if m:assignments.append(int(m.group(1)))
  m=re.fullmatch(r'Usage rate:\s*(\d+(?:\.\d+)?)/hr',line.strip())
  if m:rates.append(float(m.group(1)))
 return {'active_assignments':assignments,'usage_rates_per_hour':rates,'zero_observed':assignments==[0]and rates==[0.0]}

def run(a):
 packet=a.packet.resolve();out=a.output.resolve();C.require(not out.exists(),'FRESH_DRIVER_OUTPUT');C.require(C.sha(a.packet_zip)==a.packet_sha256,'PACKET_ZIP_SHA');m=C.packet_preflight(packet,a.packet_zip,a.packet_sha256);gate(a.packet_sha256,a.cli_identity,C.read(a.dispatch));T.verify_cli_identity(a.cli_python,C.read(a.cli_identity));out.mkdir(parents=True);api=T.OfficialCLI(a.cli_python,out)
 adoption=None
 if a.browser_adoption:
  adoption=BA.record(a.browser_adoption,a.browser_adoption_sha256);C.require(adoption['packet_sha256']==a.packet_sha256 and adoption['driver_sha256']==C.sha(__file__)and adoption['cli_identity_sha256']==C.sha(a.cli_identity),'BROWSER_GATE_BINDINGS')
 session=adoption['session']if adoption else a.session;C.require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,127}',session or''),'EXPLICIT_SESSION_ID');base='/content/m4-exact-'+session;birth=None;owned=False;registered=False;serial=0;last=None;controls_ready=False
 record={'kind':'M4_EXACT_NESTED_DRIVER','status':'BLOCKED','packet_sha256':a.packet_sha256,'remote_base':base,'session':session,'commands':[],'provider_calls':0,'allocation_attempts':0,'m4_status':'NOT_QUALIFIED'};C.write(out/'driver.json',record)
 def call(label,argv,limit=60,reserve=C.CLOSURE_RESERVE,cleanup=False):
  nonlocal serial;serial+=1;limit=limit if cleanup or birth is None else C.budget(birth,limit,reserve);r,text=api(f'{serial:03d}-'+label,argv,limit);record['commands'].append({'label':label,'status':r['status'],'exit_code':r.get('exit_code'),'cleanup':r['cleanup']});record['provider_calls']+=1;C.write(out/'driver.json',record);C.require(r['status']=='PASS','CLI_FAILED:'+label);return text
 def kernel(label,code,limit=60,reserve=C.CLOSURE_RESERVE,cleanup=False,artifact=None,artifact_expected=None):
  identity={str(Path(base)/name):r['sha256']for name,r in m['files'].items()if name.startswith('control/')and name.endswith('.py')}
  modules={Path(name).stem:(str(Path(base)/name),r['sha256'])for name,r in m['files'].items()if name.startswith('control/')and name.endswith('.py')and not Path(name).stem.startswith('__')}
  for name in ('generation_gate','exact_admission'):
   relative='nested-exact/source/'+name+'.py';modules[name]=(str(Path(base)/'payload'/relative),m['files'][relative]['sha256'])
  request=K.request(code,label,base,a.packet_sha256,birth,identity,artifact=artifact,artifact_expected=artifact_expected)
  source=K.render(code,request,identity if controls_ready else{},modules if controls_ready else{})
  p=out/(label+'.py');p.write_text(source);timeout=math.floor(limit if cleanup else C.budget(birth,limit,reserve))
  text=call(label,['exec','-s',session,'--timeout',str(timeout),'-f',str(p)],timeout,reserve,cleanup)
  K.marker(text,request)
  receipt=out/'kernel-readbacks'/(label+'.json');download(label+'-completion',request['completion_path'],receipt)
  local_artifact=None
  if artifact is not None:
   local_artifact=out/'kernel-readbacks'/(label+'.artifact.json');download(label+'-artifact',str(artifact),local_artifact)
  proof=K.verify(text,request,receipt,local_artifact);proof['submitted_source_sha256']=C.sha(p);C.write(out/'kernel-readbacks'/(label+'-proof.json'),proof)
  return text
 def upload(label,source,dest):return call(label,['upload','-s',session,str(source),dest.lstrip('/')])
 def download(label,source,dest):dest.parent.mkdir(parents=True,exist_ok=True);return call(label,['download','-s',session,source.lstrip('/'),str(dest)],90)
 def operation(phase,limit=1200):
  identity={name.removeprefix('control/'):r['sha256']for name,r in m['files'].items()if name.startswith('control/')and name.endswith('.py')}
  code='import hashlib,runpy,sys\nfrom pathlib import Path\n'+f'b=Path({base!r}); expected={identity!r}\n'+'for name,digest in expected.items():\n p=b/"control"/name\n if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest: raise ValueError("CONTROL_SOURCE_BYTES")\n'+f'sys.path.insert(0,str(b/"control"));sys.argv=["remote.py",{phase!r},"--base",str(b),"--packet-sha256",{a.packet_sha256!r},"--allocation-epoch",{str(birth)!r}]\nrunpy.run_path(str(b/"control/remote.py"),run_name="__main__")\n'
  artifact=Path(base)/'records'/('allocation.json'if phase=='unpack'else'phase.json')
  expected={'packet_sha256':a.packet_sha256,**({'birth_epoch':birth,'original_deadline_epoch':birth+3600}if phase=='unpack'else{'phase':phase,'allocation_epoch':birth,'status':'TERMINAL_LOCAL_REVIEW_REQUIRED'})}
  return kernel('remote-'+phase,code,limit,artifact=artifact,artifact_expected=expected)
 def recover(phase,diagnostic=False):
  nonlocal last
  label='recovery-'+phase;code='import runpy,sys\nfrom pathlib import Path\n'+f'b=Path({base!r});sys.path.insert(0,str(b/"control"));sys.argv=["remote.py","export","--base",str(b),"--allocation-epoch",{str(birth)!r},"--label",{label!r}];runpy.run_path(str(b/"control/remote.py"),run_name="__main__")\n';kernel('export-'+phase,code,120)
  d=out/label;d.mkdir();download(label+'-summary',base+'/'+label+'-export.json',d/'summary.json');s=C.read(d/'summary.json');C.require(s['bytes']<=C.MAX_BYTES and s['members']<=C.MAX_FILES and s['physical_bytes']<=C.INNER_BYTES,'REMOTE_FULL_ENVELOPE')
  download(label+'-parts',base+'/'+label+'-parts/manifest.json',d/'manifest.json');parts=transport.validate_manifest(C.read(d/'manifest.json'),s['sha256'],s['bytes']);(d/'parts').mkdir()
  for i,p in enumerate(parts['parts']):download(label+'-'+str(i),base+'/'+label+'-parts/'+p['name'],d/'parts'/p['name'])
  transport.assemble(d/'manifest.json',d/'parts',d/'evidence.zip',expected_manifest_sha256=C.sha(d/'manifest.json'),expected_sha256=s['sha256'],expected_bytes=s['bytes']);C.recover(d/'evidence.zip',s['sha256'],d/'records');shutil.rmtree(d/'parts');(d/'evidence.zip').unlink()
  # Delete only verified transfer copies, after full host readback.
  kernel('dispose-'+phase,'from pathlib import Path\nimport shutil\n'+f'b=Path({base!r});label={label!r}\n'+'(b/(label+".zip")).unlink();shutil.rmtree(b/(label+"-parts"));(b/(label+"-export.json")).unlink()\n',30)
  if diagnostic:
   record['diagnostic_recovered_records']=str(d/'records');record['diagnostic_recovered_envelope']=s;C.write(out/'driver.json',record);return d/'records',None
  audit=out/(phase+'-host-audit.json');argv=[sys.executable,'-B',str(HERE/'host_audit.py'),phase,'--packet',str(packet),'--records',str(d/'records'),'--remote-base',base,'--output',str(audit),'--packet-sha256',a.packet_sha256];r=run_step(out,'host-'+phase,argv,birth+3600-C.CLOSURE_RESERVE,300,cwd=HERE);record['last_recovered_records']=str(d/'records');record['last_recovered_envelope']=s;C.write(out/'driver.json',record);C.require(r['status']=='PASS','HOST_AUDIT:'+phase)
  result=C.read(audit)
  if phase.startswith('device'):C.require(result.get('record_validation')==result.get('numerical_status')=='PASS','SCIENTIFIC_GATES:'+phase)
  previous=last;last=d
  if previous:shutil.rmtree(previous/'records');C.write(previous/'disposed.json',{'records_removed':True,'replacement':str(d/'records'),'verification_sha256':C.sha(audit)})
  return d/'records',audit
 def initialize(phase,limit=1200):
  try:return operation(phase,limit)
  except BaseException as error:
   C.write(out/(phase+'-operation-error.json'),{'type':type(error).__name__,'message':str(error),'action':'ATTEMPT_DIAGNOSTIC_RECOVERY_BEFORE_CLOSURE'})
   try:recover(phase,diagnostic=True)
   except BaseException as recovery_error:C.write(out/(phase+'-recovery-error.json'),{'type':type(recovery_error).__name__,'message':str(recovery_error)})
   raise
 def scientific(phase,limit=1200):
  error=None
  try:operation(phase,limit)
  except BaseException as e:
   error=e;C.write(out/(phase+'-operation-error.json'),{'type':type(e).__name__,'message':str(e),'action':'RECOVER_TERMINAL_OR_PARTIAL_RECORDS_BEFORE_CLOSURE'})
  recovered=recover(phase)
  if error is not None:raise error
  return recovered
 try:
  before=call('sessions-before',['sessions']);C.require(adoption is not None or'No active sessions found on server.'in before,'FRESH_SESSION_SERVER_EMPTY');call('usage-before',['usage'])
  birth=adoption['allocation_epoch']if adoption else time.time();record.update(allocation_epoch=birth,original_deadline_epoch=birth+3600,closure_reserve_seconds=C.CLOSURE_RESERVE,allocation_attempts=0 if adoption else 1);C.write(out/'driver.json',record)
  if adoption:registered=True;call('browser-registration',['--adopt-root-browser-v1',str(a.browser_adoption),a.browser_adoption_sha256]);registered=True
  else:owned=True;call('one-allocation',['--allocation-transport-v1','new','-s',session,'--tpu','v6e1'],360);owned=True
  after=call('sessions-after',['sessions']);lines=[l for l in after.splitlines()if'Hardware:'in l]
  if adoption:
   segments=lines[0].split(' | ')if len(lines)==1 else[]
   hardware=[part for part in segments[1:]if part.startswith('Hardware:')]
   variant=[part for part in segments[1:]if part.startswith('Variant:')]
   C.require(bool(segments)and segments[0]=='['+session+'] '+adoption['endpoint']and hardware==['Hardware: '+adoption['hardware']]and variant==['Variant: TPU'],'BROWSER_EXACT_ASSIGNMENT')
   owned=True
   proof=kernel('browser-marker','import hashlib\nfrom pathlib import Path\n'+f'p=Path({adoption["marker_path"]!r});assert p.is_file()and not p.is_symlink()and hashlib.sha256(p.read_bytes()).hexdigest()=={adoption["marker_sha256"]!r}\nprint("EXACT_ROOT_MARKER_MATCH")\n',30);C.require('EXACT_ROOT_MARKER_MATCH'in proof,'BROWSER_MARKER_READBACK')
  else:C.require(len(lines)==1 and'['+session+']'in lines[0]and'Hardware: V6E1'in lines[0],'EXACT_ALLOCATED_SESSION')
  kernel('readiness','from pathlib import Path\n'+f'b=Path({base!r});assert not b.exists()and not b.is_symlink();(b/"control/ownership").mkdir(parents=True)\n',30)
  for name in sorted(m['files']):
   if name.startswith('control/')and name.endswith('.py'):upload('control-'+name.replace('/','-'),packet/name,base+'/'+name)
  controls_ready=True
  upload('packet',a.packet_zip,base+'/payload.zip');initialize('unpack',60);initialize('install');records,cpu=scientific('cpu79');upload('cpu-root-gate',cpu,base+'/records/cpu-gate.json');records,actual=scientific('device79')
  request={'kind':'M4_EXACT_NESTED_ROOT_REVIEW_REQUEST','status':'ACTUAL79_RECOVERED_WAIT_ROOT','packet_sha256':a.packet_sha256,'records':str(records),'host_audit_sha256':C.sha(actual),'acceptance_file':str(a.accept79_file),'birth_epoch':birth,'actual79_finished_epoch':C.read(records/'device79/receipt.json')['finished_epoch'],'original_deadline_epoch':birth+3600,'state_start_reserve_seconds':C.STATE_RESERVE};C.write(out/'root-review-request.json',request);record['status']='ACTUAL79_RECOVERED_WAIT_ROOT';C.write(out/'driver.json',record)
  if a.stop_after_79:record['status']='ACTUAL79_RECOVERED_ROOT_REVIEW_REQUIRED';return
  C.require(a.accept79_file is not None and not a.accept79_file.is_symlink(),'EXPLICIT_ROOT_ACCEPTANCE_PATH')
  while not a.accept79_file.exists():C.budget(birth,1,C.STATE_RESERVE);time.sleep(.5)
  accepted_epoch=time.time();accepted=C.accepted79(a.accept79_file,C.sha(a.accept79_file),packet_sha=a.packet_sha256,receipt_sha=C.sha(records/'device79/receipt.json'),oracle_sha=C.sha(records/'cpu79/oracle-seal.json'),completed_epoch=request['actual79_finished_epoch'],birth=birth)
  boundary={'kind':'M4_EXACT_NESTED_ACTUAL_REVIEW_BOUNDARY','accepted79_sha256':C.sha(a.accept79_file),'accepted_epoch':accepted_epoch,'actual79_finished_epoch':request['actual79_finished_epoch'],'birth_epoch':birth,'original_deadline_epoch':birth+3600,'remaining_seconds':birth+3600-accepted_epoch,'state_reserve_seconds':C.STATE_RESERVE};C.write(out/'root-review-boundary.json',boundary);upload('accepted79',a.accept79_file,base+'/records/accepted79.json');upload('actual-review-boundary',out/'root-review-boundary.json',base+'/records/root-review-boundary.json')
  records,statecpu=scientific('cpustate',600);upload('state-cpu-launch-audit',statecpu,base+'/records/state-cpu-launch-audit.json');records,actualstate=scientific('devicestate',600);record['status']='STATE8_RECOVERED_ROOT_REVIEW_REQUIRED'
 except BaseException as error:record['error']={'type':type(error).__name__,'message':str(error)};raise
 finally:
  # No fresh allocation or repeated stop after a failed/unknown identity check.
  if owned:
   for label,argv,limit in [('sessions-before-stop',['sessions'],15),('stop-exact',['stop','-s',session],25),('sessions-after-stop',['sessions'],15),('usage-after',['usage'],15)]:
    try:record[label]=call(label,argv,limit,cleanup=True)
    except BaseException as e:record.setdefault('closure_errors',[]).append({'step':label,'type':type(e).__name__,'message':str(e)})
   record['server_empty_observed']='No active sessions found on server.'in record.get('sessions-after-stop','')
  elif registered:
   try:call('remove-provisional',['--remove-browser-provisional-v1',str(a.browser_adoption),a.browser_adoption_sha256],20,cleanup=True)
   except BaseException as e:record.setdefault('closure_errors',[]).append({'step':'remove-provisional','message':str(e)})
  record['finished_epoch']=time.time()
  if owned:
   record['usage_readback']=usage_zero(record.get('usage-after',''));record['usage_zero_observed']=record['usage_readback']['zero_observed'];record['elapsed_lifecycle_seconds']=record['finished_epoch']-birth
   if not record.get('server_empty_observed')or not record['usage_zero_observed']or record.get('closure_errors')or record['elapsed_lifecycle_seconds']>3600:record['status']='BLOCKED_CLEANUP'
  C.write(out/'driver.json',record)

def main():
 p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--packet-zip',type=Path,required=True);p.add_argument('--packet-sha256',required=True);p.add_argument('--cli-python',type=Path,required=True);p.add_argument('--cli-identity',type=Path,required=True);p.add_argument('--dispatch',type=Path,required=True);p.add_argument('--session');p.add_argument('--browser-adoption',type=Path);p.add_argument('--browser-adoption-sha256');p.add_argument('--output',type=Path,required=True);p.add_argument('--accept79-file',type=Path);p.add_argument('--stop-after-79',action='store_true');a=p.parse_args();C.require(a.stop_after_79 or a.accept79_file is not None,'ROOT_ACCEPTANCE_PATH_REQUIRED');run(a)
if __name__=='__main__':
 prior={sig:signal.getsignal(sig)for sig in(signal.SIGTERM,signal.SIGINT)}
 def interrupted(signum,frame):
  for sig in prior:signal.signal(sig,signal.SIG_IGN)
  raise KeyboardInterrupt('owned exact-nested driver interrupted')
 try:
  for sig in prior:signal.signal(sig,interrupted)
  main()
 finally:
  for sig,handler in prior.items():signal.signal(sig,handler)

