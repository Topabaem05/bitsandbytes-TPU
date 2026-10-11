"""One root-authorized allocation, recovered gates, explicit root review, exact closure."""
import argparse,json,math,os,re,shutil,signal,sys,time
from pathlib import Path
import contract as C
import cli_transport as T
import browser_adoption as BA
import transport
import kernel_protocol as K
import kernel_entries as E
from remote import run_step
HERE=Path(__file__).resolve().parent

def gate(packet,packet_sha,identity,dispatch,output):
 C.require(dispatch.get('kind')=='ROOT_M5_EXACT1024_DISPATCH'and dispatch.get('status')=='ACTUAL_DISPATCH_AUTHORIZED'and dispatch.get('packet_sha256')==packet_sha and dispatch.get('driver_sha256')==C.sha(__file__)and dispatch.get('cli_identity_sha256')==C.sha(identity)and dispatch.get('allocation_seconds')==3600 and dispatch.get('allocation_attempts')==1 and dispatch.get('science_manifest_sha256')==C.science(packet),'ROOT_DISPATCH_GATE');C.require(dispatch.get('output')==str(output.resolve()),'EXACT_DISPATCH_OUTPUT')

def usage_zero(text):
 clean=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',text);assignments=[];rates=[]
 for line in clean.splitlines():
  m=re.fullmatch(r'Active assignments:\s*(\d+)',line.strip())
  if m:assignments.append(int(m.group(1)))
  m=re.fullmatch(r'Usage rate:\s*(\d+(?:\.\d+)?)/hr',line.strip())
  if m:rates.append(float(m.group(1)))
 return {'active_assignments':assignments,'usage_rates_per_hour':rates,'zero_observed':assignments==[0]and rates==[0.0]}

def run(a):
 packet=a.packet.resolve();out=a.output.resolve();C.require(not out.exists(),'FRESH_DRIVER_OUTPUT');C.require(C.sha(a.packet_zip)==a.packet_sha256,'PACKET_ZIP_SHA');m=C.packet_preflight(packet,a.packet_zip,a.packet_sha256);gate(packet,a.packet_sha256,a.cli_identity,C.read(a.dispatch),out);T.verify_cli_identity(a.cli_python,C.read(a.cli_identity));out.mkdir(parents=True);api=T.OfficialCLI(a.cli_python,out)
 adoption=None
 if a.browser_adoption:
  adoption=BA.record(a.browser_adoption,a.browser_adoption_sha256);C.require(adoption['packet_sha256']==a.packet_sha256 and adoption['driver_sha256']==C.sha(__file__)and adoption['cli_identity_sha256']==C.sha(a.cli_identity),'BROWSER_GATE_BINDINGS');dispatch=C.read(a.dispatch);C.require(dispatch.get('browser_adoption_sha256')==a.browser_adoption_sha256 and dispatch.get('original_allocation_epoch')==adoption['allocation_epoch']and adoption['hardware']=='V5E1','ROOT_ADOPTION_DISPATCH_BINDING')
 session=adoption['session']if adoption else a.session;C.require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,127}',session or''),'EXPLICIT_SESSION_ID');base='/content/m5-exact1024-'+session;birth=None;owned=False;registered=False;serial=0;last=None;controls_ready=False
 record={'kind':'M5_EXACT1024_DRIVER','status':'BLOCKED','packet_sha256':a.packet_sha256,'remote_base':base,'session':session,'commands':[],'provider_calls':0,'allocation_attempts':0,'m5_status':'NOT_QUALIFIED'};C.write(out/'driver.json',record)
 def call(label,argv,limit=60,reserve=C.CLOSURE_RESERVE,cleanup=False):
  nonlocal serial;serial+=1;limit=limit if cleanup or birth is None else C.budget(birth,limit,reserve);r,text=api(f'{serial:03d}-'+label,argv,limit);record['commands'].append({'label':label,'status':r['status'],'exit_code':r.get('exit_code'),'cleanup':r['cleanup']});record['provider_calls']+=1;C.write(out/'driver.json',record);C.require(r['status']=='PASS','CLI_FAILED:'+label);return text
 def kernel(label,code,limit=60,reserve=C.CLOSURE_RESERVE,cleanup=False,artifact=None,artifact_expected=None):
  identity,modules=E.identities(base,m);entry=E.entry(label,base,a.packet_sha256,birth,m,adoption)
  C.require(code==entry['body']and (str(artifact)if artifact is not None else None)==entry['artifact_path']and(artifact_expected or{})==entry['artifact_expected']and controls_ready==entry['controls_ready'],'CANONICAL_KERNEL_ENTRY')
  request=K.request(entry['body'],label,base,a.packet_sha256,birth,identity,artifact=entry['artifact_path'],artifact_expected=entry['artifact_expected'])
  source=K.render(entry['body'],request,identity if entry['controls_ready']else{},modules if entry['controls_ready']else{})
  p=out/(label+'.py');p.write_text(source);timeout=max(1,math.floor(limit if cleanup else C.budget(birth,limit,reserve)))
  text=call(label,['exec','-s',session,'--timeout',str(timeout),'-f',str(p)],timeout,reserve,cleanup)
  K.marker(text,request)
  receipt=out/'kernel-readbacks'/(label+'.json');download(label+'-completion',request['completion_path'],receipt)
  local_artifact=None
  if artifact is not None:
   local_artifact=out/'kernel-readbacks'/(label+'.artifact.json');download(label+'-artifact',str(artifact),local_artifact)
  proof=K.verify(text,request,receipt,local_artifact);proof['submitted_source_sha256']=C.sha(p);proofpath=out/'kernel-readbacks'/(label+'-proof.json');C.write(proofpath,proof)
  record.setdefault('kernel_proofs',[]).append({'label':label,'proof_sha256':C.sha(proofpath),'submitted_source_sha256':C.sha(p)});C.write(out/'driver.json',record)
  return text
 def upload(label,source,dest):return call(label,['upload','-s',session,str(source),dest.lstrip('/')])
 def download(label,source,dest):dest.parent.mkdir(parents=True,exist_ok=True);return call(label,['download','-s',session,source.lstrip('/'),str(dest)],90,reserve=60)
 def operation(phase,limit=1200):
  identity={name.removeprefix('control/'):r['sha256']for name,r in m['files'].items()if name.startswith('control/')and name.endswith('.py')}
  code='import hashlib,runpy,sys\nfrom pathlib import Path\n'+f'b=Path({base!r}); expected={identity!r}\n'+'for name,digest in expected.items():\n p=b/"control"/name\n if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest: raise ValueError("CONTROL_SOURCE_BYTES")\n'+f'sys.path.insert(0,str(b/"control"));sys.argv=["remote.py",{phase!r},"--base",str(b),"--packet-sha256",{a.packet_sha256!r},"--allocation-epoch",{str(birth)!r}]\nrunpy.run_path(str(b/"control/remote.py"),run_name="__main__")\n'
  artifact=Path(base)/'records'/('allocation.json'if phase=='unpack'else'phase.json')
  expected={'packet_sha256':a.packet_sha256,**({'birth_epoch':birth,'original_deadline_epoch':birth+3600}if phase=='unpack'else{'kind':'M5_EXACT1024_STAGE','phase':phase,'allocation_epoch':birth,'status':'TERMINAL_LOCAL_REVIEW_REQUIRED','m5_status':'NOT_QUALIFIED'})}
  return kernel('remote-'+phase,code,limit,artifact=artifact,artifact_expected=expected)
 def recover(phase,diagnostic=False):
  nonlocal last
  label='recovery-'+phase;code='import runpy,sys\nfrom pathlib import Path\n'+f'b=Path({base!r});sys.path.insert(0,str(b/"control"));sys.argv=["remote.py","export","--base",str(b),"--allocation-epoch",{str(birth)!r},"--label",{label!r}];runpy.run_path(str(b/"control/remote.py"),run_name="__main__")\n';kernel('export-'+phase,code,120,reserve=60,artifact=Path(base)/(label+'-export.json'))
  d=out/label;d.mkdir();download(label+'-summary',base+'/'+label+'-export.json',d/'summary.json');s=C.read(d/'summary.json');C.require(s['bytes']<=C.MAX_BYTES and s['members']<=C.MAX_FILES and s['physical_bytes']<=C.INNER_BYTES,'REMOTE_FULL_ENVELOPE')
  download(label+'-parts',base+'/'+label+'-parts/manifest.json',d/'manifest.json');parts=transport.validate_manifest(C.read(d/'manifest.json'),s['sha256'],s['bytes']);(d/'parts').mkdir()
  for i,p in enumerate(parts['parts']):download(label+'-'+str(i),base+'/'+label+'-parts/'+p['name'],d/'parts'/p['name'])
  transport.assemble(d/'manifest.json',d/'parts',d/'evidence.zip',expected_manifest_sha256=C.sha(d/'manifest.json'),expected_sha256=s['sha256'],expected_bytes=s['bytes']);C.recover(d/'evidence.zip',s['sha256'],d/'records');shutil.rmtree(d/'parts')
  # Delete only verified transfer copies, after full host readback.
  kernel('dispose-'+phase,'from pathlib import Path\nimport shutil\n'+f'b=Path({base!r});label={label!r}\n'+'(b/(label+".zip")).unlink();shutil.rmtree(b/(label+"-parts"));(b/(label+"-export.json")).unlink()\n',30,reserve=60)
  if diagnostic:
   record['diagnostic_recovered_records']=str(d/'records');record['diagnostic_recovered_envelope']=s;C.write(out/'driver.json',record);return d/'records',None
  audit=out/(phase+'-host-audit.json');argv=[sys.executable,'-B',str(HERE/'host_audit.py'),phase,'--packet',str(packet),'--records',str(d/'records'),'--remote-base',base,'--output',str(audit),'--packet-sha256',a.packet_sha256];r=run_step(out,'host-'+phase,argv,birth+3600-60,300,cwd=HERE);record['last_recovered_records']=str(d/'records');record['last_recovered_envelope']=s;C.write(out/'driver.json',record);C.require(r['status']in('PASS','CHILD_FAILED'),'HOST_AUDIT_PROCESS:'+phase)
  result=C.read(audit)
  C.require(result.get('record_validation')==result.get('numerical_status')=='PASS','SCIENTIFIC_GATES:'+phase)
  previous=last;last=d
  if previous:
   shutil.rmtree(previous/'records');(previous/'evidence.zip').unlink();C.write(previous/'disposed.json',{'records_removed':True,'replacement':str(d/'records'),'verification_sha256':C.sha(audit)})
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
  recovered=recover(phase,diagnostic=error is not None)
  if error is not None:raise error
  return recovered
 try:
  before=call('sessions-before',['sessions']);C.require(adoption is not None,'EXACT_ROOT_BROWSER_ADOPTION_REQUIRED');call('usage-before',['usage'])
  birth=adoption['allocation_epoch']if adoption else time.time();record.update(allocation_epoch=birth,original_deadline_epoch=birth+3600,closure_reserve_seconds=C.CLOSURE_RESERVE,allocation_attempts=0 if adoption else 1);C.write(out/'driver.json',record)
  if adoption:registered=True;call('browser-registration',['--adopt-root-browser-v1',str(a.browser_adoption),a.browser_adoption_sha256]);registered=True
  else:owned=True;call('one-allocation',['--allocation-transport-v1','new','-s',session,'--tpu','v6e1'],360);owned=True
  after=call('sessions-after',['sessions']);lines=[l for l in after.splitlines()if'Hardware:'in l]
  if adoption:
   C.require(len(lines)==1 and lines[0].split(' | ')[0]=='['+session+'] '+adoption['endpoint']and'Hardware: '+adoption['hardware']in lines[0]and'Variant: TPU'in lines[0],'BROWSER_EXACT_ASSIGNMENT')
   proof=kernel('browser-marker','import hashlib\nfrom pathlib import Path\n'+f'p=Path({adoption["marker_path"]!r});assert p.is_file()and not p.is_symlink()and hashlib.sha256(p.read_bytes()).hexdigest()=={adoption["marker_sha256"]!r}\nprint("EXACT_ROOT_MARKER_MATCH")\n',30);C.require('EXACT_ROOT_MARKER_MATCH'in proof,'BROWSER_MARKER_READBACK');owned=True
  else:C.require(len(lines)==1 and'['+session+']'in lines[0]and'Hardware: V6E1'in lines[0],'EXACT_ALLOCATED_SESSION')
  kernel('readiness','from pathlib import Path\n'+f'b=Path({base!r});assert not b.exists()and not b.is_symlink();(b/"control/ownership").mkdir(parents=True)\n',30)
  for name in sorted(m['files']):
   if name.startswith('control/')and name.endswith('.py'):upload('control-'+name.replace('/','-'),packet/name,base+'/'+name)
  controls_ready=True
  upload('packet',a.packet_zip,base+'/payload.zip');initialize('unpack',60)
  if a.runtime_wheels:
   wheels=C.runtime_cache(packet,a.runtime_wheels)
   kernel('wheel-cache','from pathlib import Path\n'+f'Path({(base+"/wheel-cache")!r}).mkdir()\n',30)
   for row in wheels:upload('locked-wheel-'+row['filename'],a.runtime_wheels/row['filename'],base+'/wheel-cache/'+row['filename'])
  initialize('install',C.INSTALL_SECONDS);records,cpu=scientific('cpu',C.CPU_SECONDS);upload('cpu-root-gate',cpu,base+'/records/cpu-gate.json');records,actual=scientific('device',C.DEVICE_SECONDS);record['status']='TRAINING20_RESUME10_RECOVERED_ROOT_REVIEW_REQUIRED'

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
 p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--packet-zip',type=Path,required=True);p.add_argument('--packet-sha256',required=True);p.add_argument('--cli-python',type=Path,required=True);p.add_argument('--cli-identity',type=Path,required=True);p.add_argument('--dispatch',type=Path,required=True);p.add_argument('--session');p.add_argument('--browser-adoption',type=Path);p.add_argument('--browser-adoption-sha256');p.add_argument('--output',type=Path,required=True);p.add_argument('--runtime-wheels',type=Path);a=p.parse_args();C.require(a.browser_adoption is not None and a.browser_adoption_sha256 is not None,'ROOT_BROWSER_ADOPTION_REQUIRED');run(a)
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

