"""Independent complete successful-run inventory, canonical scripts, CLI ownership/readback."""
import math,re,sys
from pathlib import Path
import contract as C,cli_transport as T,browser_adoption as BA,kernel_protocol as K,kernel_entries as E
HERE=Path(__file__).resolve().parent

def closed_process(summary,ownership,argv,owner_pid=None):
 C.closed(summary);C.require(len(summary.get('groups',[]))==1,'ONE_OWNED_CLI_GROUP');g=summary['groups'][0]
 C.require(all(type(ownership.get(k))is int and ownership[k]>0 for k in('pid','pgid','owner_pid'))and ownership['pid']==ownership['pgid']==g['pid']==g['pgid']and ownership.get('exit_status')==g.get('exit_status')==0 and ownership.get('argv')==argv and ownership.get('cleanup')==g and g.get('errors')==[],'OWNED_CLI_PROCESS_OUTCOME')
 if owner_pid is not None:C.require(ownership['owner_pid']==owner_pid,'CLI_DRIVER_PARENT')
 restoration=summary.get('restoration',{});C.require(restoration.get('handlers_restored')is True and restoration.get('timer_restored')is True,'CLI_OWNER_RESTORATION')

# These parsers admit only the installed official CLI 0.7.4 display contract.
# A displayed rate of 0.00 is an observation, not an unrounded billing claim.
CLI_DISPLAY_SOURCES={
 'colab_cli/commands/session.py':'91aeef7b4936c8462a969347096611301ef3c11737f220a9cbb774771c3cbb43',
 'colab_cli/consumption.py':'b6e936db0ddd6e4c31ef7b516c456be7549c34394a70a3d1356b679ec168a594',
 'colab_cli/common.py':'56cc48f56a8dbe223b04728d075b8c8382c801925ad6a508e8eaa2870b887276',
}

def output_lines(text):
 C.require(isinstance(text,str),'CLI_RAW_TEXT')
 clean=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',text)
 return [line.strip()for line in clean.splitlines()if line.strip()]

def sessions_output(text):
 lines=output_lines(text);pruned=0
 if lines and re.fullmatch(r'\[colab\] Pruned [1-9][0-9]* stale local session\(s\)\.',lines[0]):
  pruned=int(lines.pop(0).split()[2])
 empty=lines==['[colab] No active sessions found on server.']
 rows=[]
 if not empty:
  C.require(bool(lines),'SESSIONS_RAW_REQUIRED')
  for line in lines:
   match=re.fullmatch(r'\[([A-Za-z0-9][A-Za-z0-9_-]{0,127}|\?)\] ([A-Za-z0-9_-]{1,256}) \| Hardware: ([A-Za-z0-9_-]+) \| Shape: ([A-Za-z0-9 _-]+) \| Variant: ([A-Za-z0-9_-]+)',line)
   C.require(match is not None,'OFFICIAL_SESSIONS_RAW')
   rows.append(dict(zip(('session','endpoint','hardware','shape','variant'),match.groups())))
 C.require(len({(r['session'],r['endpoint'])for r in rows})==len(rows),'DUPLICATE_SESSION_ROW')
 return {'empty':empty,'sessions':rows,'pruned_local_sessions':pruned}

def usage_output(text):
 lines=output_lines(text)
 patterns=(r'Current balance: (-?[0-9]+\.[0-9]{2}) compute units',r'Usage rate: ([0-9]+\.[0-9]{2})/hr',r'Active assignments: ([0-9]+)')
 C.require(len(lines)==3,'OFFICIAL_USAGE_LABELS')
 values=[]
 for line,pattern in zip(lines,patterns):
  match=re.fullmatch(pattern,line);C.require(match is not None,'OFFICIAL_USAGE_LABELS');values.append(match[1])
 return {'displayed_balance':values[0],'displayed_rate_per_hour':values[1],'active_assignments':int(values[2]),'displayed_zero':values[1]=='0.00'and int(values[2])==0}

def resource_empty(session_text,usage_text):
 sessions=sessions_output(session_text);usage=usage_output(usage_text)
 C.require(sessions['empty']and usage['displayed_zero'],'PHYSICAL_EMPTY_SERVER_ZERO_DISPLAYED_USAGE')
 return {'sessions':sessions,'usage':usage}

def owned_session(text,adoption,*,unregistered=False,allow_empty=False):
 observed=sessions_output(text)
 if allow_empty and observed['empty']:return observed
 expected={'session':'?'if unregistered else adoption['session'],'endpoint':adoption['endpoint'],'hardware':adoption['hardware'],'shape':'Standard','variant':'TPU'}
 C.require(observed['sessions']==[expected]and not observed['empty'],'PHYSICAL_ADOPTED_SESSION_IDENTITY')
 return observed

def utc_epoch(value):
 from datetime import datetime
 C.require(isinstance(value,str),'LIFECYCLE_UTC_TIME')
 try:parsed=datetime.fromisoformat(value)
 except(ValueError,TypeError):raise ValueError('LIFECYCLE_UTC_TIME')
 C.require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds()==0,'LIFECYCLE_UTC_TIME')
 return parsed.timestamp()

def lifecycle_ledger(path,ownerships,summary):
 """Reconcile every event with admitted ownerships in launch order."""
 import json
 lines=path.read_text().splitlines();C.require(bool(lines),'CLI_LIFECYCLE_REQUIRED')
 events=[json.loads(line,object_pairs_hook=C.json_pairs)for line in lines]
 C.require(len(summary['groups'])==len(ownerships),'LIFECYCLE_GROUP_INVENTORY')
 expected={}
 remaining=list(summary['groups'])
 for own in ownerships:
  matches=[i for i,group in enumerate(remaining)if group==own.get('cleanup')];C.require(len(matches)==1,'LIFECYCLE_GROUP_INVENTORY');group=remaining.pop(matches[0])
  key=(own['pid'],own['started_at']);C.require(key not in expected,'DUPLICATE_LAUNCH_GENERATION')
  C.require(own['pid']==own['pgid']==group['pid']==group['pgid']and own['cleanup']==group and own['exit_status']==group['exit_status']==0 and group['errors']==[],'LIFECYCLE_OWNERSHIP_GROUP')
  start=utc_epoch(own['started_at']);cleanup_start=utc_epoch(group['started_at']);end=utc_epoch(group['ended_at'])
  C.require(own['ended_at']==group['ended_at'],'LIFECYCLE_OWNERSHIP_TIMES')
  expected[key]={'own':own,'start':start,'cleanup_start':cleanup_start,'end':end}
 active={};started=set();closed=set()
 for event in events:
  C.require(type(event)is dict and event.get('kind')in('child_started','child_cleanup_checks'),'LIFECYCLE_UNEXPECTED_EVENT')
  utc_epoch(event.get('recorded_at'))  # Wall-clock UTC is not a monotonic timer.
  pid=event.get('pid');C.require(type(pid)is int and pid>0,'LIFECYCLE_PID')
  if event['kind']=='child_started':
   C.require(set(event)=={'kind','pid','pgid','recorded_at'}and event['pgid']==pid and pid not in active,'LIFECYCLE_START_FIELDS_OR_DUPLICATE')
   candidates=[key for key in expected if key[0]==pid and key not in started]
   C.require(bool(candidates),'LIFECYCLE_UNOWNED_OR_AMBIGUOUS_CHILD');key=candidates[0];active[pid]=key;started.add(key)
  else:
   C.require(set(event)=={'kind','pid','errors','recorded_at'}and event['errors']==[]and pid in active,'LIFECYCLE_ORPHAN_DUPLICATE_OR_FAILED_CLEANUP')
   key=active.pop(pid);closed.add(key)
 C.require(not active and started==closed==set(expected),'LIFECYCLE_COMPLETE_LAUNCH_AND_CLEANUP_SET')
 return {'launch_generations':len(expected),'events':len(events)}

def cli_record(directory,record,prefix,argv,owner_pid=None):
 C.require(set(C.inventory(directory))=={'result.json','ownership.json','cleanup.json','stdout-stderr.raw','lifecycle.jsonl'},'COMPLETE_PHYSICAL_CLI_RECORDS')
 result=C.read(directory/'result.json');own=C.read(directory/'ownership.json');cleanup=C.read(directory/'cleanup.json');full=result['argv']
 C.require(full==own['argv']and full[:3]==prefix,'ADMITTED_CLI_PREFIX_AND_ARGV')
 if argv is not None:C.require(full[3:]==argv,'CANONICAL_CLI_OPERATION')
 C.require(result.get('status')==record.get('status')=='PASS'and result.get('exit_code')==record.get('exit_code')==0 and not result.get('error')and result.get('original_outcome')=={'status':'PASS','exit_code':0,'error':None}and result['cleanup']==cleanup==record['cleanup'],'CLI_SUCCESS_AND_CLEANUP_BINDING')
 C.require(type(result.get('timeout_seconds'))in(int,float)and math.isfinite(result['timeout_seconds'])and result['timeout_seconds']>0,'CLI_FINITE_TIMEOUT')
 closed_process(cleanup,own,full,owner_pid);lifecycle_ledger(directory/'lifecycle.jsonl',[own],cleanup)
 return result,own,(directory/'stdout-stderr.raw').read_text()

def fresh_resource_closure(packet,actual,d,review):
 ctx=context(packet,actual,d,review);fresh=C.read(review/'fresh-resource-closure.json');root=review/'fresh-resource-closure'
 C.require(fresh.get('status')=='NO_ACTIVE_SESSION_OR_USAGE'and type(fresh.get('epoch'))in(int,float)and math.isfinite(fresh['epoch'])and fresh['epoch']>=d['finished_epoch']and fresh['cli_identity_sha256']==ctx['dispatch']['cli_identity_sha256']and fresh.get('allocation')=='NOT_RUN','FRESH_EMPTY_SERVER_ZERO_USAGE')
 C.require(len(fresh.get('commands',[]))==2 and {p.name for p in root.iterdir()}=={'cli'}and {p.name for p in(root/'cli').iterdir()}=={'0-sessions','1-usage'},'FRESH_PHYSICAL_CLI_SET')
 texts={};parents=set()
 for i,label in enumerate(('sessions','usage')):
  directory=root/'cli'/(str(i)+'-'+label);result,own,text=cli_record(directory,fresh['commands'][i],ctx['cli_prefix'],[label])
  C.require(result==fresh['commands'][i],'FRESH_SUMMARY_PHYSICAL_RESULT');parents.add(own['owner_pid'])
  C.require(d['finished_epoch']<=utc_epoch(own['started_at'])<=utc_epoch(own['ended_at'])<=fresh['epoch'],'FRESH_CLI_OBSERVATION_TIME');texts[label]=text
 C.require(len(parents)==1,'FRESH_CLI_COMMON_OWNER')
 return {**resource_empty(texts['sessions'],texts['usage']),'status':'FRESH_PHYSICAL_EMPTY_SERVER_ZERO_DISPLAYED_USAGE','raw_sha256':{label:C.sha(root/'cli'/(str(i)+'-'+label)/'stdout-stderr.raw')for i,label in enumerate(('sessions','usage'))}}


def context(packet,actual,d,review):
 import driver as D
 packet=packet.resolve();actual=actual.resolve();review=review.resolve();dispatch=C.read(review/'dispatch/root-dispatch.json');config=C.read(review/'dispatch/host-config.json');host=C.read(review/'host-owner/host-finalization.json');own=C.read(review/'host-owner/ownership.json');adopt_path=review/'dispatch/browser-adoption.json'
 argv=config.get('argv',[]);C.require(len(argv)>=3 and argv[:3]==[sys.executable,'-B',str(HERE/'driver.py')],'ADMITTED_HOST_DRIVER_PREFIX');tail=argv[3:];C.require(len(tail)%2==0 and all(isinstance(x,str)for x in tail),'ADMITTED_HOST_CONFIG_ARGV');opts=dict(zip(tail[::2],tail[1::2]));required=['--packet','--packet-zip','--packet-sha256','--cli-python','--cli-identity','--dispatch','--browser-adoption','--browser-adoption-sha256','--output'];keys=required+(['--runtime-wheels']if'--runtime-wheels'in opts else[])
 C.require(tail[::2]==keys and len(opts)==len(keys),'EXACT_ADMITTED_HOST_CONFIG_FLAGS');cli_python=opts['--cli-python'];identity=Path(opts['--cli-identity']);T.verify_cli_identity(Path(cli_python),C.read(identity));C.require(all(C.read(identity)['files'].get(n)==digest for n,digest in CLI_DISPLAY_SOURCES.items()),'ADMITTED_CLI_DISPLAY_PROFILE');C.require(C.sha(HERE/'official_cli_no_oauth.py')==C.read(packet/'manifest.json')['files']['control/official_cli_no_oauth.py']['sha256'],'ADMITTED_OFFICIAL_CLI_SCRIPT')
 C.require(opts['--packet']==str(packet)and opts['--packet-sha256']==d['packet_sha256']==config['packet_sha256']and opts['--output']==str(actual)and opts['--dispatch']==str(review/'dispatch/root-dispatch.json')and opts['--browser-adoption']==str(adopt_path)and opts['--browser-adoption-sha256']==dispatch['browser_adoption_sha256'],'HOST_CONFIG_DISPATCH_BINDING')
 C.require(Path(opts['--packet-zip']).is_file()and C.sha(Path(opts['--packet-zip']))==d['packet_sha256'],'HOST_CONFIG_PACKET_ARCHIVE');D.gate(packet,d['packet_sha256'],identity,dispatch,actual)
 adoption=BA.record(adopt_path,dispatch['browser_adoption_sha256'],check_time=False);birth=dispatch['original_allocation_epoch'];C.require(adoption['allocation_epoch']==d['allocation_epoch']==birth and adoption['packet_sha256']==d['packet_sha256']and adoption['driver_sha256']==dispatch['driver_sha256']and adoption['cli_identity_sha256']==dispatch['cli_identity_sha256']==C.sha(identity)and adoption['hardware']=='V5E1'and adoption['session']==d['session']and d['remote_base']=='/content/m5-exact1024-'+adoption['session'],'ROOT_ADOPTED_DRIVER_IDENTITY')
 C.require(d['original_deadline_epoch']==config['absolute_deadline_epoch']==host['absolute_deadline_epoch']==birth+3600 and birth<=d['finished_epoch']<=host['finished_epoch']<=birth+3600,'ORIGINAL_LIFECYCLE_DEADLINE')
 C.require(d['status']=='TRAINING20_RESUME10_RECOVERED_ROOT_REVIEW_REQUIRED'and not d.get('error')and d['allocation_attempts']==0,'PARTIAL_EXECUTION_NOT_COMPLETE')
 C.require(host['status']=='DRIVER_EXITED'and host['exit_code']==0 and not host.get('error')and host['config_sha256']==C.sha(review/'dispatch/host-config.json')and host['argv']==argv and host['packet_sha256']==d['packet_sha256']and config['host_output']==str(review/'host-owner'),'HOST_DRIVER_OUTCOME');closed_process(host['cleanup'],own,argv);hostdir=review/'host-owner';C.require(set(C.inventory(hostdir))=={'host-finalization.json','ownership.json','cleanup.json','lifecycle.jsonl','stdout.raw','stderr.raw'}and C.read(hostdir/'cleanup.json')==host['cleanup'],'HOST_PHYSICAL_OWNER_RECORDS');lifecycle_ledger(hostdir/'lifecycle.jsonl',[own],host['cleanup'])
 return {'dispatch':dispatch,'config':config,'adoption':adoption,'birth':birth,'owner_pid':own['pid'],'cli_prefix':[cli_python,'-B',str(HERE/'official_cli_no_oauth.py')],'runtime_wheels':opts.get('--runtime-wheels'),'options':opts}

def verify(packet,actual,d,review):
 packet=packet.resolve();actual=actual.resolve();ctx=context(packet,actual,d,review);manifest=C.read(packet/'manifest.json');base=d['remote_base'];birth=ctx['birth'];session=d['session'];required=E.labels(ctx['runtime_wheels']is not None);proofs=d.get('kernel_proofs',[])
 C.require(len(proofs)==len(required)and len({r.get('label')for r in proofs})==len(required)and {r.get('label')for r in proofs}==set(required),'COMPLETE_KERNEL_READBACK_SET');physical_scripts={p.name for p in actual.glob('*.py')};C.require(physical_scripts=={n+'.py'for n in required}and all(not(actual/(n+'.py')).is_symlink()for n in required),'EXACT_PHYSICAL_SUBMITTED_SCRIPTS')
 sources,modules=E.identities(base,manifest);bylabel={r['label']:r for r in proofs};requests={};nonces=set();readback_files=set()
 for label in required:
  row=bylabel[label];path=actual/'kernel-readbacks'/(label+'-proof.json');C.require(C.sha(path)==row['proof_sha256'],'KERNEL_PROOF_SHA');proof=C.read(path);observed=proof['request'];nonce=observed.get('nonce');C.require(isinstance(nonce,str)and re.fullmatch('[0-9a-f]{32}',nonce)is not None and nonce not in nonces,'UNIQUE_KERNEL_NONCE');nonces.add(nonce);spec=E.entry(label,base,d['packet_sha256'],birth,manifest,ctx['adoption']);req=K.request(spec['body'],label,base,d['packet_sha256'],birth,sources,artifact=spec['artifact_path'],artifact_expected=spec['artifact_expected']);req['nonce']=nonce;req['completion_path']=str(Path(base).parent/(Path(base).name+'-kernel-completions')/(label+'-'+nonce+'.json'));C.require(observed==req,'CANONICAL_KERNEL_REQUEST')
  script=actual/(label+'.py');expected=K.render(spec['body'],req,sources if spec['controls_ready']else{},modules if spec['controls_ready']else{}).encode();C.require(script.read_bytes()==expected and C.sha(script)==proof['submitted_source_sha256']==row['submitted_source_sha256'],'CANONICAL_SUBMITTED_KERNEL_SOURCE');requests[label]=req
  readback_files.update(label+s for s in('.json','-proof.json'))
  if req['artifact_path']is not None:readback_files.add(label+'.artifact.json')
 C.require(set(C.inventory(actual/'kernel-readbacks'))==readback_files,'EXACT_PHYSICAL_KERNEL_READBACKS')
 expected=[]
 def call(label,argv):expected.append((label,argv))
 def kernel(label):
  req=requests[label];call(label,None);call(label+'-completion',['download','-s',session,req['completion_path'].lstrip('/'),str(actual/'kernel-readbacks'/(label+'.json'))])
  if req['artifact_path']is not None:call(label+'-artifact',['download','-s',session,req['artifact_path'].lstrip('/'),str(actual/'kernel-readbacks'/(label+'.artifact.json'))])
 def recovery(phase):
  lab='recovery-'+phase;kernel('export-'+phase);summary=C.read(actual/lab/'summary.json');parts=C.read(actual/lab/'manifest.json');import transport;transport.validate_manifest(parts,summary['sha256'],summary['bytes'])
  C.require((actual/'kernel-readbacks'/('export-'+phase+'.artifact.json')).read_bytes()==(actual/lab/'summary.json').read_bytes(),'EXPORT_SUMMARY_INDEPENDENT_BYTES')
  C.require(summary['bytes']<=C.MAX_BYTES and summary['members']<=C.MAX_FILES and summary['physical_bytes']<=C.INNER_BYTES and(phase!='cpu'or summary['physical_bytes']<=C.CPU_BYTES),'EXPORT_SUMMARY_BOUND')
  call(lab+'-summary',['download','-s',session,(base+'/'+lab+'-export.json').lstrip('/'),str(actual/lab/'summary.json')]);call(lab+'-parts',['download','-s',session,(base+'/'+lab+'-parts/manifest.json').lstrip('/'),str(actual/lab/'manifest.json')])
  for i,p in enumerate(parts['parts']):call(lab+'-'+str(i),['download','-s',session,(base+'/'+lab+'-parts/'+p['name']).lstrip('/'),str(actual/lab/'parts'/p['name'])])
  kernel('dispose-'+phase)
 call('sessions-before',['sessions']);call('usage-before',['usage']);call('browser-registration',['--adopt-root-browser-v1',ctx['options']['--browser-adoption'],ctx['options']['--browser-adoption-sha256']]);call('sessions-after',['sessions']);kernel('browser-marker');kernel('readiness')
 for n in sorted(manifest['files']):
  if n.startswith('control/')and n.endswith('.py'):call('control-'+n.replace('/','-'),['upload','-s',session,str(packet/n),(base+'/'+n).lstrip('/')])
 call('packet',['upload','-s',session,ctx['options']['--packet-zip'],(base+'/payload.zip').lstrip('/')]);kernel('remote-unpack')
 if ctx['runtime_wheels']is not None:
  kernel('wheel-cache')
  for r in C.read(packet/'runtime/requirements.lock.json')['wheels']:call('locked-wheel-'+r['filename'],['upload','-s',session,str(Path(ctx['runtime_wheels'])/r['filename']),(base+'/wheel-cache/'+r['filename']).lstrip('/')])
 kernel('remote-install');kernel('remote-cpu');recovery('cpu');call('cpu-root-gate',['upload','-s',session,str(actual/'cpu-host-audit.json'),(base+'/records/cpu-gate.json').lstrip('/')]);kernel('remote-device');recovery('device')
 for lab,argv in [('sessions-before-stop',['sessions']),('stop-exact',['stop','-s',session]),('sessions-after-stop',['sessions']),('usage-after',['usage'])]:call(lab,argv)
 commands=d.get('commands',[]);C.require([r.get('label')for r in commands]==[lab for lab,_ in expected]and d['provider_calls']==len(expected),'COMPLETE_CANONICAL_CLI_COMMAND_SEQUENCE')
 directories={f'{i:03d}-'+lab for i,(lab,_)in enumerate(expected,1)};C.require({p.name for p in(actual/'cli').iterdir()}==directories,'EXACT_PHYSICAL_CLI_DIRECTORIES');execs=[];pidset=set();parsed={}
 for i,((label,argv),record)in enumerate(zip(expected,commands),1):
  directory=actual/'cli'/(f'{i:03d}-'+label);result,own,text=cli_record(directory,record,ctx['cli_prefix'],argv,ctx['owner_pid']);full=result['argv'];suffix=full[3:]
  if argv is None:
   C.require(len(suffix)==7 and suffix[:3]==['exec','-s',session]and suffix[3]=='--timeout'and suffix[5:] ==['-f',str(actual/(label+'.py'))],'EXACT_SUBMITTED_CLI_EXEC')
   try:seconds=int(suffix[4])
   except(ValueError,TypeError):raise ValueError('KERNEL_CLI_TIMEOUT')
   C.require(str(seconds)==suffix[4]and 0<seconds<=3600,'KERNEL_CLI_TIMEOUT');execs.append(label)
  generation=(own['pid'],own['started_at']);C.require(generation not in pidset,'UNIQUE_CLI_PROCESS_GENERATION');pidset.add(generation);parsed[label]=(directory,result)
 C.require(tuple(execs)==required,'COMPLETE_PHYSICAL_CLI_EXEC_SET')
 for label in required:
  proof=C.read(actual/'kernel-readbacks'/(label+'-proof.json'));req=requests[label];artifact=actual/'kernel-readbacks'/(label+'.artifact.json')if req['artifact_path']is not None else None;verified=K.verify((parsed[label][0]/'stdout-stderr.raw').read_text(),req,actual/'kernel-readbacks'/(label+'.json'),artifact);C.require(all(proof[k]==v for k,v in verified.items()),'KERNEL_PROOF_RECOMPUTED')
 closure=resource_empty((parsed['sessions-after-stop'][0]/'stdout-stderr.raw').read_text(),(parsed['usage-after'][0]/'stdout-stderr.raw').read_text())
 for label in ('sessions-before-stop','stop-exact','sessions-after-stop','usage-after'):C.require(d[label]==(parsed[label][0]/'stdout-stderr.raw').read_text(),'DRIVER_PHYSICAL_CLOSURE_TEXT')
 import driver as D
 C.require(d.get('server_empty_observed')is closure['sessions']['empty']and d.get('usage_zero_observed')is closure['usage']['displayed_zero']and d.get('usage_readback')==D.usage_zero(d['usage-after'])and not d.get('closure_errors'),'DRIVER_PHYSICAL_CLOSURE_OBSERVATIONS')
 owned_session((parsed['sessions-before'][0]/'stdout-stderr.raw').read_text(),ctx['adoption'],unregistered=True)
 owned_session((parsed['sessions-after'][0]/'stdout-stderr.raw').read_text(),ctx['adoption'])
 owned_session(d['sessions-before-stop'],ctx['adoption'],allow_empty=True)
 return {'resource_closure':closure,'status':'COMPLETE_CANONICAL_KERNEL_AND_CLI_READBACK_VERIFIED','count':len(required),'cli_commands':len(expected),'runtime_wheels_enabled':ctx['runtime_wheels']is not None,'all_source_scripts_regenerated':True}
