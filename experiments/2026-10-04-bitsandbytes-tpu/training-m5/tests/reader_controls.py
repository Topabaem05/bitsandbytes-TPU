"""Mutate disposable production-shaped records; mandatory complete-run failures."""
import copy,json,shutil
from pathlib import Path
import contract as C,inspect_actual as I

def exercise(packet,out,driver,review,cache=False):
 rows=[]
 def verify(d=None):return I.kernel_readbacks(packet,out,driver if d is None else d,review)
 good=verify();assert good['count']==(11 if cache else 10);rows.append({'name':'correct_complete_'+str(good['count']),'status':'PASS','evidence':good})
 # Exercise the complete record-reader path with an explicit numerical stub.
 # This is not a scientific or milestone acceptance fixture.
 import datetime
 from unittest.mock import patch
 import kernel_reader as KR
 def full_inspect(d):
  C.write(out/'driver.json',d);cfg=C.read(review/'dispatch/host-config.json')['argv'];archive=Path(cfg[cfg.index('--packet-zip')+1])
  def numerical_stub(*args,**kwargs):return {'record_validation':'PASS','numerical_status':'PASS','fixture_only_numerical_audit_stub':True}
  with patch.object(I.H,'audit',numerical_stub):result=I.inspect(packet,archive,d['packet_sha256'],out,review)
  assert result['fixture_only_numerical_audit_stub']is True and result['m5_status']=='NOT_QUALIFIED'
  return result
 def make_fresh():
  cfg=C.read(review/'dispatch/host-config.json')['argv'];prefix=[cfg[cfg.index('--cli-python')+1],'-B',str(KR.HERE/'official_cli_no_oauth.py')];commands=[]
  for i,label in enumerate(('sessions','usage')):
   directory=review/'fresh-resource-closure/cli'/(str(i)+'-'+label);directory.mkdir(parents=True);stamp=datetime.datetime.now(datetime.timezone.utc).isoformat();pid=30000+i;g={'pid':pid,'pgid':pid,'started_at':stamp,'ended_at':stamp,'exit_status':0,'leader_reaped':True,'status':'CLEANUP_VERIFIED','group_absence':'OBSERVED_NO_SUCH_GROUP','errors':[]};cleanup={'status':'NO_CLEANUP_ERRORS','errors':[],'groups':[g],'restoration':{'handlers_restored':True,'timer_restored':True}};argv=prefix+[label];own={'argv':argv,'pid':pid,'pgid':pid,'owner_pid':29000,'started_at':stamp,'ended_at':stamp,'exit_status':0,'cleanup':g};result={'argv':argv,'timeout_seconds':45,'status':'PASS','exit_code':0,'original_outcome':{'status':'PASS','exit_code':0,'error':None},'cleanup':cleanup};C.write(directory/'result.json',result);C.write(directory/'ownership.json',own);C.write(directory/'cleanup.json',cleanup);(directory/'lifecycle.jsonl').write_text(json.dumps({'kind':'child_started','pid':pid,'pgid':pid,'recorded_at':stamp})+'\n'+json.dumps({'kind':'child_cleanup_checks','pid':pid,'errors':[],'recorded_at':stamp})+'\n');(directory/'stdout-stderr.raw').write_text('[colab] No active sessions found on server.\n'if label=='sessions'else'Current balance: 100.00 compute units\nUsage rate: 0.00/hr\nActive assignments: 0\n');commands.append(result)
  import time
  C.write(review/'fresh-resource-closure.json',{'status':'NO_ACTIVE_SESSION_OR_USAGE','epoch':time.time(),'commands':commands,'cli_identity_sha256':C.read(review/'dispatch/root-dispatch.json')['cli_identity_sha256'],'allocation':'NOT_RUN'});C.write(review/'browser-tab-closure.json',{'status':'OWNED_BROWSER_TAB_CLOSED','tab_id':'synthetic-tab','tab_absent':True,'epoch':time.time(),'fixture_only':True})
 make_fresh();full_inspect(driver);rows.append({'name':'correct_complete_full_record_inspect_'+str(good['count']),'status':'PASS','numerical_audit':'EXPLICIT_SYNTHETIC_STUB_NOT_MILESTONE'})
 if cache:
  changed=copy.deepcopy(driver);changed['kernel_proofs']=[r for r in changed['kernel_proofs']if r['label']!='wheel-cache']
  try:verify(changed)
  except ValueError as e:assert 'COMPLETE_KERNEL_READBACK_SET'in str(e);rows.append({'name':'required_cache_missing','status':'PASS','rejected':str(e)})
  else:raise AssertionError('required cache accepted missing')
  return rows
 backup=out.parent/'reader-pristine';shutil.copytree(out,backup);review_backup=out.parent/'review-pristine';shutil.copytree(review,review_backup)
 def mutate(name,fn,expected=None,full=False):
  d=copy.deepcopy(driver)
  try:
   fn(d);full_inspect(d)if full else verify(d)
  except (ValueError,FileNotFoundError,KeyError)as e:
   if expected:assert expected in str(e),(name,str(e),expected)
   rows.append({'name':name,'status':'PASS','rejected':str(e)})
  else:raise AssertionError('reader accepted '+name)
  finally:
   shutil.rmtree(out);shutil.copytree(backup,out);shutil.rmtree(review);shutil.copytree(review_backup,review)
 def proof(d,change,label='remote-install'):
  p=out/'kernel-readbacks'/(label+'-proof.json');v=C.read(p);change(v);C.write(p,v)
  next(r for r in d['kernel_proofs']if r['label']==label)['proof_sha256']=C.sha(p)
 def cli(d,label,change):
  row=next(r for r in d['commands']if r['label']==label);directory=next((out/'cli').glob('*-'+label));r=C.read(directory/'result.json');o=C.read(directory/'ownership.json');cl=C.read(directory/'cleanup.json');change(r,o,cl,row);C.write(directory/'result.json',r);C.write(directory/'ownership.json',o);C.write(directory/'cleanup.json',cl)
 def argv(d,label,change):
  def edit(r,o,cl,row):change(r['argv']);o['argv']=r['argv']
  cli(d,label,edit)
 def remove(d,labels,physical=True):
  d['kernel_proofs']=[r for r in d['kernel_proofs']if r['label']not in labels];d['commands']=[r for r in d['commands']if r['label']not in labels];d['provider_calls']=len(d['commands'])
  if physical:
   for label in labels:
    for p in(out/(label+'.py'),):
     if p.exists():p.unlink()
    for p in(out/'kernel-readbacks').glob(label+'*'):p.unlink()
    for p in(out/'cli').glob('*-'+label):shutil.rmtree(p)
 def all_removed(d):
  d.update(commands=[],kernel_proofs=[],provider_calls=0)
  for p in out.glob('*.py'):p.unlink()
  shutil.rmtree(out/'cli');(out/'cli').mkdir();shutil.rmtree(out/'kernel-readbacks');(out/'kernel-readbacks').mkdir()
 mutate('all_rows_and_all_physical_removed',all_removed,'COMPLETE_KERNEL_READBACK_SET')
 mutate('all_mutable_rows_removed_keep_physical',lambda d:d.update(commands=[],kernel_proofs=[],provider_calls=0),'COMPLETE_KERNEL_READBACK_SET')
 mutate('install_pair_and_physical_removed',lambda d:remove(d,{'remote-install','remote-install-completion','remote-install-artifact'}),'COMPLETE_KERNEL_READBACK_SET')
 mutate('missing_export_and_dispose',lambda d:remove(d,{'export-device','dispose-device'}),'COMPLETE_KERNEL_READBACK_SET')
 mutate('empty_physical_cli_inventory',lambda d:shutil.rmtree(out/'cli'))
 mutate('missing_submitted_install_script',lambda d:(out/'remote-install.py').unlink(),'EXACT_PHYSICAL_SUBMITTED_SCRIPTS')
 mutate('extra_submitted_exec_script',lambda d:(out/'unexpected.py').write_text('print(1)'),'EXACT_PHYSICAL_SUBMITTED_SCRIPTS')
 mutate('duplicate_phase',lambda d:d['kernel_proofs'].__setitem__(-1,copy.deepcopy(d['kernel_proofs'][0])),'COMPLETE_KERNEL_READBACK_SET')
 nonce=C.read(out/'kernel-readbacks/browser-marker-proof.json')['request']['nonce']
 mutate('duplicate_nonce',lambda d:proof(d,lambda v:v['request'].update(nonce=nonce)),'UNIQUE_KERNEL_NONCE')
 for label in ['artifact_path','artifact_expected']:
  for name,value in([('none',None),('wrong_path','/content/other.json')]if label=='artifact_path'else[('dropped',{})]):
   mutate('weakened_'+label+'_'+name,lambda d,label=label,value=value:proof(d,lambda v:v['request'].__setitem__(label,value)),'CANONICAL_KERNEL_REQUEST')
 mutate('wrong_request_source_sha',lambda d:proof(d,lambda v:v['request'].update(admitted_source_sha256='a'*64)),'CANONICAL_KERNEL_REQUEST')
 mutate('completion_path_inside_base',lambda d:proof(d,lambda v:v['request'].update(completion_path=driver['remote_base']+'/completion.json')),'CANONICAL_KERNEL_REQUEST')
 def body(d):
  p=out/'remote-install.py';p.write_text(p.read_text()+'\nprint("changed body")\n');proof(d,lambda v:v.update(submitted_source_sha256=C.sha(p)));next(r for r in d['kernel_proofs']if r['label']=='remote-install')['submitted_source_sha256']=C.sha(p)
 mutate('resealed_wrong_submitted_source',body,'CANONICAL_SUBMITTED_KERNEL_SOURCE')
 mutate('wrong_exec_session',lambda d:argv(d,'remote-install',lambda a:a.__setitem__(5,'another-session')),'EXACT_SUBMITTED_CLI_EXEC')
 mutate('wrong_exec_file',lambda d:argv(d,'remote-install',lambda a:a.__setitem__(-1,str(out/'remote-cpu.py'))),'EXACT_SUBMITTED_CLI_EXEC')
 mutate('bare_cli_prefix',lambda d:argv(d,'remote-install',lambda a:a.__delitem__(slice(0,3))),'ADMITTED_CLI_PREFIX_AND_ARGV')
 mutate('module_cli_prefix',lambda d:argv(d,'remote-install',lambda a:a.__setitem__(2,'-m')),'ADMITTED_CLI_PREFIX_AND_ARGV')
 mutate('different_cli_python',lambda d:argv(d,'remote-install',lambda a:a.__setitem__(0,'/other/python')),'ADMITTED_CLI_PREFIX_AND_ARGV')
 mutate('completion_download_wrong_source',lambda d:argv(d,'remote-install-completion',lambda a:a.__setitem__(-2,'content/another.json')),'CANONICAL_CLI_OPERATION')
 mutate('artifact_download_wrong_destination',lambda d:argv(d,'remote-install-artifact',lambda a:a.__setitem__(-1,str(out/'other.json'))),'CANONICAL_CLI_OPERATION')
 mutate('completion_download_absent',lambda d:remove(d,{'remote-install-completion'}),'COMPLETE_CANONICAL_CLI_COMMAND_SEQUENCE')
 def extra(d):
  d['commands'].insert(4,copy.deepcopy(d['commands'][4]));d['provider_calls']+=1
 mutate('extra_exec_command',extra,'COMPLETE_CANONICAL_CLI_COMMAND_SEQUENCE')
 def failed_child(d):
  def edit(r,o,c,row):o['exit_status']=1;o['cleanup']['exit_status']=1;c['groups'][0]['exit_status']=1;r['cleanup']=c;row['cleanup']=c
  cli(d,'remote-install',edit)
 mutate('cli_zero_failed_owned_child',failed_child,'OWNED_CLI_PROCESS_OUTCOME')
 def dirty(d):
  def edit(r,o,c,row):c['errors']=['synthetic dirty closure'];r['cleanup']=c;row['cleanup']=c
  cli(d,'remote-install',edit)
 mutate('resealed_dirty_cli_closure',dirty,'OWNED_CLEANUP_ERRORS')
 def missing_result(d):next((out/'cli').glob('*-remote-install/result.json')).unlink()
 mutate('physical_result_absent',missing_result,'COMPLETE_PHYSICAL_CLI_RECORDS')
 mutate('physical_raw_error_after_marker',lambda d:next((out/'cli').glob('*-remote-install/stdout-stderr.raw')).write_text('ValueError: kernel failed\n'),'KERNEL_COMPLETION_LINE')
 mutate('extra_physical_exec_directory',lambda d:shutil.copytree(next((out/'cli').glob('*-remote-install')),out/'cli/999-extra-exec'),'EXACT_PHYSICAL_CLI_DIRECTORIES')
 def ownership_argv(d):
  p=next((out/'cli').glob('*-remote-install/ownership.json'));o=C.read(p);o['argv']=['wrong'];C.write(p,o)
 mutate('result_ownership_argv_disagree',ownership_argv,'ADMITTED_CLI_PREFIX_AND_ARGV')
 def export_summary(d):
  p=out/'recovery-cpu/summary.json';v=C.read(p);v['unexpected']='resealed mutable summary';C.write(p,v)
 mutate('independent_export_summary_bytes_changed',export_summary,'EXPORT_SUMMARY_INDEPENDENT_BYTES')
 def lifecycle(d):
  p=next((out/'cli').glob('*-remote-install/lifecycle.jsonl'));events=[json.loads(line)for line in p.read_text().splitlines()];events[0].update(pid=1,pgid=1);p.write_text(json.dumps(events[0])+'\n')
 mutate('wrong_physical_lifecycle',lifecycle,'LIFECYCLE_UNOWNED_OR_AMBIGUOUS_CHILD')
 mutate('partial_execution_not_complete',lambda d:d.update(status='BLOCKED'), 'PARTIAL_EXECUTION_NOT_COMPLETE')
 mutate('wrong_host_owner_argv',lambda d:C.write(review/'host-owner/ownership.json',{**C.read(review/'host-owner/ownership.json'),'argv':['wrong']}),'OWNED_CLI_PROCESS_OUTCOME')
 mutate('wrong_host_config_source',lambda d:C.write(review/'dispatch/host-config.json',{**C.read(review/'dispatch/host-config.json'),'argv':['wrong']}),'ADMITTED_HOST_DRIVER_PREFIX')
 def cache_required(d):
  p=review/'dispatch/host-config.json';v=C.read(p);v['argv']+=['--runtime-wheels','/synthetic/cache'];C.write(p,v)
  p=review/'host-owner/host-finalization.json';h=C.read(p);h.update(argv=v['argv'],config_sha256=C.sha(review/'dispatch/host-config.json'));h['cleanup']['groups'][0]['exit_status']=0;C.write(p,h)
  p=review/'host-owner/ownership.json';o=C.read(p);o['argv']=v['argv'];C.write(p,o)
 mutate('required_cache_config_with_no_cache',cache_required,'COMPLETE_KERNEL_READBACK_SET')

 # Physical output inconsistencies must fail even when all mutable summaries say PASS.
 def raw(d,label,text,bind=True):
  next((out/'cli').glob('*-'+label+'/stdout-stderr.raw')).write_text(text)
  if bind and label in d:d[label]=text
 empty='[colab] No active sessions found on server.\n';zero='Current balance: 100.00 compute units\nUsage rate: 0.00/hr\nActive assignments: 0\n'
 active='['+driver['session']+'] '+C.read(review/'dispatch/browser-adoption.json')['endpoint']+' | Hardware: V5E1 | Shape: Standard | Variant: TPU\n'
 for name,text in [('empty_raw',''),('active',active),('empty_active',empty+active),('empty_twice',empty+empty),('auth_warning','Authentication required.\n'+empty),('unrecognized','No active sessions\n')]:
  mutate('physical_sessions_'+name,lambda d,text=text:raw(d,'sessions-after-stop',text),full=True)
 for name,text in [('empty_raw',''),('nonzero_assignments',zero.replace('assignments: 0','assignments: 1')),('nonzero_rate',zero.replace('0.00/hr','0.10/hr')),('duplicate_rate',zero+'Usage rate: 0.00/hr\n'),('missing_rate',zero.replace('Usage rate: 0.00/hr\n','')),('ambiguous_rate',zero.replace('0.00/hr','0.00 or 0.10/hr')),('wrong_order','Usage rate: 0.00/hr\nCurrent balance: 100.00 compute units\nActive assignments: 0\n')]:
  mutate('physical_usage_'+name,lambda d,text=text:raw(d,'usage-after',text),full=True)
 mutate('driver_raw_closure_text_disagrees',lambda d:raw(d,'usage-after',zero.replace('100.00','99.00'),bind=False),'DRIVER_PHYSICAL_CLOSURE_TEXT',full=True)
 mutate('driver_closure_flag_disagrees',lambda d:d.update(usage_zero_observed=False),'DRIVER_PHYSICAL_CLOSURE_OBSERVATIONS',full=True)
 for label in ('sessions-before','sessions-after','sessions-before-stop'):
  for field,value in [('endpoint','wrong-endpoint'),('session','wrong-session')]:
   text=active.replace(C.read(review/'dispatch/browser-adoption.json')['endpoint'],value)if field=='endpoint'else active.replace('['+driver['session']+']','['+value+']')
   mutate('physical_'+label+'_wrong_'+field,lambda d,label=label,text=text:raw(d,label,text),'PHYSICAL_ADOPTED_SESSION_IDENTITY',full=True)
 mutate('physical_multiple_sessions',lambda d:raw(d,'sessions-after',active+active.replace(driver['session'],'other-session').replace(C.read(review/'dispatch/browser-adoption.json')['endpoint'],'another-endpoint')),'PHYSICAL_ADOPTED_SESSION_IDENTITY',full=True)
 for label,text in [('sessions',active),('usage',zero.replace('0.00/hr','0.10/hr')),('usage',zero+'Active assignments: 0\n')]:
  mutate('fresh_only_bad_'+label+'_'+str(len(text)),lambda d,label=label,text=text:(review/'fresh-resource-closure/cli'/('0-sessions'if label=='sessions'else'1-usage')/'stdout-stderr.raw').write_text(text),full=True)
 mutate('fresh_extra_child_started',lambda d:(review/'fresh-resource-closure/cli/0-sessions/lifecycle.jsonl').open('a').write(json.dumps({'kind':'child_started','pid':99999,'pgid':99999,'recorded_at':'2026-10-11T00:00:00+00:00'})+'\n'),full=True)
 mutate('fresh_command_physical_disagreement',lambda d:C.write(review/'fresh-resource-closure.json',{**C.read(review/'fresh-resource-closure.json'),'commands':[]}),'FRESH_PHYSICAL_CLI_SET',full=True)
 def ledger_path(host=False):return review/'host-owner/lifecycle.jsonl'if host else next((out/'cli').glob('*-remote-install/lifecycle.jsonl'))
 def ledger(d,change,host=False):
  path=ledger_path(host);events=[json.loads(line)for line in path.read_text().splitlines()];change(events);path.write_text(''.join(json.dumps(v)+'\n'for v in events))
 def extra_events(events,closed=False):
  events.append({**events[0],'pid':99999,'pgid':99999})
  if closed:events.append({**events[1],'pid':99999})
 cases=[('extra_unclosed_child',lambda e:extra_events(e)),('extra_fully_closed_unowned_child',lambda e:extra_events(e,True)),('orphan_cleanup',lambda e:e.insert(0,dict(e[1]))),('duplicate_cleanup',lambda e:e.append(dict(e[1]))),('missing_cleanup',lambda e:e.pop()),('missing_start',lambda e:e.pop(0)),('cleanup_before_start',lambda e:e.reverse()),('wrong_pgid',lambda e:e[0].update(pgid=e[0]['pid']+1)),('duplicate_start',lambda e:e.insert(1,dict(e[0]))),('ignored_signal',lambda e:e.append({'kind':'signal','signal':15,'recorded_at':e[0]['recorded_at']})),('unknown_event',lambda e:e.append({'kind':'ignored','recorded_at':e[0]['recorded_at']})),('cleanup_errors',lambda e:e[1].update(errors=['unclosed']))]
 for host in (False,True):
  for name,change in cases:mutate(('host_'if host else'cli_')+name,lambda d,change=change,host=host:ledger(d,change,host),full=True)
 # Valid production alternatives: stale-local pruning, displayed zero with debt,
 # a reused PID in a distinct launch generation, and wall-clock adjustment.
 def positive(name,fn):
  d=copy.deepcopy(driver)
  try:fn(d);full_inspect(d);rows.append({'name':name,'status':'PASS','numerical_audit':'EXPLICIT_SYNTHETIC_STUB_NOT_MILESTONE'})
  finally:shutil.rmtree(out);shutil.copytree(backup,out);shutil.rmtree(review);shutil.copytree(review_backup,review)
 def prune(d):
  text='[colab] Pruned 2 stale local session(s).\n'+empty;raw(d,'sessions-after-stop',text)
  (review/'fresh-resource-closure/cli/0-sessions/stdout-stderr.raw').write_text(text)
 positive('correct_prune_then_empty',prune)
 def debt(d):raw(d,'usage-after',zero.replace('100.00','-1.00'));(review/'fresh-resource-closure/cli/1-usage/stdout-stderr.raw').write_text(zero.replace('100.00','-1.00'))
 positive('correct_displayed_zero_with_negative_balance',debt)
 def reuse(d):
  directories=sorted((out/'cli').iterdir());first=C.read(directories[0]/'ownership.json');directory=directories[1];own=C.read(directory/'ownership.json');result=C.read(directory/'result.json');cleanup=C.read(directory/'cleanup.json');row=d['commands'][1];pid=first['pid'];stamp=(datetime.datetime.fromisoformat(own['started_at'])+datetime.timedelta(microseconds=1)).isoformat();own.update(pid=pid,pgid=pid,started_at=stamp,ended_at=stamp);cleanup['groups'][0].update(pid=pid,pgid=pid,started_at=stamp,ended_at=stamp);own['cleanup']=cleanup['groups'][0];result['cleanup']=cleanup;row['cleanup']=cleanup;C.write(directory/'ownership.json',own);C.write(directory/'result.json',result);C.write(directory/'cleanup.json',cleanup);events=[{'kind':'child_started','pid':pid,'pgid':pid,'recorded_at':stamp},{'kind':'child_cleanup_checks','pid':pid,'errors':[],'recorded_at':stamp}];(directory/'lifecycle.jsonl').write_text(''.join(json.dumps(e)+'\n'for e in events))
 positive('correct_reused_pid_distinct_generation',reuse)
 positive('correct_wall_clock_adjustment_in_ledger',lambda d:ledger(d,lambda e:e[1].update(recorded_at='2001-01-01T00:00:00+00:00')))
 # Generic ParentOwnership can close more than one known child.
 own=C.read(next((out/'cli').glob('*-remote-install/ownership.json')));group=own['cleanup'];own2=copy.deepcopy(own);own2.update(pid=own['pid']+100000,pgid=own['pid']+100000);own2['cleanup'].update(pid=own2['pid'],pgid=own2['pid']);path=out/'generic-ledger.jsonl';stamp=own['started_at'];events=[{'kind':'child_started','pid':v['pid'],'pgid':v['pid'],'recorded_at':stamp}for v in(own,own2)]+[{'kind':'child_cleanup_checks','pid':v['pid'],'errors':[],'recorded_at':stamp}for v in(own,own2)];path.write_text(''.join(json.dumps(e)+'\n'for e in events));assert KR.lifecycle_ledger(path,[own,own2],{'groups':[own2['cleanup'],group]})=={'launch_generations':2,'events':4};path.unlink();rows.append({'name':'correct_generic_multiple_owned_children','status':'PASS'})
 shutil.rmtree(backup);shutil.rmtree(review_backup);return rows
