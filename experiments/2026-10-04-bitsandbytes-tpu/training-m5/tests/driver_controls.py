"""Real M5 driver/remote branches; all CLI/science/audit fixtures synthetic only."""
import ast,copy,datetime,json,shutil,sys,tempfile,time,types
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'control'),str(ROOT/'tests')]
import contract as C,driver as D,remote as R,kernel_protocol as K,transport,inspect_actual as I
from kernel_controls import packet

def main(output):
 output=Path(output);C.require(not output.exists(),'FRESH_DRIVER_CONTROLS');reports=[]
 with tempfile.TemporaryDirectory(prefix='bnb-m5-v2-driver-',dir='/tmp')as td:
  shared=Path(td).resolve();pkt,z,pktsha=packet(shared)
  def case(name):
   with tempfile.TemporaryDirectory(prefix='scenario-',dir=shared)as sd:
    root=Path(sd).resolve();out=root/'host';server=root/'server';server.mkdir();birth=time.time()-2;session='synthetic-session';base='/content/m5-exact1024-'+session;events=[];sciences=[];deadlines=[]
    review=root/'review';(review/'dispatch').mkdir(parents=True);identity=root/'identity.json';shutil.copyfile('/Users/topamini/code/bitsandbytes-TPU/.work/task2-cloud/cli-identity.json',identity);adoption=review/'dispatch/browser-adoption.json';C.write(adoption,{'format':'bnb-tpu.browser-adoption.v1','status':'ROOT_CREATED_BROWSER_RUNTIME_READY','endpoint':'synthetic-endpoint','session':session,'hardware':'V5E1','variant':'TPU','authuser':'0','allocation_epoch':birth,'observed_epoch':time.time(),'packet_sha256':pktsha,'driver_sha256':C.sha(ROOT/'control/driver.py'),'cli_identity_sha256':C.sha(identity),'marker_path':'/tmp/bnb-tpu-root-browser-'+('b'*32)+'.json','marker_sha256':'c'*64})
    dispatch=review/'dispatch/root-dispatch.json';C.write(dispatch,{'kind':'ROOT_M5_EXACT1024_DISPATCH','status':'ACTUAL_DISPATCH_AUTHORIZED','packet_sha256':pktsha,'driver_sha256':C.sha(ROOT/'control/driver.py'),'cli_identity_sha256':C.sha(identity),'allocation_seconds':3600,'allocation_attempts':1,'science_manifest_sha256':C.science(pkt),'output':str(out.resolve()),'browser_adoption_sha256':C.sha(adoption),'original_allocation_epoch':birth,'fixture_only':True})
    def remote_path(s):
     sibling=base+'-kernel-completions/'
     if s.lstrip('/').startswith(sibling.lstrip('/')):return root/'kernel-completions'/s.split('-kernel-completions/',1)[1]
     assert s.startswith(base+'/')or s.startswith(base.lstrip('/')+'/'),s
     return server/s.split('m5-exact1024-'+session+'/',1)[1]
    class FakeCLI:
     def __init__(self,python,output):self.stopped=False;self.output=output;self.prefix=[str(python),'-B',str(D.HERE/'official_cli_no_oauth.py')]
     def __call__(self,label,argv,timeout):
      rec={'status':'PASS','exit_code':0,'cleanup':{'status':'NO_CLEANUP_ERRORS','errors':[],'groups':[]}};text='';events.append('cli:'+argv[0]);stem=label.split('-',1)[1]
      if argv[0]=='sessions':text='[colab] No active sessions found on server.'if self.stopped else'['+('?'if stem=='sessions-before'else session)+'] synthetic-endpoint | Hardware: V5E1 | Shape: Standard | Variant: TPU'
      elif argv[0]=='usage':text='Current balance: 100.00 compute units\nUsage rate: '+('0.10'if self.stopped and name=='wrong_usage_closure'else'0.00')+'/hr\nActive assignments: 0\n'
      elif argv[0]=='--adopt-root-browser-v1':pass
      elif argv[0]=='stop':self.stopped=True
      elif argv[0]=='upload'and stem.startswith('locked-wheel-'):pass
      elif argv[0]=='upload':
       dest=remote_path(argv[-1]);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(argv[-2],dest)
      elif argv[0]=='download':
       if name=='missing_completion_download'and stem=='remote-install-completion':raise FileNotFoundError('synthetic missing independent completion')
       shutil.copyfile(remote_path(argv[-2]),argv[-1])
      elif argv[0]=='exec':
       source=Path(argv[-1]);stem=source.stem
       if stem=='browser-marker':text='EXACT_ROOT_MARKER_MATCH\n'
       elif stem=='readiness':(server/'control/ownership').mkdir(parents=True)
       elif stem=='wheel-cache':(server/'wheel-cache').mkdir()
       elif stem.startswith('remote-'):
        phase=stem.removeprefix('remote-');events.append('remote:'+phase)
        if phase=='install':C.write(server/'records/phase.json',{'kind':'M5_EXACT1024_STAGE','phase':'install','packet_sha256':pktsha,'allocation_epoch':birth,'status':'TERMINAL_LOCAL_REVIEW_REQUIRED','m5_status':'NOT_QUALIFIED'})
        else:
         with patch.object(R,'run_step',science_step):R.operation(types.SimpleNamespace(command=phase,base=server,allocation_epoch=birth,packet_sha256=pktsha))
        if name=='science_cli_failure'and phase=='device':rec.update(status='CLI_FAILED',exit_code=1)
       elif stem.startswith('export-'):
        phase=stem.removeprefix('export-');events.append('recover:'+phase);lab='recovery-'+phase;summary=C.export(server/'records',server/(lab+'.zip'));C.write(server/(lab+'-export.json'),summary);transport.split(server/(lab+'.zip'),server/(lab+'-parts'),expected_sha256=summary['sha256'],expected_bytes=summary['bytes'])
       elif stem.startswith('dispose-'):
        lab='recovery-'+stem.removeprefix('dispose-');(server/(lab+'.zip')).unlink();shutil.rmtree(server/(lab+'-parts'));(server/(lab+'-export.json')).unlink()
       else:raise ValueError('UNEXPECTED_FAKE_KERNEL:'+stem)
       if rec['status']=='PASS':
        req=next(ast.literal_eval(n.value)for n in ast.parse(source.read_text()).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='_m4kv3_request'for t in n.targets));receipt={**req,'status':'COMPLETED','artifact':None}
        if req['artifact_path']is not None:receipt['artifact']=K.row(remote_path(req['artifact_path']))
        path=remote_path(req['completion_path']);C.write(path,receipt);mark={'nonce':req['nonce'],'phase':req['phase'],'admitted_source_sha256':req['admitted_source_sha256'],'readback_sha256':C.sha(path)}
        if stem=='remote-install':
         if name=='wrong_nonce':mark['nonce']='wrong'
         if name=='wrong_phase':mark['phase']='wrong'
         if name=='wrong_source':mark['admitted_source_sha256']='e'*64
        text+=K.VERSION+' '+json.dumps(mark)+'\n'
        if stem=='remote-install':
         if name=='cli_zero_kernel_error':text='ValueError: synthetic kernel failure\n'
         if name=='duplicate_marker':text+=text
         if name=='post_marker_error':text+='ValueError: synthetic late error\n'
         if name=='wrong_artifact':C.write(server/'records/phase.json',{'changed':'after completion'})
        if stem=='export-cpu'and name=='export_cli_zero_no_marker':text='ValueError: synthetic export failure\n'
      else:raise ValueError('UNEXPECTED_FAKE_CLI:'+str(argv))
      d=self.output/'cli'/label;d.mkdir(parents=True);(d/'stdout-stderr.raw').write_text(text);pid=20000+int(label.split('-',1)[0]);stamp=datetime.datetime.now(datetime.timezone.utc).isoformat();g={'started_at':stamp,'ended_at':stamp,'pid':pid,'pgid':pid,'exit_status':rec['exit_code'],'leader_reaped':True,'status':'CLEANUP_VERIFIED','group_absence':'OBSERVED_NO_SUCH_GROUP','errors':[]};rec.update(argv=self.prefix+argv,timeout_seconds=timeout,original_outcome={'status':rec['status'],'exit_code':rec['exit_code'],'error':None},cleanup={'status':'NO_CLEANUP_ERRORS','errors':[],'groups':[g],'restoration':{'handlers_restored':True,'timer_restored':True}});C.write(d/'ownership.json',{'started_at':stamp,'ended_at':stamp,'argv':rec['argv'],'pid':pid,'pgid':pid,'owner_pid':19000,'exit_status':rec['exit_code'],'cleanup':g});C.write(d/'cleanup.json',rec['cleanup']);C.write(d/'result.json',rec);(d/'lifecycle.jsonl').write_text(json.dumps({'kind':'child_started','pid':pid,'pgid':pid,'recorded_at':stamp})+'\n'+json.dumps({'kind':'child_cleanup_checks','pid':pid,'errors':[],'recorded_at':stamp})+'\n');return rec,text
    def science_step(records,label,argv,deadline,limit,**kw):
     events.append('science:'+label);sciences.append(label);deadlines.append(deadline);dest=Path(argv[argv.index('--output')+1]);dest.mkdir();C.write(dest/('oracle-seal.json'if label=='cpu'else'receipt.json'),{'fixture_only':True,'finished_epoch':time.time(),'deadline_epoch':deadline,'process_token':argv[argv.index('--process-token')+1]});return {'status':'PASS','exit_code':0}
    def host_fixture(records,label,argv,deadline,limit,**kw):
     phase=label.removeprefix('host-');audit={'fixture_only':True,'status':'QUALIFIED_LINUX20_RESUME10_VERIFIED'if phase=='cpu'else'SYNTHETIC_DEVICE_AUDIT','record_validation':'PASS','numerical_status':'PASS','packet_sha256':pktsha,'oracle_sha256':C.sha(server/'records/cpu/oracle-seal.json'),'m4_acceptance_sha256':C.sha(pkt/'accepted-m4.json')};C.write(Path(argv[argv.index('--output')+1]),audit);return {'status':'PASS','exit_code':0}
    args=types.SimpleNamespace(packet=pkt,packet_zip=z,packet_sha256=pktsha,cli_python=Path('/Users/topamini/code/Post2TPU/.runtime/cloud-cli/bin/python'),cli_identity=identity,dispatch=dispatch,session=None,browser_adoption=adoption,browser_adoption_sha256=C.sha(adoption),output=out,runtime_wheels=(root/'synthetic-wheel-cache'if name=='correct_cache'else None));error=None
    with patch.object(D.T,'verify_cli_identity',lambda *a:None),patch.object(D.T,'OfficialCLI',FakeCLI),patch.object(D,'run_step',host_fixture),patch.object(C,'runtime_cache',lambda *a:C.read(pkt/'runtime/requirements.lock.json')['wheels']):
     try:D.run(args)
     except BaseException as e:error={'type':type(e).__name__,'message':str(e)}
    driver=C.read(out/'driver.json');assert driver['server_empty_observed']and driver['allocation_attempts']==0;assert driver['allocation_epoch']==birth and driver['original_deadline_epoch']==birth+3600;assert all(d==birth+3600-C.CLOSURE_RESERVE for d in deadlines);assert 'cli:stop'in events
    reader_controls=[]
    if name in('correct','correct_cache'):
     assert error is None and sciences==['cpu','device']and driver['status']=='TRAINING20_RESUME10_RECOVERED_ROOT_REVIEW_REQUIRED',(error,driver)
     host_argv=[sys.executable,'-B',str(D.HERE/'driver.py')]
     for flag,val in[('--packet',pkt.resolve()),('--packet-zip',z),('--packet-sha256',pktsha),('--cli-python',args.cli_python),('--cli-identity',identity),('--dispatch',dispatch),('--browser-adoption',adoption),('--browser-adoption-sha256',C.sha(adoption)),('--output',out.resolve())]:host_argv.extend([flag,str(val)])
     if args.runtime_wheels is not None:host_argv.extend(['--runtime-wheels',str(args.runtime_wheels)])
     config=review/'dispatch/host-config.json';C.write(config,{'argv':host_argv,'packet_sha256':pktsha,'absolute_deadline_epoch':birth+3600,'host_output':str(review.resolve()/'host-owner')})
     hoststamp=datetime.datetime.fromtimestamp(birth+1,datetime.timezone.utc).isoformat();hostend=datetime.datetime.now(datetime.timezone.utc).isoformat();hg={'started_at':hostend,'ended_at':hostend,'pid':19000,'pgid':19000,'exit_status':0,'leader_reaped':True,'status':'CLEANUP_VERIFIED','group_absence':'OBSERVED_NO_SUCH_GROUP','errors':[]};hc={'status':'NO_CLEANUP_ERRORS','errors':[],'groups':[hg],'restoration':{'handlers_restored':True,'timer_restored':True}}
     C.write(review/'host-owner/ownership.json',{'started_at':hoststamp,'ended_at':hostend,'argv':host_argv,'pid':19000,'pgid':19000,'owner_pid':18000,'exit_status':0,'cleanup':hg});C.write(review/'host-owner/host-finalization.json',{'status':'DRIVER_EXITED','exit_code':0,'packet_sha256':pktsha,'argv':host_argv,'config_sha256':C.sha(config),'absolute_deadline_epoch':birth+3600,'finished_epoch':time.time(),'cleanup':hc})
     C.write(review/'host-owner/cleanup.json',hc);(review/'host-owner/stdout.raw').write_text('synthetic driver stdout\n');(review/'host-owner/stderr.raw').write_text('');(review/'host-owner/lifecycle.jsonl').write_text(json.dumps({'kind':'child_started','pid':19000,'pgid':19000,'recorded_at':hoststamp})+'\n'+json.dumps({'kind':'child_cleanup_checks','pid':19000,'errors':[],'recorded_at':hostend})+'\n')
     from reader_controls import exercise
     reader_controls=exercise(pkt,out,driver,review,cache=name=='correct_cache')
    elif name=='wrong_usage_closure':assert error is None and driver['status']=='BLOCKED_CLEANUP'and not driver['usage_zero_observed']
    elif name=='science_cli_failure':assert error and sciences==['cpu','device']and 'recover:device'in events and 'diagnostic_recovered_records'in driver
    elif name=='export_cli_zero_no_marker':assert error and sciences==['cpu']and 'recover:cpu'in events
    else:assert error and sciences==[]and 'recover:install'in events and 'diagnostic_recovered_records'in driver,(name,error,events)
    if 'recover:install'in events:assert events.index('recover:install')<events.index('cli:stop')
    report={'name':name,'status':'PASS','driver_status':driver['status'],'error':error,'science_fixture_phases':sciences,'reader_controls':reader_controls,'kernel_proofs':len(driver.get('kernel_proofs',[])),'original_clock_preserved':True,'exact_runtime_stop_observed':True,'usage_zero_observed':driver['usage_zero_observed'],'disposable_bytes':sum(q.stat().st_size for q in root.rglob('*')if q.is_file())}
   report['fixtures_deleted']=not Path(sd).exists();reports.append(report)
  for name in ['correct','correct_cache','cli_zero_kernel_error','duplicate_marker','post_marker_error','wrong_nonce','wrong_phase','wrong_source','wrong_artifact','missing_completion_download','science_cli_failure','export_cli_zero_no_marker','wrong_usage_closure']:case(name)
  shared_bytes=sum(q.stat().st_size for q in shared.rglob('*')if q.is_file())
 C.write(output,{'status':'PASS','scenarios':reports,'count':len(reports),'independent_reader_controls':sum(len(r['reader_controls'])for r in reports),'shared_bytes_deleted':shared_bytes,'fixtures_deleted':not Path(td).exists(),'provider_calls':0,'synthetic_transport_and_audits_only':True,'actual_m4_acceptance':'NOT_SUPPLIED','m5_status':'NOT_QUALIFIED'});print(json.dumps({'status':'PASS','scenarios':len(reports),'fixtures_deleted':True,'provider_calls':0}))
if __name__=='__main__':main(Path(sys.argv[1]))
