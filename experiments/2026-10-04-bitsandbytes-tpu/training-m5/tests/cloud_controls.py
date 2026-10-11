"""Isolated synthetic contract/real root-prepare branches. No provider or qualification."""
import copy,hashlib,json,os,shutil,sys,tempfile,time,types,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'control'));sys.path.insert(0,str(ROOT/'source'))
import contract as C,admission as A,root_records as R,build_packet as B,cli_transport as T,browser_adoption as BA,driver as D,host_audit as H,training as M,record_validation as V
rows=[]
def check(name,fn,why=None):
 try:fn()
 except Exception as e:
  if why is None or why not in str(e):raise
  rows.append({'name':name,'status':'PASS','rejected':str(e)});return
 C.require(why is None,'INCORRECT_ACCEPTANCE:'+name);rows.append({'name':name,'status':'PASS'})
def m4():return {'kind':'ROOT_M4_EXACT_NESTED_ACCEPTANCE','status':'ACCEPTED_M4_EXACT79_AND_STATE8','m4_status':'QUALIFIED','record_validation':'PASS','numerical_status':'PASS','source_variant':'nested-exact-v1','source_admission_sha256':A.BASE_ADMISSION,'science_map_sha256':A.BASE_SCIENCE,'integration_map_sha256':A.BASE_MAP,'runtime_lock_sha256':A.RUNTIME_LOCK,'nested79':{'case_count':79,'record_validation':'PASS','numerical_status':'PASS'},'saved_state8':{'case_count':8,'record_validation':'PASS','numerical_status':'PASS','fresh_restore_process':True,'checkpoint_tensor_state':'EXACT'},'resource_closure':{'server_empty':True,'usage_zero':True,'owned_groups_absent':True,'owned_browser_tab_absent':True},'provenance':{k:hashlib.sha256(k.encode()).hexdigest()for k in ['accepted79_sha256','actual79_receipt_sha256','actual_state8_receipt_sha256','fresh_cpu79_oracle_sha256','fresh_state8_oracle_sha256','independent79_readback_sha256','independent_state8_readback_sha256','complete_archive_sha256','closure_readback_sha256']},'accepted_epoch':time.time()-10}
cli_python=Path('/Users/topamini/code/Post2TPU/.runtime/cloud-cli/bin/python');identity=Path('/Users/topamini/code/bitsandbytes-TPU/.work/task2-cloud/cli-identity.json')
check('actual_readonly_cli_identity_symlink',lambda:T.verify_cli_identity(cli_python,C.read(identity)))
check('fixed_allocation_budget',lambda:C.require(C.budget(100,1500,660,now=500)==1500,'BUDGET'))
check('expired_original_budget_rejected',lambda:C.budget(100,1,660,now=3040),'ORIGINAL_ALLOCATION_DEADLINE_RESERVE')
check('future_allocation_rejected',lambda:C.budget(100,1,660,now=99),'FINITE_ORIGINAL_ALLOCATION_BUDGET')
with tempfile.TemporaryDirectory(prefix='bnb-m5-cloud-controls-')as td:
 t=Path(td);accepted=t/'synthetic-only-m4.json';C.write(accepted,m4());digest=C.sha(accepted)
 check('synthetic_m4_consumer_shape_only',lambda:A.m4_dependency(accepted,digest))
 for name,mutate,why in [('missing_state8',lambda r:r.pop('saved_state8'),'M4_COMPLETE79_AND_SAVED8'),('wrong_nested_count',lambda r:r['nested79'].update(case_count=78),'M4_COMPLETE79_AND_SAVED8'),('wrong_m4_source',lambda r:r.update(science_map_sha256='a'*64),'M4_BASE_SOURCE_RUNTIME_BINDINGS'),('unclosed_m4',lambda r:r['resource_closure'].update(usage_zero=False),'M4_ACCEPTED_RESOURCE_CLOSURE'),('missing_actual_provenance',lambda r:r['provenance'].pop('independent_state8_readback_sha256'),'M4_ACTUAL_PROVENANCE'),('future_m4_acceptance',lambda r:r.update(accepted_epoch=time.time()+600),'M4_ACCEPTANCE_TIME'),('failed_m4',lambda r:r.update(numerical_status='FAIL'),'GENUINE_FUTURE_ACCEPTED_M4_REQUIRED')]:
  r=m4();mutate(r);q=t/(name+'.json');C.write(q,r);check(name,lambda q=q:A.m4_dependency(q,C.sha(q)),why)
 check('wrong_external_m4_hash',lambda:A.m4_dependency(accepted,'a'*64),'EXTERNAL_ACCEPTED_M4_SHA')
 review=t/'synthetic-only-review.json';R.source_review(ROOT,review,'ACCEPT_SOURCE_AFTER_ROOT_REVIEW','6553e22ca52bee23f2378eac14998c19e857219f')
 built=B.build(ROOT,review,C.sha(review),accepted,digest,t/'packet');packet=Path(built['packet']);archive=Path(built['archive']);packet_sha=built['packet_sha256']
 check('actual_packet_build_source_and_dependency',lambda:C.packet_preflight(packet,archive,packet_sha))
 check('missing_real_dependency_blocks_build',lambda:B.build(ROOT,review,C.sha(review),t/'missing-acceptance.json','a'*64,t/'blocked-packet'),'No such file')
 check('wrong_source_review_decision',lambda:R.source_review(ROOT,t/'bad-review.json','UNREVIEWED','6553e22ca52bee23f2378eac14998c19e857219f'),'EXPLICIT_ROOT_SOURCE_REVIEW')
 C.write(t/'data.json',{'value':'retained'});data=t/'records';data.mkdir();shutil.copyfile(t/'data.json',data/'data.json');envelope=C.export(data,t/'data.zip')
 check('actual_archive_export_and_recovery',lambda:C.recover(t/'data.zip',envelope['sha256'],t/'readback'))
 check('wrong_archive_whole_sha',lambda:C.recover(t/'data.zip','a'*64,t/'bad-readback'),'WHOLE_ARCHIVE_SHA_SIZE')
 with zipfile.ZipFile(t/'traversal.zip','w')as z:z.writestr('../escape',b'x')
 check('traversal_archive_rejected',lambda:C.recover(t/'traversal.zip',C.sha(t/'traversal.zip'),t/'traversal'),'CONFINED_NAME')
 with zipfile.ZipFile(t/'duplicate.zip','w')as z:z.writestr('same',b'x');z.writestr('same',b'y')
 check('duplicate_archive_rejected',lambda:C.recover(t/'duplicate.zip',C.sha(t/'duplicate.zip'),t/'duplicate'),'ARCHIVE_MEMBER_COUNT')
 q=t/'dup.json';q.write_text('{"a":1,"a":2}')
 check('duplicate_json_rejected',lambda:C.read(q),'DUPLICATE_JSON_KEY')
 cache=C.runtime_cache;calls=[]
 class FakeCLI:
  def __init__(self,python,out):self.out=out
  def __call__(self,label,argv,timeout):
   calls.append(argv);body='[?] synthetic_endpoint | Hardware: V5E1 | Shape: Standard | Variant: TPU'if self.out.name=='dispatch-read'else'No active sessions found on server.'if argv==['sessions']else'Active assignments: 0\nUsage rate: 0.00/hr'
   return {'status':'PASS','exit_code':0,'cleanup':{'status':'NO_CLEANUP_ERRORS','errors':[],'groups':[]}},body
 prior=T.OfficialCLI;T.OfficialCLI=FakeCLI;C.runtime_cache=lambda *a:[]
 def call(command,review_dir,extra=()):
  argv=[str(ROOT/'control/root_records.py'),command,'--packet',str(packet),'--packet-zip',str(archive),'--packet-sha256',packet_sha,'--review',str(review_dir),'--output',str(t/'actual'),'--cli-python',str(cli_python),'--cli-identity',str(identity),'--runtime-wheels',str(t/'mock-cache'),*extra];old=sys.argv;sys.argv=argv
  try:R.main()
  finally:sys.argv=old
 try:
  d=t/'prepare';check('real_prepare_preconnect_branch_synthetic_transport',lambda:call('preconnect',d));check('real_prepare_exclusive_request_branch',lambda:call('request',d));request_before=C.sha(d/'allocation-request.json')
  check('real_prepare_repeat_request_no_clock_reset',lambda:call('request',d),'FRESH_ROOT_RECORD_NO_CLOCK_RESET');C.require(C.sha(d/'allocation-request.json')==request_before,'IMMUTABLE_REQUEST')
  marker=C.read(d/'browser-marker.json');stdout='M5_BROWSER_MARKER '+marker['sha256']+' '+str(marker['bytes'])
  check('real_prepare_wrong_observed_marker',lambda:call('observe',d,['--marker-output','WRONG']),'OBSERVED_EXACT_ORIGINAL_MARKER')
  check('real_prepare_observe_branch_synthetic_ui',lambda:call('observe',d,['--marker-output',stdout]))
  def bad_case(name,mutator,why):
   c=t/name;shutil.copytree(d,c);mutator(c);check(name,lambda:call('dispatch',c,['--reviewer-decision','AUTHORIZE_ONE_ALLOCATION_AFTER_ROOT_REVIEW']),why)
  def mutate(c,name,changes):q=c/name;r=C.read(q);r.update(changes);C.write(q,r)
  bad_case('stale_ui_observation',lambda c:mutate(c,'browser-observation.json',{'epoch':time.time()-181}),'FRESH_ROOT_UI_MARKER')
  bad_case('future_ui_observation',lambda c:mutate(c,'browser-observation.json',{'epoch':time.time()+1}),'FRESH_ROOT_UI_MARKER')
  bad_case('wrong_ui_hardware',lambda c:mutate(c,'browser-observation.json',{'hardware':'V6E1'}),'FRESH_ROOT_UI_MARKER')
  bad_case('wrong_ui_marker_hash',lambda c:mutate(c,'browser-observation.json',{'marker_sha256':'a'*64}),'FRESH_ROOT_UI_MARKER')
  bad_case('wrong_ui_request_binding',lambda c:mutate(c,'browser-observation.json',{'allocation_request_sha256':'a'*64}),'FRESH_ROOT_UI_MARKER')
  bad_case('reset_original_deadline',lambda c:mutate(c,'allocation-request.json',{'deadline_epoch':time.time()+7200}),'ORIGINAL_REQUEST_BINDING_DEADLINE')
  check('real_prepare_dispatch_branch_synthetic_transport',lambda:call('dispatch',d,['--reviewer-decision','AUTHORIZE_ONE_ALLOCATION_AFTER_ROOT_REVIEW']))
  adopted=d/'dispatch/browser-adoption.json';dispatch=C.read(d/'dispatch/root-dispatch.json')
  check('browser_adoption_profile_and_original_epoch',lambda:BA.record(adopted,C.sha(adopted)))
  check('actual_driver_gate_bindings',lambda:D.gate(packet,packet_sha,identity,dispatch,t/'actual'))
  bad=copy.deepcopy(dispatch);bad['driver_sha256']='a'*64;check('wrong_driver_sha_rejected',lambda:D.gate(packet,packet_sha,identity,bad,t/'actual'),'ROOT_DISPATCH_GATE')
  check('wrong_actual_output_rejected',lambda:D.gate(packet,packet_sha,identity,dispatch,t/'elsewhere'),'EXACT_DISPATCH_OUTPUT')
 finally:T.OfficialCLI=prior;C.runtime_cache=cache
 quota=t/'quota-records';quota.mkdir()
 with(quota/'sparse.bin').open('wb')as f:f.truncate(C.CPU_BYTES+1)
 check('cpu_outer_archive64mib_enforced',lambda:C.export(quota,t/'quota.zip'),'CPU_ARCHIVE_64MIB_BOUND')
 shutil.rmtree(quota)
 # Actual frame-parameter role helper at the extended count.
 literals=V.literals(ROOT/'integration/plugin/src/bitsandbytes_tpu',1024);x={'dtype':'float32','shape':[1024],'values':[0.0]*1024,'bytes_sha256':hashlib.sha256(b'\x00'*4096).hexdigest()};raw={'input_parameter_ids':[0],'parameters':{'0':x}}
 check('typed1024_parameter_roles',lambda:C.require(V.frame_parameter_roles(raw,[x],literals)[1]=={'argument_0':'PASS'},'PARAMETER'))
 wrong=copy.deepcopy(x);wrong['values'][256]=1.0;check('typed_changed_parameter_value_detected',lambda:C.require(V.frame_parameter_roles(raw,[wrong],literals)[1]=={'argument_0':'FAIL'},'WRONG_PARAMETER'))
 check('unsupported_composite_count514',lambda:V.literals(ROOT/'integration/plugin/src/bitsandbytes_tpu',514),'FIXED_COMPOSITE_COUNT')
 C.require(C.budget(100,1500,660,now=1000)==1500,'FINITE_DEVICE_BUDGET')
 # Complete built-wheel Python inventory and source byte checks.
 built_wheel=t/'plugin.whl';expected=A.source()['plugin_python_files']
 with zipfile.ZipFile(built_wheel,'w')as z:
  for n in expected:z.write(ROOT/'integration/plugin/src/bitsandbytes_tpu'/n,'bitsandbytes_tpu/'+n)
 check('complete_built_plugin_wheel',lambda:H.wheel(built_wheel,'bitsandbytes_tpu',expected))
 with zipfile.ZipFile(t/'wrong-plugin.whl','w')as z:
  for n in expected:
   if n!='exact.py':z.write(ROOT/'integration/plugin/src/bitsandbytes_tpu'/n,'bitsandbytes_tpu/'+n)
 check('missing_built_plugin_module',lambda:H.wheel(t/'wrong-plugin.whl','bitsandbytes_tpu',expected),'BUILT_COMPLETE_WHEEL_SOURCE')
 qualified={'python':'3.12.14','torch':'2.9.0+cpu','platform':'Linux','machine':'x86_64','torch_git':'0fabc3ba44823f257e70ce397d989c8de5e362c1','capability':'AVX2','num_threads':1,'xla_or_jax_imported':False,'build':'synthetic contract fixture only'}
 check('qualified_cpu_contract_shape_only',lambda:V.cpu_evidence(qualified))
 for name,changes in [('wrong_cpu_capability',{'capability':'AVX512'}),('wrong_cpu_threads',{'num_threads':2}),('cpu_imported_xla',{'xla_or_jax_imported':True}),('wrong_cpu_torch',{'torch':'2.14.1'}),('wrong_cpu_platform',{'platform':'Darwin'})]:check(name,lambda changes=changes:V.cpu_evidence({**qualified,**changes}),'QUALIFIED_CPU_SOURCE_CAPABILITY')
 P=M.load(ROOT/'legacy/probe_precision.py','control_P');ctx=types.SimpleNamespace(P=P,B=types.SimpleNamespace(require=C.require))
 hlo='HloModule Training\nENTRY main {\n p0 = f32[8,256] parameter(0)\n p1 = f32[4,256] parameter(1)\n ROOT out = f32[8,4] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={1}, operand_precision={highest,highest}\n}\n'
 check('training_highest_reachable_hlo',lambda:M.highest_precision(ctx,hlo))
 check('training_default_precision_rejected',lambda:M.highest_precision(ctx,hlo.replace('highest,highest','default,default')),'TRAINING_FP32_HIGHEST')
 check('training_wrong_contraction_rejected',lambda:M.highest_precision(ctx,hlo.replace('f32[8,4] dot','f32[4,8] dot')),'HLO_DOT_CONTRACTION')
 sourcecopy=t/'source-copy';shutil.copytree(packet,sourcecopy);(sourcecopy/'source/unadmitted.py').write_text('pass\n')
 check('unadmitted_source_file_rejected',lambda:C.source(sourcecopy),'EXACT_SCIENCE_CLOSURE');shutil.rmtree(sourcecopy)
 # Recovered outer launch records, with synthetic identities only.
 launchrecords=t/'launch-records';(launchrecords/'cpu').mkdir(parents=True);birth=1000.0;deadline=birth+3600-660;parent={'pid':100,'pgid':100,'parent_pid':99,'process_token':'b'*32,'deadline_epoch':deadline,'allocation_epoch':birth,'started_epoch':1100.0,'finished_epoch':1200.0};C.write(launchrecords/'cpu/oracle-seal.json',parent)
 remote=Path('/content/synthetic-m5');argv=[str(remote/'venv/bin/python'),'-B',str(remote/'payload/source/training.py'),*H.D.science_arguments(remote/'payload',remote/'records','cpu',parent['process_token'],birth,deadline,m4_sha=digest,oracle_sha=C.sha(launchrecords/'cpu/oracle-seal.json'))]
 close={'pid':100,'pgid':100,'status':'CLEANUP_VERIFIED','leader_reaped':True,'group_absence':'OBSERVED_NO_SUCH_GROUP','errors':[],'exit_status':0};own={'pid':100,'pgid':100,'owner_pid':99,'argv':argv,'exit_status':0,'cleanup':close};result={'status':'PASS','exit_code':0,'argv':argv,'environment':{'CPU':H.D.CPU_ENV,'PJRT_DEVICE':None},'cleanup':{'groups':[close],'errors':[],'restoration':{'handlers_restored':True,'timer_restored':True}}};C.write(launchrecords/'steps/cpu/ownership.json',own);C.write(launchrecords/'steps/cpu/result.json',result)
 check('actual_outer_launch_helper_correct_synthetic',lambda:H.launch(launchrecords,'cpu',parent,packet,str(remote),birth,deadline))
 for name,mutation,why in [('wrong_outer_launch_pid',lambda x:x.update(pid=101),'ACTUAL_OWNED_TRAINING_LAUNCH'),('wrong_outer_launch_argv',lambda x:x.update(argv=argv+['--unreviewed']),'ACTUAL_OWNED_TRAINING_LAUNCH'),('wrong_outer_group_closure',lambda x:x['cleanup'].update(group_absence='UNVERIFIED'),'ACTUAL_TRAINING_OUTER_CLOSURE')]:
  changed=copy.deepcopy(own);mutation(changed);C.write(launchrecords/'steps/cpu/ownership.json',changed);check(name,lambda:H.launch(launchrecords,'cpu',parent,packet,str(remote),birth,deadline),why)
 C.write(launchrecords/'steps/cpu/ownership.json',own);changed=copy.deepcopy(result);changed['environment']['CPU']={**H.D.CPU_ENV,'ATEN_CPU_CAPABILITY':'avx512'};C.write(launchrecords/'steps/cpu/result.json',changed)
 check('wrong_outer_cpu_environment',lambda:H.launch(launchrecords,'cpu',parent,packet,str(remote),birth,deadline),'ACTUAL_TRAINING_ENVIRONMENT')
 result={'status':'PASS','count':len(rows),'controls':rows,'synthetic_contract_fixtures_only':True,'actual_m4_acceptance':'NOT_SUPPLIED','provider_calls':0,'fixtures_deleted':True,'m5_status':'NOT_QUALIFIED','synthetic_transport_calls':len(calls),'science_manifest_sha256':C.science(ROOT),'source_bundle_sha256':M.bundle()}
C.write(Path(sys.argv[1]),result);print(json.dumps({'status':'PASS','controls':len(rows),'provider_calls':0}))
