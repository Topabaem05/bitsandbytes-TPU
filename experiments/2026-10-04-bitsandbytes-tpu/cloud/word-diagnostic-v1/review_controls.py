"""Local synthetic controls. They make no provider calls or TPU claims."""
import argparse,copy,json,shutil,subprocess,sys,tempfile,time,types
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent;P=ROOT/'packet';sys.path[:0]=[str(P/'control'),str(P/'source')]
import contract as C,driver as D,host_run as H,inspect_diagnostic as I,run_diagnostic as R
checks=[]
def test(name,fn):fn();checks.append({'name':name,'status':'PASS'})
def rejects(fn,contains):
 try:fn()
 except Exception as e:assert contains in str(e),(contains,str(e));return
 raise AssertionError('accepted incorrect fixture:'+contains)
def main():
 with tempfile.TemporaryDirectory(prefix='m4-word-local-controls-')as tmp:
  tmp=Path(tmp);cpu=tmp/'cpu';proc=subprocess.run([sys.executable,'-B',str(P/'source/run_diagnostic.py'),'--mode','cpu','--output',str(cpu),'--deadline-epoch',str(time.time()+90),'--unqualified-cpu'],capture_output=True,text=True,timeout=100);assert proc.returncode==0,(proc.stdout,proc.stderr);report=C.read(cpu/'report.json');C.write(ROOT/'local-cpu-control-report.json',report);test('actual_host_torch_24_unqualified_cpu',lambda: R.check_cpu(report['rows']))
  def wrong(label,keys):
   rows=copy.deepcopy(report['rows']);row=next(r for r in rows if r['label']==label)
   for key in keys:
    v=row['outputs'][key];v['int_values']=[0]*len(v['int_values']);v['hex_words']=['0'*len(w)for w in v['hex_words']];v['storage_sha256']=R.digest(b'\0'*sum(len(w)//2 for w in v['hex_words']))
   return rows
  for label,keys,error in [('bitcast-raw-vector',['words','roundtrip'],'CPU_BITCAST_INDEPENDENT_STORAGE_ORACLE'),('scalar-compare-select',list(next(r for r in report['rows']if r['label']=='scalar-compare-select')['outputs']),'CPU_TYPED_SCALAR_INDEPENDENT_ORACLE'),('shift-left',['legacy','tensor'],'CPU_INDEPENDENT_INTEGER_SHIFT_ORACLE'),('shift-right',['legacy','tensor'],'CPU_INDEPENDENT_INTEGER_SHIFT_ORACLE')]:test('reject_coherently_wrong_'+label,lambda label=label,keys=keys,error=error:rejects(lambda:R.check_cpu(wrong(label,keys)),error))
  seal=C.read(ROOT/'packet-seal.json');test('correct_packet_preflight',lambda:C.packet_preflight(P,ROOT/'packet.zip',seal['packet_sha256']));test('wrong_archive_sha_rejected',lambda:rejects(lambda:C.packet_preflight(P,ROOT/'packet.zip','0'*64),'WHOLE_ARCHIVE_SHA_SIZE'))
  bad=tmp/'bad-packet';shutil.copytree(P,bad);(bad/'source/selected-inputs.json').write_text('{}');test('wrong_loose_source_rejected',lambda:rejects(lambda:C.packet_preflight(bad,ROOT/'packet.zip',seal['packet_sha256']),'LOOSE_PACKET_DIFFERS'));shutil.rmtree(bad)
  records=tmp/'records';shutil.copytree(cpu,records/'cpu');tpu=copy.deepcopy(report);tpu.update(mode='tpu',status='TPU_DIAGNOSTIC_COMPLETE_WITH_EXACT_COMPARISONS',returned=24,errors=0,exact_matches=24,physical_graphs=[]);runtime=dict(python='3.12.14',system='Linux',machine='x86_64',versions=I.CORE,status='PINNED_TPU_DIAGNOSTIC_RUNTIME',requested_backend='TPU',observed_backend='TPU',devices=['TPU:0']);tpu['runtime']=runtime
  qualified_cpu=copy.deepcopy(report);qualified_cpu['runtime']=dict(runtime,status='PINNED_LINUX_CPU_DIAGNOSTIC_ORACLE',requested_backend=None);C.write(records/'cpu/report.json',qualified_cpu)
  for row in tpu['rows']:row.update(observation_status='ERROR',observation_error={'type':'SYNTHETIC_OBSERVATION','message':'synthetic graph failure fixture'},cpu_exact_comparison={k:True for k in row['outputs']},cpu_exact_match=True);C.write(records/'tpu/rows'/(row['label']+'.json'),row)
  C.write(records/'tpu/report.json',tpu);C.write(records/'tpu/inputs.json',tpu['inputs']);(records/'tpu/graphs').mkdir()
  for mode in('cpu','tpu'):C.write(records/'steps'/mode/'result.json',{'status':'PASS','exit_code':0,'environment':{'PJRT_DEVICE':'TPU'if mode=='tpu'else None},'cleanup':{'status':'NO_CLEANUP_ERRORS','errors':[],'groups':[]}})
  test('synthetic_graph_error_keeps_exact_rows',lambda:I.inspect(P,records));rowpath=records/'tpu/rows/shift-left.json';saved=rowpath.read_bytes();rowpath.write_text('{}');test('contradictory_physical_row_rejected',lambda:rejects(lambda:I.inspect(P,records),'DIAGNOSTIC_PHYSICAL_ROW_EQUAL'));rowpath.write_bytes(saved)
  old=tpu['runtime']['machine'];tpu['runtime']['machine']='arm64';C.write(records/'tpu/report.json',tpu);test('wrong_runtime_machine_rejected',lambda:rejects(lambda:I.inspect(P,records),'DIAGNOSTIC_RUNTIME_PROFILE'));tpu['runtime']['machine']=old;C.write(records/'tpu/report.json',tpu)
  birth=time.time();adoption={'allocation_epoch':birth,'session':'fixture','endpoint':'endpoint','hardware':'V5E1','packet_sha256':seal['packet_sha256'],'driver_sha256':C.sha(P/'control/driver.py'),'cli_identity_sha256':'f'*64,'marker_path':'/tmp/fixture','marker_sha256':'e'*64}
  for badmarker in(False,True):
   calls=[];out=tmp/('driver-badmarker'if badmarker else'driver-validmarker');dispatch=tmp/'dispatch.json';C.write(dispatch,{'kind':'ROOT_M4_WORD_DIAGNOSTIC_DISPATCH_V1','status':'ACTUAL_DISPATCH_AUTHORIZED','generation':C.GENERATION,'packet_sha256':seal['packet_sha256'],'driver_sha256':C.sha(P/'control/driver.py'),'cli_identity_sha256':'f'*64,'diagnostic_seconds':600,'allocation_attempts':0,'allocation_seconds':3600,'m4_status':'NOT_QUALIFIED'})
   args=types.SimpleNamespace(packet=P,output=out,packet_zip=ROOT/'packet.zip',packet_sha256=seal['packet_sha256'],cli_python=Path(sys.executable),cli_identity=tmp/'identity.json',dispatch=dispatch,browser_adoption=tmp/'adopt.json',browser_adoption_sha256='a'*64);C.write(args.cli_identity,{})
   def api(label,argv,timeout):
    calls.append(list(argv))
    if argv[0]=='download':Path(argv[-1]).parent.mkdir(parents=True,exist_ok=True);Path(argv[-1]).write_text('{}')
    return {'status':'PASS','cleanup':{'status':'NO_CLEANUP_ERRORS','errors':[],'groups':[]}},'synthetic output'
   def marker(text,req):
    if req['label']==('browser-marker'if badmarker else'readiness'):raise ValueError('SYNTHETIC_STOP_POINT')
   original_sha=C.sha
   with patch.object(C,'packet_preflight',return_value={'files':{}}),patch.object(C,'sha',side_effect=lambda path:'f'*64 if Path(path)==args.cli_identity else original_sha(path)),patch.object(D.T,'verify_cli_identity'),patch.object(D.T,'OfficialCLI',return_value=api),patch.object(D.B,'record',return_value=adoption),patch.object(D.cli_display,'sessions_output',return_value={'sessions':[{'session':'fixture','endpoint':'endpoint','hardware':'V5E1','shape':'Standard','variant':'TPU'}]}),patch.object(D.cli_display,'resource_empty',return_value={'status':'SYNTHETIC_EMPTY'}),patch.object(D.K,'request',side_effect=lambda body,label,*a,**k:{'label':label,'completion_path':'/tmp/completion'}),patch.object(D.K,'render',return_value='synthetic'),patch.object(D.K,'marker',side_effect=marker),patch.object(D.K,'verify',return_value={}):
    rejects(lambda:D.run(args),'SYNTHETIC_STOP_POINT')
   assert calls.count(['stop','-s','fixture'])==1 and not any(c[0]=='--remove-browser-provisional-v1'for c in calls),calls;checks.append({'name':'exact_assignment_'+('bad_marker'if badmarker else'valid_marker')+'_stops_once','status':'PASS'})
  config=tmp/'host.json';C.write(config,{'argv':[sys.executable,'-B',str(P/'control/driver.py'),'--browser-adoption',str(tmp/'adopt.json'),'--browser-adoption-sha256','a'*64],'absolute_deadline_epoch':birth+601,'host_output':str(tmp/'unused'),'packet_sha256':seal['packet_sha256']})
  with patch.object(H.B,'record',return_value=adoption):test('host_601_second_deadline_rejected',lambda:rejects(lambda:H.run(config),'EXACT_DIAGNOSTIC_HOST_DEADLINE'))
 C.write(ROOT/'review-controls.json',{'status':'PASS','checks':checks,'count':len(checks),'scope':'HOST_TORCH_UNQUALIFIED_AND_SYNTHETIC_CONTROL_ONLY','provider_calls':0,'actual_tpu':False,'fixtures':'DELETED_AFTER_TEST','packet_sha256':seal['packet_sha256'],'m4_status':'NOT_QUALIFIED'});print(json.dumps({'status':'PASS','checks':len(checks),'fixtures':'DELETED'}))
if __name__=='__main__':main()
