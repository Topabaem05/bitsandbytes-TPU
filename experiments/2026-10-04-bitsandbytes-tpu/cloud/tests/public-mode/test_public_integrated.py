"""Independent arithmetic/graph replay and cloud plumbing with explicit fake API.

All qualified-shaped records are synthetic. No provider, installed runtime,
actual public device, root acceptance, or paid job is created by these controls.
"""
import copy,os,sys,time
from pathlib import Path
import pytest
from control_helpers import HERE,U,R
from integrated_fixture import Fixture
sys.path.insert(0,str(HERE/'cloud'));import owner as O
PACKET=Path(os.environ['PUBLIC15_TEST_PACKET']).resolve()

@pytest.fixture(scope='module')
def evidence(tmp_path_factory):
 return Fixture(PACKET,tmp_path_factory.mktemp('one-generated-fixture'),Path(os.environ['PUBLIC15_RUNTIME_WHEELS']),Path(os.environ['PUBLIC15_JAX071_PYTHON']))

@pytest.mark.parametrize('fault',[None,'unqualified','output','source','parent','deadline','wheel','stale_time','source_argv'])
def test_independent_cpu_gate_correct_and_altered(evidence,fault):
 f=evidence;r=f.receipt();r.update(public_cpu_inventory_sha256='b'*64,public_cpu_evidence_sha256='c'*64);path=f.out/'cpu-oracle/oracle-seal.json';old={p:p.read_bytes()for p in (path,f.out/'cpu-oracle/rows/direct-float32.json',f.out/'installed-source.json',Path(f.args.cpu_ownership))}
 try:
  s=U.read(path)
  if fault=='unqualified':s.update(runtime=f.original_seal['runtime'],qualified_runtime=False,scope=f.original_seal['scope'])
  if fault=='output':
   q=f.out/'cpu-oracle/rows/direct-float32.json';row=U.read(q);row['outputs']['y']['values'][0]+=1.;U.write(q,row);s['artifacts']=f.C.inventory(f.out/'cpu-oracle',('oracle-seal.json',))
  if fault=='source':
   q=f.out/'installed-source.json';v=U.read(q);v['installed_python_files']['bitsandbytes_tpu_pallas']['adapter.py']='0'*64;U.write(q,v)
  if fault=='parent':
   q=Path(f.args.cpu_ownership);v=U.read(q);v['owner_pid']+=1;U.write(q,v)
  if fault=='stale_time':s['started_epoch']=r['allocation_epoch']-100
  if fault=='source_argv':
   q=Path(f.args.cpu_ownership);v=U.read(q);v['argv'][v['argv'].index('--package-root')+1]='/content/unadmitted/package';U.write(q,v)
  if fault=='deadline':r['science_deadline_epoch']=r['allocation_epoch']-1
  if fault=='wheel':r['built_wheels'][0]['sha256']='0'*64
  U.write(path,s);r['oracle_sha256']=U.sha(path)
  if fault:
   with pytest.raises(ValueError):U.cpu_gate(PACKET,f.out,r,f.args.runtime_wheels)
  else:
   gate,report=U.cpu_gate(PACKET,f.out,r,f.args.runtime_wheels);assert gate==U.gate_bindings(f.m,r)and report['case_count']==15 and report['record_validation']=='PASS'
 finally:
  for p,b in old.items():p.write_bytes(b)


def test_complete_synthetic_graph_conversion_recovery(evidence):
 f=evidence;args=copy.copy(f.args);args.actual=str(f.actual);args.receipt_sha256=U.sha(f.actual/'receipt.json');args.ownership=str(f.out/'steps/12-public/ownership.json');args.ownership_sha256=U.sha(args.ownership);args.native_acceptance=str(PACKET/U.BINDINGS['public_native_acceptance_sha256']);args.native_acceptance_sha256=U.NATIVE_ACCEPTANCE_SHA;args.audit_deadline_epoch=time.time()+90
 V=f.C.load(PACKET/'public/verify_public.py','public_integration_V');report=V.verify(args)
 assert report['record_validation']=='PASS'and U.validate_report(report,PACKET)=='PASS'and report['case_count']==15
 assert U.read(f.base/'fixture-provenance.json')['actual_device']=='NOT_RUN'

@pytest.mark.parametrize('fault',[None,'oracle','boundary'])
def test_full_fake_owner_gate_device_recovery_and_closure(evidence,tmp_path,fault):
 from fake_owner_api import CompletePublic
 f=evidence;cli=Path(os.environ['PUBLIC15_CLI_PYTHON']);site=cli.absolute().parent.parent/'lib/python3.12/site-packages';identity=tmp_path/'cli-identity.json';U.write(identity,{'distribution':'google-colab-cli','version':'0.7.4','python_sha256':U.sha(cli),'venv_config_sha256':U.sha(cli.absolute().parent.parent/'pyvenv.cfg'),'metadata_sha256':U.sha(site/'google_colab_cli-0.7.4.dist-info/METADATA'),'record_sha256':U.sha(site/'google_colab_cli-0.7.4.dist-info/RECORD'),'files':{p.relative_to(site).as_posix():U.sha(p)for p in(site/'colab_cli').rglob('*.py')}})
 m=f.m;jax=Path(os.environ['PUBLIC15_JAX071_PYTHON']);cache=Path(os.environ['PUBLIC15_RUNTIME_WHEELS']);host=tmp_path/'host';gate={'scope':'SYNTHETIC_FAKE_API_ONLY_NOT_ACTUAL_AUTHORIZATION','status':'ACTUAL_DISPATCH_AUTHORIZED','packet_sha256':U.sha(PACKET/'payload.zip'),'driver_sha256':U.sha(HERE/'cloud/owner.py'),'output':str(host.resolve()),'plugin_source_manifest_sha256':m['plugin_source_manifest_sha256'],'runtime_lock_sha256':m['runtime_lock_sha256'],'budget':m['budget'],'one_allocation_only':True,'provider_or_solver':'FORBIDDEN','cli_identity_sha256':U.sha(identity),'experiment':U.MODE,'precision':'highest',**{k:m[k]for k in R.TRANSFER_BINDINGS},**{k:m[k]for k in U.FIELDS},'mosaic_audit_python':str(jax),'mosaic_audit_python_sha256':U.sha(jax),'public_runtime_wheels':str(cache)};U.write(tmp_path/'gate.json',gate)
 api=CompletePublic(PACKET,tmp_path,f,fault=fault);result=O.drive(PACKET,host,U.sha(PACKET/'payload.zip'),tmp_path/'gate.json',U.sha(tmp_path/'gate.json'),cli,identity,api=api,simulated=True,mosaic_audit_python=jax,public_runtime_wheels=cache)
 assert result['mode']=='SIMULATED_NO_CLOUD'and result['whole_archive_verified']and result['server_empty_observed']and result['usage_zero_observed']and not result['cleanup_errors']
 if fault=='oracle':assert result['status']=='BLOCKED'and result['original_error']['message']=='PUBLIC_ROOT_CPU_GATE_REJECTED'and'13-tpu'not in api.calls
 elif fault=='boundary':assert result['status']=='BLOCKED_PUBLIC_RECORDS'and result['public_record_validation']=='INVALID_RECORDS'and result['local_readback_error']['message']=='PUBLIC_RECOVERED_BOUNDARY_SEALS'
 else:assert result['status']=='PASS_PUBLIC15_RECORDS'and result['numerical_status']=='PASS'and result['local_verifier_exit_code']==0
 assert U.read(f.base/'fixture-provenance.json')['actual_device']=='NOT_RUN'

@pytest.mark.parametrize('fault',['not_run_first','pass_after_error'])
def test_report_first_error_rule(fault):
 ids=[v['id']for v in U.read(PACKET/'public/public-protocol.json')['cases']];r={'record_validation':'PASS','scope':'ACTUAL_PUBLIC_API','case_count':15,'native_case_count':11,'reference_case_count':4,'rows':[{'case':v,'status':'PASS'}for v in ids],'numerical_status':'PASS','m4_status':'NOT_QUALIFIED','m6_memory_status':'NOT_QUALIFIED','performance':'NOT_MEASURED'}
 if fault=='not_run_first':r['rows'][0]['status']='NOT_RUN'
 else:r['rows'][0]['status']='ERROR';r['numerical_status']='ERROR'
 with pytest.raises(ValueError,match='PUBLIC_REPORT_'):U.validate_report(r,PACKET)
