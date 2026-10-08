"""Offline source, topology and lifecycle controls. No provider or device work."""
import copy,importlib.util,json,os,sys,time,zipfile
from pathlib import Path
import pytest
from control_helpers import HERE,U,R,receipt,synthetic_direct_run
sys.path.insert(0,str(HERE/'cloud'));import owner as O;import public_transport as T
PACKET=Path(os.environ['PUBLIC15_TEST_PACKET']).resolve()

def test_genuine_packet_and_owner_preflight():
 m=U.read(PACKET/'manifest.json');U.payload(PACKET,m);assert O.preflight(PACKET,U.sha(PACKET/'payload.zip'))==m
 assert m['experiment']==U.MODE and len(m['files'])>100

@pytest.mark.parametrize('field',U.FIELDS)
def test_public_fields_rejected_in_other_modes(field):
 m={'experiment':'api42','files':{},field:'x'}
 with pytest.raises(ValueError,match='PUBLIC_NOT_REQUESTED'):U.verify_manifest(m)

@pytest.mark.parametrize('prefix',U.PREFIXES)
def test_public_prefix_rejected_in_other_modes(prefix):
 with pytest.raises(ValueError,match='PUBLIC_NOT_REQUESTED'):U.verify_manifest({'experiment':'api42','files':{prefix+'other.py':{'sha256':'0'*64,'bytes':0}}})

@pytest.mark.parametrize('field',U.FIELDS)
def test_wrong_explicit_public_binding(field):
 m=U.read(PACKET/'manifest.json');m[field]='0'*64
 with pytest.raises(ValueError):U.verify_manifest(m)

def test_complete_map_omission_even_with_same_pin():
 m=U.read(PACKET/'manifest.json');m['files'].pop('public-plugin/src/bitsandbytes_tpu_pallas/adapter.py')
 with pytest.raises(ValueError,match='PUBLIC_COMPLETE_PLUGIN_MAP'):U.payload(PACKET,m)

def test_no_public_experiment_with_correct_sources():
 m=U.read(PACKET/'manifest.json');m['experiment']='transfer-api42'
 with pytest.raises(ValueError,match='PUBLIC_NOT_REQUESTED'):R.verify_experiment(m)

def test_runtime_cache_exact_existing_body_and_wrong_path():
 C,a=U.payload(PACKET,U.read(PACKET/'manifest.json'));cache=Path(os.environ['PUBLIC15_RUNTIME_WHEELS']);rows=C.runtime_wheels(a,PACKET/'public-baseline',cache,qualified=True)
 assert rows['wheel_source_check']=='PASS' and len(rows['records'])==35
 with pytest.raises(ValueError,match='EXACT_RUNTIME_WHEEL_BODY'):C.runtime_wheels(a,PACKET/'public-baseline',cache/'missing',qualified=True)

@pytest.mark.parametrize('fault',[None,'pid','parent','token','source','deadline'])
def test_real_direct_group_topology_and_binding(tmp_path,monkeypatch,fault):
 base=tmp_path/'base';(base/'payload').mkdir(parents=True);(base/'records/steps/11-cpu-oracle').mkdir(parents=True);U.write(base/'records/steps/11-cpu-oracle/ownership.json',{'fixture':'SYNTHETIC_ONLY'})
 m=U.read(PACKET/'manifest.json');U.write(base/'payload/manifest.json',m);r=receipt(m);U.write(base/'launch.json',U.gate_bindings(m,r));monkeypatch.setattr(T.U,'payload',lambda *a:(None,None));monkeypatch.setattr(T.U,'source_args',lambda *a:[])
 runner=synthetic_direct_run(base,fault);deadline=min(time.time()+45,r['science_deadline_epoch'])
 if fault:
  with pytest.raises(ValueError,match='PUBLIC_DIRECT_OWNER_IDENTITY|PUBLIC_RECEIPT_SOURCE_ORACLE_BINDING'):T.execute_public(base,r,m,runner,deadline)
 else:
  T.execute_public(base,r,m,runner,deadline);assert r['status']=='PUBLIC_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED'
 own=U.read(base/'records/steps/12-public/ownership.json');assert own['cleanup']['status']=='CLEANUP_VERIFIED'
 with pytest.raises(ProcessLookupError):os.killpg(own['pid'],0)

@pytest.mark.parametrize('fault',['wrong_gate','expired','wrong_phase'])
def test_prelaunch_incorrect_rejected(tmp_path,monkeypatch,fault):
 base=tmp_path;(base/'payload').mkdir();m=U.read(PACKET/'manifest.json');U.write(base/'payload/manifest.json',m);r=receipt(m);g=U.gate_bindings(m,r);deadline=time.time()+45
 if fault=='wrong_gate':g['oracle_sha256']='0'*64
 if fault=='wrong_phase':r['tpu_status']='RUNNING'
 if fault=='expired':deadline=time.time()-1
 U.write(base/'launch.json',g);monkeypatch.setattr(T.U,'payload',lambda *a:(None,None));called=[]
 with pytest.raises(ValueError):T.execute_public(base,r,m,lambda *a,**k:called.append(True),deadline)
 assert called==[]

@pytest.mark.parametrize('fault',['terminal','gate','seal'])
def test_early_failure_readback_is_not_verified(tmp_path,fault):
 m=U.read(PACKET/'manifest.json');r=receipt(m);owner={'original_error':{'type':'ModuleNotFoundError','message':'primary fixture failure'}};original=copy.deepcopy(owner['original_error']);gate=U.gate_bindings(m,r)
 if fault=='gate':gate=None
 if fault=='seal':owner.pop('recovered_oracle_sha256',None)
 missing=U.readback_missing(owner,r,tmp_path,gate);assert missing and owner['original_error']==original

@pytest.mark.parametrize('numerical',['PASS','FAIL','ERROR'])
def test_numerical_status_remains_distinct(numerical):
 ids=[c['id']for c in U.read(PACKET/'public/public-protocol.json')['cases']];r={'record_validation':'PASS','scope':'ACTUAL_PUBLIC_API','case_count':15,'native_case_count':11,'reference_case_count':4,'rows':[{'case':n,'status':'PASS'}for n in ids],'numerical_status':numerical,'m4_status':'NOT_QUALIFIED','m6_memory_status':'NOT_QUALIFIED','performance':'NOT_MEASURED'}
 if numerical!='PASS':r['rows'][0]['status']=numerical
 if numerical=='ERROR':
  for item in r['rows'][1:]:item['status']='NOT_RUN'
 assert U.validate_report(r,PACKET)==numerical
 if numerical!='PASS':r['numerical_status']='PASS'
 else:r['rows'].pop()
 with pytest.raises(ValueError):U.validate_report(r,PACKET)


def test_ordinary_native_phase_accepts_gate_without_compiler_fields(tmp_path,monkeypatch):
    """Exercise the plain native predicate; scientific transport is an isolated stub."""
    base=tmp_path/'ordinary-native';out=base/'records';payload=base/'payload';payload.mkdir(parents=True);out.mkdir()
    original=U.read(PACKET/'manifest.json');m={k:v for k,v in original.items()if k not in U.FIELDS};m.update(experiment=R.NC.MODE,files={},native_generation='mosaic-serde7-gather-bf16-fp32-v1',native_manifest_sha256=R.NC.MANIFEST_SHA,native_source_variant=R.NC.VARIANT,native_scope='BOUNDED_NATIVE_BOUNDARY_ONLY')
    U.write(payload/'manifest.json',m);(base/'payload.zip').write_bytes(b'synthetic branch fixture')
    epoch=time.time()-2;r={'status':'CPU_ORACLE_READY_TPU_NOT_RUN','runtime_status':'PASS_TPU_RUNTIME_PROBE_ONLY','cpu_status':'PASS','tpu_status':'NOT_RUN','packet_sha256':U.sha(base/'payload.zip'),'manifest_sha256':U.sha(payload/'manifest.json'),'runtime_lock_sha256':m['runtime_lock_sha256'],'source_admission_sha256':m['source_admission_sha256'],'allocation_epoch':epoch,'steps':[],'error':None,'experiment':m['experiment'],'precision':'highest','science_deadline_epoch':time.time()+90,'oracle_sha256':'a'*64,**{k:m[k]for k in R.TRANSFER_BINDINGS}}
    U.write(out/'receipt.json',r);gate={'kind':'ROOT_NATIVE_CPU_ORACLE_GATE','status':'QUALIFIED_LINUX_CPU_ORACLE_VERIFIED','oracle_sha256':r['oracle_sha256'],'source_admission_sha256':m['source_admission_sha256'],'native_manifest_sha256':R.NC.MANIFEST_SHA,'native_cpu_inventory_sha256':'b'*64,'native_cpu_evidence_sha256':'c'*64,'source_variant':R.NC.VARIANT};r.update(native_cpu_inventory_sha256=gate['native_cpu_inventory_sha256'],native_cpu_evidence_sha256=gate['native_cpu_evidence_sha256']);U.write(out/'receipt.json',r);U.write(base/'launch.json',gate)
    assert not any(k.startswith('compiler_')for k in gate)
    monkeypatch.setattr(R,'verify_experiment',lambda value:R.NC.MODE);monkeypatch.setattr(R,'verify_transfer_payload',lambda *a:None);monkeypatch.setattr(R.NC,'payload',lambda *a:None);called=[]
    def stage(*args,**kwargs):
        called.append(args[1]);U.write(out/'native/parent.json',{'scope':'SYNTHETIC_BRANCH_ONLY'});U.write(out/'native/expected.json',{'scope':'SYNTHETIC_BRANCH_ONLY'});U.write(out/'steps/12-native-parent/ownership.json',{'scope':'SYNTHETIC_BRANCH_ONLY'});return {'status':'PASS','exit_code':0,'cleanup':{'errors':[]},'error':None}
    monkeypatch.setattr(R,'run_step',stage)
    assert R.execute(base,U.sha(base/'payload.zip'),epoch,'native')==0
    assert called==['12-native-parent']and U.read(out/'receipt.json')['status']=='NATIVE_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED'


def test_actual_pinned_frontend_preflight_owned(tmp_path):
    jax=Path(os.environ['PUBLIC15_JAX071_PYTHON']);report=O.host_frontend_preflight(PACKET,jax,tmp_path,native_source=PACKET/'public-baseline/experiments/2026-10-04-bitsandbytes-tpu/native')
    assert report['runtime']['jax']==report['runtime']['jaxlib']=='0.7.1'and report['provider_calls']==0
    assert not report['cleanup']['errors']


def test_timeout_cleanup_of_owned_ignoring_term_writer(tmp_path):
    code='import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(20)'
    r=R.run_step(tmp_path,'synthetic-timeout',[sys.executable,'-B','-c',code],time.time()+0.5,1,cwd=tmp_path,tpu=False)
    assert r['status']!='PASS'and not r['cleanup']['errors']
    own=U.read(tmp_path/'steps/synthetic-timeout/ownership.json')
    assert own['cleanup']['leader_reaped']and own['cleanup']['group_absence']=='OBSERVED_NO_SUCH_GROUP'
    with pytest.raises(ProcessLookupError):os.killpg(own['pid'],0)


@pytest.mark.parametrize('fault',[None,'hash','inventory','duplicate','path','mode'])
def test_small_cpu_archive_complete_recovery(tmp_path,fault):
    base=tmp_path/'source';out=base/'records';(out/'cpu-oracle').mkdir(parents=True)
    U.write(out/'cpu-oracle/oracle-seal.json',{'scope':'SYNTHETIC_ARCHIVE_ONLY_NOT_CPU_ACCEPTANCE'})
    for n in ['source-controls.json','installed-source.json','built-source.json','built-plugin-source.json','steps/11-cpu-oracle/ownership.json','steps/11-cpu-oracle/result.json']:U.write(out/n,{'scope':'SYNTHETIC_ARCHIVE_ONLY'})
    for k in ('plugin','upstream'):
        (base/('built-'+k)).mkdir();(base/('built-'+k)/'fixture.whl').write_bytes(b'SYNTHETIC_NOT_INSTALLABLE')
    seals=U.cpu_archive(out,base);target=tmp_path/'download';target.mkdir();(target/'inventory.json').write_bytes((out/'public-cpu-inventory.json').read_bytes());(target/'evidence.zip').write_bytes((out/'public-cpu-evidence.zip').read_bytes())
    if fault=='hash':seals['public_cpu_evidence_sha256']='0'*64
    if fault=='inventory':U.write(target/'inventory.json',{})
    if fault in ('duplicate','path','mode'):
        with zipfile.ZipFile(target/'evidence.zip')as z:items=[(i,z.read(i))for i in z.infolist()]
        with zipfile.ZipFile(target/'evidence.zip','w')as z:
            for i,b in items:
                if fault=='mode':i.external_attr=0o120777<<16
                if fault=='path':i.filename='../'+i.filename
                z.writestr(i,b)
            if fault=='duplicate':
                import warnings
                with warnings.catch_warnings():warnings.simplefilter('ignore');z.writestr(items[0][0],items[0][1])
        seals.update(public_cpu_evidence_sha256=U.sha(target/'evidence.zip'),public_cpu_evidence_bytes=(target/'evidence.zip').stat().st_size)
    if fault:
        with pytest.raises(ValueError):U.recover_cpu(target,seals)
    else:
        got=U.recover_cpu(target,seals);assert U.read(got/'cpu-oracle/oracle-seal.json')['scope']=='SYNTHETIC_ARCHIVE_ONLY_NOT_CPU_ACCEPTANCE'


def test_actual_shape_blocked_cpu_receipt_preserves_primary_and_closes(tmp_path):
    """Full admitted owner flow with fake API only; no fake accepted CPU oracle."""
    from fake_owner_api import BlockedCPU
    cli=Path(os.environ['PUBLIC15_CLI_PYTHON']);site=cli.absolute().parent.parent/'lib/python3.12/site-packages';identity=tmp_path/'cli-identity.json'
    U.write(identity,{'distribution':'google-colab-cli','version':'0.7.4','python_sha256':U.sha(cli),'venv_config_sha256':U.sha(cli.absolute().parent.parent/'pyvenv.cfg'),'metadata_sha256':U.sha(site/'google_colab_cli-0.7.4.dist-info/METADATA'),'record_sha256':U.sha(site/'google_colab_cli-0.7.4.dist-info/RECORD'),'files':{p.relative_to(site).as_posix():U.sha(p)for p in (site/'colab_cli').rglob('*.py')}})
    m=U.read(PACKET/'manifest.json');jax=Path(os.environ['PUBLIC15_JAX071_PYTHON']);cache=Path(os.environ['PUBLIC15_RUNTIME_WHEELS']);host=tmp_path/'host'
    gate={'scope':'SYNTHETIC_FAKE_API_ONLY_NOT_ACTUAL_AUTHORIZATION','status':'ACTUAL_DISPATCH_AUTHORIZED','packet_sha256':U.sha(PACKET/'payload.zip'),'driver_sha256':U.sha(HERE/'cloud/owner.py'),'output':str(host.resolve()),'plugin_source_manifest_sha256':m['plugin_source_manifest_sha256'],'runtime_lock_sha256':m['runtime_lock_sha256'],'budget':m['budget'],'one_allocation_only':True,'provider_or_solver':'FORBIDDEN','cli_identity_sha256':U.sha(identity),'experiment':U.MODE,'precision':'highest',**{k:m[k]for k in R.TRANSFER_BINDINGS},**{k:m[k]for k in U.FIELDS},'mosaic_audit_python':str(jax),'mosaic_audit_python_sha256':U.sha(jax),'public_runtime_wheels':str(cache)}
    U.write(tmp_path/'gate.json',gate);api=BlockedCPU(PACKET,tmp_path)
    result=O.drive(PACKET,host,U.sha(PACKET/'payload.zip'),tmp_path/'gate.json',U.sha(tmp_path/'gate.json'),cli,identity,api=api,simulated=True,mosaic_audit_python=jax,public_runtime_wheels=cache)
    assert result['mode']=='SIMULATED_NO_CLOUD'and result['status']=='BLOCKED'
    assert result['original_error']=={'type':'ValueError','message':'REMOTE_PHASE_BLOCKED:10-cpu'}
    assert U.read(host/'receipt.json')['error']['message']=='SYNTHETIC_CPU_SOURCE_FAILURE'
    assert result['local_verifier_status']=='NOT_RUN_INCOMPLETE_PUBLIC_RECORDS'
    assert result['whole_archive_verified']and result['server_empty_observed']and result['usage_zero_observed']and not result['cleanup_errors']
    assert '13-tpu'not in api.calls and not(host/'local-verifier-ownership.json').exists()


def test_host_preflight_rejects_ambient_jax_with_closed_group(tmp_path):
    # The control interpreter is explicitly the unqualified host Python.
    with pytest.raises(ValueError,match='MOSAIC_FRONTEND_PREFLIGHT_REJECTED'):
        O.host_frontend_preflight(PACKET,Path(sys.executable),tmp_path,native_source=PACKET/'public-baseline/experiments/2026-10-04-bitsandbytes-tpu/native')
    launch=U.read(tmp_path/'audit-preflight-launch.json');assert launch['cleanup']['group_absence']=='OBSERVED_NO_SUCH_GROUP'
    with pytest.raises(ProcessLookupError):os.killpg(launch['pid'],0)
