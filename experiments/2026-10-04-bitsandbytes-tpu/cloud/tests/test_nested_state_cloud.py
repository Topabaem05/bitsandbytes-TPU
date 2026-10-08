"""State8 protocol and full recovered synthetic records. No provider or TPU calls."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import signal
import struct
import sys
import zipfile

import pytest

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import build_packet as B
import remote
import owner
import nested_contract as C
import nested_state_contract as S
import test_transfer_cloud as F
import test_nested_cloud as NF

SCIENCE=HERE.parent
base_archive=F.base_archive


def dependency():
    return {'format':'bnb-tpu.nested79-dependency.v1','status':'ACCEPTED_NESTED79_DEVICE_RECORDS',
        'record_validation':'PASS','numerical_status':'PASS','cases':79,
        'packet_sha256':'1'*64,'actual_receipt_sha256':'2'*64,'qualified_oracle_sha256':'3'*64,'accepted_result_sha256':'4'*64,
        'source_admission_sha256':'6647dbe8d4928a201020e7116105e836585bc1d4ad1adf3cc730976d9885b139',
        'plugin_source_manifest_sha256':C.NESTED_PLUGIN_MANIFEST_SHA,'nested_probe_sha256':C.NESTED_PROBE_SHA,
        'nested_inputs_sha256':C.NESTED_INPUT_SHA,'nested_source_sha256':C.NESTED_SOURCE_SHA,
        'nested_schemas_sha256':C.NESTED_SCHEMA_SHA,'patch_manifest_sha256':B.TRANSFER_PATCH_MANIFEST_SHA,
        'runtime_lock_sha256':B.RUNTIME_SHA,'profile_sha256':S.PROFILE_SHA}


def state_args(base):
    gate=base/'SYNTHETIC-accepted79-dependency.json';F.write(gate,dependency())
    args=NF.nested_args();args.update(experiment='nested-state-8',nested_state_probe=SCIENCE/'probe_nested_state.py',
        nested_state_probe_sha256=S.NESTED_STATE_PROBE_SHA,nested79_dependency=gate,nested79_dependency_sha256=remote.sha(gate))
    return args


@pytest.fixture
def state_packet(tmp_path,base_archive,monkeypatch):
    return NF.build_fixture(tmp_path,base_archive,monkeypatch,args=state_args(tmp_path))


@pytest.mark.parametrize('fault',[None,'status','numerical','error','count','profile','runtime','source','plugin','probe','schema','admission','accepted-result','extra','missing'])
def test_accepted_nested79_dependency_is_exact(fault):
    record=dependency()
    if fault=='status':record['status']='M4_COMPLETE'
    elif fault=='numerical':record['numerical_status']='FAIL'
    elif fault=='error':record['numerical_status']='ERROR'
    elif fault=='count':record['cases']=78
    elif fault=='accepted-result':record['accepted_result_sha256']='WRONG'
    elif fault=='extra':record['saved_state']='NOT_RUN'
    elif fault=='missing':record.pop('accepted_result_sha256')
    elif fault is not None:
        field={'profile':'profile_sha256','runtime':'runtime_lock_sha256','source':'nested_source_sha256','plugin':'plugin_source_manifest_sha256',
            'probe':'nested_probe_sha256','schema':'nested_schemas_sha256','admission':'source_admission_sha256'}[fault];record[field]='a'*64
    if fault is None:assert S.verify_dependency(record)==record
    else:
        with pytest.raises(ValueError,match='NESTED79_'):S.verify_dependency(record)


@pytest.mark.parametrize('fault',[None,'probe','missing','scope','variant','wrong-experiment','missing-dependency'])
def test_state_explicit_source_protocol(state_packet,fault):
    packet,manifest,expected=state_packet
    if fault=='probe':manifest['nested_state_probe_sha256']='a'*64
    elif fault=='missing':manifest['files'].pop('probe_nested_state.py')
    elif fault=='scope':manifest['nested_state_scope']='ONLY_ONE'
    elif fault=='variant':manifest['nested_state_variant']='ANY'
    elif fault=='wrong-experiment':manifest['experiment']='nested-79'
    elif fault=='missing-dependency':manifest['files'].pop('nested79-dependency.json')
    if fault is None:
        assert remote.verify_experiment(manifest)=='nested-state-8'
        assert owner.preflight(packet,expected)['nested_state_scope']==S.NESTED_STATE_SCOPE
    else:
        with pytest.raises(ValueError,match='NESTED_STATE_'):remote.verify_experiment(manifest)


def fixture_tools():
    spec=importlib.util.spec_from_file_location('portable_nested_state_saved',HERE/'tests/nested_state_saved_fixture.py')
    tools=importlib.util.module_from_spec(spec);spec.loader.exec_module(tools);return tools


def mutate(data,tools,fault):
    case=tools.S.cases(tools.B,tools.N)[0];path=data.actual/'restore/raw'/(case['id']+'.json')
    if fault in ('numeric','case-error','restore-error'):
        raw=tools.B.read(path)
        if fault=='numeric':
            array=raw['outputs']['y'];array['values'][0]=struct.unpack('<f',struct.pack('<f',array['values'][0]+1))[0]
            array['bytes_sha256']=__import__('hashlib').sha256(tools.N.binary(array)).hexdigest()
        else:raw=tools.N.case_error_record(raw,RuntimeError('SYNTHETIC_RESTORE_ERROR'))
        tools.B.write(path,raw)
    elif fault=='missing-row':path.unlink()
    elif fault in ('source','pid','token','oracle','checkpoint-link'):
        if fault=='checkpoint-link':
            raw=tools.B.read(path);raw['loaded_checkpoint_sha256']='a'*64;tools.B.write(path,raw)
        else:
            parent=tools.B.read(data.actual/'receipt.json')
            if fault=='source':parent['source_post']='wrong'
            elif fault=='pid':parent.update(pid=1999,pgid=1999)
            elif fault=='token':parent['process_token']='z'*32
            else:parent['nested_oracle_sha256']='a'*64
            tools.B.write(data.actual/'receipt.json',parent)
    elif fault=='failed-restore-launch':
        parent=tools.B.read(data.actual/'receipt.json');parent['children'][1]['exit_code']=1;tools.B.write(data.actual/'receipt.json',parent)
    if fault!='missing-row':tools.reseal(data)
    return case


@pytest.mark.parametrize('fault',[None,'numeric','restore-error','source','missing-row','pid','token','oracle','checkpoint-link','failed-restore-launch'])
def test_recovered_state8_real_verifier(tmp_path,fault):
    tools=fixture_tools();data=tools.make_saved_fixture(tmp_path/'fixture');mutate(data,tools,fault)
    if fault in (None,'numeric','restore-error'):
        report=tools.S.verify(tools.B,tools.R,tools.P,tools.A,tools.N,data)
        assert report['record_validation']=='PASS' and len(report['rows'])==8 and report['m4_status']=='NOT_QUALIFIED'
        assert report['numerical_status']=={None:'PASS','numeric':'FAIL','restore-error':'ERROR'}[fault]
    else:
        with pytest.raises(Exception):tools.S.verify(tools.B,tools.R,tools.P,tools.A,tools.N,data)


@pytest.mark.parametrize('fault',[None,'numeric','restore-error','source','missing-row','pid','token','oracle','checkpoint-link',
    'failed-restore-launch','member-missing','member-changed','plugin-wheel','phase','stop'])
def test_full_nested_state_owner_archive_verifier(state_packet,tmp_path,monkeypatch,fault):
    """Actual local verifier and full byte archive; service and device identities are synthetic."""
    import transport
    tools=fixture_tools();packet,manifest,expected=state_packet
    data=tools.make_saved_fixture(tmp_path/'fixture',admission=manifest['source_admission_sha256']);case=mutate(data,tools,fault)
    records=tmp_path/'remote-records';records.mkdir();shutil.copytree(data.nested_oracle,records/'cpu-oracle')
    shutil.copytree(data.oracle,records/'state-cpu-oracle');shutil.copytree(data.actual,records/'nested-state')
    original=(records/'nested-state/receipt.json').read_bytes();admission=json.loads((packet/'source-admission.json').read_text())
    F.write(records/'source-controls.json',F.controls_fixture())
    proof={'status':'POST_PATCH_PYTHON_SOURCE_PASS','source_admission_sha256':manifest['source_admission_sha256'],
        'patch_manifest_sha256':B.TRANSFER_PATCH_MANIFEST_SHA,'installed_python_files':{n:admission[n]['files'] for n in ('bitsandbytes','bitsandbytes_tpu')},
        'source_variant':'nested-v1','nested_source_sha256':C.NESTED_SOURCE_SHA,'plugin_source_manifest_sha256':C.NESTED_PLUGIN_MANIFEST_SHA}
    F.write(records/'installed-source.json',proof)
    built={'status':'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS','patch_manifest_sha256':B.TRANSFER_PATCH_MANIFEST_SHA,
        'python_files':admission['bitsandbytes']['files'],'wheel_name':'SYNTHETIC-upstream.whl','wheel_sha256':'SYNTHETIC_NO_WHEEL_BODY'}
    F.write(records/'built-source.json',built)
    plugin={'status':'NESTED_PLUGIN_WHEEL_PYTHON_SOURCE_PASS','python_files':copy.deepcopy(C.NESTED_FILES),
        'plugin_source_manifest_sha256':C.NESTED_PLUGIN_MANIFEST_SHA,'nested_source_sha256':C.NESTED_SOURCE_SHA,
        'wheel_name':'SYNTHETIC-plugin.whl','wheel_sha256':'SYNTHETIC_NO_PLUGIN_WHEEL_BODY'}
    if fault=='plugin-wheel':plugin['python_files']['blockwise.py']='wrong'
    F.write(records/'built-plugin-source.json',plugin)
    receipt={'status':'NESTED_STATE_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED','experiment':'nested-state-8','precision':'highest',
        'nested_scope':C.NESTED_SCOPE,'nested_source_variant':'nested-v1','nested_state_scope':S.NESTED_STATE_SCOPE,'nested_state_variant':'nested-state-v1',
        'packet_sha256':expected,'manifest_sha256':remote.sha(packet/'manifest.json'),
        'source_admission_sha256':manifest['source_admission_sha256'],'runtime_lock_sha256':manifest['runtime_lock_sha256'],
        'runtime_status':'PASS_TPU_RUNTIME_PROBE_ONLY','cpu_status':'PASS','nested_state_cpu_status':'PASS','tpu_status':'NESTED_STATE_RECORDS_COMPLETE',
        'built_source_status':'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS','installed_source_status':'POST_PATCH_PYTHON_SOURCE_PASS',
        'built_plugin_source_status':'NESTED_PLUGIN_WHEEL_PYTHON_SOURCE_PASS','source_controls_status':'PASS_QUALIFIED_LINUX_SOURCE_CONTROLS',
        'tpu_attempted':True,'oracle_sha256':data.oracle_sha256,'nested_oracle_sha256':data.nested_oracle_sha256,
        'nested_state_receipt_sha256':remote.sha(records/'nested-state/receipt.json'),
        'built_wheels':[{'name':r['wheel_name'],'sha256':r['wheel_sha256']} for r in (built,plugin)],
        'steps':[{'scope':'SYNTHETIC_NO_DEVICE_EXECUTION','cleanup':{'errors':[],'groups':[]}}],
        **{field:manifest[field] for field in (*remote.TRANSFER_BINDINGS,*C.NESTED_BINDINGS,*S.NESTED_STATE_BINDINGS)}}
    F.write(records/'receipt.json',receipt)
    out=tmp_path/'host';identity=tmp_path/'identity.json';identity.write_text('{}')
    gate={'status':'ACTUAL_DISPATCH_AUTHORIZED','packet_sha256':expected,'driver_sha256':remote.sha(HERE/'owner.py'),
        'output':str(out.resolve()),'plugin_source_manifest_sha256':manifest['plugin_source_manifest_sha256'],
        'runtime_lock_sha256':manifest['runtime_lock_sha256'],'budget':manifest['budget'],'one_allocation_only':True,
        'provider_or_solver':'FORBIDDEN','cli_identity_sha256':remote.sha(identity),'experiment':'nested-state-8','precision':'highest',
        'nested_scope':C.NESTED_SCOPE,'nested_source_variant':'nested-v1','nested_state_scope':S.NESTED_STATE_SCOPE,'nested_state_variant':'nested-state-v1',
        **{field:manifest[field] for field in (*remote.TRANSFER_BINDINGS,*C.NESTED_BINDINGS,*S.NESTED_STATE_BINDINGS)}}
    acceptance=tmp_path/'gate.json';F.write(acceptance,gate);monkeypatch.setattr(owner,'verify_cli_identity',lambda *args:None)
    calls=[];sessions=[]
    def api(label,argv,timeout):
        calls.append(label)
        if label=='03-one-allocation':
            assert argv[0]=='--allocation-transport-v1' and timeout==360
            sessions.append(argv[argv.index('-s')+1]);receipt['allocation_epoch']=json.loads((out/'owner.json').read_text())['allocation_epoch'];F.write(records/'receipt.json',receipt)
        if label in ('04-sessions-after','90-before-stop'):return {'status':'PASS'},'['+sessions[0]+']\nHardware: V6E1'
        if label=='11b-state-cpu':assert '\"remote.py\",\'nested-state-cpu\'' in Path(argv[-1]).read_text()
        if label=='13-tpu':assert '\"remote.py\",\'nested-state\'' in Path(argv[-1]).read_text()
        if label=='21-export':
            exported=remote.package(records)
            if fault in ('member-missing','member-changed'):
                archive=records/'evidence.zip';replacement=records/'modified.zip';target='nested-state/restore/raw/'+case['id']+'.json'
                with zipfile.ZipFile(archive) as original_archive,zipfile.ZipFile(replacement,'w') as changed:
                    for info in original_archive.infolist():
                        if info.filename==target and fault=='member-missing':continue
                        body=original_archive.read(info)
                        if info.filename==target and fault=='member-changed':body+=b'CORRUPT_ARCHIVE_MEMBER'
                        changed.writestr(info,body)
                replacement.replace(archive);exported.update(evidence_sha256=remote.sha(archive),evidence_bytes=archive.stat().st_size)
            F.write(records/'receipt.json',{**receipt,**exported})
        if label=='24-export-parts':transport.split(records/'evidence.zip',tmp_path/'remote-parts',expected_sha256=remote.sha(records/'evidence.zip'),expected_bytes=(records/'evidence.zip').stat().st_size)
        if argv[0]=='download':
            destination=Path(argv[-1])
            if label in ('09-install-receipt','10-cpu-receipt','11b-state-cpu-receipt'):
                phase={**receipt,'status':{'09-install-receipt':'INSTALLED_NOT_QUALIFIED','10-cpu-receipt':'CPU_ORACLE_READY_TPU_NOT_RUN',
                    '11b-state-cpu-receipt':'NESTED_STATE_CPU_ORACLE_READY_TPU_NOT_RUN'}[label]}
                if label=='10-cpu-receipt':phase['oracle_sha256']=data.nested_oracle_sha256
                if fault=='phase' and label=='11b-state-cpu-receipt':phase['nested_state_probe_sha256']='wrong'
                F.write(destination,phase)
            else:
                rel=argv[-2].removeprefix('content/bnb-tpu-first/')
                source=records/rel.removeprefix('records/') if rel.startswith('records/') else tmp_path/'remote-parts'/rel.removeprefix('result-parts/')
                shutil.copyfile(source,destination)
        if fault=='stop' and label=='91-stop-exact':return {'status':'CLI_FAILED'},'SYNTHETIC stop refusal'
        return {'status':'PASS'},'Active assignments: 0\nUsage rate: 0.00/hr' if argv==['usage'] else 'No active sessions found on server.'
    result=owner.drive(packet,out,expected,acceptance,remote.sha(acceptance),Path('fixture-python'),identity,api=api,simulated=True)
    assert result['mode']=='SIMULATED_NO_CLOUD' and result['allocation_attempts']==1
    assert calls.count('13-tpu')==(0 if fault=='phase' else 1)
    assert calls[-4:]==['90-before-stop','91-stop-exact','92-after-stop','93-usage-after']
    assert (records/'nested-state/receipt.json').read_bytes()==original
    if fault in ('member-missing','member-changed'):
        assert result.get('local_readback_error') and 'local_verifier_exit_code' not in result
    else:
        assert result['whole_archive_verified'] is True and result['retrieval']=='COMPLETE_WHOLE_ARCHIVE'
        assert not result['local_verifier_cleanup']['errors']
    if fault not in ('phase','member-missing','member-changed'):
        report=json.loads((out/'verify.json').read_text())
        if fault in ('source','missing-row','pid','token','oracle','checkpoint-link','failed-restore-launch'):
            assert report['record_validation']=='FAIL' and result['local_verifier_exit_code']==1
        else:
            expected_numeric={None:'PASS','numeric':'FAIL','restore-error':'ERROR'}.get(fault,'PASS')
            assert report['record_validation']=='PASS' and report['numerical_status']==expected_numeric and len(report['rows'])==8
            assert result['local_verifier_exit_code']==(0 if expected_numeric=='PASS' else 2)
    if fault in (None,'numeric','restore-error'):
        assert result['status']==('PASS' if fault is None else 'FAIL')+'_TPU_NESTED_STATE_RECORDS'
        assert result['record_validation']=='PASS' and result['m4_status']=='NOT_QUALIFIED'
    else:assert result['status'] not in ('PASS_TPU_NESTED_STATE_RECORDS','FAIL_TPU_NESTED_STATE_RECORDS')
    if fault=='stop':assert result['status']=='BLOCKED_CLEANUP'


def phase_fixture(state_packet,tmp_path):
    packet,manifest,expected=state_packet;packet.rename(tmp_path/'payload');payload=tmp_path/'payload'
    shutil.copyfile(payload/'payload.zip',tmp_path/'payload.zip');out=tmp_path/'records';out.mkdir()
    receipt={'status':'CPU_ORACLE_READY_TPU_NOT_RUN','cpu_status':'PASS','tpu_status':'NOT_RUN','experiment':'nested-state-8',
        'precision':'highest','packet_sha256':expected,'manifest_sha256':remote.sha(payload/'manifest.json'),
        'source_admission_sha256':manifest['source_admission_sha256'],'runtime_lock_sha256':manifest['runtime_lock_sha256'],
        'allocation_epoch':1000,'oracle_sha256':'NESTED_ORACLE','science_deadline_epoch':3000,'steps':[],'error':None,
        'runtime_status':'PASS_TPU_RUNTIME_PROBE_ONLY','built_source_status':'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS',
        'installed_source_status':'POST_PATCH_PYTHON_SOURCE_PASS','built_plugin_source_status':'NESTED_PLUGIN_WHEEL_PYTHON_SOURCE_PASS',
        'source_controls_status':'PASS_QUALIFIED_LINUX_SOURCE_CONTROLS','nested_state_scope':S.NESTED_STATE_SCOPE,'nested_state_variant':'nested-state-v1',
        **{field:manifest[field] for field in (*remote.TRANSFER_BINDINGS,*C.NESTED_BINDINGS,*S.NESTED_STATE_BINDINGS)}}
    F.write(out/'receipt.json',receipt);return payload,out,manifest,expected,receipt


@pytest.mark.parametrize('fault',[None,'wrong-nested-oracle','source-controls','wrong-state-binding','prior-state','wrong-phase'])
def test_fresh_state_cpu_oracle_stage_order(state_packet,tmp_path,monkeypatch,fault):
    payload,out,manifest,expected,receipt=phase_fixture(state_packet,tmp_path)
    F.write(tmp_path/'nested-launch.json',{'nested_oracle_sha256':'WRONG' if fault=='wrong-nested-oracle' else 'NESTED_ORACLE'})
    if fault=='source-controls':receipt['source_controls_status']='NOT_RUN'
    if fault=='wrong-state-binding':receipt['nested_state_probe_sha256']='wrong'
    if fault=='prior-state':receipt['nested_oracle_sha256']='ALREADY_RUN'
    F.write(out/'receipt.json',receipt);monkeypatch.setattr(remote.time,'time',lambda:1000);calls=[]
    def child(output,label,argv,deadline,limit,**kwargs):
        calls.append(label);assert label=='11-nested-state-cpu-oracle' and limit==300 and kwargs.get('tpu') is not True
        assert argv[2:4]==[str(payload/'probe_nested_state.py'),'prepare']
        assert argv[argv.index('--nested-oracle-sha256')+1]=='NESTED_ORACLE' and '--nested-probe' in argv
        (out/'state-cpu-oracle').mkdir();F.write(out/'state-cpu-oracle/oracle-seal.json',{'SYNTHETIC':'route only'})
        return {'status':'PASS','exit_code':0,'cleanup':{'errors':[],'groups':[]}}
    monkeypatch.setattr(remote,'run_step',child)
    if fault=='wrong-state-binding':
        with pytest.raises(ValueError,match='NESTED_STATE_PHASE_SOURCE_BINDING'):remote.execute(tmp_path,expected,1000,'nested-state-cpu')
    else:
        code=remote.execute(tmp_path,expected,1000,'nested' if fault=='wrong-phase' else 'nested-state-cpu')
        rec=json.loads((out/'receipt.json').read_text());assert code==(0 if fault is None else 1)
        assert rec['status']==('NESTED_STATE_CPU_ORACLE_READY_TPU_NOT_RUN' if fault is None else 'BLOCKED')
        if fault is None:assert rec['nested_oracle_sha256']=='NESTED_ORACLE' and rec['oracle_sha256']==remote.sha(out/'state-cpu-oracle/oracle-seal.json')
    assert calls==(['11-nested-state-cpu-oracle'] if fault is None else [])


@pytest.mark.parametrize('fault',[None,'numeric','restore-error','source','pid','pgid','token','deadline','launch','oracle','cleanup','controls','prior-phase'])
def test_remote_state_parent_actual_launch_binding(state_packet,tmp_path,monkeypatch,fault):
    payload,out,manifest,expected,receipt=phase_fixture(state_packet,tmp_path)
    receipt.update(status='NESTED_STATE_CPU_ORACLE_READY_TPU_NOT_RUN',nested_state_cpu_status='PASS',nested_oracle_sha256='NESTED_ORACLE',oracle_sha256='STATE_ORACLE')
    if fault=='controls':receipt['built_plugin_source_status']='NOT_RUN'
    if fault=='prior-phase':receipt['status']='CPU_ORACLE_READY_TPU_NOT_RUN'
    F.write(out/'receipt.json',receipt);F.write(tmp_path/'launch.json',{'oracle_sha256':'STATE_ORACLE','nested_oracle_sha256':'NESTED_ORACLE'})
    monkeypatch.setattr(remote.time,'time',lambda:1000);calls=[]
    def child(output,label,argv,deadline,limit,**kwargs):
        calls.append(label);assert label=='12-nested-state' and deadline==2500 and limit==1500 and kwargs['tpu'] is True
        assert argv[2:4]==[str(payload/'probe_nested_state.py'),'execute']
        token=argv[argv.index('--process-token')+1]
        result={'kind':'PRIVATE_NESTED_STATE_PROBE','status':'COMPLETE','state_variant':'nested-state-v1',
            'probe_sha256':manifest['nested_state_probe_sha256'],'nested_probe_sha256':manifest['nested_probe_sha256'],
            'inputs_sha256':manifest['nested_inputs_sha256'],'nested_source_sha256':manifest['nested_source_sha256'],
            'schema_sha256':manifest['nested_schemas_sha256'],'source_variant':'nested-v1',
            'patch_manifest_sha256':manifest['patch_manifest_sha256'],'source_admission_sha256':manifest['source_admission_sha256'],
            'source_pre':manifest['source_admission_sha256'],'source_post':manifest['source_admission_sha256'],
            'oracle_sha256':'STATE_ORACLE','nested_oracle_sha256':'NESTED_ORACLE','runtime_lock_sha256':manifest['runtime_lock_sha256'],
            'process_token':token,'deadline_epoch':deadline,'pid':7001,'pgid':7001}
        changes={'source':('source_post','wrong'),'pid':('pid',7002),'pgid':('pgid',7002),'token':('process_token','a'*32),
            'deadline':('deadline_epoch',3000),'oracle':('nested_oracle_sha256','wrong')}
        if fault in changes:result.update([changes[fault]])
        if fault=='pid':result['pgid']=7002
        (out/'nested-state').mkdir();F.write(out/'nested-state/receipt.json',result)
        failed=fault in ('numeric','restore-error')
        return {'status':'CHILD_FAILED' if failed else 'PASS','exit_code':2 if failed else 0,
            'cleanup':{'errors':['SYNTHETIC'] if fault=='cleanup' else [],'groups':[{'pid':7999 if fault=='launch' else 7001,'pgid':7999 if fault=='launch' else 7001}]}}
    monkeypatch.setattr(remote,'run_step',child)
    code=remote.execute(tmp_path,expected,1000,'nested-state');rec=json.loads((out/'receipt.json').read_text())
    valid=fault in (None,'numeric','restore-error');assert code==(0 if valid else 1)
    assert rec['status']==('NESTED_STATE_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED' if valid else 'BLOCKED')
    assert calls==([] if fault in ('controls','prior-phase') else ['12-nested-state'])


@pytest.mark.parametrize('fault',['missing-state-probe','wrong-state-pin','missing-dependency','wrong-dependency-sha','unaccepted-dependency'])
def test_builder_requires_exact_state_source_and_accepted79(tmp_path,base_archive,monkeypatch,fault):
    args=state_args(tmp_path)
    if fault=='missing-state-probe':args['nested_state_probe']=None
    elif fault=='wrong-state-pin':args['nested_state_probe_sha256']='a'*64
    elif fault=='missing-dependency':args['nested79_dependency']=None
    elif fault=='wrong-dependency-sha':args['nested79_dependency_sha256']='a'*64
    else:
        record=dependency();record['numerical_status']='FAIL';F.write(args['nested79_dependency'],record)
        args['nested79_dependency_sha256']=remote.sha(args['nested79_dependency'])
    with pytest.raises(ValueError,match='NESTED_STATE_|NESTED79_'):NF.build_fixture(tmp_path,base_archive,monkeypatch,args)


def test_complete_state_packet_is_deterministic(tmp_path,base_archive,monkeypatch):
    first=tmp_path/'first';second=tmp_path/'second';first.mkdir();second.mkdir()
    packet1,manifest1,sha1=NF.build_fixture(first,base_archive,monkeypatch,state_args(first))
    packet2,manifest2,sha2=NF.build_fixture(second,base_archive,monkeypatch,state_args(second))
    assert sha1==sha2 and (packet1/'payload.zip').read_bytes()==(packet2/'payload.zip').read_bytes()
    assert manifest1==manifest2
    assert {'probe_nested_state.py','nested79-dependency.json','probe_nested.py','nested-inputs.json','nested-source.json','nested-schemas.json'}<=set(manifest1['files'])
    assert owner.preflight(packet1,sha1)['nested_state_variant']=='nested-state-v1'
