"""Explicit nested packets and recovered synthetic records. No provider or TPU calls."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import signal
import sys
import zipfile

import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import build_packet as B
import remote
import owner
import nested_contract as C
import test_transfer_cloud as F

ROOT=B.PROJECT
SCIENCE=HERE.parent
base_archive=F.base_archive


def nested_args():
    paths={name:SCIENCE/file for name,file in [('nested_probe','probe_nested.py'),('nested_inputs','nested-inputs.json'),
        ('nested_source','nested-source.json'),('nested_schemas','nested-schemas.json')]}
    return dict(F.source_args(),experiment='nested-79',**paths,
        **{name+'_sha256':remote.sha(path) for name,path in paths.items()},nested_plugin=SCIENCE/'plugin')


def build_fixture(base,archive,monkeypatch,args=None):
    original_run=B.subprocess.run;original_read=B.subprocess.check_output
    def run(argv,**kwargs):
        if argv[:1]==['git'] and 'archive' in argv:
            kwargs['stdout'].write(archive.read_bytes());return
        return original_run(argv,**kwargs)
    def read(argv,**kwargs):
        return B.COMMIT+'\n' if argv[:1]==['git'] and 'rev-parse' in argv else original_read(argv,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(B.subprocess,'run',run);patch.setattr(B.subprocess,'check_output',read)
        packet=base/'packet';result=B.build(base/'synthetic-checkout',packet,C.NESTED_PLUGIN_MANIFEST_SHA,**(args or nested_args()))
    return packet,json.loads((packet/'manifest.json').read_text()),result['payload_sha256']


@pytest.fixture
def nested_packet(tmp_path,base_archive,monkeypatch):
    return build_fixture(tmp_path,base_archive,monkeypatch)


@pytest.mark.parametrize('fault',[None,'missing','probe','inputs','source','schemas','plugin','scope','variant','wrong-experiment','plugin-without-variant'])
def test_nested_explicit_source_admission(nested_packet,fault):
    packet,manifest,digest=nested_packet
    if fault=='missing':del manifest['files']['nested-source.json']
    elif fault in ('probe','inputs','source','schemas'):manifest['nested_'+fault+'_sha256']='a'*64
    elif fault=='plugin':manifest['plugin_source_manifest_sha256']='a'*64
    elif fault=='scope':manifest['nested_scope']='ONLY_ONE_CASE'
    elif fault=='variant':manifest['nested_source_variant']='ANY_PLUGIN'
    elif fault=='wrong-experiment':manifest['experiment']='transfer-api42'
    elif fault=='plugin-without-variant':
        manifest['experiment']='transfer-api42'
        for field,name in C.NESTED_BINDINGS.items():manifest.pop(field,None);manifest['files'].pop(name,None)
        manifest.pop('nested_scope',None);manifest.pop('nested_source_variant',None)
    if fault is None:
        assert remote.verify_experiment(manifest)=='nested-79'
        assert owner.preflight(packet,digest)['nested_scope']==C.NESTED_SCOPE
        admission=json.loads((packet/'source-admission.json').read_text())
        assert admission['bitsandbytes_tpu']=={'files':C.NESTED_FILES,'source_variant':'nested-v1','nested_source_sha256':C.NESTED_SOURCE_SHA}
        assert len(admission['bitsandbytes']['files'])==47
    else:
        with pytest.raises(ValueError,match='NESTED_'):remote.verify_experiment(manifest)


@pytest.mark.parametrize('fault',['wrong-plugin','changed-plugin','wrong-patch','wrong-probe','wrong-spec','wrong-variant'])
def test_builder_rejects_unreviewed_nested_source(tmp_path,base_archive,monkeypatch,fault):
    args=nested_args()
    if fault in ('wrong-plugin','changed-plugin'):
        plugin=tmp_path/'plugin';shutil.copytree(args['nested_plugin'],plugin);args['nested_plugin']=plugin
        path=plugin/('source-manifest.json' if fault=='wrong-plugin' else 'src/bitsandbytes_tpu/blockwise.py')
        path.write_bytes(path.read_bytes()+b'\n')
    elif fault=='wrong-patch':args['patch_manifest_sha256']='a'*64
    elif fault=='wrong-probe':args['nested_probe_sha256']='a'*64
    elif fault=='wrong-spec':args['nested_inputs_sha256']='a'*64
    else:args['experiment']='transfer-api42'
    with pytest.raises(ValueError,match='NESTED_|PLUGIN_|TRANSFER_'):build_fixture(tmp_path,base_archive,monkeypatch,args)


@pytest.mark.parametrize('fault',[None,'changed','missing','extra','duplicate','pyc'])
def test_nested_built_wheel_exact_inventory(tmp_path,fault):
    files={name:(SCIENCE/'plugin/src/bitsandbytes_tpu'/name).read_bytes() for name in C.NESTED_FILES}
    if fault=='changed':files['blockwise.py']+=b'\n'
    elif fault=='missing':del files['blockwise.py']
    elif fault=='extra':files['extra.py']=b'pass\n'
    elif fault=='pyc':files['extra.pyc']=b'compiled control'
    wheel=tmp_path/'synthetic.whl'
    with zipfile.ZipFile(wheel,'w') as z:
        for name,body in files.items():z.writestr('bitsandbytes_tpu/'+name,body)
        if fault=='duplicate':z.writestr('bitsandbytes_tpu/blockwise.py',files['blockwise.py'])
    if fault is None:assert C.verify_nested_wheel(wheel)==C.NESTED_FILES
    else:
        with pytest.raises(ValueError,match='NESTED_WHEEL_'):C.verify_nested_wheel(wheel)


def saved_fixtures():
    spec=importlib.util.spec_from_file_location('nested_saved_controls',HERE/'tests/nested_saved_fixture.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.mark.parametrize('fault',[None,'numeric','case-error','missing-row','source','wrong-runtime'])
def test_actual_nested_saved_verifier(tmp_path,fault):
    fixtures=saved_fixtures();data=fixtures.make_saved_fixture(tmp_path);N=fixtures.N
    case=fixtures.DATA['cases'][0];path=data.actual/'raw'/(case['id']+'.json')
    if fault in ('numeric','case-error'):
        raw=fixtures.B.read(path)
        if fault=='numeric':
            array=raw['outputs']['codes'];array['values'][0]=(array['values'][0]+1)%256
            array['bytes_sha256']=__import__('hashlib').sha256(N.binary(array)).hexdigest()
        else:raw.update(status='ERROR',error_type='RuntimeError',error_message='RETAINED_SYNTHETIC_CASE_ERROR',traceback='SYNTHETIC_ERROR_TRACE')
        fixtures.B.write(path,raw)
    elif fault=='missing-row':path.unlink()
    elif fault in ('source','wrong-runtime'):
        rec=fixtures.B.read(data.actual/'receipt.json');rec['source_post' if fault=='source' else 'runtime']='wrong';fixtures.B.write(data.actual/'receipt.json',rec)
    fixtures.reseal(data)
    if fault in (None,'numeric','case-error'):
        report=N.verify(fixtures.B,fixtures.R,fixtures.P,fixtures.A,data)
        assert report['record_validation']=='PASS' and len(report['rows'])==79
        assert report['numerical_status']==({None:'PASS','numeric':'FAIL','case-error':'ERROR'}[fault])
        assert report['m4_status']=='NOT_QUALIFIED'
    else:
        with pytest.raises(Exception):N.verify(fixtures.B,fixtures.R,fixtures.P,fixtures.A,data)


@pytest.mark.parametrize('fault',[None,'numeric','case-error','kind','source','probe','patch','pid','pgid','launch','phase','cleanup','controls','wheel'])
def test_nested_remote_phase_bindings(nested_packet,tmp_path,monkeypatch,fault):
    packet,manifest,expected=nested_packet;packet.rename(tmp_path/'payload');payload=tmp_path/'payload'
    shutil.copyfile(payload/'payload.zip',tmp_path/'payload.zip');out=tmp_path/'records';out.mkdir()
    F.write(tmp_path/'launch.json',{'oracle_sha256':'oracle'})
    receipt={'status':'CPU_ORACLE_READY_TPU_NOT_RUN','cpu_status':'PASS','tpu_status':'NOT_RUN','experiment':'nested-79',
             'precision':'highest','packet_sha256':expected,'manifest_sha256':remote.sha(payload/'manifest.json'),
             'source_admission_sha256':manifest['source_admission_sha256'],'runtime_lock_sha256':manifest['runtime_lock_sha256'],
             'allocation_epoch':1000,'oracle_sha256':'oracle','science_deadline_epoch':3000,'steps':[],'error':None,
             'built_source_status':'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS','installed_source_status':'POST_PATCH_PYTHON_SOURCE_PASS',
             'built_plugin_source_status':'wrong' if fault=='wheel' else 'NESTED_PLUGIN_WHEEL_PYTHON_SOURCE_PASS',
             'source_controls_status':'wrong' if fault=='controls' else 'PASS_QUALIFIED_LINUX_SOURCE_CONTROLS',
             **{field:manifest[field] for field in (*remote.TRANSFER_BINDINGS,*C.NESTED_BINDINGS)}}
    F.write(out/'receipt.json',receipt)
    result={'kind':'PRIVATE_NESTED_PROBE','status':'COMPLETE','probe_sha256':manifest['nested_probe_sha256'],
            'inputs_sha256':manifest['nested_inputs_sha256'],'nested_source_sha256':manifest['nested_source_sha256'],
            'schema_sha256':manifest['nested_schemas_sha256'],'source_variant':'nested-v1',
            'patch_manifest_sha256':manifest['patch_manifest_sha256'],'source_admission_sha256':manifest['source_admission_sha256'],
            'source_pre':manifest['source_admission_sha256'],'source_post':manifest['source_admission_sha256'],
            'oracle_sha256':'oracle','runtime_lock_sha256':manifest['runtime_lock_sha256'],'pid':7001,'pgid':7001}
    changes={'kind':('kind','TRANSFER_API42_PROBE'),'source':('source_post','wrong'),'probe':('probe_sha256','wrong'),
             'patch':('patch_manifest_sha256','wrong'),'pid':('pid',7002),'pgid':('pgid',7002)}
    if fault in changes:result.update([changes[fault]])
    if fault=='pid':result['pgid']=7002
    monkeypatch.setattr(remote.time,'time',lambda:1000);calls=[]
    def child(output,label,argv,deadline,limit,**kwargs):
        calls.append(label);assert label=='12-nested' and deadline==2500 and limit==1500 and kwargs['tpu'] is True
        assert argv[2]==str(payload/'probe_nested.py') and '--transfer-admission' in argv
        assert argv[argv.index('--deadline-epoch')+1]=='2500'
        (out/'nested').mkdir();F.write(out/'nested/receipt.json',result)
        failed=fault in ('numeric','case-error')
        return {'status':'CHILD_FAILED' if failed else 'PASS','exit_code':2 if failed else 0,
                'cleanup':{'errors':['SYNTHETIC'] if fault=='cleanup' else [],'groups':[{'pid':7999 if fault=='launch' else 7001,'pgid':7999 if fault=='launch' else 7001}]}}
    monkeypatch.setattr(remote,'run_step',child)
    code=remote.execute(tmp_path,expected,1000,'transfer' if fault=='phase' else 'nested')
    receipt=json.loads((out/'receipt.json').read_text());valid=fault in (None,'numeric','case-error')
    assert code==(0 if valid else 1)
    assert receipt['status']==('NESTED_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED' if valid else 'BLOCKED')
    if valid:assert receipt['nested_receipt_sha256']==remote.sha(out/'nested/receipt.json')
    assert calls==([] if fault in ('controls','wheel','phase') else ['12-nested'])


@pytest.mark.parametrize('fault',[None,'numeric','case-error','source','missing-row','pid','token','oracle','member-missing','member-changed','plugin-wheel','phase','part-bytes','stop'])
def test_full_nested_owner_archive_verifier(nested_packet,tmp_path,monkeypatch,fault):
    """Actual source packet, full byte recovery and real verifier with synthetic service data."""
    import transport
    fixture=saved_fixtures();packet,manifest,expected=nested_packet
    data=fixture.make_saved_fixture(tmp_path/'fixture',admission=manifest['source_admission_sha256'])
    case=fixture.DATA['cases'][0];path=data.actual/'raw'/(case['id']+'.json')
    if fault in ('numeric','case-error'):
        raw=fixture.B.read(path)
        if fault=='numeric':
            array=raw['outputs']['codes'];array['values'][0]=(array['values'][0]+1)%256
            array['bytes_sha256']=__import__('hashlib').sha256(fixture.N.binary(array)).hexdigest()
        else:raw.update(status='ERROR',error_type='RuntimeError',error_message='RETAINED_SYNTHETIC_CASE_ERROR',traceback='SYNTHETIC_ERROR_TRACE')
        fixture.B.write(path,raw)
    elif fault=='source':
        raw=fixture.B.read(data.actual/'receipt.json');raw['source_post']='wrong';fixture.B.write(data.actual/'receipt.json',raw)
    elif fault=='missing-row':path.unlink()
    elif fault in ('pid','token','oracle'):
        rec=fixture.B.read(data.actual/'receipt.json')
        if fault=='pid':rec.update(pid=1999,pgid=1999)
        elif fault=='token':rec['process_token']='z'*32
        else:rec['oracle_sha256']='b'*64
        fixture.B.write(data.actual/'receipt.json',rec)
    fixture.reseal(data)
    records=tmp_path/'remote-records';records.mkdir();shutil.copytree(data.oracle,records/'cpu-oracle');shutil.copytree(data.actual,records/'nested')
    original_receipt=(records/'nested/receipt.json').read_bytes()
    admission=json.loads((packet/'source-admission.json').read_text())
    F.write(records/'source-controls.json',F.controls_fixture())
    proof={'status':'POST_PATCH_PYTHON_SOURCE_PASS','source_admission_sha256':manifest['source_admission_sha256'],
           'patch_manifest_sha256':B.TRANSFER_PATCH_MANIFEST_SHA,'installed_python_files':{name:admission[name]['files'] for name in ('bitsandbytes','bitsandbytes_tpu')},
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
    receipt={'status':'NESTED_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED','experiment':'nested-79','precision':'highest',
        'nested_scope':C.NESTED_SCOPE,'nested_source_variant':'nested-v1','packet_sha256':expected,'manifest_sha256':remote.sha(packet/'manifest.json'),
        'source_admission_sha256':manifest['source_admission_sha256'],'runtime_lock_sha256':manifest['runtime_lock_sha256'],
        'runtime_status':'PASS_TPU_RUNTIME_PROBE_ONLY','cpu_status':'PASS','tpu_status':'NESTED_RECORDS_COMPLETE',
        'built_source_status':'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS','installed_source_status':'POST_PATCH_PYTHON_SOURCE_PASS',
        'built_plugin_source_status':'NESTED_PLUGIN_WHEEL_PYTHON_SOURCE_PASS','source_controls_status':'PASS_QUALIFIED_LINUX_SOURCE_CONTROLS',
        'tpu_attempted':True,'oracle_sha256':data.oracle_sha256,'nested_receipt_sha256':remote.sha(records/'nested/receipt.json'),
        'built_wheels':[{'name':row['wheel_name'],'sha256':row['wheel_sha256']} for row in (built,plugin)],
        'steps':[{'scope':'SYNTHETIC_NO_DEVICE_EXECUTION','cleanup':{'errors':[],'groups':[]}}],
        **{field:manifest[field] for field in (*remote.TRANSFER_BINDINGS,*C.NESTED_BINDINGS)}}
    F.write(records/'receipt.json',receipt)
    out=tmp_path/'host';identity=tmp_path/'identity.json';identity.write_text('{}')
    gate={'status':'ACTUAL_DISPATCH_AUTHORIZED','packet_sha256':expected,'driver_sha256':remote.sha(HERE/'owner.py'),
          'output':str(out.resolve()),'plugin_source_manifest_sha256':manifest['plugin_source_manifest_sha256'],
          'runtime_lock_sha256':manifest['runtime_lock_sha256'],'budget':manifest['budget'],'one_allocation_only':True,
          'provider_or_solver':'FORBIDDEN','cli_identity_sha256':remote.sha(identity),'experiment':'nested-79','precision':'highest',
          'nested_scope':C.NESTED_SCOPE,'nested_source_variant':'nested-v1',
          **{field:manifest[field] for field in (*remote.TRANSFER_BINDINGS,*C.NESTED_BINDINGS)}}
    acceptance=tmp_path/'gate.json';F.write(acceptance,gate)
    monkeypatch.setattr(owner,'verify_cli_identity',lambda *args:None)
    calls=[];sessions=[];handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)};timer=signal.getitimer(signal.ITIMER_REAL)
    def api(label,argv,timeout):
        calls.append(label)
        if label=='03-one-allocation':
            assert argv[0]=='--allocation-transport-v1' and timeout==360
            sessions.append(argv[argv.index('-s')+1]);receipt['allocation_epoch']=json.loads((out/'owner.json').read_text())['allocation_epoch'];F.write(records/'receipt.json',receipt)
        if label in ('04-sessions-after','90-before-stop'):return {'status':'PASS'},'['+sessions[0]+']\nHardware: V6E1'
        if label=='13-tpu':assert '\"remote.py\",\'nested\'' in Path(argv[-1]).read_text()
        if label=='21-export':
            exported=remote.package(records)
            if fault in ('member-missing','member-changed'):
                archive=records/'evidence.zip';replacement=records/'modified-evidence.zip'
                target='nested/raw/'+case['id']+'.json'
                with zipfile.ZipFile(archive) as original,zipfile.ZipFile(replacement,'w') as changed:
                    for info in original.infolist():
                        if info.filename==target and fault=='member-missing':continue
                        body=original.read(info)
                        if info.filename==target and fault=='member-changed':body+=b'CORRUPT_ARCHIVE_MEMBER'
                        changed.writestr(info,body)
                replacement.replace(archive)
                # Independently expected whole bytes agree, but retained member inventory does not.
                exported.update(evidence_sha256=remote.sha(archive),evidence_bytes=archive.stat().st_size)
            F.write(records/'receipt.json',{**receipt,**exported})
        if label=='24-export-parts':transport.split(records/'evidence.zip',tmp_path/'remote-parts',expected_sha256=remote.sha(records/'evidence.zip'),expected_bytes=(records/'evidence.zip').stat().st_size)
        if argv[0]=='download':
            destination=Path(argv[-1])
            if label in ('09-install-receipt','10-cpu-receipt'):
                phase={**receipt,'status':'INSTALLED_NOT_QUALIFIED' if label.startswith('09-') else 'CPU_ORACLE_READY_TPU_NOT_RUN'}
                if fault=='phase' and label=='09-install-receipt':phase['nested_source_sha256']='wrong'
                F.write(destination,phase)
            else:
                rel=argv[-2].removeprefix('content/bnb-tpu-first/')
                source=records/rel.removeprefix('records/') if rel.startswith('records/') else tmp_path/'remote-parts'/rel.removeprefix('result-parts/')
                shutil.copyfile(source,destination)
                if fault=='part-bytes' and label.startswith('26-part-'):destination.write_bytes(destination.read_bytes()+b'bad')
        if fault=='stop' and label=='91-stop-exact':return {'status':'CLI_FAILED'},'SYNTHETIC stop refusal'
        return {'status':'PASS'},'Active assignments: 0\nUsage rate: 0.00/hr' if argv==['usage'] else 'No active sessions found on server.'
    result=owner.drive(packet,out,expected,acceptance,remote.sha(acceptance),Path('fixture-python'),identity,api=api,simulated=True)
    assert result['mode']=='SIMULATED_NO_CLOUD' and result['allocation_attempts']==1
    assert calls.count('13-tpu')==(0 if fault=='phase' else 1)
    assert calls[-4:]==['90-before-stop','91-stop-exact','92-after-stop','93-usage-after']
    assert all(signal.getsignal(sig)==value for sig,value in handlers.items()) and signal.getitimer(signal.ITIMER_REAL)==timer
    assert (records/'nested/receipt.json').read_bytes()==original_receipt
    if fault=='part-bytes':assert result.get('retrieval_error') and 'local_verifier_exit_code' not in result
    elif fault in ('member-missing','member-changed'):
        assert result['retrieval']=='COMPLETE_WHOLE_ARCHIVE' and result.get('local_readback_error')
        assert result.get('whole_archive_verified') is not True and 'local_verifier_exit_code' not in result
    else:
        assert result['retrieval']=='COMPLETE_WHOLE_ARCHIVE' and result['whole_archive_verified'] is True
        assert (out/'recovered/nested/receipt.json').read_bytes()==original_receipt
        assert not result['local_verifier_cleanup']['errors']
        assert all(group['group_absence']=='OBSERVED_NO_SUCH_GROUP' for group in result['local_verifier_cleanup']['groups'])
    if fault=='phase':assert 'local_verifier_exit_code' not in result
    elif fault not in ('part-bytes','member-missing','member-changed'):
        report=json.loads((out/'verify.json').read_text());assert report['m4_status']=='NOT_QUALIFIED'
        if fault in ('source','missing-row','pid','token','oracle'):assert report['record_validation']=='FAIL' and result['local_verifier_exit_code']==1
        else:
            numerical='FAIL' if fault=='numeric' else 'ERROR' if fault=='case-error' else 'PASS'
            assert report['record_validation']=='PASS' and report['numerical_status']==numerical and len(report['rows'])==79
            assert result['local_verifier_exit_code']==(0 if numerical=='PASS' else 2)
    if fault in (None,'numeric','case-error'):
        assert result['status']==('PASS' if fault is None else 'FAIL')+'_TPU_NESTED_RECORDS'
        assert result['record_validation']=='PASS' and result['m4_status']=='NOT_QUALIFIED'
    else:assert result['status'] not in ('PASS_TPU_NESTED_RECORDS','FAIL_TPU_NESTED_RECORDS')
    if fault=='stop':assert result['status']=='BLOCKED_CLEANUP'


@pytest.mark.parametrize('fault',[None,'plugin-wheel','controls'])
def test_nested_cpu_oracle_exact_route(nested_packet,tmp_path,monkeypatch,fault):
    packet,manifest,expected=nested_packet;packet.rename(tmp_path/'payload');payload=tmp_path/'payload'
    shutil.copyfile(payload/'payload.zip',tmp_path/'payload.zip');out=tmp_path/'records';out.mkdir()
    receipt={'status':'INSTALLED_NOT_QUALIFIED','installation_status':'PASS','cpu_status':'NOT_RUN','tpu_status':'NOT_RUN','experiment':'nested-79',
             'precision':'highest','packet_sha256':expected,'manifest_sha256':remote.sha(payload/'manifest.json'),
             'source_admission_sha256':manifest['source_admission_sha256'],'runtime_lock_sha256':manifest['runtime_lock_sha256'],
             'allocation_epoch':1000,'science_deadline_epoch':3000,'steps':[],'error':None,
             'built_source_status':'POST_PATCH_WHEEL_PYTHON_SOURCE_PASS','installed_source_status':'POST_PATCH_PYTHON_SOURCE_PASS',
             'built_plugin_source_status':'wrong' if fault=='plugin-wheel' else 'NESTED_PLUGIN_WHEEL_PYTHON_SOURCE_PASS',
             **{field:manifest[field] for field in (*remote.TRANSFER_BINDINGS,*C.NESTED_BINDINGS)}}
    F.write(out/'receipt.json',receipt);monkeypatch.setattr(remote.time,'time',lambda:1000);calls=[]
    def child(output,label,argv,deadline,limit,**kwargs):
        calls.append(label)
        if label=='10-transfer-source-controls':
            controls=F.controls_fixture()
            if fault=='controls':controls['controls'].pop()
            F.write(out/'source-controls.json',controls)
        elif label=='10-runtime-probe':F.write(out/'runtime-probe.json',{'status':'PASS_TPU_RUNTIME_PROBE_ONLY'})
        elif label=='11-cpu-oracle':
            assert argv[2]==str(payload/'probe_nested.py') and argv[3]=='prepare' and '--transfer-admission' in argv
            assert kwargs.get('tpu') is not True
            (out/'cpu-oracle').mkdir();F.write(out/'cpu-oracle/oracle-seal.json',{'SYNTHETIC':'CLI route only; no CPU science'})
        return {'status':'PASS','exit_code':0,'cleanup':{'errors':[],'groups':[]}}
    monkeypatch.setattr(remote,'run_step',child)
    code=remote.execute(tmp_path,expected,1000,'cpu');receipt=json.loads((out/'receipt.json').read_text())
    assert code==(0 if fault is None else 1)
    assert receipt['status']==('CPU_ORACLE_READY_TPU_NOT_RUN' if fault is None else 'BLOCKED')
    assert calls==([] if fault=='plugin-wheel' else ['10-transfer-source-controls'] if fault=='controls' else ['10-transfer-source-controls','10-runtime-probe','11-cpu-oracle'])
