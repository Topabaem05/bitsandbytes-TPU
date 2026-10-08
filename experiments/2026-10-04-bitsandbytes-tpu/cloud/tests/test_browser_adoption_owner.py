"""Full browser-adoption owner mode, fake service, actual scientific verifier. No cloud."""
import copy,json,shutil,signal,sys,time,zipfile
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import remote,owner,build_packet as B,nested_contract as C,browser_adoption as A
import test_nested_cloud as NF
import test_transfer_cloud as F
base_archive=F.base_archive
nested_packet=NF.nested_packet

@pytest.mark.parametrize('hardware',['V6E1','V5E1'])
@pytest.mark.parametrize('fault',[None,'numeric','case-error','marker','wrong-endpoint','additional-runtime','registration-fail','late-extra-runtime','endpoint-prefix-collision','stop','hardware-mismatch','gate-hardware','unlisted-hardware'])
def test_browser_adoption_owned_lifecycle(nested_packet,tmp_path,monkeypatch,fault,hardware):
    """Actual source packet, full byte recovery and real verifier with synthetic service data."""
    import transport
    fixture=NF.saved_fixtures();packet,manifest,expected=nested_packet
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
    now=time.time();adoption={'format':'bnb-tpu.browser-adoption.v1','status':'ROOT_CREATED_BROWSER_RUNTIME_READY',
        'endpoint':'OFFLINE_ENDPOINT','session':'offline-browser-adopted','hardware':hardware,'variant':'TPU','authuser':'0',
        'allocation_epoch':now-100,'observed_epoch':now-1,'packet_sha256':expected,'driver_sha256':remote.sha(HERE/'owner.py'),
        'cli_identity_sha256':remote.sha(identity),'marker_path':'/tmp/bnb-tpu-root-browser-'+('1'*32)+'.json','marker_sha256':'2'*64}
    adoption_path=tmp_path/'adoption.json';F.write(adoption_path,adoption)
    gate.update(runtime_mode=A.MODE,browser_adoption_sha256=remote.sha(adoption_path),adopted_endpoint=adoption['endpoint'],
        adopted_session=adoption['session'],adopted_hardware=adoption['hardware'],original_allocation_epoch=adoption['allocation_epoch'],
        marker_path=adoption['marker_path'],marker_sha256=adoption['marker_sha256'])
    if fault=='gate-hardware':gate['adopted_hardware']='V5E1' if hardware=='V6E1' else 'V6E1'
    if fault=='unlisted-hardware':adoption['hardware']='V4';F.write(adoption_path,adoption);gate['browser_adoption_sha256']=remote.sha(adoption_path);gate['adopted_hardware']='V4'
    acceptance=tmp_path/'gate.json';F.write(acceptance,gate)
    monkeypatch.setattr(owner,'verify_cli_identity',lambda *args:None)
    calls=[];sessions=[];handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)};timer=signal.getitimer(signal.ITIMER_REAL)
    def api(label,argv,timeout):
        calls.append(label)
        assert '--allocation-transport-v1' not in argv and 'new' not in argv
        if label=='01-sessions-before':return {'status':'PASS'},'[?] OFFLINE_ENDPOINT | Hardware: '+hardware+' | Variant: TPU'
        if label=='03-browser-registration':
            assert argv==[A.FLAG,str(adoption_path),remote.sha(adoption_path)]
            if fault in ('wrong-endpoint','additional-runtime','registration-fail'):return {'status':'CLI_FAILED'},'SYNTHETIC registration refusal'
            sessions.append(adoption['session']);receipt['allocation_epoch']=adoption['allocation_epoch'];F.write(records/'receipt.json',receipt)
        if label in ('04-sessions-after','90-before-stop'):return {'status':'PASS'},'['+sessions[0]+'] '+('OFFLINE_ENDPOINT_SUFFIX' if fault=='endpoint-prefix-collision' else 'OFFLINE_ENDPOINT')+' | Hardware: '+(('V5E1' if hardware=='V6E1' else 'V6E1') if fault=='hardware-mismatch' else hardware)+' | Variant: TPU'+('\n[?] EXTRA_ENDPOINT | Hardware: V6E1 | Variant: TPU' if fault=='late-extra-runtime' else '')
        if label=='04b-browser-marker':
            code=Path(argv[-1]).read_text();assert adoption['marker_path'] in code and adoption['marker_sha256'] in code
            return {'status':'PASS'},'SYNTHETIC marker missing traceback' if fault=='marker' else json.dumps({'status':'ROOT_BROWSER_MARKER_MATCH','marker_sha256':adoption['marker_sha256']})
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
    if fault in ('gate-hardware','unlisted-hardware'):
        with pytest.raises((PermissionError,ValueError)):
            owner.drive(packet,out,expected,acceptance,remote.sha(acceptance),Path('fixture-python'),identity,api=api,simulated=True,browser_adoption=adoption_path,browser_adoption_sha256=remote.sha(adoption_path))
        assert calls==[]
        return
    result=owner.drive(packet,out,expected,acceptance,remote.sha(acceptance),Path('fixture-python'),identity,api=api,simulated=True,browser_adoption=adoption_path,browser_adoption_sha256=remote.sha(adoption_path))
    assert result['allocation_attempts']==0 and '03-one-allocation' not in calls
    if fault in ('marker','wrong-endpoint','additional-runtime','registration-fail','late-extra-runtime','endpoint-prefix-collision','hardware-mismatch'):
        assert not result['adoption_verified'] and result['remote_termination']=='NOT_RUN_UNPROVEN_BROWSER_IDENTITY'
        assert calls[-1]=='89-remove-provisional' and '91-stop-exact' not in calls and '13-tpu' not in calls
        assert result['status']=='BLOCKED'
    else:
        assert result['adoption_verified'] is True and result['allocation_epoch']==adoption['allocation_epoch']
        assert result['elapsed_lifecycle_seconds']>=100
        assert calls[-4:]==['90-before-stop','91-stop-exact','92-after-stop','93-usage-after']
        assert result['status']==('BLOCKED_CLEANUP' if fault=='stop' else 'FAIL_TPU_NESTED_RECORDS' if fault else 'PASS_TPU_NESTED_RECORDS')
        assert result['whole_archive_verified'] is True and result['local_verifier_exit_code']==(2 if fault in ('numeric','case-error') else 0)
