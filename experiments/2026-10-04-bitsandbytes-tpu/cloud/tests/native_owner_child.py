"""Real owner/recovery control with a declared no-service API fixture."""
import argparse, importlib.util, json, shutil, sys
from pathlib import Path
from unittest.mock import patch

parser=argparse.ArgumentParser();parser.add_argument('--cloud',type=Path,required=True);parser.add_argument('--packet',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--case',required=True);parser.add_argument('--fixture-driver',type=Path,required=True);a=parser.parse_args();cloud=a.cloud.resolve();sys.path.insert(0,str(cloud));import owner,remote,native_contract as NC
spec=importlib.util.spec_from_file_location('bounded_owner_fixture',a.fixture_driver.resolve());FIX=importlib.util.module_from_spec(spec);spec.loader.exec_module(FIX);FIX.CLOUD=cloud;FIX.owner=owner;FIX.remote=remote;FIX.OUT=a.output.resolve();FIX.OUT.mkdir(exist_ok=False);FIX.unpack_cpu_fixture(FIX.OUT/'cpu-fixture');packet=a.packet.resolve();manifest=NC.read(packet/'manifest.json');original_drive=owner.drive
case=a.case;fault=case if case not in ('correct',)else None
if case in ('cpu-import-failure','cpu-import-failure-cleanup','cpu-blocked-receipt','native-parent-missing'):
    def drive(*args,api=None,**kwargs):
        closure={name:cell.cell_contents for name,cell in zip(api.__code__.co_freevars,api.__closure__)};receipt=closure['receipt'];records=closure['records']
        def incomplete():
            for name in list(receipt):
                if name.startswith('native_') and not name.startswith('native_cpu_'):receipt.pop(name)
            receipt.update(status='BLOCKED',tpu_status='NOT_RUN',error={'type':'ModuleNotFoundError','message':"No module named 'transport'"},cpu_status='PASS',runtime_status='PASS_TPU_RUNTIME_PROBE_ONLY')
            shutil.rmtree(records/'native');shutil.rmtree(records/'steps/12-native-parent');FIX.write(records/'receipt.json',receipt)
        if case!='native-parent-missing':incomplete()
        def observed(label,argv,timeout):
            if case=='cpu-blocked-receipt' and label=='10-cpu-receipt':
                closure['calls'].append(label);FIX.write(Path(argv[-1]),receipt);return {'status':'PASS'},'SYNTHETIC_COMMAND_SUCCESS_BLOCKED_CPU_RECEIPT'
            if case=='cpu-blocked-receipt' and label=='10-cpu':return api(label,argv,timeout)
            if label==('13-tpu' if case=='native-parent-missing' else '10-cpu'):
                closure['calls'].append(label)
                if case=='native-parent-missing':incomplete();receipt['error']={'type':'SYNTHETIC_PARENT_FAILURE','message':'No terminal native proof'};FIX.write(records/'receipt.json',receipt)
                return {'status':'CLI_FAILED'},'SYNTHETIC_NO_SERVICE_EARLY_FAILURE'
            if case=='cpu-import-failure-cleanup' and label=='91-stop-exact':
                closure['calls'].append(label);return {'status':'CLI_FAILED'},'SYNTHETIC_NO_SERVICE_STOP_FAILURE'
            return api(label,argv,timeout)
        return original_drive(*args,api=observed,**kwargs)
else:drive=original_drive
with patch.object(owner,'drive',drive):row=FIX.chain(packet,manifest,fault)
base=FIX.OUT/(fault or'correct');report=NC.read(base/'host/owner.json')
if case in ('cpu-import-failure','cpu-import-failure-cleanup','cpu-blocked-receipt','native-parent-missing'):
    assert report['whole_archive_verified']is True and report['retrieval']=='COMPLETE_WHOLE_ARCHIVE'
    expected_error={'type':'ValueError','message':'REMOTE_PHASE_BLOCKED:10-cpu'} if case=='cpu-blocked-receipt'else{'type':'RuntimeError','message':'CLI_STEP_BLOCKED:'+('13-tpu' if case=='native-parent-missing'else'10-cpu')}
    assert report['original_error']==expected_error
    assert NC.read(base/'host/recovered/receipt.json')['error']==({'type':'SYNTHETIC_PARENT_FAILURE','message':'No terminal native proof'} if case=='native-parent-missing'else{'type':'ModuleNotFoundError','message':"No module named 'transport'"})
    if cloud.name=='original-cloud':assert report['local_readback_error']=={'type':'KeyError','message':"'recovered_oracle_sha256'"}
    else:
        assert report['local_verifier_status']=='NOT_RUN_INCOMPLETE_NATIVE_RECORDS'and report['native_record_validation']=='NOT_RUN'and'local_readback_error'not in report
        assert not(base/'host/local-verifier-ownership.json').exists()
    assert report['status']==('BLOCKED_CLEANUP'if case=='cpu-import-failure-cleanup'else'BLOCKED')
    assert report['server_empty_observed']and report['usage_zero_observed']
elif case=='correct':assert report['status']=='PASS_BOUNDED_NATIVE_DIAGNOSTIC'and report['local_verifier_exit_code']==0
else:assert report['status']!='PASS_BOUNDED_NATIVE_DIAGNOSTIC'
row.update(original_error=report.get('original_error'),local_readback_error=report.get('local_readback_error'),local_verifier_status=report.get('local_verifier_status'),whole_archive_verified=report.get('whole_archive_verified'),provider_calls=0);FIX.write(FIX.OUT/'result.json',row);FIX.compact_closed_case(base);print(json.dumps(row))
