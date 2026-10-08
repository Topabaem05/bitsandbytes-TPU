"""Exercise the generated owner runpy snippet with isolated sibling loading."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import owner
import remote
import test_nested_cloud as N
import test_nested_state_cloud as NS
import test_transfer_cloud as F

state_packet=NS.state_packet
base_archive=F.base_archive
HERE=Path(__file__).resolve().parents[1]
OLD="""from nested_contract import (NESTED_BINDINGS,NESTED_SCOPE,NESTED_SOURCE_SHA,NESTED_PLUGIN_MANIFEST_SHA,
    NESTED_FILES,verify_nested_manifest,verify_nested_payload,verify_nested_wheel,nested_cli)
"""
NEW="""# runpy.run_path does not add this admitted control directory to sys.path.
# Load the exact sibling without consulting or replacing installed module caches.
_contract_spec = importlib.util.spec_from_file_location('_bnb_admitted_nested_contract', HERE / 'nested_contract.py')
_contract = importlib.util.module_from_spec(_contract_spec)
_contract_spec.loader.exec_module(_contract)
for _name in ('NESTED_BINDINGS', 'NESTED_SCOPE', 'NESTED_SOURCE_SHA', 'NESTED_PLUGIN_MANIFEST_SHA',
              'NESTED_FILES', 'verify_nested_manifest', 'verify_nested_payload', 'verify_nested_wheel', 'nested_cli'):
    globals()[_name] = getattr(_contract, _name)
"""


@pytest.mark.parametrize('mode',['repaired','cached-module','bad-source'])
def test_nested_state_generated_operation_without_sibling_path(state_packet,tmp_path,monkeypatch,mode):
    """The full owner generates code; only its remote base path is adapted locally."""
    packet,manifest,expected=state_packet
    neutral=tmp_path/'neutral';neutral.mkdir();base=tmp_path/'remote-base';base.mkdir()
    out=tmp_path/'host';identity=tmp_path/'identity.json';identity.write_text('{}')
    C=N.C
    gate={'status':'ACTUAL_DISPATCH_AUTHORIZED','packet_sha256':expected,'driver_sha256':remote.sha(HERE/'owner.py'),
        'output':str(out.resolve()),'plugin_source_manifest_sha256':manifest['plugin_source_manifest_sha256'],
        'runtime_lock_sha256':manifest['runtime_lock_sha256'],'budget':manifest['budget'],'one_allocation_only':True,
        'provider_or_solver':'FORBIDDEN','cli_identity_sha256':remote.sha(identity),'experiment':'nested-state-8','precision':'highest',
        'nested_scope':C.NESTED_SCOPE,'nested_source_variant':'nested-v1',
        'nested_state_scope':NS.S.NESTED_STATE_SCOPE,'nested_state_variant':'nested-state-v1',
        **{field:manifest[field] for field in (*remote.TRANSFER_BINDINGS,*C.NESTED_BINDINGS,*NS.S.NESTED_STATE_BINDINGS)}}
    acceptance=tmp_path/'gate.json';F.write(acceptance,gate)
    monkeypatch.setattr(owner,'verify_cli_identity',lambda *args:None)
    calls=[];sessions=[];observed=[]
    def api(label,argv,timeout):
        calls.append(label)
        if label=='03-one-allocation':sessions.append(argv[argv.index('-s')+1])
        if label in ('04-sessions-after','90-before-stop'):return {'status':'PASS'},'['+sessions[0]+']\nHardware: V6E1'
        if argv[0]=='upload':
            target=base/argv[-1].removeprefix('content/bnb-tpu-first/');target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path(argv[-2]),target)
        if label=='08-unpack':
            code=Path(argv[-1]).read_text();source=base/'control/remote.py'
            assert code.count("b=Path('/content/bnb-tpu-first')")==1
            code=code.replace("b=Path('/content/bnb-tpu-first')",f'b=Path({str(base)!r})')
            if mode=='old':
                text=source.read_text();assert text.count(NEW)==1;legacy=text.replace(NEW,OLD)
                legacy_sha=hashlib.sha256(legacy.encode()).hexdigest()
                source.write_text(legacy)
                # Inject the retained legacy import boundary into the current control source.
                # The exact historical source replay remains in the accepted bootstrap records.
                code=code.replace(manifest['files']['cloud/remote.py']['sha256'],legacy_sha)
            if mode=='bad-source':
                contract=base/'control/nested_state_contract.py';contract.write_bytes(contract.read_bytes()+b'\nCORRUPT_SOURCE\n')
            runner=neutral/'runner.py'
            runner.write_text("import sys,types,socket\nfrom pathlib import Path\n"
                "assert 'nested_contract' not in sys.modules and 'nested_state_contract' not in sys.modules\n"
                f"assert {str(base/'control')!r} not in sys.path\n"
                "def forbidden(*args,**kwargs):raise AssertionError('PROVIDER_CALL_FORBIDDEN')\n"
                "socket.socket.connect=forbidden\n"
                +( "cached=types.ModuleType('nested_contract');sys.modules['nested_contract']=cached;state_cached=types.ModuleType('nested_state_contract');sys.modules['nested_state_contract']=state_cached\n" if mode=='cached-module' else '')
                +"try:\n exec(compile(Path(sys.argv[1]).read_text(),sys.argv[1],'exec'),{'__name__':'__main__'})\n"
                +"except SystemExit:\n"
                +(" assert sys.modules['nested_contract'] is cached and sys.modules['nested_state_contract'] is state_cached\n" if mode=='cached-module' else " assert 'nested_contract' not in sys.modules\n")
                +" raise\n")
            snippet=neutral/'generated-operation.py';snippet.write_text(code)
            child=subprocess.run([sys.executable,'-I','-B',str(runner),str(snippet)],cwd=neutral,capture_output=True,text=True,timeout=15)
            observed.append(child)
            (tmp_path/'operation.stdout').write_text(child.stdout);(tmp_path/'operation.stderr').write_text(child.stderr)
            # A notebook CLI may report success while the executed code raises.
            return {'status':'PASS'},child.stdout+child.stderr
        if label=='09-install':return {'status':'PASS'},'SYNTHETIC notebook traceback with CLI exit zero; installation was not executed'
        if argv[0]=='download':return {'status':'PASS'},'SYNTHETIC missing receipt despite CLI zero'
        return {'status':'PASS'},'Active assignments: 0\nUsage rate: 0.00/hr' if argv==['usage'] else 'No active sessions found on server.'
    result=owner.drive(packet,out,expected,acceptance,remote.sha(acceptance),Path('fixture-python'),identity,api=api,simulated=True)
    assert len(observed)==1
    child=observed[0]
    if mode=='old':
        assert child.returncode!=0 and "ModuleNotFoundError: No module named 'nested_contract'" in child.stderr
        assert not (base/'payload').exists()
    elif mode=='bad-source':
        assert child.returncode!=0 and 'CONTROL_SOURCE_HASH' in child.stderr
        assert 'CORRUPT_SOURCE' not in child.stderr and not (base/'payload').exists()
    else:
        assert child.returncode==0 and (base/'payload/manifest.json').is_file()
        assert remote.sha(base/'payload/probe_nested.py')==C.NESTED_PROBE_SHA
        assert (base/'control/nested_contract.py').read_bytes()==(packet/'cloud/nested_contract.py').read_bytes()
    assert result['status']=='BLOCKED' and result['original_error']['type']=='FileNotFoundError'
    assert '09-install-receipt' in calls and '10-cpu' not in calls and '13-tpu' not in calls
    assert calls[-4:]==['90-before-stop','91-stop-exact','92-after-stop','93-usage-after']
