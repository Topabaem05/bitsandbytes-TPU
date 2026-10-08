"""Portable M6 packet/ownership/recovery controls. No TPU or provider calls."""
import json,os
from pathlib import Path
import subprocess,sys
import pytest

HERE=Path(__file__).resolve().parent

def test_native_portable_contract_and_owner_chain(tmp_path):
    value=os.environ.get('BNB_TRANSFER_BASE_ARCHIVE')
    if not value:pytest.skip('Set BNB_TRANSFER_BASE_ARCHIVE to the retained pinned upstream archive.')
    output=tmp_path/'owner-controls'
    commands=[('native_controls.py',['--upstream-archive',value,'--output',str(output)],output/'control-results.json',13),
              ('native_additional_controls.py',['--packet',str(output/'packet-fixture'),'--seed-archive',str(output/'correct.tar.gz'),'--output',str(tmp_path/'additional')],tmp_path/'additional/results.json',12),
              ('native_phase_controls.py',['--packet',str(output/'packet-fixture'),'--seed-archive',str(output/'correct.tar.gz'),'--output',str(tmp_path/'phases')],tmp_path/'phases/results.json',8)]
    for script,args,result,count in commands:
        proc=subprocess.run([sys.executable,'-B',str(HERE/script),*args],capture_output=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),timeout=180)
        (tmp_path/(script+'.stdout')).write_bytes(proc.stdout);(tmp_path/(script+'.stderr')).write_bytes(proc.stderr)
        assert proc.returncode==0,proc.stderr.decode(errors='replace')
        report=json.loads(result.read_text());assert report['status']=='PASS' and len(report['controls'])==count and report['provider_calls']==0
