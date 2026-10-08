"""Portable import and early native recovery controls. All science records are fixtures."""
import json,os,sys,subprocess
from pathlib import Path
import pytest
import native_controls as FIX

HERE=Path(__file__).resolve().parent

def test_native_fresh_runpy_and_recovery(tmp_path,monkeypatch):
    archive=os.environ.get('BNB_TRANSFER_BASE_ARCHIVE')
    if not archive:pytest.skip('Set BNB_TRANSFER_BASE_ARCHIVE to the retained pinned archive.')
    monkeypatch.setattr(FIX,'OUT',tmp_path/'seed');monkeypatch.setattr(FIX,'ARCHIVE',Path(archive).resolve())
    packet,manifest,_=FIX.prepare();assert FIX.chain(packet,manifest,None)['status']=='PASS';FIX.compact_closed_case(FIX.OUT/'correct')
    output=tmp_path/'recovery';proc=subprocess.run([sys.executable,'-B',str(HERE/'native_recovery_controls.py'),'--packet',str(packet),'--seed-archive',str(FIX.OUT/'correct.tar.gz'),'--output',str(output)],capture_output=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),timeout=180)
    (tmp_path/'controls.stdout').write_bytes(proc.stdout);(tmp_path/'controls.stderr').write_bytes(proc.stderr)
    assert proc.returncode==0,proc.stderr.decode(errors='replace');report=json.loads((output/'results.json').read_text());assert report['status']=='PASS'and report['count']==12 and report['provider_calls']==0
