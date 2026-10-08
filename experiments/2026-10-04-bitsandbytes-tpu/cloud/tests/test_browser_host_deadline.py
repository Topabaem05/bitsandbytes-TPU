"""Actual outer Ownership, harmless local child, original absolute deadline."""
import importlib.util,json,subprocess,sys,time
from pathlib import Path
import pytest
WRAPPER=Path(__file__).resolve().parents[2]/'host_run.py'
def load():
 s=importlib.util.spec_from_file_location('browser_host_wrapper',WRAPPER);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_guard_clamps_original_epoch():
 m=load();assert m.guard_seconds({'absolute_deadline_epoch':110},now=100)==70
 assert m.guard_seconds({},now=100)==3660

@pytest.mark.parametrize('deadline',[90,float('inf')])
def test_guard_rejects_expired_or_invalid(deadline):
 with pytest.raises(ValueError):load().guard_seconds({'absolute_deadline_epoch':deadline},now=100)

@pytest.mark.parametrize('expired',[False,True])
def test_actual_host_entry_path(tmp_path,expired):
 output=tmp_path/'host';cfg=tmp_path/'config.json';deadline=time.time()+(-1 if expired else 30)
 cfg.write_text(json.dumps({'host_output':str(output),'argv':[sys.executable,'-B','-c','pass'],'absolute_deadline_epoch':deadline}))
 p=subprocess.run([sys.executable,'-B',str(WRAPPER),str(cfg)],capture_output=True,text=True,timeout=10)
 if expired:assert p.returncode!=0 and not output.exists()
 else:
  assert p.returncode==0;rec=json.loads((output/'host-finalization.json').read_text());assert 0<rec['outer_guard_seconds']<=90 and rec['absolute_deadline_epoch']==deadline and rec['cleanup']['errors']==[]
