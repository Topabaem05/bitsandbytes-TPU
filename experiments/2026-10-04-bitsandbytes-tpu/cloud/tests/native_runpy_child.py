"""Actual remote CLI through fresh runpy; only runtime/science steps are explicit doubles."""
import argparse, hashlib, json, os, runpy, sys, time, types
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);parser.add_argument('--cached-transport',action='store_true');a=parser.parse_args();base=a.base.resolve();control=base/'control';packet=base/'payload';records=base/'records';script=control/'remote.py'
assert not any(name in sys.modules for name in ('remote','transport','native_contract','protocol','torch','torch_xla','jax'))
assert str(control) not in sys.path and str(packet/'native') not in sys.path and Path.cwd()!=control
manifest=json.loads((packet/'manifest.json').read_text())
for name,r in manifest['files'].items():
    if name.startswith('cloud/') and name.endswith('.py'):
        p=control/name.removeprefix('cloud/');assert not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],'CONTROL_SOURCE_HASH'
if a.cached_transport:
    cache=types.ModuleType('transport')
    def forbidden(*args,**kwargs):raise AssertionError('AMBIENT_TRANSPORT_MUST_NOT_RUN')
    cache.split=forbidden;sys.modules['transport']=cache
calls=[]
def step(out,label,argv,deadline,limit,**kwargs):
    calls.append(label);assert label in ('10-transfer-source-controls','10-runtime-probe','11-cpu-oracle')
    if label=='10-runtime-probe':(records/'runtime-probe.json').write_text(json.dumps({'status':'PASS_TPU_RUNTIME_PROBE_ONLY','scope':'SYNTHETIC_NO_RUNTIME_OR_DEVICE'}))
    return {'status':'PASS','exit_code':0,'cleanup':{'errors':[]},'scope':'SYNTHETIC_NO_RUNTIME_OR_DEVICE'}
def inject(frame,event,arg):
    if event=='call' and frame.f_code.co_name=='execute' and Path(frame.f_code.co_filename).resolve()==script:
        frame.f_globals['run_step']=step;sys.settrace(None)
    return inject
epoch=json.loads((records/'receipt.json').read_text())['allocation_epoch'];digest=hashlib.sha256((base/'payload.zip').read_bytes()).hexdigest();sys.argv=['remote.py','cpu','--base',str(base),'--packet-sha256',digest,'--allocation-epoch',str(epoch)];sys.settrace(inject)
code=None
try:runpy.run_path(str(script),run_name='__main__')
except SystemExit as error:code=error.code
finally:sys.settrace(None)
receipt=json.loads((records/'receipt.json').read_text());assert calls==['10-transfer-source-controls','10-runtime-probe','11-cpu-oracle']
assert str(control) not in sys.path and str(packet/'native') not in sys.path and not any(name in sys.modules for name in ('torch','torch_xla','jax'))
report={'scope':'FRESH_RUNPY_ACTUAL_REMOTE_CPU_PHASE_SYNTHETIC_CHILD_STEPS','pid':os.getpid(),'pgid':os.getpgid(0),'cwd':str(Path.cwd()),'remote_cli_exit_code':code,'calls':calls,'receipt_status':receipt['status'],'receipt_error':receipt.get('error'),'transport_parts_present':(base/'native-cpu-parts/manifest.json').is_file(),'ambient_transport_fixture':a.cached_transport,'provider_calls':0,'device_science':'NOT_RUN'};(base/'fresh-runpy-result.json').write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(0)
