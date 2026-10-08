"""Fresh remote CPU CLI boundary; child/runtime computations are explicit synthetic doubles."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import socket
import sys
import types

p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--cached-transport',action='store_true');a=p.parse_args()
base=a.base.resolve();control=base/'control';records=base/'records';packet=base/'payload';script=control/'remote.py'
assert str(control) not in sys.path and Path.cwd()!=control
assert not any(name in sys.modules for name in ('remote','transport','torch','torch_xla','jax'))
manifest=json.loads((packet/'manifest.json').read_text())
for name,row in manifest['files'].items():
    if name.startswith('cloud/') and name.endswith('.py'):
        source=control/name.removeprefix('cloud/');assert hashlib.sha256(source.read_bytes()).hexdigest()==row['sha256'],'CONTROL_SOURCE_HASH'
def forbidden(*args,**kwargs):raise AssertionError('NETWORK_OR_AMBIENT_TRANSPORT_FORBIDDEN')
socket.socket.connect=forbidden
if a.cached_transport:
    cache=types.ModuleType('transport');cache.split=forbidden;sys.modules['transport']=cache
calls=[]
def step(output,label,argv,deadline,limit,**kwargs):
    calls.append(label);assert label in ('10-transfer-source-controls','10-runtime-probe','11-cpu-oracle')
    return {'status':'PASS','exit_code':0,'cleanup':{'errors':[]},'scope':'SYNTHETIC_NO_RUNTIME_OR_DEVICE'}
def trace(frame,event,arg):
    if event=='call' and frame.f_code.co_name=='execute' and Path(frame.f_code.co_filename).resolve()==script:
        frame.f_globals['run_step']=step;sys.settrace(None)
    return trace
receipt=json.loads((records/'receipt.json').read_text());epoch=receipt['allocation_epoch'];digest=hashlib.sha256((base/'payload.zip').read_bytes()).hexdigest()
sys.argv=['remote.py','cpu','--base',str(base),'--packet-sha256',digest,'--allocation-epoch',str(epoch)]
sys.settrace(trace);code=None
try:runpy.run_path(str(script),run_name='__main__')
except SystemExit as error:code=error.code
finally:sys.settrace(None)
receipt=json.loads((records/'receipt.json').read_text())
assert calls==['10-transfer-source-controls','10-runtime-probe','11-cpu-oracle']
assert str(control) not in sys.path and not any(name in sys.modules for name in ('torch','torch_xla','jax'))
if a.cached_transport:assert sys.modules['transport'] is cache and cache.split is forbidden
else:assert 'transport' not in sys.modules
report={'scope':'SYNTHETIC_CHILD_STEPS_REAL_REMOTE_CPU_CLI','pid':os.getpid(),'pgid':os.getpgid(0),
        'remote_exit_code':code,'receipt_status':receipt['status'],'receipt_error':receipt.get('error'),
        'parts_present':(base/'primitive-cpu-parts/manifest.json').is_file(),'provider_calls':0,'device_science':'NOT_RUN'}
(base/'fresh-runpy-result.json').write_text(json.dumps(report,indent=2)+'\n')
