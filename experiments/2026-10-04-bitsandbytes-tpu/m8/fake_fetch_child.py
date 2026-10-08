"""Offline owned child. This fixture has no network path."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--output',required=True)
a=parser.parse_args();out=Path(a.output);request=json.loads(Path(a.request).read_text())
spec=json.loads((out.parent/'fixture.json').read_text());mode=spec['mode']
if mode=='timeout':
    signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)
if mode=='descendant':
    p=subprocess.Popen([sys.executable,'-B','-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)'])
    (out/'descendant-pid.json').write_text(json.dumps({'pid':p.pid,'pgid':os.getpgid(p.pid)}))
    signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)
data=Path(spec['packet']).read_bytes()
if mode=='hash':data=bytes([data[0]^1])+data[1:]
if mode=='truncated':data=data[:-1]
if mode=='extra':data+=b'x'
(out/'packet.zip').write_bytes(data)
child={'pid':os.getpid(),'pgid':os.getpgid(0),'parent_pid':os.getppid(),'process_token':request['process_token'],
       'source_sha256':request['source_sha256'],'http':{'status':200,'bytes':request['packet_bytes'],'sha256':request['packet_sha256']}}
if mode=='pid':child['pid']+=1
(out/'child.json').write_text(json.dumps(child))
