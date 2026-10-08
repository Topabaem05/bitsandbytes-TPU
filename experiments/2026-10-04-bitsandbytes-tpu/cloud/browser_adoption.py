"""Admit one root-created browser runtime through official session APIs. No allocation."""
import hashlib
import json
import math
from pathlib import Path
import re
import time

FLAG='--adopt-root-browser-v1'
REMOVE_FLAG='--remove-browser-provisional-v1'
MODE='ADOPT_ROOT_CREATED_BROWSER_RUNTIME'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(ok,name):
    if not ok:raise ValueError(name)


def record(path,expected,*,now=None,check_time=True):
    path=Path(path);require(not path.is_symlink() and path.is_file() and sha(path)==expected,'BROWSER_ADOPTION_RECORD_SHA')
    r=json.loads(path.read_text());fields={'format','status','endpoint','session','hardware','variant','authuser','allocation_epoch','observed_epoch',
        'packet_sha256','driver_sha256','cli_identity_sha256','marker_path','marker_sha256'}
    require(type(r) is dict and set(r)==fields,'BROWSER_ADOPTION_RECORD_FIELDS')
    require(r['format']=='bnb-tpu.browser-adoption.v1' and r['status']=='ROOT_CREATED_BROWSER_RUNTIME_READY' and
        r['hardware']=='V6E1' and r['variant']=='TPU' and r['authuser']=='0','BROWSER_ADOPTION_PROFILE')
    require(isinstance(r['endpoint'],str) and re.fullmatch('[A-Za-z0-9_-]{1,256}',r['endpoint']) and
        isinstance(r['session'],str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,127}',r['session']),'BROWSER_ADOPTION_IDENTITY')
    require(isinstance(r['marker_path'],str) and re.fullmatch('/tmp/bnb-tpu-root-browser-[0-9a-f]{32}\\.json',r['marker_path']),'BROWSER_ADOPTION_MARKER_PATH')
    require(all(isinstance(r[k],str) and re.fullmatch('[0-9a-f]{64}',r[k]) for k in ('packet_sha256','driver_sha256','cli_identity_sha256','marker_sha256')),'BROWSER_ADOPTION_BINDINGS')
    require(all(type(r[k]) in (int,float) and math.isfinite(r[k]) for k in ('allocation_epoch','observed_epoch')),'BROWSER_ADOPTION_EPOCH')
    if check_time:
        now=time.time() if now is None else now
        require(r['allocation_epoch']<=r['observed_epoch']<=now and now-r['observed_epoch']<=180 and now-r['allocation_epoch']<3600-660,'BROWSER_ADOPTION_STALE_OR_EXPIRED')
    return r


def register(common,r,*,now=None):
    """Register proxy fields from one fresh official listing; never assign or log tokens."""
    from colab_cli.state import SessionState
    now=time.time() if now is None else now
    require(common.state.store.list()=={},'BROWSER_ADOPTION_LOCAL_STORE_NOT_EMPTY')
    assignments=common.state.client.list_assignments()
    require(len(assignments)==1,'BROWSER_ADOPTION_EXACT_ONE_ASSIGNMENT')
    a=assignments[0]
    require(a.endpoint==r['endpoint'] and a.accelerator.value==r['hardware'] and a.variant.name==r['variant'],'BROWSER_ADOPTION_ASSIGNMENT_MISMATCH')
    info=a.runtime_proxy_info;expiry=info.expires_at()
    require(isinstance(info.token,str) and bool(info.token) and isinstance(info.url,str) and info.url.startswith('https://') and expiry.timestamp()>now+120,'BROWSER_ADOPTION_PROXY_EXPIRED')
    s=SessionState(name=r['session'],endpoint=a.endpoint,token=info.token,url=info.url,token_expires_at=expiry,
        variant=a.variant.name,accelerator=a.accelerator.value,machine_shape=a.machine_shape.name)
    common.state.store.add(s)
    loaded=common.state.store.get(r['session'])
    require(loaded is not None and loaded.endpoint==r['endpoint'] and loaded.token==info.token,'BROWSER_ADOPTION_STORE_READBACK')
    return {'status':'PROVISIONAL_BROWSER_RUNTIME_REGISTERED','endpoint_sha256':hashlib.sha256(r['endpoint'].encode()).hexdigest(),
        'session':r['session'],'hardware':r['hardware'],'allocation_epoch':r['allocation_epoch'],'proxy_expires_epoch':expiry.timestamp()}


def remove_provisional(common,r):
    s=common.state.store.get(r['session'])
    if s is not None:
        require(s.endpoint==r['endpoint'],'BROWSER_PROVISIONAL_REMOVE_ENDPOINT')
        common.state.store.remove(r['session'])
    require(common.state.store.get(r['session']) is None,'BROWSER_PROVISIONAL_REMOVE_READBACK')
    return {'status':'PROVISIONAL_LOCAL_REGISTRATION_REMOVED','remote_termination':'NOT_RUN'}


def command(argv,common):
    require(len(argv)==3 and argv[0] in (FLAG,REMOVE_FLAG),'EXACT_BROWSER_ADOPTION_OPT_IN_REQUIRED')
    r=record(argv[1],argv[2],check_time=argv[0]==FLAG)
    return register(common,r) if argv[0]==FLAG else remove_provisional(common,r)
