"""Owned, explicit-interpreter frontend replay for the recovered native verifier."""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import verifier as V
HERE=Path(__file__).resolve().parent
def request(records,deadline):
    return {'phase':'CONVERSION_REPLAY' if records else 'CPU_RUNTIME_PREFLIGHT','cases':records,'sources':{n:V.digest((HERE/n).read_bytes()) for n in ('mosaic_audit.py','mosaic_compatibility.py')},'deadline_epoch':deadline}
def artifacts(prefix,row):
    """Bind original and converted artifacts before independent CPU parsing."""
    for key,suffix,body_suffix in [('original_payload','.original-native-config.json','.original-mosaic-body.bin'),('payload','.native-config.json','.mosaic-body.bin')]:
        payload=row[key];V.require(prefix.with_suffix(suffix).read_text()==payload,'CONVERSION_CONFIG_ARTIFACT')
        body=base64.b64decode(json.loads(payload)['custom_call_config']['body'],validate=True);V.require(prefix.with_suffix(body_suffix).read_bytes()==body,'CONVERSION_BODY_ARTIFACT')
    V.require(json.loads(prefix.with_suffix('.conversion-audit.json').read_text())==row['payload_conversion'],'CONVERSION_AUDIT_ARTIFACT')
def replay(records,python,deadline):
    V.require(python is not None and Path(python).is_absolute() and Path(python).is_file(),'EXPLICIT_PINNED_AUDIT_INTERPRETER_REQUIRED')
    V.require(type(deadline)in(int,float) and math.isfinite(deadline) and time.time()<deadline,'RECOVERY_AUDIT_DEADLINE')
    value=request(records,deadline);raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    env=dict(os.environ,JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None);env.pop('PJRT_DEVICE',None)
    argv=[str(python),'-B',str(HERE/'mosaic_audit.py')];started=time.time()
    # No new session: the external verifier owner can terminate this entire group.
    proc=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,cwd=HERE)
    try:
        group=os.getpgid(proc.pid);V.require(group==os.getpgid(0),'AUDIT_INHERITED_OWNER_GROUP')
        stdout,stderr=proc.communicate(raw,timeout=max(0.001,deadline-time.time()));finished=time.time()
    except BaseException:
        if proc.poll()is None:proc.kill()
        proc.wait(timeout=5);raise
    V.require(len(stdout)<=1024*1024 and len(stderr)<=1024*1024,'AUDIT_OUTPUT_BOUND')
    report=json.loads(stdout);V.require(proc.returncode==0,'INDEPENDENT_CONVERSION_REJECTED:'+str(report))
    V.require(report.get('status')=='REPRODUCED_EXACT_CONVERSION' and report.get('kind')=='INDEPENDENT_MOSAIC_CPU_FRONTEND_AUDIT','AUDIT_TERMINAL')
    V.require(report['pid']==proc.pid and report['parent_pid']==os.getpid() and report['pgid']==group and report['deadline_epoch']==deadline,'AUDIT_OBSERVED_PROCESS_IDENTITY')
    V.require(started<=report['started_epoch']<=report['finished_epoch']<=finished<=deadline,'AUDIT_OBSERVED_INTERVAL')
    V.require(report['request_sha256']==hashlib.sha256(raw).hexdigest() and report['source_pre']==report['source_post']==value['sources'],'AUDIT_REQUEST_SOURCE_BINDING')
    V.require(report['runtime']==report['runtime_post'],'AUDIT_RUNTIME_PRE_POST')
    V.require(set(report['cases'])==set(records),'AUDIT_CASE_SET')
    return {'argv':argv,'python_sha256':V.digest(Path(python).read_bytes()),'pid':proc.pid,'pgid':group,'parent_pid':os.getpid(),'started_epoch':started,'finished_epoch':finished,'deadline_epoch':deadline,'exit_code':proc.returncode,'reaped':proc.poll()is not None,'inherited_group':True,'report':report,'stderr_sha256':V.digest(stderr),'frontend_scope':'PINNED_JAX071_CPU_ONLY'}
