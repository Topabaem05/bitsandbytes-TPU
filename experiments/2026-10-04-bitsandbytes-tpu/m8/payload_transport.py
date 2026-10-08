"""Exact public-commit ZIP transport. This module does not run science."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import ssl
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import uuid

TRANSPORT_MODE='PUBLIC_COMMIT_ZIP_V1'
TRANSPORT_MEMBER='records/payload-transport.json'
PACKET_BYTES=983676
PACKET_SHA='654641d70a43bb1154097301b8ba41dfe5d2f95d16a2dd0e01c181e69eead80a'
HOST='raw.githubusercontent.com'
REPOSITORY='Topabaem05/bitsandbytes-TPU'
PACKET_PATH='experiments/2026-10-04-bitsandbytes-tpu/m8/payloads/api42-transfer4-v1.zip'
WORK_RESERVE=660
FETCH_LIMIT=60
CLEANUP_LIMIT=5
MAX_EXPORT_BYTES=100*1024*1024

def need(condition, code):
    if not condition:raise ValueError(code)

def digest(data):return hashlib.sha256(data).hexdigest()

def write_json(path, value):
    data=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    temporary=path.with_suffix(path.suffix+'.tmp')
    with open(temporary,'wb') as f:
        os.chmod(temporary,0o600);f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)

def packet_url(commit):
    need(type(commit)is str and re.fullmatch('[0-9a-f]{40}',commit),'FULL_COMMIT_REQUIRED')
    return 'https://'+HOST+'/'+REPOSITORY+'/'+commit+'/'+PACKET_PATH

class RejectRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        error=ValueError('PAYLOAD_REDIRECT_REJECTED')
        try:fp.close()
        except BaseException as close_error:
            try:error.add_note('PAYLOAD_REDIRECT_CLOSE_FAILURE:'+type(close_error).__name__)
            except BaseException:pass
        raise error

def https_opener():
    context=ssl.create_default_context()
    need(context.check_hostname and context.verify_mode==ssl.CERT_REQUIRED,'TLS_REQUIRED')
    context.set_alpn_protocols(['http/1.1'])
    # Explicitly disable proxy and authentication handlers.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}),
             urllib.request.HTTPSHandler(context=context),RejectRedirect())

def read_https(commit, deadline, *, opener=None):
    url=packet_url(commit)
    need(type(deadline) in (float,int) and math.isfinite(deadline),'FETCH_DEADLINE')
    remaining=deadline-time.time();need(remaining>0,'FETCH_DEADLINE')
    request=urllib.request.Request(url,headers={'Accept-Encoding':'identity'},method='GET')
    response=None;original_error=None
    try:
        response=(opener or https_opener()).open(request,timeout=min(10,remaining))
        need(type(response.status)is int and response.status==200,'PAYLOAD_HTTP_STATUS')
        need(response.geturl()==url,'PAYLOAD_URL_CHANGED')
        length=response.headers.get('Content-Length')
        need(length is None or length==str(PACKET_BYTES),'PAYLOAD_CONTENT_LENGTH')
        need(response.headers.get('Content-Encoding','identity')=='identity','PAYLOAD_CONTENT_ENCODING')
        chunks=[];size=0;hasher=hashlib.sha256()
        while True:
            need(time.time()<deadline,'FETCH_DEADLINE')
            block=response.read(min(65536,PACKET_BYTES-size+1))
            need(type(block)is bytes,'PAYLOAD_STREAM_TYPE')
            if not block:break
            size+=len(block);need(size<=PACKET_BYTES,'PAYLOAD_STREAM_CAP')
            hasher.update(block);chunks.append(block)
        need(time.time()<deadline,'FETCH_DEADLINE')
        need(size==PACKET_BYTES,'PAYLOAD_TRUNCATED')
        need(hasher.hexdigest()==PACKET_SHA,'PAYLOAD_SHA')
        return b''.join(chunks),{'status':200,'bytes':size,'sha256':hasher.hexdigest()}
    except BaseException as error:
        original_error=error
        if response is None and isinstance(error,urllib.error.HTTPError):response=error
        raise
    finally:
        if response is not None:
            try:response.close()
            except BaseException as error:
                if original_error is None:raise
                try:original_error.add_note('PAYLOAD_RESPONSE_CLOSE_FAILURE:'+type(error).__name__)
                except BaseException:pass

def group_absent(pgid):
    try:os.killpg(pgid,0);return False
    except ProcessLookupError:return True

def close_owned(process, deadline):
    """Close only the supplied direct-child handle; retain probe uncertainty."""
    result={'exit_code':None,'reaped':False,'group_absent':False,'errors':[]}
    end=min(deadline,time.time()+CLEANUP_LIMIT)
    mono_end=time.monotonic()+max(0,end-time.time())
    def remaining():return max(0,min(end-time.time(),mono_end-time.monotonic()))
    def failure(stage,error):result['errors'].append({'stage':stage,'type':type(error).__name__})
    def poll():
        try:return process.poll()
        except BaseException as error:failure('poll',error);return None
    def absent():
        try:return group_absent(process.pid)
        except BaseException as error:failure('group_probe',error);return False
    def send(sig):
        try:os.killpg(process.pid,sig)
        except ProcessLookupError:pass
        except BaseException as error:failure('signal_'+str(sig),error)
    def wait(limit,stage,expected_timeout=False):
        try:process.communicate(timeout=max(0,min(limit,remaining())))
        except subprocess.TimeoutExpired as error:
            if not expected_timeout:failure(stage,error)
        except BaseException as error:failure(stage,error)
    if poll() is None or not absent():
        send(signal.SIGTERM);wait(0.25,'term_wait',True)
    if poll() is None or not absent():send(signal.SIGKILL)
    wait(remaining(),'reap')
    result['reaped']=poll() is not None
    try:result['exit_code']=process.returncode
    except BaseException as error:failure('exit_code',error)
    while remaining()>0:
        before=len(result['errors']);observed=absent()
        if observed:result['group_absent']=True;break
        if len(result['errors'])>before:break
        time.sleep(min(0.01,remaining()))
    if not result['group_absent']:result['group_absent']=absent()
    # A failed absence probe cannot authorize closure, even after a later signal.
    if any(e['stage']=='group_probe' for e in result['errors']):result['group_absent']=False
    return result

def payload_child(request_path, directory):
    request=json.loads(Path(request_path).read_text());directory=Path(directory)
    need(set(request)=={'commit','url','packet_bytes','packet_sha256','deadline_epoch',
                       'fetch_deadline_epoch','parent_pid','process_token','source_sha256'},'FETCH_REQUEST_SCHEMA')
    need(request['url']==packet_url(request['commit']) and request['packet_bytes']==PACKET_BYTES
         and request['packet_sha256']==PACKET_SHA,'FETCH_REQUEST_SOURCE')
    need(re.fullmatch('[0-9a-f]{32}',request['process_token']),'FETCH_TOKEN')
    need(request['parent_pid']==os.getppid(),'FETCH_PARENT')
    need(request['fetch_deadline_epoch']<=request['deadline_epoch']-WORK_RESERVE-CLEANUP_LIMIT,'FETCH_RESERVE')
    source=Path(__file__).read_bytes();need(digest(source)==request['source_sha256'],'FETCH_SOURCE_PRE')
    data,http=read_https(request['commit'],request['fetch_deadline_epoch'])
    need(digest(Path(__file__).read_bytes())==request['source_sha256'],'FETCH_SOURCE_POST')
    with open(directory/'packet.zip','xb') as f:os.chmod(directory/'packet.zip',0o600);f.write(data)
    write_json(directory/'child.json',{'pid':os.getpid(),'pgid':os.getpgid(0),'parent_pid':os.getppid(),
        'process_token':request['process_token'],'source_sha256':request['source_sha256'],'http':http})

def fetch_packet(directory, commit, deadline, *, worker_command=None):
    url=packet_url(commit)
    need(type(deadline) in (float,int) and math.isfinite(deadline),'ORIGINAL_DEADLINE')
    directory=Path(directory);directory.mkdir(mode=0o700,parents=True,exist_ok=False)
    started=time.time();remaining=deadline-WORK_RESERVE-started
    budget=min(FETCH_LIMIT,remaining-CLEANUP_LIMIT);need(budget>0,'NOT_RUN_FETCH_BUDGET')
    source_path=Path(__file__).resolve();source_sha=digest(source_path.read_bytes())
    command=worker_command or [sys.executable,'-B',str(source_path),'--payload-child']
    token=uuid.uuid4().hex
    request={'commit':commit,'url':url,'packet_bytes':PACKET_BYTES,'packet_sha256':PACKET_SHA,
        'deadline_epoch':deadline,'fetch_deadline_epoch':min(deadline-WORK_RESERVE-CLEANUP_LIMIT,started+FETCH_LIMIT),
        'parent_pid':os.getpid(),'process_token':token,'source_sha256':source_sha}
    write_json(directory/'request.json',request)
    record={'kind':'M8_PUBLIC_COMMIT_TRANSPORT','mode':TRANSPORT_MODE,
        'execution_mode':'SYNTHETIC_FIXTURE' if worker_command else 'ACTUAL_REQUIRED',
        'commit':commit,'url':url,'packet_bytes':PACKET_BYTES,'packet_sha256':PACKET_SHA,
        'deadline_epoch':deadline,'work_reserve_seconds':WORK_RESERVE,'fetch_limit_seconds':FETCH_LIMIT,
        'cleanup_limit_seconds':CLEANUP_LIMIT,'fetch_deadline_epoch':request['fetch_deadline_epoch'],
        'started_epoch':started,'body_admitted_epoch':None,'finished_epoch':None,'parent_pid':os.getpid(),'process_token':token,
        'source_pre':source_sha,'source_post':None,'status':'LAUNCHING','child':None,
        'launch':{'argv':command,'launched_pid':None,'pid':None,'pgid':None,'exit_code':None,
                  'reaped':False,'group_absent':True},'error_type':None,'cleanup_errors':[],'provider_closure':'UNCONFIRMED'}
    write_json(directory/'receipt.json',record)
    process=None;original_error=None;data=None
    try:
        process=subprocess.Popen(command+['--request',str(directory/'request.json'),'--output',str(directory)],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        record['launch'].update(launched_pid=process.pid,pid=process.pid,pgid=process.pid,group_absent=False)
        write_json(directory/'receipt.json',record)  # Durable before wait.
        try:process.communicate(timeout=max(0.001,request['fetch_deadline_epoch']-time.time()))
        except subprocess.TimeoutExpired:raise TimeoutError('PAYLOAD_FETCH_TIMEOUT')
        need(process.returncode==0,'PAYLOAD_CHILD_FAILED')
        child=json.loads((directory/'child.json').read_text());record['child']=child
        need(child['pid']==child['pgid']==process.pid and child['parent_pid']==os.getpid(), 'FETCH_CHILD_IDENTITY')
        need(child['process_token']==token and child['source_sha256']==source_sha,'FETCH_CHILD_SEAL')
        need(child['http']=={'status':200,'bytes':PACKET_BYTES,'sha256':PACKET_SHA},'FETCH_CHILD_HTTP')
        path=directory/'packet.zip';need(path.is_file() and not path.is_symlink(),'FETCH_FILE_TYPE')
        need(path.stat().st_size==PACKET_BYTES,'FETCH_FILE_BYTES')
        data=path.read_bytes();need(digest(data)==PACKET_SHA,'FETCH_FILE_SHA')
        record['body_admitted_epoch']=time.time()
        need(record['body_admitted_epoch']<=request['fetch_deadline_epoch'],'FETCH_FINISHED_LATE')
    except BaseException as error:original_error=error
    finally:
        if process is not None:
            try:cleanup=close_owned(process,deadline-WORK_RESERVE)
            except BaseException as error:
                cleanup={'exit_code':None,'reaped':False,'group_absent':False,
                         'errors':[{'stage':'cleanup_internal','type':type(error).__name__}]}
            record['cleanup_errors']=cleanup.pop('errors')
            record['launch'].update(cleanup)
            if (record['cleanup_errors'] or not(record['launch']['reaped'] and record['launch']['group_absent'])) and original_error is None:
                original_error=RuntimeError('FETCH_CLEANUP_UNCERTAIN')
        try:record['source_post']=digest(source_path.read_bytes())
        except BaseException as error:
            record['cleanup_errors'].append({'stage':'source_post','type':type(error).__name__})
            if original_error is None:original_error=error
        record['finished_epoch']=time.time()
        if record['source_post']!=source_sha and original_error is None:original_error=ValueError('FETCH_POST_SOURCE_CHANGED')
        if record['finished_epoch']>deadline-WORK_RESERVE and original_error is None:original_error=TimeoutError('FETCH_CLEANUP_LATE')
        record.update(status='FAIL' if original_error else 'BYTE_EXACT_CLOSED',
                      error_type=type(original_error).__name__ if original_error else None)
        try:write_json(directory/'receipt.json',record)
        except BaseException:
            if original_error is not None:raise original_error
            raise
    if original_error:raise original_error
    return data,record

def audit_record(record, commit, deadline, wrapper_sha, *, allow_synthetic=False):
    keys={'kind','mode','execution_mode','commit','url','packet_bytes','packet_sha256','deadline_epoch',
          'work_reserve_seconds','fetch_limit_seconds','cleanup_limit_seconds','fetch_deadline_epoch',
          'started_epoch','body_admitted_epoch','finished_epoch','parent_pid','process_token','source_pre','source_post',
          'status','child','launch','error_type','cleanup_errors','provider_closure'}
    need(type(record)is dict and set(record)==keys,'TRANSPORT_RECORD_SCHEMA')
    need(type(record['launch'])is dict and set(record['launch'])=={'argv','launched_pid','pid','pgid','exit_code','reaped','group_absent'},'TRANSPORT_LAUNCH_SCHEMA')
    need(type(record['child'])is dict and set(record['child'])=={'pid','pgid','parent_pid','process_token','source_sha256','http'},'TRANSPORT_CHILD_SCHEMA')
    need(record['kind']=='M8_PUBLIC_COMMIT_TRANSPORT' and record['mode']==TRANSPORT_MODE,'TRANSPORT_MODE')
    need(record['execution_mode']=='ACTUAL_REQUIRED' or (allow_synthetic and record['execution_mode']=='SYNTHETIC_FIXTURE'),'TRANSPORT_SYNTHETIC')
    need(record['commit']==commit and record['url']==packet_url(commit),'TRANSPORT_REF')
    need(record['packet_bytes']==PACKET_BYTES and record['packet_sha256']==PACKET_SHA,'TRANSPORT_PACKET')
    need(record['deadline_epoch']==deadline and record['work_reserve_seconds']==WORK_RESERVE
         and record['fetch_limit_seconds']==FETCH_LIMIT and record['cleanup_limit_seconds']==CLEANUP_LIMIT,'TRANSPORT_BUDGET')
    need(record['source_pre']==record['source_post']==wrapper_sha,'TRANSPORT_SOURCE')
    need(record['status']=='BYTE_EXACT_CLOSED' and record['error_type'] is None,'TRANSPORT_TERMINAL')
    need(record['cleanup_errors']==[],'TRANSPORT_CLEANUP_ERROR')
    need(record['provider_closure']=='UNCONFIRMED','NO_PROVIDER_CLOSURE_CLAIM')
    for key in ['started_epoch','body_admitted_epoch','finished_epoch','fetch_deadline_epoch']:
        need(type(record[key])in(int,float) and math.isfinite(record[key]),'TRANSPORT_TIME')
    need(record['started_epoch']<record['fetch_deadline_epoch']<=min(record['started_epoch']+FETCH_LIMIT,deadline-WORK_RESERVE-CLEANUP_LIMIT),'TRANSPORT_FETCH_BOUND')
    need(record['started_epoch']<=record['body_admitted_epoch']<=record['fetch_deadline_epoch'],'TRANSPORT_ADMISSION_BOUND')
    need(record['body_admitted_epoch']<=record['finished_epoch']<=deadline-WORK_RESERVE,'TRANSPORT_CLOSE_BOUND')
    need(re.fullmatch('[0-9a-f]{32}',record['process_token']),'TRANSPORT_TOKEN')
    launch=record['launch'];child=record['child'];pid=launch['launched_pid']
    need(type(pid)is int and pid>0 and launch['pid']==launch['pgid']==child['pid']==child['pgid']==pid,'TRANSPORT_PID')
    need(type(record['parent_pid'])is int and record['parent_pid']>0 and child['parent_pid']==record['parent_pid'] and pid!=record['parent_pid'],'TRANSPORT_PARENT')
    need(type(launch['argv'])is list and all(type(v)is str for v in launch['argv']),'TRANSPORT_ARGV')
    if record['execution_mode']=='ACTUAL_REQUIRED':
        need(len(launch['argv'])==4 and launch['argv'][1]=='-B' and launch['argv'][3]=='--payload-child','TRANSPORT_ACTUAL_ARGV')
    need(type(launch['exit_code'])is int and launch['exit_code']==0 and launch['reaped'] is True and launch['group_absent'] is True,'TRANSPORT_CLOSURE')
    need(child['source_sha256']==wrapper_sha and child['process_token']==record['process_token'],'TRANSPORT_CHILD_SEAL')
    need(child['http']=={'status':200,'bytes':PACKET_BYTES,'sha256':PACKET_SHA},'TRANSPORT_HTTP')

def append_transport(output, binding, record, scientific_names):
    output=Path(output);archive=output/'m8-evidence.tar';manifest_path=output/'m8-evidence-manifest.json'
    manifest=json.loads(manifest_path.read_text());original=archive.read_bytes()
    need(manifest['binding']==binding and manifest['archive']=={'file':archive.name,'bytes':len(original),'sha256':digest(original)},'ORIGINAL_EXPORT_BINDING')
    need(set(manifest['members'])==set(scientific_names),'ORIGINAL_SCIENCE_INVENTORY')
    raw=(json.dumps(record,sort_keys=True,allow_nan=False)+'\n').encode();buffer=io.BytesIO();seen=set()
    with tarfile.open(fileobj=io.BytesIO(original),mode='r:') as source,tarfile.open(fileobj=buffer,mode='w') as target:
        for member in source:
            need(member.isfile() and member.name in scientific_names and member.name not in seen,'ORIGINAL_EXPORT_MEMBER')
            data=source.extractfile(member).read();need(manifest['members'][member.name]=={'bytes':len(data),'sha256':digest(data)},'ORIGINAL_EXPORT_SHA')
            target.addfile(member,io.BytesIO(data));seen.add(member.name)
        need(seen==set(scientific_names),'ORIGINAL_EXPORT_MISSING')
        member=tarfile.TarInfo(TRANSPORT_MEMBER);member.size=len(raw);member.mode=0o600;member.mtime=0
        target.addfile(member,io.BytesIO(raw))
    data=buffer.getvalue();need(len(data)<=MAX_EXPORT_BYTES,'FINAL_TRANSPORT_TAR_BYTES')
    manifest['members'][TRANSPORT_MEMBER]={'bytes':len(raw),'sha256':digest(raw)}
    manifest['archive']={'file':archive.name,'bytes':len(data),'sha256':digest(data)}
    temporary=archive.with_suffix('.tmp');temporary.write_bytes(data);os.replace(temporary,archive)
    write_json(manifest_path,manifest)

if __name__=='__main__' and sys.argv[1:2]==['--payload-child']:
    parser=argparse.ArgumentParser();parser.add_argument('--payload-child',action='store_true')
    parser.add_argument('--request',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    try:payload_child(args.request,args.output)
    except BaseException as error:
        # Retain only a type. Do not print HTTP bodies, headers, or external URLs.
        print(json.dumps({'stage':'PAYLOAD_FETCH','error_type':type(error).__name__}),file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(0)
