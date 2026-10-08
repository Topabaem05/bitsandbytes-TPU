"""Bound provider operations in externally owned process groups."""
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass

@dataclass
class Outcome:
    status: str
    response: dict | None
    launch: dict
    directory: Path


def write_json(path, value):
    data=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    temp=path.with_name(path.name+'.tmp')
    with open(temp,'wb') as f:
        os.chmod(temp,0o600);f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def absent(pgid):
    try:os.killpg(pgid,0);return False
    except ProcessLookupError:return True


class ProcessProvider:
    """The caller supplies one reviewed worker command. There is no default live path."""
    def __init__(self, worker_argv, output, deadline_epoch, *, cleanup_seconds=3, register_process=None):
        if not isinstance(worker_argv,list) or not worker_argv or not all(type(x) is str for x in worker_argv):
            raise ValueError('WORKER_ARGV')
        if len(worker_argv) not in (3,4) or worker_argv[1]!='-B' or not worker_argv[2].endswith('.py') or (len(worker_argv)==4 and worker_argv[3]!='--live-provider'):raise ValueError('UNREVIEWED_WORKER_COMMAND')
        if type(deadline_epoch) not in (int,float) or not math.isfinite(deadline_epoch):raise ValueError('DEADLINE')
        if not 0.1<=cleanup_seconds<=5:raise ValueError('CLEANUP_RESERVE')
        self.argv=worker_argv;self.root=Path(output).resolve();self.root.mkdir(parents=True,exist_ok=False)
        self.deadline_epoch=deadline_epoch;self.deadline_mono=time.monotonic()+deadline_epoch-time.time()
        self.cleanup_seconds=cleanup_seconds;self.counter=0;self.submit_launched=False
        self.register_process=register_process
        self.command_files={x:hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in worker_argv if Path(x).is_file()}

    def remaining(self):
        return min(self.deadline_epoch-time.time(),self.deadline_mono-time.monotonic())

    def invoke(self, operation, request, *, call_seconds):
        if any(hashlib.sha256(Path(x).read_bytes()).hexdigest()!=seal for x,seal in self.command_files.items()):raise ValueError('WORKER_SOURCE_CHANGED')
        if operation=='submit' and self.submit_launched:raise ValueError('SUBMIT_REPLAY_FORBIDDEN')
        if type(call_seconds) not in (int,float) or not math.isfinite(call_seconds) or call_seconds<=0:raise ValueError('CALL_BUDGET')
        budget=min(call_seconds,self.remaining()-self.cleanup_seconds)
        self.counter+=1;directory=self.root/f'{self.counter:03d}-{operation}';directory.mkdir(mode=0o700)
        descriptor={'operation':operation,'worker_argv':self.argv,'worker_file_sha256':self.command_files,'request_sha256':hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'parent_pid':os.getpid(),'process_token':uuid.uuid4().hex,'deadline_epoch':self.deadline_epoch,'budget_seconds':max(budget,0),'launched_pid':None,'pid':None,'pgid':None,'exit_code':None,'reaped':False,'group_absent':True,'status':'NOT_RUN','started_monotonic':time.monotonic(),'finished_monotonic':None}
        if budget<=0:
            descriptor['finished_monotonic']=time.monotonic();write_json(directory/'launch.json',descriptor)
            return Outcome('NOT_RUN',None,descriptor,directory)
        write_json(directory/'request.json',{'operation':operation,'request':request,'deadline_epoch':self.deadline_epoch,'call_deadline_epoch':min(self.deadline_epoch,time.time()+budget),'process_token':descriptor['process_token']})
        p=subprocess.Popen(self.argv+['--request',str(directory/'request.json'),'--output',str(directory)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        if operation=='submit':self.submit_launched=True
        descriptor.update(launched_pid=p.pid,pid=p.pid,pgid=p.pid,group_absent=False,status='RUNNING')
        if self.register_process is not None:self.register_process(p,dict(descriptor),directory)
        write_json(directory/'launch.json',descriptor)  # Durable before wait.
        timeout=False;stdout=b'';stderr=b''
        try:
            try:stdout,stderr=p.communicate(timeout=budget)
            except subprocess.TimeoutExpired:timeout=True
            if timeout or not absent(p.pid):
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                end=min(time.monotonic()+self.cleanup_seconds,self.deadline_mono)
                try:stdout,stderr=p.communicate(timeout=max(0.01,end-time.monotonic()))
                except subprocess.TimeoutExpired:
                    try:os.killpg(p.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    stdout,stderr=p.communicate(timeout=0.5)
                if not absent(p.pid):
                    try:os.killpg(p.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
            closed=absent(p.pid)
            result=None
            status='TIMEOUT' if timeout else 'ERROR'
            if p.returncode==0 and not timeout and closed:
                try:
                    result=json.loads((directory/'response.json').read_text())
                    if type(result) is dict:status='SUCCESS'
                except (OSError,ValueError):pass
            if not closed:status='UNCLOSED_PROCESS_GROUP'
            return Outcome(status,result,descriptor,directory)
        finally:
            cleanup_errors=[]
            if p.poll() is None or not absent(p.pid):
                try:
                    try:os.killpg(p.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:stdout,stderr=p.communicate(timeout=max(0.01,min(self.cleanup_seconds,self.deadline_mono-time.monotonic())))
                    except subprocess.TimeoutExpired:
                        try:os.killpg(p.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        stdout,stderr=p.communicate(timeout=0.5)
                except Exception as cleanup_error:cleanup_errors.append({'stage':'owned_group_reap','type':type(cleanup_error).__name__})
            descriptor['cleanup_errors']=cleanup_errors
            descriptor.update(exit_code=p.poll(),reaped=p.poll() is not None,group_absent=absent(p.pid),finished_monotonic=time.monotonic(),status=('TIMEOUT' if timeout else 'EXITED'),stdout_sha256=hashlib.sha256(stdout).hexdigest(),stderr_sha256=hashlib.sha256(stderr).hexdigest())
            # Keep stderr private. It can contain provider failure text.
            for name,data in (('stdout.private',stdout),('stderr.private',stderr)):
                (directory/name).write_bytes(data);os.chmod(directory/name,0o600)
            write_json(directory/'launch.json',descriptor)
