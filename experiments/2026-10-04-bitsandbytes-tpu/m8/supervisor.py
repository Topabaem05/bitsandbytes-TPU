"""Own the host owner and each SDK worker as direct children. No provider API."""
import hashlib,json,math,os,signal,socket,subprocess,threading,time,uuid
from multiprocessing.connection import Connection
from pathlib import Path
from process_provider import ProcessProvider,Outcome,write_json

class BrokerProvider:
    """The owner has no Popen authority for SDK calls. The supervisor launches them."""
    def __init__(self,fd,token,deadline):self.connection=Connection(fd);self.token=token;self.deadline_epoch=deadline
    def invoke(self,operation,request,*,call_seconds):
        self.connection.send({'token':self.token,'operation':operation,'request':request,'call_seconds':call_seconds})
        remaining=self.deadline_epoch-time.time()
        if remaining<=0 or not self.connection.poll(remaining):raise TimeoutError('SUPERVISOR_REPLY_DEADLINE')
        result=self.connection.recv()
        if result.get('token')!=self.token:raise ValueError('BROKER_RESPONSE_TOKEN')
        if result.get('error'):raise RuntimeError('SUPERVISOR_CALL_'+result['error']['type'])
        return Outcome(result['status'],result['response'],result['launch'],Path(result['directory']))

class Registry:
    def __init__(self,root,owner_pid,deadline):
        self.root=root;self.owner_pid=owner_pid;self.deadline=deadline;self.procs={};self.rows=[];self.lock=threading.Lock();self.closing=False
    def register(self,proc,descriptor,directory):
        # Store the actual Popen object before any durable-write exception.
        with self.lock:
            if proc.pid in self.procs:raise ValueError('DUPLICATE_OWNED_PROCESS')
            self.procs[proc.pid]=proc
            self.rows.append({'pid':proc.pid,'pgid':proc.pid,'supervisor_pid':os.getpid(),'owner_pid':self.owner_pid,
                'process_token':descriptor['process_token'],'operation':descriptor['operation'],
                'worker_argv':descriptor['worker_argv'],'worker_file_sha256':descriptor['worker_file_sha256'],
                'provider_directory':str(directory),'registered_epoch':time.time(),'cleanup':None})
            write_json(self.root/'provider-registry.json',{'kind':'DIRECT_CHILD_PROVIDER_REGISTRY','rows':self.rows})
            if self.closing:
                self.rows[-1]['cleanup']=close_direct(proc,self.deadline)
                write_json(self.root/'provider-registry.json',{'kind':'DIRECT_CHILD_PROVIDER_REGISTRY','rows':self.rows})
                raise RuntimeError('OWNER_ALREADY_CLOSED')
    def complete(self,descriptor):
        with self.lock:
            row=next(r for r in self.rows if r['process_token']==descriptor['process_token'])
            if descriptor['launched_pid']!=row['pid']:raise ValueError('REGISTRY_COMPLETION_IDENTITY')
            row['terminal_descriptor']=dict(descriptor)
            if descriptor['reaped'] and descriptor['group_absent'] and not descriptor.get('cleanup_errors'):
                if row.get('cleanup') is None:
                    row['cleanup']={'pid':row['pid'],'pgid':row['pgid'],'reaped':True,'group_absent':True,'exit_code':descriptor['exit_code'],'errors':[],'status':'CLOSED','method':'ORIGINAL_PROVIDER_FINAL_PROOF'}
            write_json(self.root/'provider-registry.json',{'kind':'DIRECT_CHILD_PROVIDER_REGISTRY','rows':self.rows})
    def cleanup(self):
        with self.lock:
            self.closing=True;results=[]
            for row in self.rows:
                proc=self.procs[row['pid']]
                if row.get('cleanup') is not None and row['cleanup']['status']=='CLOSED':
                    results.append(row['cleanup']);continue
                row['cleanup']=close_direct(proc,self.deadline);results.append(row['cleanup'])
            write_json(self.root/'provider-registry.json',{'kind':'DIRECT_CHILD_PROVIDER_REGISTRY','rows':self.rows})
            return results

def group_absent(pgid):
    try:os.killpg(pgid,0);return False
    except ProcessLookupError:return True

def close_direct(proc,deadline):
    """Only a registered Popen direct child enters this function."""
    record={'pid':proc.pid,'pgid':proc.pid,'reaped':False,'group_absent':False,'signals':[],'errors':[],'probe_refusals':0}
    def send(sig):
        try:os.killpg(proc.pid,sig);record['signals'].append(sig)
        except ProcessLookupError:pass
        except OSError as e:record['errors'].append({'stage':'signal','type':type(e).__name__})
    end=min(deadline,time.time()+2.5)
    try:
        proc.poll()
        try:absent=group_absent(proc.pid)
        except PermissionError:absent=False;record['probe_refusals']+=1
        if not absent:
            send(signal.SIGTERM)
            try:proc.wait(timeout=max(.01,min(.2,end-time.time())))
            except subprocess.TimeoutExpired:pass
            # The leader can be reaped while a descendant still holds this group.
            try:absent=group_absent(proc.pid)
            except PermissionError:absent=False;record['probe_refusals']+=1
            if not absent:send(signal.SIGKILL)
        try:proc.wait(timeout=max(.01,min(.5,end-time.time())))
        except subprocess.TimeoutExpired:record['errors'].append({'stage':'reap','type':'TimeoutExpired'})
        record['reaped']=proc.poll() is not None
        while time.time()<end:
            try:
                if group_absent(proc.pid):record['group_absent']=True;break
            except PermissionError:record['probe_refusals']+=1
            time.sleep(.01)
        if not record['group_absent']:
            try:record['group_absent']=group_absent(proc.pid)
            except PermissionError:record['probe_refusals']+=1
        record['exit_code']=proc.returncode
    except Exception as e:record['errors'].append({'stage':'cleanup','type':type(e).__name__})
    record['status']='CLOSED' if record['reaped'] and record['group_absent'] and not record['errors'] else 'UNCLOSED'
    return record

def supervise(owner_argv,worker_argv,root,deadline,*,on_owner_launch=None):
    if type(deadline) not in (int,float) or not math.isfinite(deadline):raise ValueError('SUPERVISOR_DEADLINE')
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    mono=time.monotonic()+deadline-time.time();token=uuid.uuid4().hex
    report={'kind':'HOST_OWNER_SUPERVISION','deadline_epoch':deadline,'status':'NOT_RUN','owner_launch':None,
        'owner_cleanup':None,'provider_cleanup':[],'remote_resource_closure':'UNCONFIRMED','error':None}
    if deadline-time.time()<=3:write_json(root/'supervisor.json',report);return report
    parent_socket,child_socket=socket.socketpair();connection=Connection(parent_socket.detach());child_fd=child_socket.fileno()
    proc=None;registry=None;thread=None;result={};current=None
    broker_closed=False
    previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    stop={'requested':False}
    def interrupted(signum,frame):stop['requested']=True;report['signal']=signum
    try:
        for s in previous:signal.signal(s,interrupted)
        stdout=(root/'owner.stdout.private').open('wb');stderr=(root/'owner.stderr.private').open('wb')
        os.fchmod(stdout.fileno(),0o600);os.fchmod(stderr.fileno(),0o600)
        argv=owner_argv+['--broker-fd',str(child_fd),'--broker-token',token,'--deadline-epoch',str(deadline)]
        proc=subprocess.Popen(argv,pass_fds=(child_fd,),start_new_session=True,stdout=stdout,stderr=stderr)
        child_socket.close()
        report['owner_launch']={'pid':proc.pid,'pgid':proc.pid,'parent_pid':os.getpid(),'argv':argv,'process_token':token,'started_epoch':time.time()}
        write_json(root/'owner-launch.json',report['owner_launch'])
        registry=Registry(root,proc.pid,deadline)
        provider=ProcessProvider(worker_argv,root/'provider',deadline,register_process=registry.register)
        report['status']='RUNNING';write_json(root/'supervisor.json',report)
        if on_owner_launch:on_owner_launch(proc,root)
        def call(message):
            try:
                out=provider.invoke(message['operation'],message['request'],call_seconds=message['call_seconds'])
                if out.launch.get('launched_pid') is not None:registry.complete(out.launch)
                result['value']={'token':token,'status':out.status,'response':out.response,'launch':out.launch,'directory':str(out.directory)}
            except BaseException as e:result['value']={'token':token,'error':{'type':type(e).__name__}}
        while proc.poll() is None and not stop['requested'] and min(deadline-time.time(),mono-time.monotonic())>3:
            if thread is not None:
                if thread.is_alive():time.sleep(.01);continue
                thread.join();thread=None
                try:connection.send(result['value'])
                except (EOFError,BrokenPipeError,OSError):break
                result={};current=None
            if broker_closed:
                time.sleep(.01);continue
            if connection.poll(.02):
                try:message=connection.recv()
                except EOFError:
                    broker_closed=True;continue
                if type(message)is not dict or set(message)!={'token','operation','request','call_seconds'} or message['token']!=token or message['operation'] not in ('submit','source','status','outputs','download') or type(message['request'])is not dict:raise ValueError('UNADMITTED_BROKER_REQUEST')
                current=message['operation'];thread=threading.Thread(target=call,args=(message,),daemon=False);thread.start()
        report['owner_exit_before_cleanup']=proc.poll();report['active_operation_at_owner_exit']=current
        report['status']='OWNER_FINISHED' if proc.poll() is not None and proc.returncode==0 else 'OWNER_FAILED_OR_INTERRUPTED'
    except BaseException as e:report.update(status='SUPERVISION_ERROR',error={'type':type(e).__name__})
    finally:
        # Stop the owner first. Then stop only actual registered SDK child groups.
        if proc is not None:report['owner_cleanup']=close_direct(proc,deadline)
        if registry is not None:report['provider_cleanup']=registry.cleanup()
        if thread is not None:
            thread.join(timeout=max(.01,deadline-time.time()))
            if thread.is_alive():report['status']='SUPERVISION_THREAD_UNCLOSED'
            elif result.get('value'):write_json(root/'interrupted-call.json',result['value'])
        if registry is not None:
            # Refresh after the provider's finally path has drained logs and reaped.
            report['provider_cleanup']=registry.cleanup()
        connection.close();child_socket.close()
        for stream in ('stdout','stderr'):
            if stream in locals():locals()[stream].close()
        for s,h in previous.items():signal.signal(s,h)
        if report['owner_cleanup'] and report['owner_cleanup']['status']!='CLOSED' or any(r['status']!='CLOSED' for r in report['provider_cleanup']):report['status']='SUPERVISION_CLEANUP_UNVERIFIED'
        write_json(root/'supervisor.json',report)
    return report
