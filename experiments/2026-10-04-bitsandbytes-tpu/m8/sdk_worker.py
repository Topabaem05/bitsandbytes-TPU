"""Opt-in future root worker. Tests must never use the live-provider option."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlparse

SAVE_URL='https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel'
HERE=Path(__file__).resolve().parent


def sha(data):return hashlib.sha256(data).hexdigest()


@contextmanager
def bounded_transport(deadline, requests_module):
    """Worker-local transport adaptation. Preserve original CLI implementation."""
    original=requests_module.sessions.Session.send
    saves=[]
    def send(session, request, **kwargs):
        parsed=urlparse(request.url)
        if parsed.scheme!='https':raise ValueError('TLS_REQUIRED')
        if kwargs.get('verify',True) is not True:raise ValueError('CUSTOM_OR_WEAK_TLS_REJECTED')
        remaining=deadline-time.time()
        if not math.isfinite(remaining) or remaining<=0:raise TimeoutError('TRANSPORT_DEADLINE')
        # This owned worker never uses SDK retries. Requests adapter retries must
        # also remain disabled. Do not change original retry configuration.
        adapter=session.get_adapter(request.url)
        if getattr(adapter.max_retries,'total',None)!=0:raise ValueError('TRANSPORT_RETRY_REJECTED')
        kwargs['timeout']=(min(10,remaining),min(30,remaining))
        if parsed.path.endswith('/SaveKernel') and request.url!=SAVE_URL:raise ValueError('SAVE_HOST_OR_PATH')
        if request.url==SAVE_URL:
            if request.method!='POST' or saves:raise ValueError('SAVE_REPLAY_OR_METHOD')
            saves.append(sha(request.body if isinstance(request.body,bytes) else str(request.body).encode()))
            kwargs['allow_redirects']=False
        response=original(session,request,**kwargs)
        if request.url==SAVE_URL and 300<=response.status_code<400:
            raise ValueError('SAVE_REDIRECT_AMBIGUOUS')
        return response
    requests_module.sessions.Session.send=send
    try:yield saves
    finally:requests_module.sessions.Session.send=original


def assert_sources():
    import importlib.metadata
    if importlib.metadata.version('kaggle')!='2.2.4' or importlib.metadata.version('kagglesdk')!='0.1.37':raise ValueError('INSTALLED_VERSION_CHANGED')
    pins=json.loads((HERE/'installed-source-pins.json').read_text())
    for name,expected in pins.items():
        distribution=importlib.metadata.distribution('kaggle' if name.startswith('kaggle/') else 'kagglesdk')
        source=Path(distribution.locate_file(name))
        if source.is_symlink() or sha(source.read_bytes())!=expected:raise ValueError('INSTALLED_SOURCE_CHANGED')


def run(packet,output):
    if type(packet['deadline_epoch']) not in (int,float) or not math.isfinite(packet['deadline_epoch']) or packet['deadline_epoch']<=time.time():raise ValueError('DEADLINE')
    assert_sources()
    op,r=packet['operation'],packet['request']
    if op=='submit':
        if set(r)!={'folder','timeout','acc','wrapper_sha256','metadata_sha256'} or r['timeout']!='3600' or r['acc']!='TpuV6E8':raise ValueError('SAVE_REQUEST')
        folder=Path(r['folder']);wrapper=(folder/'wrapper.py').read_bytes();metadata=(folder/'kernel-metadata.json').read_bytes()
        if sha(wrapper)!=r['wrapper_sha256'] or sha(metadata)!=r['metadata_sha256']:raise ValueError('SUBMIT_SOURCE_CHANGED')
        stage=output/'sealed-submit';stage.mkdir(mode=0o700)
        (stage/'wrapper.py').write_bytes(wrapper);(stage/'kernel-metadata.json').write_bytes(metadata)
    call_deadline=packet.get('call_deadline_epoch',packet['deadline_epoch'])
    if type(call_deadline) not in (int,float) or not math.isfinite(call_deadline) or call_deadline>packet['deadline_epoch']:raise ValueError('CALL_DEADLINE')
    # Source admission occurs before this explicit live import. The official
    # module uses existing configured auth. No token is copied into this packet.
    if any(os.environ.get(k) for k in ('KAGGLE_API_ENVIRONMENT','KAGGLE_CONFIG_DIR')):raise ValueError('AUTH_ENV_DRIFT')
    import requests
    with bounded_transport(call_deadline,requests):
        from kaggle import api
        if any(x in api.args for x in ('--staging','--admin','--local','--verbose','-v')):raise ValueError('CLI_ARGUMENT_DRIFT')
        op,r=packet['operation'],packet['request']
        if op=='submit':
            if set(r)!={'folder','timeout','acc','wrapper_sha256','metadata_sha256'} or r['timeout']!='3600' or r['acc']!='TpuV6E8':raise ValueError('SAVE_REQUEST')
            response=api.kernels_push(str(stage),timeout=r['timeout'],acc=r['acc'])
            result={n:getattr(response,n) for n in ('ref','url','version_number','kernel_id','error','invalid_tags','invalid_dataset_sources','invalid_kernel_sources','invalid_competition_sources','invalid_model_sources')}
        else:
            from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest,ApiGetKernelSessionStatusRequest,ApiListKernelSessionOutputRequest,ApiDownloadKernelOutputRequest
            mapping={'source':(ApiGetKernelRequest,'get_kernel'),'status':(ApiGetKernelSessionStatusRequest,'get_kernel_session_status'),'outputs':(ApiListKernelSessionOutputRequest,'list_kernel_session_output'),'download':(ApiDownloadKernelOutputRequest,'download_kernel_output')}
            if op not in mapping:raise ValueError('OPERATION')
            cls,method=mapping[op];request=cls.from_dict(r)
            if request.to_dict(request)!=r:raise ValueError('REQUEST_SERIALIZATION_DRIFT')
            if op=='download':
                if type(request.version_number) is not int or request.version_number<=0:raise ValueError('EXACT_DOWNLOAD_VERSION')
            elif not request.version_label.startswith('v') or not request.version_label[1:].isdigit() or int(request.version_label[1:])<=0:raise ValueError('EXACT_REQUEST_VERSION')
            with api.build_kaggle_client() as client:
                response=getattr(client.kernels.kernels_api_client,method)(request)
                if op=='source':
                    m=response.metadata;result={'ref':m.ref,'kernel_id':m.id,'current_version_number':m.current_version_number,'is_private':m.is_private,'source_sha256':sha(response.blob.source.encode()),'language':m.language,'kernel_type':m.kernel_type,'docker_image':m.docker_image,'machine_shape':m.machine_shape}
                elif op=='status':result={'status':response.status.name,'failure_message':response.failure_message}
                elif op=='outputs':result={'files':[f.file_name for f in response.files],'next_page_token':response.next_page_token}
                else:
                    # HttpRedirect.prepare_from returns the streamed Requests
                    # Response. Never print the signed download URL or headers.
                    response.raise_for_status();size=0;digest=hashlib.sha256()
                    try:
                        with open(output/'download.bin','xb') as f:
                            os.chmod(output/'download.bin',0o600)
                            for block in response.iter_content(1024*1024):
                                if time.time()>=call_deadline:raise TimeoutError('DOWNLOAD_DEADLINE')
                                size+=len(block)
                                if size>512*1024*1024:raise ValueError('DOWNLOAD_SIZE')
                                f.write(block);digest.update(block)
                    finally:response.close()
                    result={'artifact':'download.bin','bytes':size,'sha256':digest.hexdigest()}
    assert_sources()
    return {'operation':packet['operation'],'request':packet['request'],'result':result,'source_pins_match':True,'pid':os.getpid(),'pgid':os.getpgid(0),'parent_pid':os.getppid(),'process_token':packet['process_token']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--live-provider',action='store_true');p.add_argument('--request',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if not args.live_provider:raise SystemExit('LIVE_PROVIDER_OPT_IN_REQUIRED')
    packet=json.loads(args.request.read_text());response=run(packet,args.output)
    data=(json.dumps(response,allow_nan=False)+'\n').encode()
    with open(args.output/'response.json','xb') as f:os.chmod(args.output/'response.json',0o600);f.write(data)
