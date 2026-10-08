"""Offline bootstrap for the actual admitted worker and source-defined API methods."""
import ast,contextlib,hashlib,json,os,runpy,socket,sys,time,types,typing
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def forbidden(*a,**k):raise AssertionError('OFFLINE_NETWORK_FORBIDDEN')
socket.socket.connect=forbidden;socket.create_connection=forbidden
def audit(event,args):
    if event.startswith("socket.") and event not in ("socket.__new__",):raise AssertionError("OFFLINE_NETWORK_FORBIDDEN")
    if event=="open" and isinstance(args[0],(str,bytes,os.PathLike)) and any(x in os.fsdecode(args[0]) for x in ("/.kaggle/","access_token","credentials.json")):raise AssertionError("CREDENTIAL_READ_FORBIDDEN")
sys.addaudithook(audit)
# Read only an owned fixture descriptor. The fake worker has no live path.
request_path=Path(sys.argv[sys.argv.index('--request')+1]);packet=json.loads(request_path.read_text())
if packet['operation']!='submit':raise AssertionError('FIXTURE_OPERATION')
folder=Path(packet['request']['folder']);mode_spec=json.loads((folder/'fixture-mode.json').read_text())
if set(mode_spec)!={'mode','host_source'} or mode_spec['mode']!='good':raise AssertionError('FIXTURE_MODE')
HOST=Path(mode_spec['host_source']).resolve()
# Bind installed original code before importing SDK classes or Requests.
import importlib.metadata
pinfile=HOST/'installed-source-pins.json'
if pinfile.is_symlink() or hashlib.sha256(pinfile.read_bytes()).hexdigest()!='2a4b414f644bd8920deb6ab89f03b86aa9bb81e381d258b1de7cb5b4a92304b2':raise AssertionError('INSTALLED_PIN_MAP_CHANGED')
if importlib.metadata.version('kaggle')!='2.2.4' or importlib.metadata.version('kagglesdk')!='0.1.37':raise AssertionError('INSTALLED_VERSION_CHANGED')
for name,digest in json.loads(pinfile.read_text()).items():
    distribution=importlib.metadata.distribution('kaggle' if name.startswith('kaggle/') else 'kagglesdk')
    source=Path(distribution.locate_file(name))
    if source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest()!=digest:raise AssertionError('INSTALLED_SOURCE_CHANGED')
import requests
from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest,ApiSaveKernelResponse
site=Path(sys.executable).absolute().parent.parent/'lib/python3.12/site-packages'
def method(path,cls,name,env):
    c=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==cls);f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name);f.decorator_list=[];exec(compile(ast.Module(body=[f],type_ignores=[]),str(path),'exec'),env);return env[name]
push=method(site/'kaggle/api/kaggle_api_extended.py','KaggleApi','kernels_push',{'os':os,'json':json,'Optional':typing.Optional,'List':typing.List,'cast':typing.cast,'ApiSaveKernelRequest':ApiSaveKernelRequest,'ApiSaveKernelResponse':ApiSaveKernelResponse,'slugify':lambda x:x.lower().replace(' ','-')})
http=site/'kagglesdk/kaggle_http_client.py';prepare_request=method(http,'KaggleHttpClient','_prepare_request',{'requests':requests,'KaggleObject':object});prepare_response=method(http,'KaggleHttpClient','_prepare_response',{'requests':requests})
request_path=Path(sys.argv[sys.argv.index('--request')+1]);packet=json.loads(request_path.read_text());folder=Path(packet['request']['folder']);meta=json.loads((folder/'kernel-metadata.json').read_text());owner,slug=meta['id'].split('/')
# The fixture adds its mode in a separate file. The actual sealed metadata remains unchanged.
mode=mode_spec['mode']
sent=[]
def fake_send(session,request,**kwargs):
    if request.url!='https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel' or request.method!='POST' or sent:raise AssertionError('FIXTURE_ENDPOINT_OR_REPLAY')
    sent.append(request.url)
    body=json.loads(request.body)
    output=Path(sys.argv[sys.argv.index('--output')+1])
    (output/'wire-request.sanitized.json').write_text(json.dumps({'machineShape':body.get('machineShape'),'sessionTimeoutSeconds':body.get('sessionTimeoutSeconds'),'isPrivate':body.get('isPrivate'),'text_sha256':hashlib.sha256(body['text'].encode()).hexdigest()},sort_keys=True)+'\n')
    if mode=='timeout':raise requests.ReadTimeout('synthetic original timeout')
    if mode=='deadline':time.sleep(30)
    r=requests.Response();r.status_code=400 if mode=='400' else 200;r.headers={'Content-Type':'application/json'};r.url=request.url
    r._content=json.dumps({'code':400,'message':'The kernel source size exceeds limit. SYNTHETIC_SECRET https://signed.invalid/?token=SECRET'} if mode=='400' else {'ref':meta['id'],'versionNumber':1,'kernelId':123}).encode();return r
requests.sessions.Session.send=fake_send
session=requests.Session();session.headers.clear();session.auth=None
http_object=types.SimpleNamespace(_session=session,_get_request_url=lambda *_:'https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel',_print_request=lambda _:None,_print_response=lambda _:None,_response_processor=None)
class Client:
    def save_kernel(self,r):
        req=prepare_request(http_object,'kernels.KernelsApiService','SaveKernel',r);response=session.send(req,verify=True);return prepare_response(http_object,ApiSaveKernelResponse,response)
@contextlib.contextmanager
def client():yield types.SimpleNamespace(kernels=types.SimpleNamespace(kernels_api_client=Client()))
fake=types.SimpleNamespace(KERNEL_METADATA_FILE='kernel-metadata.json',get_or_default=lambda d,k,v:d.get(k,v),get_bool=lambda d,k,v:d.get(k,v),parse_kernel_string=lambda _: (owner,slug,None),valid_push_language_types=['python'],valid_push_kernel_types=['script'],valid_push_pinning_types=['original','latest'],build_kaggle_client=client)
api=types.SimpleNamespace(args=[],kernels_push=lambda folder,timeout,acc:push(fake,folder,timeout=timeout,acc=acc))
kaggle=types.ModuleType('kaggle');kaggle.api=api;sys.modules['kaggle']=kaggle
sys.path.insert(0,str(HOST));sys.argv=[str(HOST/'sdk_worker.py'),'--live-provider',*sys.argv[1:]];runpy.run_path(str(HOST/'sdk_worker.py'),run_name='__main__')
