"""V3 interactive-kernel completion and independent readback protocol."""
import hashlib,inspect,json,re,secrets
from pathlib import Path
VERSION='M4_KERNEL_COMPLETION_V3'

def require(value,message):
 if not value:raise ValueError(message)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def row(path):
 p=Path(path);require(p.is_file()and not p.is_symlink(),'KERNEL_READBACK_REGULAR');b=p.read_bytes();return {'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def request(body,phase,base,packet_sha,birth,sources,*,artifact=None,artifact_expected=None):
 require(re.fullmatch('[a-z0-9-]+',phase)is not None,'KERNEL_PHASE')
 nonce=secrets.token_hex(16)
 return {'kind':VERSION,'nonce':nonce,'phase':phase,'packet_sha256':packet_sha,'allocation_epoch':birth,'original_deadline_epoch':birth+3600,'admitted_source_sha256':digest(sources),'body_sha256':hashlib.sha256(body.encode()).hexdigest(),'completion_path':str(Path(base).parent/(Path(base).name+'-kernel-completions')/(phase+'-'+nonce+'.json')),'artifact_path':str(artifact)if artifact else None,'artifact_expected':artifact_expected or{}}
def validate_cached_module(module,path):
 """Compare cached definitions and literal state to the admitted source bytes."""
 import ast,types
 from pathlib import Path
 path=Path(path).resolve();text=path.read_text();tree=ast.parse(text);compiled=compile(text,str(path),'exec',dont_inherit=True)
 def fail():raise ValueError('KERNEL_STALE_MODULE_CODE:'+module.__name__)
 def literals(namespace,body):
  for node in body:
   targets=node.targets if isinstance(node,ast.Assign)else[node.target]if isinstance(node,ast.AnnAssign)else[]
   if not targets or getattr(node,'value',None)is None:continue
   try:value=ast.literal_eval(node.value)
   except(ValueError,TypeError):continue
   for target in targets:
    if isinstance(target,ast.Name)and(target.id not in namespace or namespace[target.id]!=value):fail()
 def function(value,code,node):
  seen=set()
  while hasattr(value,'__wrapped__'):
   if id(value)in seen:fail()
   seen.add(id(value));value=value.__wrapped__
  if not isinstance(value,types.FunctionType)or value.__module__!=module.__name__ or value.__code__!=code or Path(value.__code__.co_filename).resolve()!=path:fail()
  try:defaults=tuple(ast.literal_eval(v)for v in node.args.defaults)
  except(ValueError,TypeError):defaults=None
  if defaults is not None and tuple(value.__defaults__ or())!=defaults:fail()
  for key,default in zip(node.args.kwonlyargs,node.args.kw_defaults):
   if default is None:continue
   try:expected=ast.literal_eval(default)
   except(ValueError,TypeError):continue
   if key.arg not in(value.__kwdefaults__ or{})or value.__kwdefaults__[key.arg]!=expected:fail()
 def definitions(namespace,body,parent_code):
  for node in body:
   if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):continue
   codes=[c for c in parent_code.co_consts if isinstance(c,types.CodeType)and c.co_name==node.name]
   if not codes or node.name not in namespace:fail()
   value=namespace[node.name]
   if isinstance(node,ast.ClassDef):
    if not isinstance(value,type)or value.__module__!=module.__name__:fail()
    definitions(vars(value),node.body,codes[-1]);literals(vars(value),node.body)
   elif isinstance(value,property):
    candidates=[v for v in(value.fget,value.fset,value.fdel)if v is not None]
    matching=[c for c in codes if any(isinstance(v,types.FunctionType)and v.__code__==c for v in candidates)]
    if len(matching)!=len(codes):fail()
   else:
    if isinstance(value,(staticmethod,classmethod)):value=value.__func__
    function(value,codes[-1],node)
 definitions(vars(module),tree.body,compiled);literals(vars(module),tree.body)

def render(body,req,sources,module_paths):
 require(hashlib.sha256(body.encode()).hexdigest()==req['body_sha256'],'KERNEL_BODY_BINDING')
 # This prefix precedes every import of sealed source in the current kernel.
 return ('import sys as _m4kv3_sys\n_m4kv3_sys.dont_write_bytecode=True\n'
  'import os as _m4kv3_os\n_m4kv3_os.environ["PYTHONDONTWRITEBYTECODE"]="1"\n'
  'import hashlib as _m4kv3_hashlib,json as _m4kv3_json\nfrom pathlib import Path as _m4kv3_Path\n'
  +inspect.getsource(validate_cached_module)+'\n'
  +f'_m4kv3_request={req!r}\n_m4kv3_sources={sources!r}\n_m4kv3_modules={module_paths!r}\n'
  +'for _m4kv3_path,_m4kv3_sha in _m4kv3_sources.items():\n'
  +' _m4kv3_p=_m4kv3_Path(_m4kv3_path)\n if _m4kv3_p.is_symlink() or _m4kv3_hashlib.sha256(_m4kv3_p.read_bytes()).hexdigest()!=_m4kv3_sha: raise ValueError("KERNEL_ADMITTED_SOURCE_BYTES")\n'
  +'for _m4kv3_name,(_m4kv3_path,_m4kv3_sha) in _m4kv3_modules.items():\n'
  +' _m4kv3_module=_m4kv3_sys.modules.get(_m4kv3_name)\n'
  +' if _m4kv3_module is not None:\n'
  +'  _m4kv3_p=_m4kv3_Path(getattr(_m4kv3_module,"__file__",""))\n'
  +'  if _m4kv3_p.resolve()!=_m4kv3_Path(_m4kv3_path).resolve() or _m4kv3_p.is_symlink() or _m4kv3_hashlib.sha256(_m4kv3_p.read_bytes()).hexdigest()!=_m4kv3_sha: raise ValueError("KERNEL_STALE_MODULE:"+_m4kv3_name)\n'
  +'  validate_cached_module(_m4kv3_module,_m4kv3_path)\n'
  +body+'\n'
  +'_m4kv3_receipt={**_m4kv3_request,"status":"COMPLETED","artifact":None}\n'
  +'if _m4kv3_request["artifact_path"] is not None:\n'
  +' _m4kv3_p=_m4kv3_Path(_m4kv3_request["artifact_path"])\n'
  +' if not _m4kv3_p.is_file() or _m4kv3_p.is_symlink(): raise ValueError("KERNEL_ARTIFACT_REGULAR")\n'
  +' _m4kv3_bytes=_m4kv3_p.read_bytes();_m4kv3_value=_m4kv3_json.loads(_m4kv3_bytes)\n'
  +' if any(_m4kv3_value.get(k)!=v for k,v in _m4kv3_request["artifact_expected"].items()): raise ValueError("KERNEL_ARTIFACT_BINDING")\n'
  +' _m4kv3_receipt["artifact"]={"sha256":_m4kv3_hashlib.sha256(_m4kv3_bytes).hexdigest(),"bytes":len(_m4kv3_bytes)}\n'
  +'_m4kv3_path=_m4kv3_Path(_m4kv3_request["completion_path"]);_m4kv3_path.parent.mkdir(parents=True,exist_ok=True)\n'
  +'with _m4kv3_path.open("x") as _m4kv3_file:_m4kv3_file.write(_m4kv3_json.dumps(_m4kv3_receipt,sort_keys=True,separators=(",",":"),allow_nan=False)+"\\n")\n'
  +'_m4kv3_readback_sha=_m4kv3_hashlib.sha256(_m4kv3_path.read_bytes()).hexdigest()\n'
  +'print(_m4kv3_request["kind"]+" "+_m4kv3_json.dumps({"nonce":_m4kv3_request["nonce"],"phase":_m4kv3_request["phase"],"admitted_source_sha256":_m4kv3_request["admitted_source_sha256"],"readback_sha256":_m4kv3_readback_sha},sort_keys=True,separators=(",",":")),flush=True)\n')
def marker(text,req):
 clean=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',text);lines=[l.strip()for l in clean.splitlines()if l.strip()]
 marked=[l for l in lines if l.startswith(VERSION+' ')]
 require(len(marked)==1 and lines[-1]==marked[0],'KERNEL_COMPLETION_LINE')
 def pairs(values):
  result={}
  for k,v in values:require(k not in result,'KERNEL_MARKER_DUPLICATE_JSON');result[k]=v
  return result
 try:value=json.loads(marked[0][len(VERSION)+1:],object_pairs_hook=pairs)
 except (json.JSONDecodeError,TypeError):raise ValueError('KERNEL_MARKER_JSON')
 expected={'nonce':req['nonce'],'phase':req['phase'],'admitted_source_sha256':req['admitted_source_sha256']}
 require(set(value)==set(expected)|{'readback_sha256'}and all(value[k]==v for k,v in expected.items())and re.fullmatch('[0-9a-f]{64}',value['readback_sha256'])is not None,'KERNEL_COMPLETION_BINDING')
 return value

def strict_json(text):
 def pairs(values):
  result={}
  for k,v in values:require(k not in result,'KERNEL_READBACK_DUPLICATE_JSON');result[k]=v
  return result
 return json.loads(text,object_pairs_hook=pairs)

def verify(text,req,receipt,artifact=None):
 mark=marker(text,req);proof=row(receipt);require(proof['sha256']==mark['readback_sha256'],'KERNEL_COMPLETION_READBACK_SHA');value=strict_json(Path(receipt).read_text())
 require(set(value)==set(req)|{'status','artifact'}and all(value[k]==v for k,v in req.items())and value['status']=='COMPLETED','KERNEL_COMPLETION_READBACK_BINDING')
 if req['artifact_path']is not None:
  require(artifact is not None and value['artifact']==row(artifact),'KERNEL_ARTIFACT_READBACK_SHA');body=strict_json(Path(artifact).read_text());require(all(body.get(k)==v for k,v in req['artifact_expected'].items()),'KERNEL_ARTIFACT_READBACK_BINDING')
 else:require(value['artifact']is None and artifact is None,'KERNEL_UNREQUESTED_ARTIFACT')
 return {'status':'KERNEL_COMPLETED_AND_READBACK_VERIFIED','request':req,'completion_record':proof,'artifact':value['artifact'],'scientific_acceptance':'NOT_EVALUATED'}
