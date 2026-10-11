"""Distinct M5 source admission and external accepted-M4 dependency. No acceptance writer."""
import ast,hashlib,importlib.util,json,math,re,sys,types,stat,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
GENERATION='nested-exact-m5-1024-scope-v3'
BASE_ADMISSION='75790f1d4613e20daa06d99111288463ef24b05aaa29922a7580512078f9912f'
BASE_SCIENCE='9e71efc73b0f2b3a210f8131c10972569a5e2d04cf298fcf5a8da59772cf4f5a'
BASE_MAP='cc237461aa97a4b33ebca3d794cba0b158139c137ce2323cbb9b2ebda0cd50b6'
RUNTIME_LOCK='323371ff61c5fbcc4f79fd6a358cf2ba17cb72b907382a5ceaac07f91dc66ed6'
def require(ok,code):
 if not ok:raise ValueError(code)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pairs(items):
 out={}
 for k,v in items:require(k not in out,'DUPLICATE_JSON');out[k]=v
 return out
def read(path):return json.loads(Path(path).read_text(),object_pairs_hook=pairs)
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
 with tmp.open('x')as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
 tmp.replace(path)
def files(root,suffix=None):
 root=Path(root);result={}
 for p in sorted(root.rglob('*')):
  require(not p.is_symlink(),'SOURCE_SYMLINK')
  if p.is_file()and(suffix is None or p.suffix==suffix):
   require(p.suffix!='.pyc','SOURCE_BYTECODE');result[p.relative_to(root).as_posix()]=sha(p)
 return result
def source_binding():
 manifest=read(HERE/'science-manifest.json')
 actual=set()
 for directory in ['source','integration','legacy','runtime','control']:
  for p in (ROOT/directory).rglob('*'):
   require(not p.is_symlink()and(p.is_dir()or stat.S_ISREG(p.lstat().st_mode)),'SCIENCE_REGULAR_TREE')
   if p.is_file()and p!=HERE/'science-manifest.json':actual.add(p.relative_to(ROOT).as_posix())
 require(actual==set(manifest['files']),'EXACT_SCIENCE_CLOSURE')
 for name,row in manifest['files'].items():
  require(not Path(name).is_absolute()and '..'not in Path(name).parts and Path(name).as_posix()==name,'CONFINED_SCIENCE_NAME')
  p=ROOT/name;require(p.is_file()and not p.is_symlink()and sha(p)==row['sha256']and p.stat().st_size==row['bytes'],'SCIENCE_FILE:'+name)
 require(manifest['generation']==GENERATION,'SCIENCE_GENERATION')
 return sha(HERE/'science-manifest.json')
def source():
 source_binding();a=read(ROOT/'integration/source-admission.json')
 require(a['generation']==GENERATION and a['m4_base_admission_sha256']==BASE_ADMISSION and a['m4_base_science_sha256']==BASE_SCIENCE,'M5_DISTINCT_ADMISSION')
 require(files(ROOT/'integration/upstream/bitsandbytes','.py')==a['upstream_python_files'],'ADMITTED_UPSTREAM_SOURCE')
 require(files(ROOT/'integration/plugin/src/bitsandbytes_tpu','.py')==a['plugin_python_files'],'ADMITTED_PLUGIN_SOURCE')
 require(a['plugin_python_files']['exact.py']=='49511fbc22e93ca9ca765da9930c8975e44ae389b605fe85747ddf7c73486830' and
  a['plugin_python_files']['_source_order.py']=='54728c36190c64334ffceb2b12d1cabcfff87ca35a8387a24766ae4dba191b75','FIXED_1024_BODY_DELTA')
 return a

def m4_dependency(path,expected):
 """A proposed consumer schema. Root supplies it only after actual79+state8 acceptance."""
 require(re.fullmatch('[0-9a-f]{64}',expected or'')and expected!='0'*64 and sha(path)==expected,'EXTERNAL_ACCEPTED_M4_SHA')
 r=read(path);require(r.get('kind')=='ROOT_M4_EXACT_NESTED_ACCEPTANCE'and r.get('status')=='ACCEPTED_M4_EXACT79_AND_STATE8'and
  r.get('m4_status')=='QUALIFIED'and r.get('record_validation')==r.get('numerical_status')=='PASS','GENUINE_FUTURE_ACCEPTED_M4_REQUIRED')
 require(r.get('source_variant')=='nested-exact-v1'and r.get('source_admission_sha256')==BASE_ADMISSION and
  r.get('science_map_sha256')==BASE_SCIENCE and r.get('integration_map_sha256')==BASE_MAP and
  r.get('runtime_lock_sha256')==RUNTIME_LOCK,'M4_BASE_SOURCE_RUNTIME_BINDINGS')
 require(r.get('nested79')=={'case_count':79,'record_validation':'PASS','numerical_status':'PASS'}and
  r.get('saved_state8')=={'case_count':8,'record_validation':'PASS','numerical_status':'PASS','fresh_restore_process':True,
  'checkpoint_tensor_state':'EXACT'},'M4_COMPLETE79_AND_SAVED8')
 require(r.get('resource_closure')=={'server_empty':True,'usage_zero':True,'owned_groups_absent':True,'owned_browser_tab_absent':True},'M4_ACCEPTED_RESOURCE_CLOSURE')
 provenance=r.get('provenance',{});required={'accepted79_sha256','actual79_receipt_sha256','actual_state8_receipt_sha256',
  'fresh_cpu79_oracle_sha256','fresh_state8_oracle_sha256','independent79_readback_sha256',
  'independent_state8_readback_sha256','complete_archive_sha256','closure_readback_sha256'}
 require(set(provenance)==required and all(re.fullmatch('[0-9a-f]{64}',v or'')and v!='0'*64 for v in provenance.values()),'M4_ACTUAL_PROVENANCE')
 require(type(r.get('accepted_epoch'))in(int,float)and math.isfinite(r['accepted_epoch'])and 0<r['accepted_epoch']<=time.time(),'M4_ACCEPTANCE_TIME')
 return r

def installed(a,roots):
 for name,key in [('bitsandbytes','upstream_python_files'),('bitsandbytes_tpu','plugin_python_files')]:
  require(files(roots[name],'.py')==a[key],'INSTALLED_COMPLETE_SOURCE:'+name)
 require(sha(Path(roots['bitsandbytes'])/'py.typed')==a['upstream_package_files']['py.typed'],'INSTALLED_PY_TYPED')
 return True

def loaded_plugin(a,root):
 root=Path(root).resolve()
 for name in a['plugin_python_files']:
  module_name='bitsandbytes_tpu'+(''if name=='__init__.py'else'.'+name[:-3]);module=sys.modules.get(module_name)
  require(module is not None and Path(module.__file__).resolve()==root/name,'LOADED_PLUGIN_MODULE:'+name)
  code=compile((root/name).read_text(),str(root/name),'exec');expected={c.co_name:c for c in code.co_consts if isinstance(c,types.CodeType)}
  for n in (n.name for n in ast.parse((root/name).read_text()).body if isinstance(n,ast.FunctionDef)):
   f=vars(module).get(n);require(isinstance(f,types.FunctionType)and f.__module__==module_name and f.__code__==expected[n],'LOADED_PLUGIN_FUNCTION:'+name+':'+n)
 return {'source_generation':GENERATION,'loaded_plugin_code':'COMPILED_ADMITTED_SOURCE','python_files':11}
