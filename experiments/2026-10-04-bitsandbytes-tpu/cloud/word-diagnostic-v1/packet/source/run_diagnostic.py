"""Retain exact arrays and bounded HLO for a diagnostic. No milestone claim."""
import argparse,gzip,hashlib,importlib.metadata,importlib.util,json,os,platform,struct,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
import torch
import probes as P
PACKET=Path(__file__).resolve().parents[1]
CORE={'torch':'2.9.0+cpu','torch-xla':'2.9.0','libtpu':'0.0.21','jax':'0.7.1','jaxlib':'0.7.1'}
RAW_OBJECT=8*1024*1024;RAW_TOTAL=96*1024*1024;PHYSICAL_TOTAL=30*1024*1024
REPORT_RESERVE=6*1024*1024;GRAPH_TOTAL=20*1024*1024;ROW_LIMIT=128*1024

def require(value,message):
 if not value:raise ValueError(message)
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
def digest(raw):return hashlib.sha256(raw).hexdigest()
def unwrap(t):
 if torch._is_functional_tensor(t):torch._functionalize_sync(t);return torch._from_functional_tensor(t)
 return t

def array(t):
 t=unwrap(t).detach().cpu().contiguous();dtype=str(t.dtype).removeprefix('torch.');shape=list(t.shape)
 if dtype=='float32':values=t.view(torch.int32).reshape(-1).tolist();width=8
 elif dtype in('int32','int64','uint8','bool'):values=t.reshape(-1).tolist();width={'int32':8,'int64':16,'uint8':2,'bool':2}[dtype]
 else:raise ValueError('DIAGNOSTIC_ARRAY_DTYPE:'+dtype)
 ints=[int(v)for v in values];words=[format(v&((1<<(4*width))-1),'0'+str(width)+'x')for v in ints]
 return {'dtype':dtype,'shape':shape,'int_values':ints,'hex_words':words,'storage_sha256':digest(b''.join(int(w,16).to_bytes(width//2,'little')for w in words))}

def qualification(mode,unqualified):
 versions={n:importlib.metadata.version(n)for n in CORE if importlib.util.find_spec(n.replace('-','_'))is not None or n=='torch-xla'and importlib.util.find_spec('torch_xla')is not None}
 result={'python':platform.python_version(),'system':platform.system(),'machine':platform.machine(),'versions':versions,'requested_backend':os.environ.get('PJRT_DEVICE'),'scope':'EXACT_WORD_DIAGNOSTIC_ONLY','m4_status':'NOT_QUALIFIED'}
 if unqualified:
  require(mode=='cpu','UNQUALIFIED_CPU_ONLY');result['status']='UNQUALIFIED_LOCAL_CPU_CONTROL';return result
 require(platform.system()=='Linux'and platform.machine()=='x86_64'and platform.python_version()=='3.12.14','PINNED_LINUX_PYTHON_REQUIRED')
 require(versions==CORE,'PINNED_RUNTIME_VERSIONS_REQUIRED')
 require(os.environ.get('ATEN_CPU_CAPABILITY')=='avx2'and os.environ.get('OMP_NUM_THREADS')=='1','CPU_ORACLE_ENVIRONMENT')
 distribution=importlib.metadata.distribution('torch-xla');members=[f for f in distribution.files or [] if str(f)=='torch_xla/core/xla_builder.py'];require(len(members)==1,'BUILDER_DISTRIBUTION_RECORD_MEMBERSHIP');member=members[0];builder=Path(distribution.locate_file(member));require(not builder.is_symlink()and member.hash is not None and member.hash.mode=='sha256','BUILDER_DISTRIBUTION_RECORD_HASH');import base64;require(base64.urlsafe_b64encode(bytes.fromhex(digest(builder.read_bytes()))).rstrip(b'=').decode()==member.hash.value,'BUILDER_DISTRIBUTION_RECORD_HASH')
 require(builder.is_file()and digest(builder.read_bytes())==P.load('original')[0].BUILDER_SHA,'PINNED_BUILDER_SOURCE')
 result['builder_sha256']=digest(builder.read_bytes());result['torch_git_version']=torch.version.git_version
 result['status']='PINNED_LINUX_CPU_DIAGNOSTIC_ORACLE'if mode=='cpu'else'PINNED_TPU_DIAGNOSTIC_RUNTIME'
 if mode=='tpu':
  require(os.environ.get('PJRT_DEVICE')=='TPU','EXPLICIT_TPU_REQUIRED');import torch_xla.runtime as xr;import torch_xla;import libtpu
  devices=torch_xla._XLAC._xla_get_devices();require(xr.device_type()=='TPU'and devices and all(d.startswith('TPU:')for d in devices),'NO_CPU_FALLBACK');result.update(observed_backend=xr.device_type(),devices=devices,libtpu_sha256=digest(Path(libtpu.get_library_path()).read_bytes()))
 return result

def check_cpu(rows):
 by={r['label']:r for r in rows};require(all(r['status']=='RETURNED'for r in rows),'CPU_PROBE_ERRORS')
 def out(label,key):return by[label]['outputs'][key]
 for name in('shift-left','shift-right'):
  values=[0,1,0x7fffff,0x800000,0xffffff,0x7fffffff,-1,-2147483648];counts=[0,1,23,30];expected=torch.tensor([[P.signed((v<<c)&0xffffffff)if name=='shift-left'else v>>c for c in counts]for v in values],dtype=torch.int32)
  require(out(name,'legacy')==out(name,'tensor')==array(expected),'CPU_INDEPENDENT_INTEGER_SHIFT_ORACLE')
 for n in('amax-1','amax-255'):
  require(out(n,'omitted')==out(n,'dim0')==out(n,'dim_negative1')and out(n,'omitted')==array(torch.tensor(0x3f800000 if n=='amax-1'else 254,dtype=torch.int32)),'CPU_AMAX_SCALAR')
 for n in P.PRIMITIVES[:4]:
  w=torch.tensor([P.signed(v)for v in P.WORDS],dtype=torch.int32);w=w[6]if n.endswith('scalar')else w
  require(out(n,'words')==array(w)and out(n,'roundtrip')==array(w.view(torch.float32)),'CPU_BITCAST_INDEPENDENT_STORAGE_ORACLE')
  require(by[n]['metadata']['source_native_functional']==('functional'in n),'CPU_BITCAST_REQUESTED_INPUT_STATE')
 words=torch.tensor([0,1,0x7fffff,0x800000,0xffffff,0x7fffffff,-1,-2147483648],dtype=torch.int32);zero=torch.zeros_like(words);one=torch.ones_like(words)
 expected_select={'gt_python':words>0,'gt_s32':words>zero,'eq_python':words==0,'eq_s32':words==zero,'select_python':torch.where(words>0,1,words),'select_s32':torch.where(words>zero,one,words),'select_negative_python':torch.where(words==0,-149,words),'select_negative_s32':torch.where(words==zero,torch.full_like(words,-149),words)}
 for key,value in expected_select.items():require(out('scalar-compare-select',key)==array(value),'CPU_TYPED_SCALAR_INDEPENDENT_ORACLE:'+key)
 for variant in('true_shift','true_shift_amax'):
  for family in P.FAMILIES:require(by[variant+'-'+family]['outputs']==by['original-'+family]['outputs'],'CPU_VARIANT_EXACT_EQUAL:'+variant+'-'+family)
 for variant in('original','true_shift','true_shift_amax'):
  require(out(variant+'-midpoints','negative_count')['int_values']==[127],'CPU_MIDPOINT_NEGATIVE_COUNT_127')
  require(out(variant+'-zero-normalize','codes')['int_values']==[127]*4,'CPU_ZERO_CODE_127')
  require(out(variant+'-tail-1-255','tail1_scales')['shape']==[1]and out(variant+'-tail-1-255','tail255_scales')['shape']==[1],'CPU_TAIL_SCALE_RANK')
  words=torch.tensor([P.signed(w)for w in P.WORDS],dtype=torch.int32);f=words.view(torch.float32);zero=torch.zeros_like(f)
  expectations={'add_zero':array((f+zero).view(torch.int32)),'subtract_zero':array((f-zero).view(torch.int32)),'cancellation':array((f+(-f)).view(torch.int32))}
  for key,expected in expectations.items():require(out(variant+'-addsub',key)==expected,'CPU_FLOAT32_ARITHMETIC_ORACLE:'+key)
  for key in('multiply_one','divide_one'):require(out(variant+'-muldiv',key)==array(words),'CPU_WORD_IDENTITY:'+key)
  require(out(variant+'-muldiv','divide_count2')==array((f/2).view(torch.int32)),'CPU_NATIVE_FLOAT32_DIVIDE2')
 return {'status':'PASS','checks':'shape, storage bits, variant equality, native float32 arithmetic identities, midpoint negative count127, zero code127','tolerance':0}

def run(args):
 out=args.output.resolve();require(not out.exists(),'FRESH_DIAGNOSTIC_OUTPUT');out.mkdir(parents=True)
 result={'kind':'M4_WORD_DIAGNOSTIC_V1','generation':'m4-word-lowering-diagnostic-v1','mode':args.mode,'status':'ERROR','m4_status':'NOT_QUALIFIED','started_epoch':time.time(),'deadline_epoch':args.deadline_epoch,'rows':[],'source_manifest_sha256':digest((PACKET/'manifest.json').read_bytes()),'physical_graphs':[]}
 raw_total=0
 try:
  result['runtime']=qualification(args.mode,args.unqualified_cpu)
  device=torch.device('cpu');xla=xm=None
  if args.mode=='tpu':import torch_xla as xla;import torch_xla.core.xla_model as xm;device=xla.device()
  x=P.inputs(device)
  if xla:xla._XLAC._xla_sync_multi(list(x.values()),devices=[],wait=True,sync_xla_data=True);xm.wait_device_ops()
  result['inputs']={k:array(v)for k,v in x.items()};write(out/'inputs.json',result['inputs'])
  oracle=None
  if args.mode=='tpu':
   require(args.oracle is not None,'EXACT_CPU_ORACLE_REQUIRED');raw=args.oracle.read_bytes();require(digest(raw)==args.oracle_sha256,'CPU_ORACLE_SHA');oracle=json.loads(raw);require(oracle['status']=='CPU_ORACLE_COMPLETE'and oracle['runtime']['status']=='PINNED_LINUX_CPU_DIAGNOSTIC_ORACLE'and oracle['source_manifest_sha256']==result['source_manifest_sha256'],'QUALIFIED_SAME_SOURCE_CPU_ORACLE');require(oracle['inputs']==result['inputs'],'CPU_TPU_EXACT_INPUT_BYTES')
  for label in P.labels():
   require(time.time()<args.deadline_epoch,'DIAGNOSTIC_ORIGINAL_DEADLINE');row={'label':label,'status':'ERROR','started_epoch':time.time()};result['rows'].append(row)
   try:
    outputs,metadata=P.execute(label,x,args.mode);outputs={k:unwrap(v)for k,v in outputs.items()};row['metadata']=metadata
    if xla:
     row['observation_status']='ERROR';mapping={}
     try:
      ctx=xla._XLAC.lowering.LoweringContext('M4WordDiagnostic');ctx.build(list(outputs.values()));mapping=ctx.device_parameter_id_tensor_mapping();row['parameter_metadata']={str(k):{'shape':list(t.shape),'dtype':str(t.dtype),'device':str(t.device)}for k,t in mapping.items()}
      bodies=[('hlo.txt',xla._XLAC._get_xla_tensors_hlo(list(outputs.values())).encode()),('hlo.pb',bytes(ctx.hlo()))]
      require(all(len(body)<=RAW_OBJECT for _,body in bodies),'DIAGNOSTIC_GRAPH_RAW_OBJECT_LIMIT');candidate_raw=sum(len(body)for _,body in bodies);require(raw_total+candidate_raw<=RAW_TOTAL,'DIAGNOSTIC_GRAPH_RAW_TOTAL_LIMIT')
      packed=[(suffix,body,gzip.compress(body,mtime=0))for suffix,body in bodies];candidate_physical=sum(len(z)for _,_,z in packed)
      require(sum(r['gzip_bytes']for r in result['physical_graphs'])+candidate_physical<=GRAPH_TOTAL,'DIAGNOSTIC_GRAPH_PHYSICAL_LIMIT')
      require(sum(p.stat().st_size for p in out.rglob('*')if p.is_file())+candidate_physical<=PHYSICAL_TOTAL-REPORT_RESERVE,'DIAGNOSTIC_PHYSICAL_RESERVE')
      for suffix,body,compressed in packed:
       path=out/'graphs'/(label+'.'+suffix+'.gz');path.parent.mkdir(exist_ok=True);path.write_bytes(compressed);result['physical_graphs'].append({'path':str(path.relative_to(out)),'raw_bytes':len(body),'raw_sha256':digest(body),'gzip_bytes':len(compressed),'gzip_sha256':digest(compressed)})
      raw_total+=candidate_raw;row['observation_status']='COMPLETE'
     except Exception as e:row['observation_error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
     # Observation failure must not prevent exact numerical synchronization.
     xla._XLAC._xla_sync_multi(list(outputs.values()),devices=[],wait=True,sync_xla_data=True);xm.wait_device_ops()
     try:row['parameters']={str(k):array(t)for k,t in mapping.items()}
     except Exception as e:row['parameter_capture_error']={'type':type(e).__name__,'message':str(e)}
    row['outputs']={k:array(v)for k,v in outputs.items()};row['status']='RETURNED'
    if oracle:
     expected=next(r for r in oracle['rows']if r['label']==label);row['cpu_exact_comparison']={k:row['outputs'][k]==expected['outputs'][k]for k in expected['outputs']};row['cpu_exact_match']=all(row['cpu_exact_comparison'].values())
   except Exception as e:row.update(error_type=type(e).__name__,error_message=str(e),traceback=traceback.format_exc())
   row['finished_epoch']=time.time();require(len(json.dumps(row,sort_keys=True,indent=2,allow_nan=False).encode())+1<=ROW_LIMIT,'DIAGNOSTIC_ROW_LIMIT');require(sum(p.stat().st_size for p in out.rglob('*')if p.is_file())+ROW_LIMIT<=PHYSICAL_TOTAL-REPORT_RESERVE,'DIAGNOSTIC_ROW_PHYSICAL_RESERVE');write(out/'rows'/(label+'.json'),row)
  if args.mode=='cpu':result['cpu_controls']=check_cpu(result['rows']);result['status']='CPU_ORACLE_COMPLETE'
  else:result['status']='TPU_DIAGNOSTIC_COMPLETE_WITH_EXACT_COMPARISONS';result['returned']=sum(r['status']=='RETURNED'for r in result['rows']);result['errors']=sum(r['status']=='ERROR'for r in result['rows']);result['exact_matches']=sum(r.get('cpu_exact_match')is True for r in result['rows'])
 except BaseException as e:result.update(error_type=type(e).__name__,error_message=str(e),traceback=traceback.format_exc())
 finally:
  result['finished_epoch']=time.time();require(len(json.dumps(result,sort_keys=True,indent=2,allow_nan=False).encode())+1<=REPORT_RESERVE,'DIAGNOSTIC_REPORT_RESERVE');write(out/'report.json',result)
 print(json.dumps({'status':result['status'],'mode':args.mode,'rows':len(result['rows']),'m4_status':'NOT_QUALIFIED'}));return 0 if result['status']in('CPU_ORACLE_COMPLETE','TPU_DIAGNOSTIC_COMPLETE_WITH_EXACT_COMPARISONS')else 2
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('cpu','tpu'),required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline-epoch',type=float,required=True);p.add_argument('--oracle',type=Path);p.add_argument('--oracle-sha256');p.add_argument('--unqualified-cpu',action='store_true');raise SystemExit(run(p.parse_args()))
