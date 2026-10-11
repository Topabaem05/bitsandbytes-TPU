"""Portable frozen20/save10/resume10 M5 probe. Root owns every external action."""
import argparse,contextlib,copy,hashlib,re,importlib.metadata,importlib.util,json,math,os,platform,secrets,subprocess,sys,time,traceback,types
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(HERE))
import admission as A
import math_model as M
import record_validation as V
import public_trace as T
import archive_store as Z
import scope_rational_reference as F

def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def bundle():return M.digest({'science':A.source_binding(),'admission':A.sha(ROOT/'integration/source-admission.json'),'protocol':M.PROTOCOL_SHA})

def runtime(torch,local=False):
 evidence={'python':platform.python_version(),'torch':torch.__version__,'torch_git':torch.version.git_version,
  'platform':platform.system(),'machine':platform.machine(),'capability':torch.backends.cpu.get_cpu_capability(),
  'num_threads':torch.get_num_threads(),'build':torch.__config__.show(),
  'xla_or_jax_imported':any(n=='torch_xla'or n.startswith('torch_xla.')or n=='jax'or n.startswith('jax.')for n in sys.modules)}
 if not local:V.cpu_evidence(evidence)
 return evidence

def methods(C):
 C.A.public_methods(C.B,C.bnb,{'bitsandbytes':C.source/'bitsandbytes'})
 for relative,classname,names in [('nn/modules.py','Params4bit',['from_prequantized']),('nn/modules.py','Linear4bit',['_save_to_state_dict']),('functional.py','QuantState',['as_dict','from_dict'])]:
  p=(C.source/'bitsandbytes'/relative).resolve();codes=compile(p.read_text(),str(p),'exec').co_consts
  cls=getattr(C.bnb.functional if classname=='QuantState'else C.bnb.nn,classname)
  cc=next(v for v in codes if isinstance(v,types.CodeType)and v.co_name==classname)
  for name in names:
   fn=getattr(cls,name);fn=getattr(fn,'__func__',fn);expected=next(v for v in cc.co_consts if isinstance(v,types.CodeType)and v.co_name==name)
   A.require(fn.__code__==expected and Path(fn.__code__.co_filename).resolve()==p,'PUBLIC_ORIGINAL_METHOD:'+name)
 A.installed(C.admission,C.roots)
 if not C.local:C.loaded_plugin=A.loaded_plugin(C.admission,C.roots['bitsandbytes_tpu'])
 A.require(not C.torch.__future__.get_overwrite_module_params_on_conversion()and not C.torch.__future__.get_swap_module_params_on_conversion(),'DEFAULT_FUTURE_FLAGS')

def setup(local=False,device=False):
 a=A.source();A.require(not(local and device),'LOCAL_CANNOT_CLAIM_TPU')
 if local:
  sys.path.insert(0,str(ROOT/'integration/upstream'));discover=importlib.metadata.entry_points
  def filtered(*args,**kwargs):return ()if kwargs.get('group')=='bitsandbytes.backends'else discover(*args,**kwargs)
  importlib.metadata.entry_points=filtered
 try:
  roots={}
  for name in ['bitsandbytes']+([]if local else['bitsandbytes_tpu']):
   A.require(name not in sys.modules,'PACKAGE_PREIMPORTED:'+name);spec=importlib.util.find_spec(name);A.require(spec and spec.origin,'INSTALLED_PACKAGE_MISSING:'+name);roots[name]=Path(spec.origin).resolve().parent
  roots.setdefault('bitsandbytes_tpu',ROOT/'integration/plugin/src/bitsandbytes_tpu');A.installed(a,roots)
  import torch
  torch.set_num_threads(1);evidence=None
  if not device:evidence=runtime(torch,local)
  else:
   A.require(torch.__version__=='2.9.0+cpu'and platform.python_version()=='3.12.14'and platform.system()=='Linux'and platform.machine()=='x86_64','EXACT_TPU_RUNTIME')
   for name,version in [('torch-xla','2.9.0'),('libtpu','0.0.21'),('jax','0.7.1'),('jaxlib','0.7.1')]:A.require(importlib.metadata.version(name)==version,'EXACT_TPU_DISTRIBUTION:'+name)
   import torch_xla as xla,torch_xla.core.xla_model as xm,torch_xla.backends as backends,torch_xla.debug.metrics as metrics
   backends.set_mat_mul_precision('highest');A.require(backends.get_mat_mul_precision()=='highest','HIGHEST_BEFORE_GRAPH')
  import bitsandbytes as bnb
 finally:
  if local:importlib.metadata.entry_points=discover
 B=load(ROOT/'legacy/probe_backend.py','m5_B');OldA=load(ROOT/'legacy/transfer_admission.py','m5_legacy_A');P=load(ROOT/'legacy/probe_precision.py','m5_P')
 C=types.SimpleNamespace(torch=torch,bnb=bnb,B=B,A=OldA,P=P,admission=a,roots=roots,source=roots['bitsandbytes'].parent,
  tolerance=B.load_spec()[0]['tolerances']['fp32_forward_gradient'],device='cpu',device_mode=device,local=local,cpu_evidence=evidence)
 if device:
  C.device=xm.xla_device();A.require(C.device.type=='xla'and xm.xla_device_hw(C.device)=='TPU'and os.environ.get('PJRT_DEVICE')=='TPU','ACTUAL_TPU_ONLY')
  C.xla=xla;C.xm=xm;C.metrics=metrics
  C.precision={'requested':'highest','readback':'highest','before_graph':True,'set_calls':1}
 if not local:C.loaded_plugin=A.loaded_plugin(a,roots['bitsandbytes_tpu'])
 methods(C);return C

def references(C):
 t=C.torch;oq4,oq8,od8,od4=M.default_oracle(C);result=[]
 for index in range(2):
  packed,first=oq4(M.generated(C,'weight',index),64,'nf4',t.uint8);dynamic=C.bnb.functional.create_dynamic_map();nf4=C.bnb.functional.get_4bit_type('nf4',device='cpu')
  offset=first.mean();codes,second=oq8(first-offset,dynamic,256)
  words=lambda v:v.reshape(-1).view(t.int32).tolist()
  rational=F.nested(words(first),words(dynamic))
  A.require(rational['offset']==words(offset)[0]and rational['codes']==codes.reshape(-1).tolist()and rational['second']==words(second),'QUALIFIED1024_SOURCE_ORDER_AND_ENCODING')
  state2=C.bnb.functional.QuantState(absmax=second,code=dynamic,blocksize=256,dtype=t.float32)
  state=C.bnb.functional.QuantState(absmax=codes,code=nf4,blocksize=64,dtype=t.float32,shape=(256,256),quant_type='nf4',offset=offset,state2=state2)
  restored=od8(codes,second,dynamic,256,t.float32)+offset
  A.require(words(restored)==rational['restored']and first.numel()==1024 and second.numel()==4 and packed.numel()==32768,'EXACT1024_RESTORED_PRIMARY_BITS')
  decoded=od4(packed.reshape(-1),restored,nf4,64,(256,256),t.float32)
  result.append({'packed':packed,'first':first,'dynamic':dynamic,'nf4':nf4,'offset':offset,'codes':codes,'second':second,'state':state,'decoded':decoded,'rational':rational})
 return result

def reference_records(C):
 return [{k:C.B.tensor_record(row[k])for k in ('packed','first','dynamic','nf4','offset','codes','second','decoded')}for row in C.refs]

def model(C,checkpoint=None):
 t,bnb=C.torch,C.bnb
 class Adapter(t.nn.Module):
  def __init__(self,index):
   super().__init__();self.base=bnb.nn.Linear4bit(256,256,bias=True,compute_dtype=t.float32,compress_statistics=True,quant_type='nf4',quant_storage=t.uint8)
   self.A=t.nn.Parameter(M.generated(C,'A',index).to(C.device));self.B=t.nn.Parameter(M.generated(C,'B',index).to(C.device))
   if checkpoint is None and C.device_mode:
    with t.no_grad():self.base.weight.copy_(M.generated(C,'weight',index));self.base.bias.copy_(M.generated(C,'bias',index))
    self.base=self.base.to(C.device)
   else:
    if checkpoint:
     prefix=f'layers.{index}.base.';saved=checkpoint['model'];stats={k.removeprefix(prefix+'weight.'):v for k,v in saved.items()if k.startswith(prefix+'weight.')};packed=saved[prefix+'weight'];bias=saved[prefix+'bias']
    else:
     ref=C.refs[index];stats=ref['state'].as_dict(packed=True);packed=ref['packed'];bias=M.generated(C,'bias',index)
    self.base.weight=bnb.nn.Params4bit.from_prequantized(packed,dict(stats),requires_grad=False,device=C.device,module=self.base)
    self.base.bias=t.nn.Parameter(bias.to(C.device),requires_grad=False)
   self.base.weight.requires_grad_(False);self.base.bias.requires_grad_(False)
  def forward(self,x):return self.base(x)+(x@self.A.t())@self.B.t()
 network=t.nn.Module();network.layers=t.nn.ModuleList([Adapter(i)for i in range(2)])
 if checkpoint:
  filtered={k:v for k,v in checkpoint['model'].items()if'.weight.'not in k};loaded=network.load_state_dict(filtered,strict=True,assign=False)
  A.require(not loaded.missing_keys and not loaded.unexpected_keys and M.state_hashes(network.state_dict())==checkpoint['model_hashes'],'EXACT_STRICT_FRESH_RESTORE')
 return network.train()

def checkpoint_check(C,checkpoint,inputs_sha):
 A.require(type(checkpoint)is dict and checkpoint.get('format')=='M5_EXACT1024_CHECKPOINT_V1','CHECKPOINT_FORMAT')
 A.require(checkpoint['completed_step']==10 and type(checkpoint['completed_step'])is int and checkpoint['adapter_names']==M.ADAPTER_NAMES,'CHECKPOINT_STEP_MAPPING')
 A.require(checkpoint['protocol_sha256']==M.PROTOCOL_SHA and checkpoint['source_bundle_sha256']==bundle()and checkpoint['inputs_sha256']==inputs_sha,'CHECKPOINT_BINDINGS')
 A.require(M.state_hashes(checkpoint['model'])==checkpoint['model_hashes'],'CHECKPOINT_FULL_MODEL_HASHES')
 expected=model(types.SimpleNamespace(**{**vars(C),'device':'cpu','device_mode':False}));base=M.state_hashes(M.frozen_state(expected))
 expected_state=expected.state_dict();A.require(set(checkpoint['model'])==set(expected_state),'CHECKPOINT_COMPLETE_TENSOR_SET')
 for name,value in checkpoint['model'].items():
  wanted=expected_state[name];A.require(isinstance(value,C.torch.Tensor)and value.shape==wanted.shape and value.dtype==wanted.dtype,'CHECKPOINT_TENSOR_STRUCTURE')
  if value.is_floating_point():A.require(C.torch.isfinite(value).all().item(),'CHECKPOINT_FINITE_VALUES')
 A.require(checkpoint['frozen_base']==base==M.state_hashes({k:v for k,v in checkpoint['model'].items()if'.base.'in k}),'CHECKPOINT_FULL_FROZEN_BASE')
 A.require(checkpoint['optimizer']==M.optimizer_descriptor(M.optimizer(C,expected)),'CHECKPOINT_FULL_OPTIMIZER');return checkpoint

def dense_forward(C,adapters,x):
 for i in range(2):
  a,b=adapters[2*i:2*i+2];x=C.torch.nn.functional.linear(x,C.refs[i]['decoded'],M.generated(C,'bias',i))+(x@a.t())@b.t()
 return x

def trace(C,output,label,deadline):
 return T.make_mode(C.torch,C.B,C.P,C.xla,C.xm,C.metrics,C.roots['bitsandbytes_tpu'],output/'trace'/label,deadline)if C.device_mode else contextlib.nullcontext()

def highest_precision(C,hlo):
 dots=C.P.hlo_precision(hlo);A.require(dots,'TRAINING_REACHABLE_DOTS')
 for dot in dots:
  C.P.validate_dot_matrix(C.B,dot)
  A.require(dot['result_dtype']=='f32'and dot['operand_dtypes']==['f32','f32']and dot['operands']==['highest','highest'],'TRAINING_FP32_HIGHEST')
 return dots

def cpu_gate(args):
 r=A.read(args.cpu_gate);A.require(A.sha(args.cpu_gate)==args.cpu_gate_sha256 and r.get('kind')=='ROOT_M5_FRESH_QUALIFIED_CPU_GATE'and r.get('status')=='QUALIFIED_LINUX20_RESUME10_VERIFIED'and r.get('oracle_sha256')==args.oracle_sha256 and r.get('source_bundle_sha256')==bundle()and r.get('m4_acceptance_sha256')==args.m4_acceptance_sha256,'FRESH_QUALIFIED_M5_CPU_GATE')
 A.require(A.sha(Path(args.oracle)/'oracle-seal.json')==args.oracle_sha256,'FRESH_CPU_ORACLE_BYTES');return r

def child(args):
 output=Path(args.output);output.mkdir(parents=True,exist_ok=False);start=11 if args.phase=='resume'else 1;end=10 if args.phase=='save'else 20
 rec={'status':'ERROR','phase':args.phase,'pid':os.getpid(),'pgid':os.getpgid(0),'parent_pid':os.getppid(),'parent_process_token':args.parent_process_token,'process_token':args.process_token,'source_bundle_sha256':bundle(),'protocol_sha256':M.PROTOCOL_SHA,'started_epoch':time.time(),'deadline_epoch':args.deadline_epoch,'step_range':[start,end],'actual_tpu':args.device,'qualified_runtime':not args.local_reference,'m5_status':'NOT_QUALIFIED'}
 try:
  A.require(os.getppid()==args.parent_pid and os.getpgid(0)==args.parent_pid,'ACTUAL_FRESH_CHILD_PARENT_GROUP')
  if not args.local_reference:A.m4_dependency(args.m4_acceptance,args.m4_acceptance_sha256)
  if args.device:cpu_gate(args)
  C=setup(args.local_reference,args.device);C.refs=references(C);t=C.torch;inputs=M.input_descriptor(C);inputs_sha=M.digest(inputs)
  A.write(output/'inputs.json',inputs);A.write(output/'reference-arrays.json',reference_records(C))
  rec.update(inputs_sha256=inputs_sha,public_method_identity=True,cpu_evidence=C.cpu_evidence,
   m4_acceptance_sha256=args.m4_acceptance_sha256,plugin_loaded_code=getattr(C,'loaded_plugin',None),
   runtime={'python':platform.python_version(),'torch':t.__version__,'platform':platform.system(),'machine':platform.machine()},precision=getattr(C,'precision',None),device={'type':'xla','hardware':'TPU','pjrt':'TPU'}if args.device else{'type':'cpu','reference':'INDEPENDENT_DENSE_DEFAULT_BYTE_PRIMARY'})
  checkpoint=None
  if args.phase=='resume':
   A.require(A.sha(args.checkpoint)==args.checkpoint_sha256,'CHECKPOINT_FILE_HASH');checkpoint=checkpoint_check(C,t.load(args.checkpoint,weights_only=True,map_location='cpu'),inputs_sha)
  if args.device:Z.Store(output,create=True)
  with trace(C,output,'initial',args.deadline_epoch)as observation:network=model(C,checkpoint)
  if args.device:A.require(not observation.observation_errors,'INITIAL_TYPED_GRAPH_OBSERVATION')
  opt=M.optimizer(C,network)
  if checkpoint:opt.load_state_dict(checkpoint['optimizer']['state'])
  initial_base=M.state_hashes(M.frozen_state(network));M.assert_frozen(C,network,initial_base)
  expected=model(types.SimpleNamespace(**{**vars(C),'device':'cpu','device_mode':False}));A.require(initial_base==M.state_hashes(M.frozen_state(expected)),'EXACT_DEVICE_BASE_VS_CPU_REFERENCE')
  rec.update(initial_base=initial_base,initial_model=M.state_hashes(network.state_dict()),optimizer=M.optimizer_descriptor(opt),restored=bool(checkpoint),checkpoint_sha256=args.checkpoint_sha256 if checkpoint else None)
  dense_adapters=[t.nn.Parameter(p.detach().cpu().clone())for p in M.adapter_parameters(network)];dense_opt=t.optim.SGD(dense_adapters,lr=.01)
  rows=[]
  for step in range(start,end+1):
   A.require(time.time()<args.deadline_epoch,'ORIGINAL_CHILD_DEADLINE');raw={'step':step,'pid':os.getpid(),'process_token':args.process_token,'status':'ERROR'}
   try:
    opt.zero_grad(set_to_none=True);dense_opt.zero_grad(set_to_none=True);x=M.generated(C,'x',step=step);target=M.generated(C,'target',step=step)
    with trace(C,output,'step-%02d'%step,args.deadline_epoch)as observation:
     y=M.forward(network,x.to(C.device))if args.device else dense_forward(C,M.adapter_parameters(network),x)
     dense=dense_forward(C,dense_adapters,x);loss=t.nn.functional.mse_loss(y,target.to(C.device));dense_loss=t.nn.functional.mse_loss(dense,target);loss.backward();dense_loss.backward();opt.step();dense_opt.step()
    if args.device:
     A.require(not observation.observation_errors,'STEP_TYPED_GRAPH_OBSERVATION')
     tensors=[y,loss,*[p.grad for p in M.adapter_parameters(network)],*M.adapter_parameters(network)]
     hlo=C.xla._XLAC._get_xla_tensors_hlo(tensors);(output/'trace'/('step-%02d'%step)/'training.hlo.txt').write_text(hlo)
     raw['hlo_precision']=highest_precision(C,hlo)
     C.xla._XLAC._xla_sync_multi(tensors,devices=[],wait=True,sync_xla_data=True);C.xm.wait_device_ops()
     raw.update(T.canonical({'counters':{k:C.metrics.counter_value(k)for k in C.metrics.counter_names()},'execution_metrics':{k:v for k in C.B.load_spec()[0]['execution_metrics']if(v:=C.metrics.metric_data(k))is not None}}));C.P.validate_execution(C.B,raw)
     raw['all_trainable_and_output_devices']='XLA_TPU';A.require(all(v.device.type=='xla'for v in tensors),'TRAINING_DEVICE_PLACEMENT')
     Z.pack_directory(output,output/'trace'/('step-%02d'%step))
    M.assert_frozen(C,network,initial_base)
    gates={'output':M.compare(C,y.cpu(),dense),'loss':M.compare(C,loss.cpu().reshape(1),dense_loss.reshape(1))}
    gates.update({name:M.compare(C,p.grad.cpu(),ref.grad)for name,p,ref in zip(M.ADAPTER_NAMES,M.adapter_parameters(network),dense_adapters)})
    gates.update({'updated_'+name:M.compare(C,p.cpu(),ref)for name,p,ref in zip(M.ADAPTER_NAMES,M.adapter_parameters(network),dense_adapters)})
    raw.update(status='COMPLETE',loss=loss.item(),output=C.B.tensor_record(y),gradients={name:C.B.tensor_record(p.grad)for name,p in zip(M.ADAPTER_NAMES,M.adapter_parameters(network))},adapters_after={name:C.B.tensor_record(p)for name,p in zip(M.ADAPTER_NAMES,M.adapter_parameters(network))},base_state=M.state_hashes(M.frozen_state(network)),base_gradients='ABSENT',gates=gates)
   except BaseException as error:
    raw.update(error={'type':type(error).__name__,'message':str(error)},traceback=traceback.format_exc())
   A.write(output/'steps'/('%02d.json'%step),raw);rows.append(raw)
   if raw['status']=='ERROR':
    for remaining in range(step+1,end+1):A.write(output/'steps'/('%02d.json'%remaining),{'status':'NOT_RUN','step':remaining,'reason':'AFTER_FIRST_ERROR'})
    raise ValueError('FIRST_TRAINING_ERROR:'+str(step))
  if args.phase=='save':
   state={k:v.detach().cpu().clone()for k,v in network.state_dict().items()};saved={'format':'M5_EXACT1024_CHECKPOINT_V1','completed_step':10,'protocol_sha256':M.PROTOCOL_SHA,'source_bundle_sha256':bundle(),'inputs_sha256':inputs_sha,'adapter_names':M.ADAPTER_NAMES,'model':state,'model_hashes':M.state_hashes(state),'frozen_base':initial_base,'optimizer':M.optimizer_descriptor(opt)};t.save(saved,output/'checkpoint.pt')
  methods(C)
  if not args.device:rec['cpu_evidence']=runtime(t,args.local_reference)
  rec['plugin_loaded_code_post']=getattr(C,'loaded_plugin',None)
  rec.update(status='COMPLETE',source_post=bundle(),final_model=M.state_hashes(network.state_dict()),numerical_status='PASS'if all(g['status']=='PASS'for row in rows for g in row['gates'].values())else'FAIL')
 except BaseException as error:rec.update(error={'type':type(error).__name__,'message':str(error)},traceback=traceback.format_exc())
 finally:rec.update(finished_epoch=time.time(),artifacts=M.load(ROOT/'legacy/probe_backend.py','m5_inv_B').inventory(output,'receipt.json'));A.write(output/'receipt.json',rec)
 return 0 if rec.get('status')=='COMPLETE'and rec.get('numerical_status')=='PASS'else 2

def child_arguments(args):
 values=[]
 if args.local_reference:values+=['--local-reference']
 if args.device:
  values+=['--device']
  for key in ['oracle','oracle_sha256','cpu_gate','cpu_gate_sha256']:values+=['--'+key.replace('_','-'),str(getattr(args,key))]
 if not args.local_reference:
  for key in ['m4_acceptance','m4_acceptance_sha256']:values+=['--'+key.replace('_','-'),str(getattr(args,key))]
 return values

def coordinator(args):
 A.require(os.getpid()==os.getpgid(0),'OWNED_COORDINATOR_GROUP');output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
 parent={'kind':'M5_QUALIFIED_CPU_ORACLE_V1'if not args.device else'M5_DEVICE_TRAINING_PARENT_V1','status':'ERROR','pid':os.getpid(),'pgid':os.getpgid(0),'process_token':args.process_token,'source_bundle_sha256':bundle(),'protocol_sha256':M.PROTOCOL_SHA,'deadline_epoch':args.deadline_epoch,'allocation_epoch':args.allocation_epoch,'started_epoch':time.time(),'qualified_runtime':not args.local_reference,'m4_acceptance_sha256':args.m4_acceptance_sha256,'parent_pid':os.getppid(),'children':[],'not_run_phases':[]}
 try:
  if not args.local_reference:A.m4_dependency(args.m4_acceptance,args.m4_acceptance_sha256)
  if args.device:cpu_gate(args)
  for phase in ['baseline','save','resume']:
   token=secrets.token_hex(16);deadline=min(args.deadline_epoch,time.time()+args.phase_seconds)
   argv=[sys.executable,'-B',str(Path(__file__).resolve()),'child','--phase',phase,'--output',str(output/phase),'--parent-pid',str(os.getpid()),'--parent-process-token',parent['process_token'],'--process-token',token,'--deadline-epoch',str(deadline),*child_arguments(args)]
   if phase=='resume':argv+=['--checkpoint',str(output/'save/checkpoint.pt'),'--checkpoint-sha256',A.sha(output/'save/checkpoint.pt')]
   with(output/(phase+'.stdout')).open('wb')as stdout,(output/(phase+'.stderr')).open('wb')as stderr:
    process=subprocess.Popen(argv,stdout=stdout,stderr=stderr,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    launch={'phase':phase,'launched_pid':process.pid,'pgid':os.getpgid(process.pid),'process_token':token,'parent_pid':parent['pid'],'parent_process_token':parent['process_token'],'argv':argv,'deadline_epoch':deadline,'started_epoch':time.time(),'started_monotonic':time.monotonic()};parent['children'].append(launch);A.write(output/'parent-progress.json',parent)
    try:process.wait(timeout=max(.01,deadline-time.time()))
    except BaseException:
     process.terminate()
     try:process.wait(timeout=3)
     except subprocess.TimeoutExpired:process.kill();process.wait(timeout=3)
     raise
    try:os.kill(process.pid,0)
    except ProcessLookupError:absent=True
    else:absent=False
    launch.update(exit_code=process.returncode,reaped=True,child_absent=absent,finished_epoch=time.time(),finished_monotonic=time.monotonic(),receipt_sha256=A.sha(output/phase/'receipt.json'))
    A.require(absent and process.returncode in(0,2),'CHILD_TERMINAL_CLOSURE');receipt=A.read(output/phase/'receipt.json')
    A.require(receipt['status']=='COMPLETE'and receipt.get('numerical_status')=='PASS','PHASE_FAILED:'+phase)
  parent.update(status='COMPLETE',numerical_status='PASS'if all(A.read(output/p/'receipt.json')['numerical_status']=='PASS'for p in['baseline','save','resume'])else'FAIL')
 except BaseException as error:parent.update(error={'type':type(error).__name__,'message':str(error)},not_run_phases=[p for p in['baseline','save','resume']if p not in[r['phase']for r in parent['children']]])
 finally:
  parent.update(finished_epoch=time.time());(output/'parent-progress.json').unlink(missing_ok=True)
  B=load(ROOT/'legacy/probe_backend.py','m5_parent_B');parent['artifacts']=B.inventory(output,'receipt.json'if args.device else'oracle-seal.json');A.write(output/('receipt.json'if args.device else'oracle-seal.json'),parent)
 return 0 if parent.get('status')=='COMPLETE'and parent.get('numerical_status')=='PASS'else 2

def main():
 p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
 for command in ['prepare','execute','child']:
  a=s.add_parser(command);a.add_argument('--output',type=Path,required=True);a.add_argument('--process-token',required=True);a.add_argument('--deadline-epoch',type=float,required=True);a.add_argument('--local-reference',action='store_true');a.add_argument('--device',action='store_true')
  for name in ['m4-acceptance','m4-acceptance-sha256','oracle','oracle-sha256','cpu-gate','cpu-gate-sha256']:a.add_argument('--'+name)
  if command=='child':
   a.add_argument('--phase',choices=['baseline','save','resume'],required=True);a.add_argument('--parent-pid',type=int,required=True);a.add_argument('--parent-process-token',required=True);a.add_argument('--checkpoint');a.add_argument('--checkpoint-sha256')
  else:a.add_argument('--allocation-epoch',type=float,required=True);a.add_argument('--phase-seconds',type=float,default=240)
 args=p.parse_args();A.require(re.fullmatch('[0-9a-f]{32}',args.process_token)is not None,'PROCESS_TOKEN');A.require(math.isfinite(args.deadline_epoch)and time.time()<args.deadline_epoch,'FINITE_FUTURE_DEADLINE')
 if args.command!='child':A.require(0<args.phase_seconds<=600 and args.allocation_epoch<=time.time()and args.deadline_epoch<=args.allocation_epoch+3600-660,'ORIGINAL_ALLOCATION_SCIENCE_BOUND');args.device=args.command=='execute'
 return child(args)if args.command=='child'else coordinator(args)
if __name__=='__main__':raise SystemExit(main())
