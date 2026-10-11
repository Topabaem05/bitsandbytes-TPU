"""Private CPU QLoRA preparation. No device or milestone qualification."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import traceback
import types

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PROTOCOL={'format':'PRIVATE_QLORA_CPU_V1','layers':2,'width':256,'rank':4,'alpha':4,'adapter_scale':1,'batch':8,'dtype':'float32','loss':'mean_squared_error','optimizer':{'name':'SGD','lr':0.01,'momentum':0,'dampening':0,'weight_decay':0,'nesterov':False,'maximize':False,'foreach':None,'differentiable':False,'fused':None},'steps':20,'checkpoint_after_update':10,'resume_steps':[11,20],'composition':'base0(x)+B0(A0(x)), then base1(h)+B1(A1(h)); no activation/dropout','quantization':{'blocksize':64,'quant_type':'nf4','quant_storage':'uint8','compress_statistics':True},'base':'weights,bias,and_all_nested_state_frozen','reference':'PINNED_DEFAULT_PYTHON_BYTE_BODY_PLUS_INDEPENDENT_DENSE_FP32_AUTOGRAD','qualified':False}
ADAPTER_NAMES=['layers.0.A','layers.0.B','layers.1.A','layers.1.B']
GENERATOR={'base_weight':'((i*17+layer*31)%257-128)/4096','base_bias':'((i+layer*3)%17-8)/128','adapter_A':'((i*7+layer*3)%17-8)/256','adapter_B':'zero','step_x':'((i*13+step*7)%41-20)/32','step_target':'((i*11+step*5)%31-15)/64','rng':'NONE'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
def read(path):return json.loads(Path(path).read_text())
def require(value,name):
    if not value:raise ValueError(name)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
PROTOCOL_SHA=digest(PROTOCOL)
PINS=read(HERE/'dependency-pins.json')
def source_bundle():
    for path,expected in PINS.items():require(sha(ROOT/path)==expected,'DEPENDENCY_SOURCE')
    return digest({'probe':sha(__file__),'dependencies':PINS,'protocol':PROTOCOL_SHA})

def bootstrap(source):
    source_bundle();sys.path.insert(0,str(Path(source).resolve()));sys.path.insert(0,str(ROOT/'.work/r4-nested-preparation/prototype/src'))
    import importlib.metadata
    discover=importlib.metadata.entry_points
    def filtered(*args,**kwargs):return () if kwargs.get('group')=='bitsandbytes.backends' else discover(*args,**kwargs)
    importlib.metadata.entry_points=filtered
    try:import torch,bitsandbytes as bnb
    finally:importlib.metadata.entry_points=discover
    torch.set_num_threads(1)
    require(Path(bnb.__file__).resolve().parent==Path(source).resolve()/'bitsandbytes','ISOLATED_SOURCE')
    B=load(ROOT/'experiments/2026-10-04-bitsandbytes-tpu/probe_backend.py','r5_B');A=load(ROOT/'experiments/2026-10-04-bitsandbytes-tpu/transfer_admission.py','r5_A')
    manifest=A.load_manifest(B,ROOT/'patches/params4bit-xla-v1.json');B.verify_source_tree(Path(bnb.__file__).resolve().parent,A.python_files(manifest['post_patch_package_files']))
    require(not torch.__future__.get_overwrite_module_params_on_conversion() and not torch.__future__.get_swap_module_params_on_conversion(),'DEFAULT_FUTURE_FLAGS')
    from bitsandbytes_tpu import registration as G
    require(not any(torch._C._dispatch_has_kernel_for_dispatch_key('bitsandbytes::'+name,'CPU') for name in G.IMPLEMENTATIONS),'NATIVE_CPU_COLLISION')
    library=torch.library.Library('bitsandbytes','IMPL')
    for name,function in G.IMPLEMENTATIONS.items():library.impl(name,function,'CPU')
    profile,_=B.load_spec()
    return types.SimpleNamespace(torch=torch,bnb=bnb,B=B,A=A,manifest=manifest,library=library,source=Path(source),tolerance=profile['tolerances']['fp32_forward_gradient'])

def methods(C):
    C.A.public_methods(C.B,C.bnb,{'bitsandbytes':C.source/'bitsandbytes'})
    for relative,classname,names in [('nn/modules.py','Params4bit',['from_prequantized']),('nn/modules.py','Linear4bit',['_save_to_state_dict']),('functional.py','QuantState',['as_dict','from_dict'])]:
        path=(C.source/'bitsandbytes'/relative).resolve();codes=compile(path.read_text(),str(path),'exec').co_consts
        cls=getattr(C.bnb.functional if classname=='QuantState' else C.bnb.nn,classname)
        cc=[v for v in codes if isinstance(v,types.CodeType) and v.co_name==classname][0]
        for name in names:
            fn=getattr(cls,name);fn=getattr(fn,'__func__',fn);expected=[v for v in cc.co_consts if isinstance(v,types.CodeType) and v.co_name==name][0]
            require(fn.__code__==expected and Path(fn.__code__.co_filename).resolve()==path,'PUBLIC_STATE_METHOD_REPLACEMENT')
    C.B.verify_source_tree(C.source/'bitsandbytes',C.A.python_files(C.manifest['post_patch_package_files']))
    require(not any(name=='jax' or name.startswith(('jax.','torch_xla','torchax')) for name in sys.modules),'NO_DEVICE_RUNTIME')

def tensor_hash(tensor):
    tensor=tensor.detach().cpu().contiguous();raw=bytes(tensor.reshape(-1).view(dtype=__import__('torch').uint8).tolist())
    return {'shape':list(tensor.shape),'dtype':str(tensor.dtype).removeprefix('torch.'),'sha256':hashlib.sha256(raw).hexdigest()}

def state_hashes(state):return {key:tensor_hash(value) for key,value in sorted(state.items())}
def generated(C,kind,layer=0,step=0):
    t=C.torch
    if kind=='weight':return (((t.arange(256*256)*17+layer*31)%257-128).float()/4096).reshape(256,256)
    if kind=='bias':return ((t.arange(256)+layer*3)%17-8).float()/128
    if kind=='A':return (((t.arange(4*256)*7+layer*3)%17-8).float()/256).reshape(4,256)
    if kind=='B':return t.zeros(256,4)
    if kind=='x':return (((t.arange(8*256)*13+step*7)%41-20).float()/32).reshape(8,256)
    if kind=='target':return (((t.arange(8*256)*11+step*5)%31-15).float()/64).reshape(8,256)
    raise ValueError(kind)
def input_descriptor(C):return {'generator':GENERATOR,'base':[{key:tensor_hash(generated(C,key,layer)) for key in ('weight','bias','A','B')} for layer in range(2)],'steps':{str(step):{key:tensor_hash(generated(C,key,step=step)) for key in ('x','target')} for step in range(1,21)}}

def default_oracle(C):
    path=C.source/'bitsandbytes/backends/default/ops.py';require(sha(path)=='e39afd16dca6e6f34a14305ade0ba4acbb49dc2c7a9baf937341d7e33b682819','DEFAULT_BYTE_SOURCE')
    namespace=dict(vars(__import__('bitsandbytes.backends.default.ops',fromlist=['ops'])));tree=ast.parse(path.read_text())
    def extract(name=None,op=None):
        nodes=[v for v in tree.body if isinstance(v,ast.FunctionDef) and (v.name==name or op and any(isinstance(d,ast.Call) and d.args and isinstance(d.args[0],ast.Constant) and d.args[0].value=='bitsandbytes::'+op for d in v.decorator_list))]
        require(len(nodes)==1,'DEFAULT_FUNCTION');node=copy.deepcopy(nodes[0]);node.decorator_list=[]
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),namespace);return namespace[node.name]
    extract(name='_get_4bit_quantize_bounds')
    return extract(op='quantize_4bit'),extract(op='quantize_blockwise'),extract(name='_dequantize_blockwise_compute'),extract(name='_dequantize_4bit_compute')

def model(C,checkpoint=None):
    t,bnb=C.torch,C.bnb
    class Adapter(t.nn.Module):
        def __init__(self,index):
            super().__init__();self.base=bnb.nn.Linear4bit(256,256,bias=True,compute_dtype=t.float32,compress_statistics=True,quant_type='nf4',quant_storage=t.uint8)
            self.A=t.nn.Parameter(generated(C,'A',index));self.B=t.nn.Parameter(generated(C,'B',index))
            if checkpoint is None:
                with t.no_grad():self.base.weight.copy_(generated(C,'weight',index));self.base.bias.copy_(generated(C,'bias',index))
                self.base=self.base.to('cpu')
            else:
                prefix=f'layers.{index}.base.';saved=checkpoint['model'];stats={key.removeprefix(prefix+'weight.'):value for key,value in saved.items() if key.startswith(prefix+'weight.')}
                self.base.weight=bnb.nn.Params4bit.from_prequantized(saved[prefix+'weight'],dict(stats),requires_grad=False,device='cpu',module=self.base)
            self.base.weight.requires_grad_(False);self.base.bias.requires_grad_(False)
        def forward(self,x):return self.base(x)+(x@self.A.t())@self.B.t()
    network=t.nn.Module();network.layers=t.nn.ModuleList([Adapter(i) for i in range(2)])
    if checkpoint is not None:
        filtered={key:value for key,value in checkpoint['model'].items() if '.weight.' not in key}
        result=network.load_state_dict(filtered,strict=True,assign=False);require(not result.missing_keys and not result.unexpected_keys,'STRICT_MODEL_LOAD')
        require(state_hashes(network.state_dict())==checkpoint['model_hashes'],'EXACT_RESTORED_MODEL')
    return network.train()

def forward(network,x):
    for layer in network.layers:x=layer(x)
    return x

def adapter_parameters(network):
    named=dict(network.named_parameters());require([name for name,p in named.items() if p.requires_grad]==ADAPTER_NAMES,'ADAPTER_MAPPING')
    return [named[name] for name in ADAPTER_NAMES]
def frozen_state(network):return {key:value for key,value in network.state_dict().items() if '.base.' in key}
def assert_frozen(C,network,expected):
    require(state_hashes(frozen_state(network))==expected,'FROZEN_BASE_STATE')
    for layer in network.layers:
        base=layer.base;s=base.weight.quant_state
        require(type(base) is C.bnb.nn.Linear4bit and type(base.weight) is C.bnb.nn.Params4bit and type(s) is C.bnb.functional.QuantState and type(s.state2) is C.bnb.functional.QuantState and s.nested and base.quant_state is s and base.weight.module is base,'PUBLIC_NESTED_ALIASES')
        require(not base.weight.requires_grad and not base.bias.requires_grad and base.weight.grad is None and base.bias.grad is None,'FROZEN_BASE_GRADIENT')

def optimizer(C,network):return C.torch.optim.SGD(adapter_parameters(network),lr=0.01,momentum=0,weight_decay=0)
def optimizer_descriptor(opt):
    state=opt.state_dict();require(len(state['param_groups'])==1 and state['param_groups'][0]['params']==[0,1,2,3] and state['state']=={},'SGD_ADAPTER_MAPPING')
    group=state['param_groups'][0]
    require({key:group[key] for key in PROTOCOL['optimizer'] if key!='name'}=={key:value for key,value in PROTOCOL['optimizer'].items() if key!='name'},'OPTIMIZER_CONFIG')
    return {'class':'torch.optim.SGD','adapter_names':ADAPTER_NAMES,'state':state}
def validate_checkpoint(C,checkpoint,inputs_sha):
    require(type(checkpoint) is dict and checkpoint['format']=='PRIVATE_QLORA_CHECKPOINT_V1','CHECKPOINT_FORMAT')
    require(checkpoint['protocol_sha256']==PROTOCOL_SHA and checkpoint['source_bundle_sha256']==source_bundle() and checkpoint['inputs_sha256']==inputs_sha,'CHECKPOINT_SOURCE_INPUTS')
    require(type(checkpoint['completed_step']) is int and checkpoint['completed_step']==10,'CHECKPOINT_STEP')
    require(checkpoint['adapter_names']==ADAPTER_NAMES,'CHECKPOINT_ADAPTER_MAPPING')
    require(state_hashes(checkpoint['model'])==checkpoint['model_hashes'],'CHECKPOINT_MODEL_HASH')
    expected=model(C);require(state_hashes(frozen_state(expected))==checkpoint['frozen_base'],'CHECKPOINT_BASE_REFERENCE')
    require(state_hashes({k:v for k,v in checkpoint['model'].items() if '.base.' in k})==checkpoint['frozen_base'],'CHECKPOINT_FROZEN_STATE')
    expected_opt=optimizer_descriptor(optimizer(C,expected));require(checkpoint['optimizer']==expected_opt,'CHECKPOINT_OPTIMIZER_CONFIG')
    return checkpoint

def compare(C,actual,expected):
    require(actual.shape==expected.shape and actual.dtype==expected.dtype,'COMPARISON_SHAPE')
    return C.B.numeric(actual.detach().reshape(-1).tolist(),expected.detach().reshape(-1).tolist(),C.tolerance)

def child(args):
    require(os.getppid()==args.parent_pid and os.getpgid(0)==args.parent_pid,'ACTUAL_PARENT_GROUP')
    C=bootstrap(args.source);methods(C);t=C.torch;inputs=input_descriptor(C);inputs_sha=digest(inputs)
    checkpoint=None
    if args.phase=='resume':
        require(sha(args.checkpoint)==args.checkpoint_sha256,'CHECKPOINT_FILE_HASH');checkpoint=validate_checkpoint(C,t.load(args.checkpoint,weights_only=True,map_location='cpu'),inputs_sha)
    network=model(C,checkpoint);opt=optimizer(C,network)
    if checkpoint:opt.load_state_dict(checkpoint['optimizer']['state'])
    initial_base=state_hashes(frozen_state(network));assert_frozen(C,network,initial_base)
    oq4,oq8,od8,od4=default_oracle(C);decoded=[]
    for index,layer in enumerate(network.layers):
        packed,scales=oq4(generated(C,'weight',index),64,'nf4',t.uint8);s=layer.base.weight.quant_state;offset=scales.mean();codes,second=oq8(scales-offset,s.state2.code,256)
        require(t.equal(packed,layer.base.weight.data) and t.equal(codes,s.absmax) and t.equal(offset,s.offset) and t.equal(second,s.state2.absmax),'DEFAULT_NESTED_BYTES')
        decoded.append(od4(packed.reshape(-1),od8(codes,second,s.state2.code,256,t.float32)+offset,s.code,64,(256,256),t.float32))
    dense_adapters=[t.nn.Parameter(p.detach().clone()) for p in adapter_parameters(network)];dense_opt=t.optim.SGD(dense_adapters,lr=0.01)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    receipt={'status':'PARTIAL','phase':args.phase,'pid':os.getpid(),'pgid':os.getpgid(0),'parent_pid':args.parent_pid,'process_token':args.process_token,'parent_process_token':args.parent_process_token,'protocol_sha256':PROTOCOL_SHA,'source_bundle_sha256':source_bundle(),'probe_sha256':sha(__file__),'inputs_sha256':inputs_sha,'runtime':{'torch':t.__version__,'python':sys.version,'platform':sys.platform,'qualified':False},'initial_base':initial_base,'initial_model':state_hashes(network.state_dict()),'optimizer':optimizer_descriptor(opt),'restored':args.phase=='resume','checkpoint_sha256':args.checkpoint_sha256 if checkpoint else None,'public_method_identity':True,'tpu_cuda_performance':'NOT_RUN'}
    write(output/'inputs.json',inputs);write(output/'receipt.json',receipt)
    start=11 if checkpoint else 1;end=10 if args.phase=='save' else 20
    rows=[]
    for step in range(start,end+1):
        require(time.time()<args.deadline_epoch,'CHILD_DEADLINE');opt.zero_grad(set_to_none=True);dense_opt.zero_grad(set_to_none=True)
        x=generated(C,'x',step=step);target=generated(C,'target',step=step);y=forward(network,x);dense=x
        for index in range(2):
            a,b=dense_adapters[index*2:index*2+2];dense=t.nn.functional.linear(dense,decoded[index],network.layers[index].base.bias)+(dense@a.t())@b.t()
        loss=t.nn.functional.mse_loss(y,target);dense_loss=t.nn.functional.mse_loss(dense,target);loss.backward();dense_loss.backward()
        grads={name:C.B.tensor_record(p.grad) for name,p in zip(ADAPTER_NAMES,adapter_parameters(network))}
        gates={'output':compare(C,y,dense),'loss':compare(C,loss.reshape(1),dense_loss.reshape(1))}
        gates.update({name:compare(C,p.grad,ref.grad) for name,p,ref in zip(ADAPTER_NAMES,adapter_parameters(network),dense_adapters)})
        opt.step();dense_opt.step();assert_frozen(C,network,initial_base)
        gates.update({'updated_'+name:compare(C,p,ref) for name,p,ref in zip(ADAPTER_NAMES,adapter_parameters(network),dense_adapters)})
        raw={'step':step,'pid':os.getpid(),'process_token':args.process_token,'loss':loss.item(),'output':C.B.tensor_record(y),'gradients':grads,'adapters_after':{name:C.B.tensor_record(p) for name,p in zip(ADAPTER_NAMES,adapter_parameters(network))},'base_state':state_hashes(frozen_state(network)),'base_gradients':'ABSENT','gates':gates}
        write(output/'steps'/f'{step:02d}.json',raw);rows.append(raw)
    if args.phase=='save':
        state={key:value.detach().cpu().clone() for key,value in network.state_dict().items()}
        saved={'format':'PRIVATE_QLORA_CHECKPOINT_V1','protocol_sha256':PROTOCOL_SHA,'source_bundle_sha256':source_bundle(),'inputs_sha256':inputs_sha,'completed_step':10,'adapter_names':ADAPTER_NAMES,'model':state,'model_hashes':state_hashes(state),'frozen_base':initial_base,'optimizer':optimizer_descriptor(opt)}
        t.save(saved,output/'checkpoint.pt')
    methods(C);receipt.update(status='COMPLETE',step_range=[start,end],final_model=state_hashes(network.state_dict()),numerical_status='PASS' if all(g['status']=='PASS' for row in rows for g in row['gates'].values()) else 'FAIL',artifacts=C.B.inventory(output,'receipt.json'));write(output/'receipt.json',receipt)
    return 0 if receipt['numerical_status']=='PASS' else 2

def coordinator(args):
    require(os.getpid()==os.getpgid(0),'OWNED_COORDINATOR_GROUP');output=Path(args.output);output.mkdir(exist_ok=False)
    B=load(ROOT/'experiments/2026-10-04-bitsandbytes-tpu/probe_backend.py','r5_owner_B')
    parent={'status':'PARTIAL','pid':os.getpid(),'pgid':os.getpgid(0),'process_token':secrets.token_hex(16),'protocol_sha256':PROTOCOL_SHA,'source_bundle_sha256':source_bundle(),'deadline_epoch':time.time()+330,'children':[]}
    for phase in ('baseline','save','resume'):
        token=secrets.token_hex(16);deadline=min(parent['deadline_epoch'],time.time()+100)
        cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'child','--source',str(Path(args.source).resolve()),'--phase',phase,'--output',str(output/phase),'--parent-pid',str(parent['pid']),'--parent-process-token',parent['process_token'],'--process-token',token,'--deadline-epoch',str(deadline)]
        if phase=='resume':cmd+=['--checkpoint',str(output/'save/checkpoint.pt'),'--checkpoint-sha256',sha(output/'save/checkpoint.pt')]
        with (output/(phase+'.stdout')).open('w') as stdout,(output/(phase+'.stderr')).open('w') as stderr:
            process=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            launch={'phase':phase,'launched_pid':process.pid,'pgid':os.getpgid(process.pid),'process_token':token,'parent_pid':parent['pid'],'parent_process_token':parent['process_token'],'argv':cmd,'deadline_epoch':deadline,'started_epoch':time.time(),'started_monotonic':time.monotonic()}
            parent['children'].append(launch);write(output/'parent.json',parent)
            try:process.wait(timeout=max(deadline-time.time(),0.01))
            except subprocess.TimeoutExpired:
                process.terminate()
                try:process.wait(timeout=3)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=3)
                raise ValueError('OWNED_CHILD_TIMEOUT')
            try:os.kill(process.pid,0)
            except ProcessLookupError:absent=True
            else:absent=False
            launch.update(exit_code=process.returncode,reaped=True,child_absent=absent,finished_epoch=time.time(),finished_monotonic=time.monotonic())
            require(absent and process.returncode in (0,2),'CHILD_TERMINAL')
            launch['receipt_sha256']=sha(output/phase/'receipt.json');write(output/'parent.json',parent)
    parent.update(status='COMPLETE',artifacts=B.inventory(output,'parent.json'));write(output/'parent.json',parent);return 0

def validate_array(C,value,shape):
    require(type(value) is dict and value.get('shape')==shape and value.get('dtype')=='float32','STEP_ARRAY_SHAPE')
    require(type(value.get('values')) is list and len(value['values'])==math.prod(shape) and all(type(v) in (int,float) and math.isfinite(v) for v in value['values']),'STEP_ARRAY_VALUES')
    return C.torch.tensor(value['values'],dtype=C.torch.float32).reshape(shape)

def dense_replay(C,expected_model):
    """Independent public dense autograd replay from admitted default byte bodies."""
    t=C.torch;oq4,oq8,od8,od4=default_oracle(C);weights=[]
    for index,layer in enumerate(expected_model.layers):
        packed,scales=oq4(generated(C,'weight',index),64,'nf4',t.uint8);offset=scales.mean();state=layer.base.weight.quant_state
        codes,second=oq8(scales-offset,state.state2.code,256)
        require(t.equal(packed,layer.base.weight.data) and t.equal(codes,state.absmax) and t.equal(second,state.state2.absmax) and t.equal(offset,state.offset),'VERIFIER_DEFAULT_BYTE_REFERENCE')
        weights.append(od4(packed.reshape(-1),od8(codes,second,state.state2.code,256,t.float32)+offset,state.code,64,(256,256),t.float32))
    adapters=[t.nn.Parameter(generated(C,key,index)) for index in range(2) for key in ('A','B')];opt=t.optim.SGD(adapters,lr=0.01);result={}
    for step in range(1,21):
        opt.zero_grad(set_to_none=True);value=generated(C,'x',step=step)
        for index in range(2):
            a,b=adapters[index*2:index*2+2];value=t.nn.functional.linear(value,weights[index],generated(C,'bias',index))+(value@a.t())@b.t()
        loss=t.nn.functional.mse_loss(value,generated(C,'target',step=step));loss.backward()
        record={'loss':loss.item(),'output':C.B.tensor_record(value),'gradients':{name:C.B.tensor_record(p.grad) for name,p in zip(ADAPTER_NAMES,adapters)}}
        opt.step();record['adapters_after']={name:C.B.tensor_record(p) for name,p in zip(ADAPTER_NAMES,adapters)};result[step]=record
    return result


def verify(actual,source,closure=None):
    root=Path(actual);C=bootstrap(source);methods(C);inputs_sha=digest(input_descriptor(C));parent=read(root/'parent.json')
    require(parent['status']=='COMPLETE' and parent['protocol_sha256']==PROTOCOL_SHA and parent['source_bundle_sha256']==source_bundle(),'PARENT_SOURCE_TERMINAL');C.B.check_seal(root,'parent.json',parent)
    require(parent['pid']==parent['pgid'] and parent['pid']>0,'PARENT_GROUP')
    launches=parent['children'];require([row['phase'] for row in launches]==['baseline','save','resume'] and len({parent['pid'],*[r['launched_pid'] for r in launches]})==4,'FRESH_PROCESS_SEQUENCE')
    require(len({parent['process_token'],*[r['process_token'] for r in launches]})==4,'FRESH_TOKEN')
    expected_model=model(C);expected_base=state_hashes(frozen_state(expected_model));expected_optimizer=optimizer_descriptor(optimizer(C,expected_model));rows={};receipts={};dense_rows=dense_replay(C,expected_model)
    for index,launch in enumerate(launches):
        phase=launch['phase'];receipt=read(root/phase/'receipt.json');receipts[phase]=receipt
        require(launch['reaped'] is True and launch['child_absent'] is True and launch['exit_code'] in (0,2),'CHILD_TERMINAL')
        require(receipt['pid']==launch['launched_pid'] and receipt['pgid']==launch['pgid']==parent['pgid'] and receipt['parent_pid']==parent['pid'] and receipt['parent_process_token']==parent['process_token'] and receipt['process_token']==launch['process_token'],'ACTUAL_CHILD_BINDING')
        require(sha(root/phase/'receipt.json')==launch['receipt_sha256'],'CHILD_RECEIPT_HASH')
        require(all(type(launch[k]) in (int,float) and math.isfinite(launch[k]) for k in ('started_epoch','finished_epoch','deadline_epoch','started_monotonic','finished_monotonic')) and launch['started_epoch']<=launch['finished_epoch']<=launch['deadline_epoch']<=parent['deadline_epoch'],'CHILD_DEADLINE')
        if index:require(launches[index-1]['finished_monotonic']<=launch['started_monotonic'],'SEQUENTIAL_CHILD_CLOSURE')
        require(receipt['status']=='COMPLETE' and receipt['probe_sha256']==sha(__file__) and receipt['source_bundle_sha256']==source_bundle() and receipt['protocol_sha256']==PROTOCOL_SHA and receipt['inputs_sha256']==inputs_sha and receipt['public_method_identity'] is True,'CHILD_SOURCE_INPUTS')
        require(receipt['initial_base']==expected_base and receipt['optimizer']==expected_optimizer,'INITIAL_BASE_OPTIMIZER')
        require(read(root/phase/'inputs.json')==input_descriptor(C),'INPUT_TENSOR_SEAL');C.B.check_seal(root/phase,'receipt.json',receipt)
        expected_steps=list(range(11,21)) if phase=='resume' else list(range(1,11 if phase=='save' else 21))
        require(receipt['step_range']==[expected_steps[0],expected_steps[-1]] and sorted(p.name for p in (root/phase/'steps').iterdir())==[f'{step:02d}.json' for step in expected_steps],'STEP_SEQUENCE')
        phase_rows={}
        for step in expected_steps:
            raw=read(root/phase/'steps'/f'{step:02d}.json');require(raw['step']==step and raw['pid']==receipt['pid'] and raw['process_token']==receipt['process_token'],'STEP_IDENTITY')
            require(type(raw['loss']) in (float,int) and math.isfinite(raw['loss']) and raw['loss']>=0,'STEP_LOSS')
            output=validate_array(C,raw['output'],[8,256]);loss=C.torch.nn.functional.mse_loss(output,generated(C,'target',step=step)).item();require(abs(raw['loss']-loss)<=C.tolerance['atol']+C.tolerance['rtol']*abs(loss),'LOSS_OUTPUT_BINDING')
            require(set(raw['gradients'])==set(raw['adapters_after'])==set(ADAPTER_NAMES),'STEP_ADAPTER_SET')
            for name in ADAPTER_NAMES:
                shape=[4,256] if name.endswith('.A') else [256,4];validate_array(C,raw['gradients'][name],shape);validate_array(C,raw['adapters_after'][name],shape)
            require(raw['base_state']==expected_base and raw['base_gradients']=='ABSENT','FROZEN_BASE_STATE')
            require(set(raw['gates'])=={'output','loss',*ADAPTER_NAMES,*['updated_'+n for n in ADAPTER_NAMES]},'DENSE_GATE_SET');phase_rows[step]=raw
        rows[phase]=phase_rows
    checkpoint=C.torch.load(root/'save/checkpoint.pt',weights_only=True,map_location='cpu');validate_checkpoint(C,checkpoint,inputs_sha)
    require(receipts['resume']['checkpoint_sha256']==sha(root/'save/checkpoint.pt') and receipts['resume']['restored'] is True and receipts['resume']['initial_model']==checkpoint['model_hashes'],'EXACT_FRESH_RESTORE')
    require(checkpoint['model_hashes']==receipts['save']['final_model'],'SAVED_FINAL_MODEL')
    require(all(C.B.tensor_record(checkpoint['model'][name])==rows['save'][10]['adapters_after'][name] for name in ADAPTER_NAMES),'CHECKPOINT_STEP10_ADAPTERS')
    gates={}
    for phase in ('save','resume'):
        for step,raw in rows[phase].items():
            baseline=rows['baseline'][step];stage={}
            for key in ('output',):stage[key]=C.B.numeric(raw[key]['values'],baseline[key]['values'],C.tolerance)
            stage['loss']=C.B.numeric([raw['loss']],[baseline['loss']],C.tolerance)
            for key in ('gradients','adapters_after'):
                for name in ADAPTER_NAMES:stage[key+'/'+name]=C.B.numeric(raw[key][name]['values'],baseline[key][name]['values'],C.tolerance)
            gates[phase+'/'+str(step)]=stage
    dense_gates={}
    for phase,phase_rows in rows.items():
        for step,raw in phase_rows.items():
            reference=dense_rows[step];stage={'output':C.B.numeric(raw['output']['values'],reference['output']['values'],C.tolerance),'loss':C.B.numeric([raw['loss']],[reference['loss']],C.tolerance)}
            for key in ('gradients','adapters_after'):
                for name in ADAPTER_NAMES:stage[key+'/'+name]=C.B.numeric(raw[key][name]['values'],reference[key][name]['values'],C.tolerance)
            dense_gates[phase+'/'+str(step)]=stage
    exact=receipts['baseline']['final_model']==receipts['resume']['final_model']
    if closure is not None:
        summary=read(closure);require(summary['status']=='NO_CLEANUP_ERRORS' and len(summary['groups'])==1,'OUTER_CLOSURE');group=summary['groups'][0];require(group['pid']==parent['pid'] and group['pgid']==parent['pgid'] and group['leader_reaped'] is True and group['group_absence']=='OBSERVED_NO_SUCH_GROUP','OUTER_ACTUAL_PARENT')
    passed=exact and all(g['status']=='PASS' for stage in dense_gates.values() for g in stage.values()) and all(g['status']=='PASS' for stage in gates.values() for g in stage.values()) and all(g['status']=='PASS' for phase in rows.values() for raw in phase.values() for g in raw['gates'].values())
    return {'record_validation':'PASS','numerical_status':'PASS' if passed else 'FAIL','m5_status':'NOT_QUALIFIED','qualified_runtime':False,'steps':20,'checkpoint_step':10,'resumed_steps':10,'exact_final_model':exact,'gates':gates,'independent_dense_gates':dense_gates,'source_bundle_sha256':source_bundle(),'inputs_sha256':inputs_sha,'tpu_cuda_performance':'NOT_RUN'}

def run(args):
    source_bundle();output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(ROOT/'experiments/2026-10-04-bitsandbytes-tpu/cloud/ownership'));from cleanup_lifecycle import Ownership
    owner=Ownership(output)
    with (output/'coordinator.stdout').open('w') as stdout,(output/'coordinator.stderr').open('w') as stderr:
        with owner.guard(deadline_seconds=360):
            process=owner.launch([sys.executable,'-B',str(Path(__file__).resolve()),'coordinator','--source',str(Path(args.source).resolve()),'--output',str((output/'actual').resolve())],record=output/'ownership.json',stdout=stdout,stderr=stderr)
            process.wait(timeout=340);require(process.returncode==0,'COORDINATOR_FAILURE')
    summary=owner.summary();write(output/'closure.json',summary)
    result=verify(output/'actual',args.source,output/'closure.json');write(output/'review.json',result)
    print(json.dumps({'record_validation':result['record_validation'],'numerical_status':result['numerical_status'],'steps':20,'exact_final_model':result['exact_final_model']}))
    return 0 if result['numerical_status']=='PASS' else 2

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for command in ('run','coordinator'):
        p=sub.add_parser(command);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('child')
    for name in ('source','phase','output','parent-process-token','process-token'):p.add_argument('--'+name,required=True)
    p.add_argument('--parent-pid',type=int,required=True);p.add_argument('--deadline-epoch',type=float,required=True);p.add_argument('--checkpoint');p.add_argument('--checkpoint-sha256')
    args=parser.parse_args()
    return {'run':run,'coordinator':coordinator,'child':child}[args.command](args)
if __name__=='__main__':raise SystemExit(main())
