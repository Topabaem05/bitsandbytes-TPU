"""Private M4 nested-state supplement. No allocation or milestone acceptance."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import time
import traceback
import types

NESTED_PIN='9cfbc7c6b9841b365b47ecfd0bf1b184fda754208aec5f214fe3d0cd9198348d'
KIND='PRIVATE_NESTED_STATE_PROBE'
PROTOCOL={'format':'NESTED_STATE_8_V1','cases':'eight frozen nested79 module cases',
 'statistics':'original Linear4bit default compress_statistics=True',
 'oracle':'qualified nested79 default Python byte bodies and independent dense gradients',
 'checkpoint':'plain CPU full state_dict; torch.load(weights_only=True,map_location=cpu)',
 'restore':'fresh original classes, QuantState.from_dict, Params4bit.from_prequantized, strict load_state_dict(assign=False)',
 'state':'exact checkpoint tensors and metadata, offset, state2, code maps, aliases',
 'process':'owned parent PID==PGID, sequential fresh children inherit PGID, durable launch before wait and reap',
 'precision':'highest once before TPU graph construction','bf16_gradients':'NOT_RUN','m4_status':'NOT_QUALIFIED'}

def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def helpers(args):
 if hashlib.sha256(Path(args.nested_probe).read_bytes()).hexdigest()!=NESTED_PIN:raise ValueError('NESTED_PROBE_PIN')
 N=load(args.nested_probe,'nested_state_reference');return (*N.helpers(args),N)

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def cases(B,N):return [c for c in N.spec(B)[0]['cases'] if c['kind']=='module']
def binding(B,N,args):
 profile,_=B.load_spec()
 return {'kind':KIND,'probe_sha256':B.sha(__file__),'nested_probe_sha256':NESTED_PIN,'helper_sha256':N.PINS,
 'protocol_sha256':digest(PROTOCOL),'inputs_sha256':N.INPUT_SHA,'schema_sha256':N.SCHEMA_SHA,
 'source_variant':'nested-v1','state_variant':'nested-state-v1','nested_source_sha256':N.SOURCE_SHA,'default_body_sha256':N.DEFAULT_SHA,
 'source_admission_sha256':args.admission_sha256,'patch_manifest_sha256':B.sha(args.patch_manifest),
 'profile_sha256':B.PROFILE_SHA,'runtime_lock_sha256':profile['runtime_lock_sha256'],
 'nested_oracle_sha256':args.nested_oracle_sha256,'case_ids':[c['id'] for c in cases(B,N)],
 'm4_status':'NOT_QUALIFIED','native_cpu_golden':'NOT_RUN','cuda_golden':'NOT_RUN','hlo_scope':'PRE_OPTIMIZATION_OUTPUT_GRAPH'}

def nested_oracle(B,P,N,args):
 data=types.SimpleNamespace(**vars(args));data.oracle=args.nested_oracle;data.oracle_sha256=args.nested_oracle_sha256
 return N.validate_oracle(B,P,data)

def state_methods(B,A,N,bnb,roots):
 result=N.public_methods(B,A,bnb,roots)
 path=(roots['bitsandbytes']/'nn/modules.py').resolve();compiled=compile(path.read_text(),str(path),'exec')
 for classname,names in [('Params4bit',('from_prequantized',)),('Linear4bit',('_save_to_state_dict',))]:
  cls=getattr(bnb.nn,classname);cc=[c for c in compiled.co_consts if isinstance(c,types.CodeType) and c.co_name==classname][0]
  for name in names:
   fn=getattr(cls,name);fn=getattr(fn,'__func__',fn);expected=[c for c in cc.co_consts if isinstance(c,types.CodeType) and c.co_name==name][0]
   B.require(fn.__code__==expected and Path(fn.__code__.co_filename).resolve()==path,'PUBLIC_STATE_METHOD_REPLACEMENT')
 return dict(result,original_state_method_identity=True)

def records(B,N,state):return {k:N.tensor_record(B,v) for k,v in sorted(state.items())}
def state_keys(case):
 keys={'weight','weight.absmax','weight.quant_map','weight.nested_absmax','weight.nested_quant_map','weight.quant_state.bitsandbytes__nf4'}
 if case['bias'] is not None:keys.add('bias')
 return keys

def validate_array(B,N,v):
 B.require(type(v) is dict and v.get('dtype') in ('float32','bfloat16','uint8') and type(v.get('shape')) is list and all(type(n) is int and n>0 for n in v['shape']),'STATE_ARRAY_SCHEMA')
 values=v.get('values');B.require(type(values) is list and len(values)==math.prod(v['shape']) and all(type(x) in (int,float) and math.isfinite(x) for x in values),'STATE_ARRAY_FINITE')
 if v['dtype']=='uint8':B.require(all(type(x) is int and 0<=x<256 for x in values),'STATE_ARRAY_UINT8')
 B.require(v.get('bytes_sha256')==hashlib.sha256(N.binary(v)).hexdigest(),'STATE_ARRAY_BYTES')
 import struct
 if v['dtype']=='float32':B.require(all(struct.unpack('<f',struct.pack('<f',x))[0]==x for x in values),'STATE_ARRAY_FP32')
 if v['dtype']=='bfloat16':B.require(all(struct.pack('<f',x)[:2]==b'\0\0' for x in values),'STATE_ARRAY_BF16')

def validate_state(B,N,case,state):
 B.require(type(state) is dict and set(state)==state_keys(case),'STATE_KEYS')
 for v in state.values():validate_array(B,N,v)
 count=math.prod(case['weight_shape']);B.require(state['weight']['shape']==[(count+1)//2,1] and state['weight']['dtype']=='uint8','STATE_PACKED')
 scales=(count+63)//64
 B.require(state['weight.absmax']['shape']==[scales] and state['weight.absmax']['dtype']=='uint8' and state['weight.nested_absmax']['shape']==[(scales+255)//256] and state['weight.nested_absmax']['dtype']=='float32','STATE_SCALE_SHAPE')
 B.require(state['weight.quant_map']['shape']==[16] and state['weight.nested_quant_map']['shape']==[256] and state['weight.quant_map']['dtype']==state['weight.nested_quant_map']['dtype']=='float32','STATE_MAP_SHAPE')
 if case['bias'] is not None:B.require(state['bias']['shape']==[case['weight_shape'][0]] and state['bias']['dtype']==case['dtype'],'STATE_BIAS')
 packed=state['weight.quant_state.bitsandbytes__nf4'];B.require(packed['dtype']=='uint8' and packed['shape']==[len(packed['values'])],'PACKED_METADATA_DTYPE')
 meta=json.loads(bytes(packed['values']).decode())
 expected={'quant_type':'nf4','blocksize':64,'dtype':case['dtype'],'shape':case['weight_shape'],'nested_blocksize':256,'nested_dtype':'float32'}
 B.require(set(meta)==set(expected)|{'nested_offset'} and all(meta[k]==v for k,v in expected.items()) and type(meta['nested_offset']) in (int,float) and math.isfinite(meta['nested_offset']),'PACKED_NESTED_METADATA')

def checkpoint(B,N,torch,path,case):
 state=torch.load(path,map_location='cpu',weights_only=True)
 B.require(type(state) is dict and set(state)==state_keys(case) and all(type(v) is torch.Tensor and v.device.type=='cpu' for v in state.values()),'CHECKPOINT_PLAIN_CPU_STATE')
 raw=records(B,N,state);validate_state(B,N,case,raw);return state,raw

def original_module(B,case,bnb,torch,device):
 module=bnb.nn.Linear4bit(case['weight_shape'][1],case['weight_shape'][0],bias=case['bias'] is not None,compute_dtype=getattr(torch,case['dtype']),quant_type='nf4',quant_storage=torch.uint8).to(dtype=getattr(torch,case['dtype']))
 B.require(module.weight.compress_statistics is True,'DEFAULT_STATISTICS')
 with torch.no_grad():
  module.weight.copy_(torch.tensor(case['weight'],dtype=getattr(torch,case['dtype'])).reshape(case['weight_shape']))
  if module.bias is not None:module.bias.copy_(torch.tensor(case['bias'],dtype=getattr(torch,case['dtype'])))
 return module.to(device).train()

def target_module(B,case,bnb,torch,device):
 # Accepted M3 target-constructor boundary, with original default nested statistics.
 module=bnb.nn.Linear4bit(case['weight_shape'][1],case['weight_shape'][0],bias=case['bias'] is not None,compute_dtype=getattr(torch,case['dtype']),quant_type='nf4',quant_storage=torch.uint8,device=device).to(dtype=getattr(torch,case['dtype']))
 B.require(module.weight.compress_statistics is True and module.weight.device==torch.device(device) and (module.bias is None or module.bias.device==torch.device(device)),'TARGET_CONSTRUCTOR')
 return module.train()

def restore(B,N,case,state,bnb,torch,device):
 module=target_module(B,case,bnb,torch,device);original_bias=module.bias
 old=module.weight;stats={k.removeprefix('weight.'):v for k,v in state.items() if k.startswith('weight.')}
 # Both public restoration entry points execute with separate mutable dictionaries.
 qs=bnb.functional.QuantState.from_dict(dict(stats),device=device)
 parameter=bnb.nn.Params4bit.from_prequantized(state['weight'],dict(stats),requires_grad=False,device=device,module=module);module.weight=parameter
 B.require(parameter is not old and parameter.quant_state is not qs,'FRESH_STATE_IDENTITY')
 result=module.load_state_dict({k:state[k] for k in ('weight','bias') if k in state},strict=True,assign=False)
 B.require(not result.missing_keys and not result.unexpected_keys and module.weight is parameter and module.bias is original_bias,'STRICT_RESTORE')
 # as_dict/from_dict retain every nested tensor and packed non-tensor field.
 left=records(B,N,{k:v.detach().cpu() for k,v in qs.as_dict(packed=True).items()})
 right=records(B,N,{k:v.detach().cpu() for k,v in parameter.quant_state.as_dict(packed=True).items()})
 B.require(left==right,'QUANTSTATE_PUBLIC_ROUNDTRIP')
 return module.train()

def public_roundtrip(B,N,state,bnb,device):
 packed=state.as_dict(packed=True);restored=bnb.functional.QuantState.from_dict(dict(packed),device=device)
 left=records(B,N,{k:v.detach().cpu() for k,v in packed.items()})
 right=records(B,N,{k:v.detach().cpu() for k,v in restored.as_dict(packed=True).items()})
 B.require(left==right and restored.nested and restored is not state,'QUANTSTATE_PUBLIC_ROUNDTRIP')
 return True

def oracle_state(B,N,case,raw,bnb,torch):
 o=raw['outputs'];t=lambda key:torch.tensor(o[key]['values'],dtype=getattr(torch,o[key]['dtype'])).reshape(o[key]['shape'])
 state2=bnb.functional.QuantState(absmax=t('second_scales'),code=t('dynamic_map'),blocksize=256,dtype=torch.float32)
 state=bnb.functional.QuantState(absmax=t('scale_codes'),code=t('nf4_map'),blocksize=64,dtype=getattr(torch,case['dtype']),shape=tuple(case['weight_shape']),quant_type='nf4',offset=t('offset'),state2=state2)
 result={'weight':t('packed'),**{'weight.'+k:v for k,v in state.as_dict(packed=True).items()}}
 if case['bias'] is not None:result['bias']=t('bias')
 return result

# Passed explicitly to avoid a hidden live class or method replacement.
def compute(B,N,case,module,torch,bnb,device):
 state=module.weight.quant_state;before=module.weight.data.detach().clone();original=module.weight
 decoded=bnb.functional.dequantize_4bit(module.weight.data,state);target=torch.empty_like(decoded);decoded_out=bnb.functional.dequantize_4bit(module.weight.data,state,out=target)
 B.require(decoded_out is target,'PUBLIC_OUT_IDENTITY');x=torch.tensor(case['x'],dtype=getattr(torch,case['dtype']),device=device).reshape(case['x_shape']).requires_grad_(case['dtype']=='float32');y=module(x)
 outputs={'packed':module.weight.data,'scale_codes':state.absmax,'offset':state.offset,'second_scales':state.state2.absmax,'dynamic_map':state.state2.code,'nf4_map':state.code,
  'restored_scales':bnb.functional.dequantize_blockwise(state.absmax,quant_state=state.state2)+state.offset,'decoded':decoded,'decoded_out':decoded_out,'y':y}
 if case['dtype']=='float32':
  y.backward(torch.tensor([((i%5)-2)/8 for i in range(y.numel())],dtype=torch.float32,device=device).reshape(y.shape));outputs['dx']=x.grad
  if module.bias is not None:outputs['db']=module.bias.grad
 if module.bias is not None:outputs['bias']=module.bias
 details={'metadata':N.metadata(case),'out_identity':True,'gradients':'EXECUTED' if case['dtype']=='float32' else 'NOT_RUN','default_compress_statistics':module.weight.compress_statistics is True,
  'frozen_weight':not module.weight.requires_grad and module.weight.grad is None,'module_state_alias':module.quant_state is state,'parameter_module_alias':module.weight.module is module,'parameter_identity':module.weight is original,
  'original_classes':type(module) is bnb.nn.Linear4bit and type(module.weight) is bnb.nn.Params4bit,'bias_class':None if module.bias is None else 'torch.nn.parameter.Parameter' if type(module.bias) is torch.nn.Parameter else 'INVALID',
  '_packed_before':before,'_packed_after':module.weight.data}
 B.require(type(state) is bnb.functional.QuantState and type(state.state2) is bnb.functional.QuantState and state.nested and state.blocksize==64 and state.state2.blocksize==256 and list(state.shape)==case['weight_shape'] and state.dtype==getattr(torch,case['dtype']) and state.state2.dtype==torch.float32 and state.quant_type=='nf4','NESTED_STATE_CLASSES_FORMAT')
 B.require(all(v.device==device for v in (module.weight,state.absmax,state.code,state.offset,state.state2.absmax,state.state2.code)) and (module.bias is None or module.bias.device==device),'NESTED_DEVICE')
 return outputs,details

def validate_receipt(B,P,N,args,root,name,tpu=False,parent=False):
 rec=B.read(root/name)
 for k,v in binding(B,N,args).items():B.require(rec.get(k)==v,'BINDING_'+k)
 B.require(rec.get('status')=='COMPLETE' and rec.get('source_pre')==rec.get('source_post')==args.admission_sha256,'TERMINAL_SOURCE')
 B.require(type(rec.get('pid')) is int and rec['pid']>0 and type(rec.get('pgid')) is int and rec['pgid']>0 and isinstance(rec.get('process_token'),str) and re.fullmatch('[0-9a-f]{32}',rec['process_token']),'PROCESS_IDENTITY')
 if parent:B.require(rec['pid']==rec['pgid'],'PARENT_GROUP_LEADER')
 else:
  _,_,schemas=N.spec(B);B.require(rec.get('public_api')==dict(N.PUBLIC,original_state_method_identity=True) and rec.get('schemas')==schemas,'PUBLIC_SCHEMA_IDENTITY')
  if tpu:
   profile,_=B.load_spec();B.require(rec.get('runtime')==profile['runtime'] and rec.get('device')=={'type':'xla','hardware':'TPU','pjrt':'TPU'},'TPU_RUNTIME')
   B.require(rec.get('dispatch')=={k:True for k in schemas},'NESTED_DISPATCH');P.validate_environment(B,rec.get('precision_environment'))
   B.require(rec.get('precision')=={'requested':'highest','readback':'highest','set_calls':1,'before_graph':True},'PRECISION')
  else:
   B.require(rec.get('runtime')=={'python':'3.12','torch':'2.9.0+cpu','platform':'linux','machine':'x86_64'} and rec.get('oracle_method')=='NESTED79_DEFAULT_BODY_SEAL_PLUS_ORIGINAL_PUBLIC_PACKED_STATE','CPU_ORACLE_RUNTIME')
 B.check_seal(root,name,rec);return rec

def validate_oracle(B,P,N,args):
 _,all_refs=nested_oracle(B,P,N,args);root=Path(args.oracle);B.require(B.sha(root/'oracle-seal.json')==args.oracle_sha256,'ORACLE_HASH');rec=validate_receipt(B,P,N,args,root,'oracle-seal.json')
 required=set();references={};import torch
 for case in cases(B,N):
  stem='raw/'+case['id'];required.update(stem+s for s in ('.json','.state.pt'));raw=B.read(root/(stem+'.json'));B.require(raw.get('case_id')==case['id'] and raw.get('input_sha256')==B.case_sha(case) and raw.get('pid')==rec['pid'] and raw.get('process_token')==rec['process_token'],'ORACLE_ROW_BINDING')
  N.validate_arrays(B,P,case,raw,False);_,state=checkpoint(B,N,torch,root/(stem+'.state.pt'),case);B.require(raw.get('state_dict')==state,'ORACLE_CHECKPOINT_BINDING')
  reference=all_refs[case['id']];B.require(raw['outputs']==reference['outputs'],'ORACLE_NESTED79_EXACT_OUTPUTS')
  meta=json.loads(bytes(state['weight.quant_state.bitsandbytes__nf4']['values']).decode());B.require(meta['nested_offset']==reference['outputs']['offset']['values'][0],'ORACLE_NESTED_OFFSET')
  for name,key in [('packed','weight'),('scale_codes','weight.absmax'),('second_scales','weight.nested_absmax'),('dynamic_map','weight.nested_quant_map'),('nf4_map','weight.quant_map'),('bias','bias')]:
   if name in reference['outputs']:B.require(state[key]==reference['outputs'][name],'ORACLE_NESTED_STATE_LINK')
  references[case['id']]=raw
 B.require(set(rec['artifacts'])==required,'ORACLE_ARTIFACT_MATRIX');return rec,references

def gate(B,actual,reference,tolerance):
 result=B.numeric(actual,reference,tolerance)
 B.require(set(result)=={'status','max_abs','rmse','nrmse'} and result['status'] in ('PASS','FAIL') and all(type(v) in (int,float) and math.isfinite(v) for k,v in result.items() if k!='status'),'FINITE_GATE')
 return result

def verify(B,R,P,A,N,args):
 gold,references=validate_oracle(B,P,N,args);root=Path(args.actual);parent=validate_receipt(B,P,N,args,root,'receipt.json',parent=True)
 B.require(parent.get('oracle_sha256')==args.oracle_sha256,'PARENT_ORACLE');launches=parent.get('children');B.require(type(launches) is list and [l.get('phase') for l in launches]==['save','restore'],'CHILD_MATRIX')
 B.require(type(parent.get('deadline_epoch')) in (int,float) and math.isfinite(parent['deadline_epoch']),'PARENT_DEADLINE');pids={parent['pid'],gold['pid']};tokens={parent['process_token'],gold['process_token']};B.require(len(pids)==len(tokens)==2,'CPU_DEVICE_PROCESS')
 phases={};raws={};required=set();profile,_=B.load_spec();import torch
 for phase,launch in zip(('save','restore'),launches):
  child=root/phase;rec=validate_receipt(B,P,N,args,child,'receipt.json',tpu=True);phases[phase]=rec
  B.require(rec.get('phase')==phase and rec['pid'] not in pids and rec['process_token'] not in tokens,'FRESH_PROCESS');pids.add(rec['pid']);tokens.add(rec['process_token'])
  B.require(rec['pid']==launch.get('pid')==launch.get('launched_pid') and rec['pgid']==launch.get('pgid')==parent['pid'],'ACTUAL_LAUNCH_GROUP')
  B.require(rec.get('parent_pid')==launch.get('parent_pid')==parent['pid'] and rec.get('parent_process_token')==launch.get('parent_process_token')==parent['process_token'] and rec['process_token']==launch.get('process_token'),'PARENT_CHILD_IDENTITY')
  B.require(launch.get('reaped') is True and launch.get('child_absent') is True and launch.get('cleanup_errors')==[] and launch.get('timeout') is False and type(launch.get('exit_code')) is int and launch['exit_code'] in (0,2),'CHILD_TERMINAL')
  B.require(launch.get('receipt_sha256')==B.sha(child/'receipt.json') and launch.get('receipt')==phase+'/receipt.json','CHILD_RECEIPT_LINK')
  B.require(all(type(launch.get(k)) in (int,float) and math.isfinite(launch[k]) for k in ('started_epoch','finished_epoch','started_monotonic','finished_monotonic','deadline_epoch')) and launch['started_epoch']<=launch['finished_epoch']<=launch['deadline_epoch'] and launch['started_monotonic']<=launch['finished_monotonic'] and rec.get('deadline_epoch')==launch['deadline_epoch']<=parent['deadline_epoch'],'CHILD_DEADLINE')
  B.require(rec.get('oracle_sha256')==args.oracle_sha256,'CHILD_ORACLE');artifacts=set()
  if phase=='restore':B.require(rec.get('saved_sha256')==B.sha(root/'save/receipt.json') and rec.get('saved_pid')==phases['save']['pid'] and rec.get('saved_process_token')==phases['save']['process_token'],'SAVED_PROCESS_LINK')
  for case in cases(B,N):
   stem='raw/'+case['id'];artifacts.update(stem+s for s in ('.json','.state.pt','.hlo.txt','.metrics.txt'));raw=B.read(child/(stem+'.json'));raws[phase,case['id']]=raw
   B.require(raw.get('case_id')==case['id'] and raw.get('input_sha256')==B.case_sha(case) and raw.get('pid')==rec['pid'] and raw.get('process_token')==rec['process_token'],'ROW_IDENTITY')
   if raw.get('status')=='ERROR':
    B.require(all(isinstance(raw.get(k),str) and raw[k] for k in ('error_type','error_message','traceback')),'ERROR_RECORD');continue
   B.require(not any(k in raw for k in ('error_type','error_message','traceback')),'PASS_WITH_ERROR_METADATA')
   N.validate_arrays(B,P,case,raw,True);P.validate_execution(B,raw);_,state=checkpoint(B,N,torch,child/(stem+'.state.pt'),case);B.require(state==raw.get('state_dict'),'CHECKPOINT_RECORD_BINDING')
   B.require(raw.get('quant_state_public_roundtrip') is True and raw.get('packed_serialization_original') is True and raw.get('module_training') is True,'PUBLIC_STATE_SERIALIZATION')
   hlo=(child/(stem+'.hlo.txt')).read_text();B.require(hlo.strip() and (child/(stem+'.metrics.txt')).stat().st_size>0,'EXECUTION_DIAGNOSTIC')
   if case['dtype']=='float32' and len(case['x_shape'])==2:
    parsed=P.hlo_precision(hlo);B.require(raw.get('hlo_precision')==parsed,'HLO_RECORD');dots=[d for d in parsed if d['result_dtype']=='f32' and d['operand_dtypes']==['f32','f32']]
    for d in dots:P.validate_dot_matrix(B,d)
    B.require(dots and all(d['operands']==['highest','highest'] for d in dots) and {tuple(case['x_shape']),tuple(case['x_shape'][:-1]+[case['weight_shape'][0]])}<={tuple(d['result_shape']) for d in dots},'HLO_FORWARD_DX')
   else:B.require(raw.get('hlo_precision') is None,'HLO_OBSERVER_SCOPE')
  B.require(set(rec['artifacts'])==artifacts,'PHASE_ARTIFACT_MATRIX');required.update(phase+'/'+p for p in artifacts|{'receipt.json'});required.update((phase+'-logs/stdout.raw',phase+'-logs/stderr.raw'))
 B.require(launches[0]['finished_monotonic']<=launches[1]['started_monotonic'] and launches[0]['finished_epoch']<=launches[1]['started_epoch'],'SEQUENTIAL_REAP')
 B.require(set(parent['artifacts'])==required,'PARENT_ARTIFACT_MATRIX');rows=[]
 for case in cases(B,N):
  name=case['id'];s=raws['save',name];r=raws['restore',name]
  if s.get('status')=='ERROR' or r.get('status')=='ERROR':rows.append({'case_id':name,'status':'ERROR','gates':{}});continue
  B.require(B.sha(root/'save/raw'/(name+'.state.pt'))==B.sha(root/'restore/raw'/(name+'.state.pt'))==r.get('loaded_checkpoint_sha256'),'EXACT_CHECKPOINT_BYTES')
  B.require(s['state_dict']==r.get('loaded_state_before')==r['state_dict'] and r.get('fresh_reconstruction') is True and r.get('saved_receipt_sha256')==B.sha(root/'save/receipt.json'),'EXACT_SAVED_RESTORED_STATE')
  gates={};ref=references[name]
  for phase,raw in [('save',s),('restore',r)]:
   for key,array in raw['outputs'].items():
    result=gate(B,array['values'],ref['outputs'][key]['values'],N.tolerance(profile,case,key))
    if key in N.EXACT and array['bytes_sha256']!=ref['outputs'][key]['bytes_sha256']:result['status']='FAIL'
    gates[phase+'/'+key]=result
   for key,array in raw['state_dict'].items():
    result=gate(B,array['values'],ref['state_dict'][key]['values'],{'exact':True})
    if array['bytes_sha256']!=ref['state_dict'][key]['bytes_sha256']:result['status']='FAIL'
    gates[phase+'/state/'+key]=result
  for key,array in r['outputs'].items():gates['roundtrip/'+key]=gate(B,array['values'],s['outputs'][key]['values'],N.tolerance(profile,case,key))
  B.require(gates,'EMPTY_GATES');rows.append({'case_id':name,'status':'PASS' if all(g['status']=='PASS' for g in gates.values()) else 'FAIL','gates':gates})
 status='ERROR' if any(r['status']=='ERROR' for r in rows) else 'PASS' if len(rows)==8 and all(r['status']=='PASS' for r in rows) else 'FAIL'
 return {'record_validation':'PASS','numerical_status':status,'m4_status':'NOT_QUALIFIED','rows':rows,'checkpoint_pairs':sum(r['status']!='ERROR' for r in rows),'gates':sum(len(r['gates']) for r in rows),'native_cpu_golden':'NOT_RUN','cuda_golden':'NOT_RUN'}

def setup(B,P,A,N,args,tpu):
 admission,roots=N.admit(B,A,args);versions=B.runtime_check(tpu);import torch
 B.require(torch.__version__=='2.9.0+cpu','LOADED_TORCH');C=types.SimpleNamespace(torch=torch,admission=admission,roots=roots,versions=versions,device='cpu')
 if tpu:
  import torch_xla
  import torch_xla.backends as backends
  backends.set_mat_mul_precision('highest');C.precision={'requested':'highest','readback':backends.get_mat_mul_precision(),'set_calls':1,'before_graph':True};B.require(C.precision['readback']=='highest' and torch_xla.__version__.split('+')[0]=='2.9.0','LOADED_XLA_PRECISION')
  import torch_xla.core.xla_model as xm
  import torch_xla.debug.metrics as metrics
  C.device=xm.xla_device();B.require(C.device.type=='xla' and xm.xla_device_hw(C.device)=='TPU','ACTUAL_TPU');C.xla=torch_xla;C.xm=xm;C.metrics=metrics
 import bitsandbytes as bnb
 C.bnb=bnb;C.public=state_methods(B,A,N,bnb,roots);_,_,schemas=N.spec(B)
 C.schemas={k:str(torch._C._dispatch_find_schema_or_throw('bitsandbytes::'+k.partition('.')[0],k.partition('.')[2]).schema()) for k in schemas};B.require(C.schemas==schemas,'SCHEMA_IDENTITY')
 if tpu:C.dispatch={k:torch._C._dispatch_has_kernel_for_dispatch_key('bitsandbytes::'+k,'XLA') for k in schemas};B.require(all(C.dispatch.values()),'DISPATCH')
 B.require(not torch.__future__.get_overwrite_module_params_on_conversion() and not torch.__future__.get_swap_module_params_on_conversion(),'DEFAULT_CONVERSION_FLAGS');return C

def run_phase(B,R,P,A,N,args):
 prepare=args.command=='prepare';output=Path(args.output);output.mkdir(parents=True,exist_ok=False);C=None
 rec=dict(binding(B,N,args),status='PARTIAL',pid=os.getpid(),pgid=os.getpgid(0),process_token=secrets.token_hex(16) if prepare else args.process_token)
 name='oracle-seal.json' if prepare else 'receipt.json'
 try:
  if prepare:_,refs=nested_oracle(B,P,N,args)
  else:
   _,refs=validate_oracle(B,P,N,args)
   B.require(os.getppid()==args.parent_pid and os.getpgid(0)==args.parent_pid and os.getpid()!=args.parent_pid and re.fullmatch('[0-9a-f]{32}',args.process_token) and re.fullmatch('[0-9a-f]{32}',args.parent_process_token),'ACTUAL_PARENT_GROUP')
   rec.update(phase=args.phase,parent_pid=args.parent_pid,parent_process_token=args.parent_process_token,oracle_sha256=args.oracle_sha256,deadline_epoch=args.deadline_epoch)
   if args.phase=='restore':
    B.require(B.sha(Path(args.saved)/'receipt.json')==args.saved_sha256,'SAVED_SEAL');saved=validate_receipt(B,P,N,args,Path(args.saved),'receipt.json',tpu=True)
    B.require(saved['pid']!=rec['pid'] and saved['process_token']!=rec['process_token'],'FRESH_RESTORE_PROCESS');rec.update(saved_sha256=args.saved_sha256,saved_pid=saved['pid'],saved_process_token=saved['process_token'])
  C=setup(B,P,A,N,args,not prepare);rec.update(source_pre=args.admission_sha256,public_api=C.public,schemas=C.schemas)
  if prepare:rec.update(runtime={'python':'3.12','torch':'2.9.0+cpu','platform':sys.platform,'machine':__import__('platform').machine()},oracle_method='NESTED79_DEFAULT_BODY_SEAL_PLUS_ORIGINAL_PUBLIC_PACKED_STATE')
  else:rec.update(runtime=C.versions,device={'type':'xla','hardware':'TPU','pjrt':'TPU'},dispatch=C.dispatch,precision=C.precision,precision_environment=P.precision_environment())
  profile,_=B.load_spec();failed=False
  for case in cases(B,N):
   stem=output/'raw'/case['id'];stem.parent.mkdir(exist_ok=True);raw={'case_id':case['id'],'input_sha256':B.case_sha(case),'pid':rec['pid'],'process_token':rec['process_token'],'status':'ERROR'}
   try:
    state_methods(B,A,N,C.bnb,C.roots)
    if prepare:
     state=oracle_state(B,N,case,refs[case['id']],C.bnb,C.torch);module=restore(B,N,case,state,C.bnb,C.torch,'cpu');restored={k:v.detach().cpu().clone() for k,v in module.state_dict().items()};B.require(records(B,N,state)==records(B,N,restored),'CPU_PUBLIC_STATE_ROUNDTRIP')
     C.torch.save(restored,stem.with_suffix('.state.pt'));raw.update({k:v for k,v in refs[case['id']].items() if k not in ('pid','process_token')});raw.update(pid=rec['pid'],process_token=rec['process_token'],state_dict=records(B,N,restored),quant_state_public_roundtrip=public_roundtrip(B,N,module.weight.quant_state,C.bnb,'cpu'),packed_serialization_original=True,module_training=True,status='PASS')
    else:
     B.require(math.isfinite(args.deadline_epoch) and time.time()<args.deadline_epoch,'OWNER_DEADLINE')
     if args.phase=='save':module=original_module(B,case,C.bnb,C.torch,C.device)
     else:
      source=Path(args.saved)/'raw'/(case['id']+'.state.pt');B.require(B.sha(source)==saved['artifacts']['raw/'+case['id']+'.state.pt']['sha256'],'SAVED_CHECKPOINT_HASH')
      state,state_raw=checkpoint(B,N,C.torch,source,case);module=restore(B,N,case,state,C.bnb,C.torch,C.device);B.require(records(B,N,{k:v.detach().cpu() for k,v in module.state_dict().items()})==state_raw,'EXACT_RESTORE_INITIAL')
      stem.with_suffix('.state.pt').write_bytes(source.read_bytes());raw.update(loaded_state_before=state_raw,loaded_checkpoint_sha256=B.sha(source),saved_receipt_sha256=args.saved_sha256,fresh_reconstruction=True)
     C.metrics.clear_all();outputs,details=compute(B,N,case,module,C.torch,C.bnb,C.device)
     hlo=C.xla._XLAC._get_xla_tensors_hlo(list(outputs.values()));stem.with_suffix('.hlo.txt').write_text(hlo);raw['hlo_precision']=P.hlo_precision(hlo) if case['dtype']=='float32' and len(case['x_shape'])==2 else None
     C.xm.mark_step(wait=True);C.xm.wait_device_ops();raw['counters']={k:C.metrics.counter_value(k) for k in C.metrics.counter_names()};raw['execution_metrics']={k:list(v) for k in profile['execution_metrics'] if (v:=C.metrics.metric_data(k)) is not None};stem.with_suffix('.metrics.txt').write_text(C.metrics.metrics_report())
     state={k:v.detach().cpu().clone() for k,v in module.state_dict().items()};raw.update(N.serialize_details(B,details),outputs={k:N.tensor_record(B,v) for k,v in outputs.items()},placements={k:str(v.device) for k,v in outputs.items()},state_dict=records(B,N,state),module_training=module.training,quant_state_public_roundtrip=public_roundtrip(B,N,module.weight.quant_state,C.bnb,C.device),packed_serialization_original=True,status='PASS')
     if args.phase=='save':C.torch.save(state,stem.with_suffix('.state.pt'))
     N.validate_arrays(B,P,case,raw,True);validate_state(B,N,case,raw['state_dict']);P.validate_execution(B,raw)
    validate_state(B,N,case,raw['state_dict'])
   except Exception as error:
    raw=N.case_error_record(raw,error);failed=True
    for suffix in ('.state.pt','.hlo.txt','.metrics.txt') if not prepare else ('.state.pt',):
     if not stem.with_suffix(suffix).exists():stem.with_suffix(suffix).write_text('NOT_RUN_CASE_ERROR\n')
   B.write(stem.with_suffix('.json'),raw)
  rec['status']='FAILED' if prepare and failed else 'COMPLETE'
 except BaseException:
  rec['status']='FAILED';(output/'error.log').write_text(traceback.format_exc())
 finally:
  if C is not None:
   try:A.verify_installed(B,C.admission,C.roots);state_methods(B,A,N,C.bnb,C.roots);rec['source_post']=args.admission_sha256
   except Exception:rec['status']='FAILED';(output/'source-post.error.log').write_text(traceback.format_exc())
  rec['artifacts']=B.inventory(output,name);B.write(output/name,rec)
 if rec['status']!='COMPLETE':return {'record_validation':'FAIL','numerical_status':'ERROR','m4_status':'NOT_QUALIFIED'},1
 return {'oracle_status':'SEALED','oracle_sha256':B.sha(output/name),'cases':8,'m4_status':'NOT_QUALIFIED'} if prepare else {'record_validation':'PASS','numerical_status':'ERROR' if failed else 'PENDING_PARENT_VERIFY','m4_status':'NOT_QUALIFIED'},0 if prepare or not failed else 2


def require(condition,name):
 if not condition:raise ValueError(name)


# Unchanged reviewed M3 helper body, from experiments/2026-10-04-bitsandbytes-tpu/cloud/ownership/lifecycle.py
def durable_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as file:
        file.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

# Unchanged reviewed M3 helper body, from experiments/2026-10-04-bitsandbytes-tpu/cloud/state_coordinator.py
def run_child(argv, *, output, descriptor, parent, parent_path, deadline, env=None, interrupts=None):
    """Launch one direct child in the inherited group; persist identity before waiting."""
    require(math.isfinite(deadline) and deadline <= parent['deadline_epoch'] and time.time() < deadline,
            'STATE_CHILD_DEADLINE')
    streams = []
    process = None
    descriptor.update(argv=list(argv), deadline_epoch=deadline, parent_pid=parent['pid'],
                      parent_process_token=parent['process_token'], started_epoch=time.time(),
                      started_monotonic=time.monotonic(), reaped=False, child_absent=False,
                      cleanup_errors=[], timeout=False, exit_code=None)
    try:
        for filename in ('stdout.raw', 'stderr.raw'):
            streams.append((output / filename).open('wb', buffering=0))
        # A new PID is sufficient. The outer owner must retain the same PGID even
        # if this coordinator receives SIGKILL and cannot execute its finally.
        if interrupts is not None:
            interrupts['launching'] = True
        try:
            process = subprocess.Popen(argv, stdout=streams[0], stderr=streams[1], env=env)
            descriptor.update(pid=process.pid, launched_pid=process.pid, pgid=os.getpgid(process.pid))
            parent['children'].append(descriptor)
            durable_json(parent_path, parent)
        finally:
            if interrupts is not None:
                interrupts['launching'] = False
        if interrupts is not None and interrupts['pending'] is not None:
            raise InterruptedError('STATE_PARENT_SIGNAL:' + str(interrupts['pending']))
        require(process.pid != parent['pid'] and descriptor['pgid'] == parent['pgid'], 'STATE_INHERITED_GROUP')
        try:
            process.wait(timeout=max(0.001, deadline - time.time()))
        except subprocess.TimeoutExpired:
            descriptor['timeout'] = True
            raise TimeoutError('STATE_CHILD_DEADLINE_EXPIRED')
    finally:
        if process is not None:
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)
                else:
                    process.wait(timeout=2)
                descriptor.update(exit_code=process.returncode, reaped=True)
                try:
                    os.kill(process.pid, 0)
                except ProcessLookupError:
                    descriptor['child_absent'] = True
                require(descriptor['child_absent'], 'STATE_CHILD_STILL_PRESENT')
            except BaseException as error:
                descriptor['cleanup_errors'].append({'type': type(error).__name__, 'message': str(error)})
            descriptor.update(finished_epoch=time.time(), finished_monotonic=time.monotonic())
        for stream in streams:
            try:
                os.fsync(stream.fileno())
                stream.close()
            except Exception as error:
                descriptor['cleanup_errors'].append({'type': type(error).__name__, 'message': str(error)})
        if process is not None:
            durable_json(parent_path, parent)
    require(descriptor['cleanup_errors'] == [] and descriptor['reaped'] and descriptor['child_absent'] and
            not descriptor['timeout'] and descriptor['exit_code'] in (0, 2), 'STATE_CHILD_TERMINAL')
    return descriptor

def audit_after_execution(B,R,P,A,N,args,output,parent):
 try:return verify(B,R,P,A,N,args)
 except Exception:
  parent['status']='FAILED';(output/'post-validation.error.log').write_text(traceback.format_exc())
  parent['artifacts']=B.inventory(output,'receipt.json');durable_json(output/'receipt.json',parent);raise


def execute(B,R,P,A,N,args):
 output=Path(args.output);B.require(os.getpid()==os.getpgid(0) and not output.exists(),'FRESH_OWNED_PARENT')
 B.require(math.isfinite(args.deadline_epoch) and time.time()+15<args.deadline_epoch,'PARENT_DEADLINE')
 validate_oracle(B,P,N,args);admission,roots=N.admit(B,A,args);P.validate_environment(B,P.precision_environment());output.mkdir(parents=True)
 token=args.process_token or secrets.token_hex(16);B.require(re.fullmatch('[0-9a-f]{32}',token),'PARENT_TOKEN')
 parent=dict(binding(B,N,args),status='PARTIAL',pid=os.getpid(),pgid=os.getpgid(0),process_token=token,oracle_sha256=args.oracle_sha256,source_pre=args.admission_sha256,deadline_epoch=args.deadline_epoch,children=[])
 path=output/'receipt.json';durable_json(path,parent);previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)};interrupts={'launching':False,'pending':None}
 def interrupted(signum,frame):
  interrupts['pending']=signum
  if not interrupts['launching']:raise InterruptedError('NESTED_STATE_PARENT_SIGNAL:'+str(signum))
 for sig in previous:signal.signal(sig,interrupted)
 try:
  for index,phase in enumerate(('save','restore')):
   remaining=args.deadline_epoch-time.time()-10;B.require(remaining>0,'PHASE_BUDGET');deadline=time.time()+remaining/(2-index);token=secrets.token_hex(16)
   cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'device-child']
   for name in (*N.PINS,'nested_probe','patch_manifest','admission_sha256','nested_oracle','nested_oracle_sha256','oracle','oracle_sha256','admission'):cmd+=['--'+name.replace('_','-'),str(getattr(args,name))]
   cmd+=['--phase',phase,'--output',str(output/phase),'--parent-pid',str(parent['pid']),'--parent-process-token',parent['process_token'],'--process-token',token,'--deadline-epoch',str(deadline)]
   if phase=='restore':cmd+=['--saved',str(output/'save'),'--saved-sha256',B.sha(output/'save/receipt.json')]
   logs=output/(phase+'-logs');logs.mkdir();launch=run_child(cmd,output=logs,descriptor={'phase':phase,'process_token':token},parent=parent,parent_path=path,deadline=deadline,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),interrupts=interrupts)
   launch.update(receipt=phase+'/receipt.json',receipt_sha256=B.sha(output/phase/'receipt.json'));durable_json(path,parent)
  A.verify_installed(B,admission,roots);parent.update(source_post=args.admission_sha256,status='COMPLETE',artifacts=B.inventory(output,'receipt.json'));durable_json(path,parent)
  args.actual=output
  result=audit_after_execution(B,R,P,A,N,args,output,parent)
  return result,0 if result['numerical_status']=='PASS' else 2
 except BaseException as error:
  parent['status']='FAILED';parent['error']={'type':type(error).__name__,'message':str(error)};parent['artifacts']=B.inventory(output,'receipt.json');durable_json(path,parent);raise
 finally:
  for sig,handler in previous.items():signal.signal(sig,handler)

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
 for command in ('prepare','execute','verify','device-child'):
  cmd=sub.add_parser(command)
  for name in ('backend_probe','route_probe','precision_probe','transfer_admission','nested_probe','patch_manifest','admission_sha256','nested_oracle','nested_oracle_sha256'):cmd.add_argument('--'+name.replace('_','-'),required=True)
  if command=='verify':cmd.add_argument('--actual',required=True)
  else:
   cmd.add_argument('--admission',required=True);cmd.add_argument('--output',required=True)
  if command!='prepare':
   cmd.add_argument('--oracle',required=True);cmd.add_argument('--oracle-sha256',required=True)
  if command in ('execute','device-child'):
   cmd.add_argument('--deadline-epoch',type=float,required=True);cmd.add_argument('--process-token',required=command=='device-child')
  if command=='device-child':
   cmd.add_argument('--phase',choices=('save','restore'),required=True);cmd.add_argument('--parent-pid',type=int,required=True);cmd.add_argument('--parent-process-token',required=True);cmd.add_argument('--saved');cmd.add_argument('--saved-sha256')
 args=parser.parse_args(argv)
 try:
  B,R,P,A,N=helpers(args)
  if args.command=='verify':result=verify(B,R,P,A,N,args);code=0 if result['numerical_status']=='PASS' else 2
  elif args.command=='execute':result,code=execute(B,R,P,A,N,args)
  else:result,code=run_phase(B,R,P,A,N,args)
 except Exception as error:result={'record_validation':'FAIL','numerical_status':'ERROR','m4_status':'NOT_QUALIFIED','reason':str(error)};code=1
 print(json.dumps(result,allow_nan=False));return code

if __name__=='__main__':raise SystemExit(main())
