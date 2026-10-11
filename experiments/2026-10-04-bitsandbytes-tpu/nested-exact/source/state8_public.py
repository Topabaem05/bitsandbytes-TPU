"""Original public saved-state operations, rebound to the exact nested generation.

Source: canonical probe_nested_state.py. Numerical gates belong to the pinned probe_nested.py.
"""
import hashlib
import json
import math
from pathlib import Path
import types

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
  'restored_scales':torch.ops.bnb_tpu_exact.restore_scales.default(state.absmax,state.state2.absmax,state.state2.code,state.offset),'decoded':decoded,'decoded_out':decoded_out,'y':y}
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

