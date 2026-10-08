"""Private M4 prepare/execute/verify candidate. No allocation or milestone acceptance."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import struct
import sys
import time
import traceback
import types

HERE=Path(__file__).resolve().parent
PINS={'precision_probe':'e0534955507e5f91e67ecddfffc738a367ad752e69a4eea7ae8052c6d76a8852',
      'route_probe':'feae751734e57c741b1bdade004ff7ca3c041ee7eb3b7086b531bc6be433038e',
      'backend_probe':'e8743e33e6a0454d5d05b77075d47a5c9f983cf32f3956c38f2b544f86f5ef6d',
      'transfer_admission':'6a7c925211b14aba990318360c8f0ca2d4d803e4528c634135c8bde86326816b'}
INPUT_SHA='b7490f4c622aef793d14bd9a7a38727754cb1fb8e6e1dec7ff91ef8c45fbafbc'
SOURCE_SHA='c8d7519c92202e0238fbee55a63ffff32860f821b8957c8a135046a4294f13d2'
SCHEMA_SHA='bd7dd210b2451749556f053b4472d4cbb820320d6608d246153888d0dcda0b71'
DEFAULT_SHA='e39afd16dca6e6f34a14305ade0ba4acbb49dc2c7a9baf937341d7e33b682819'
EXACT={'packed','codes','scale_codes','offset','second_scales','dynamic_map','nf4_map'}
PUBLIC={'Linear4bit':'bitsandbytes.nn.modules.Linear4bit','Params4bit':'bitsandbytes.nn.modules.Params4bit',
        'original_class_identity':True,'original_method_identity':True,'functional_methods_original':True,
        'QuantState':'bitsandbytes.functional.QuantState','quant_state_methods_original':True}
KIND='PRIVATE_NESTED_PROBE'

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module); return module

def helpers(args):
    for key,sha in PINS.items():
        if hashlib.sha256(Path(getattr(args,key)).read_bytes()).hexdigest()!=sha: raise ValueError('HELPER_PIN_'+key)
    P=load(args.precision_probe,'nested_precision'); R=P.route_probe(args.route_probe); B=R.backend(args.backend_probe)
    A=load(args.transfer_admission,'nested_transfer_admission'); A.load_manifest(B,args.patch_manifest)
    return B,R,P,A

def spec(B):
    for name,sha in [('nested-inputs.json',INPUT_SHA),('nested-source.json',SOURCE_SHA),('nested-schemas.json',SCHEMA_SHA)]:
        B.require(B.sha(HERE/name)==sha,'NESTED_SPEC_PIN')
    data=B.read(HERE/'nested-inputs.json'); source=B.read(HERE/'nested-source.json'); schemas=B.read(HERE/'nested-schemas.json')
    B.require(len(data['cases'])==79 and len({c['id'] for c in data['cases']})==79,'NESTED_MATRIX')
    return data,source,schemas

def admit(B,A,args):
    admission,roots=A.admit(B,args.admission,args.admission_sha256,args.patch_manifest)
    validate_admission(B,A,admission,A.load_manifest(B,args.patch_manifest))
    B.require(B.sha(roots['bitsandbytes']/'backends/default/ops.py')==DEFAULT_SHA,'DEFAULT_SOURCE')
    return admission,roots

def validate_admission(B,A,admission,manifest):
    _,source,_=spec(B); A.validate_admission(B,admission,manifest)
    plugin=admission['bitsandbytes_tpu']
    B.require(plugin.get('source_variant')=='nested-v1' and plugin.get('nested_source_sha256')==SOURCE_SHA,'NESTED_VARIANT_LINK')
    B.require(plugin['files']==source['files'],'NESTED_PLUGIN_EXACT_MAP')
    return admission

def public_methods(B,A,bnb,roots):
    A.public_methods(B,bnb,roots)
    path=(roots['bitsandbytes']/'functional.py').resolve(); code=compile(path.read_text(),str(path),'exec')
    for name in ('quantize_4bit','dequantize_4bit','quantize_blockwise','dequantize_blockwise'):
        expected=[c for c in code.co_consts if isinstance(c,types.CodeType) and c.co_name==name]
        function=getattr(bnb.functional,name)
        B.require(expected and function.__code__==expected[-1] and Path(function.__code__.co_filename).resolve()==path,'FUNCTIONAL_METHOD_REPLACEMENT')
    cls=bnb.functional.QuantState
    expected_class=[c for c in code.co_consts if isinstance(c,types.CodeType) and c.co_name=='QuantState']
    B.require(len(expected_class)==1 and cls.__module__=='bitsandbytes.functional' and cls.__name__=='QuantState','QUANTSTATE_CLASS')
    for name in ('__init__','as_dict','from_dict'):
        function=getattr(cls,name); function=getattr(function,'__func__',function)
        expected=[c for c in expected_class[0].co_consts if isinstance(c,types.CodeType) and c.co_name==name]
        B.require(expected and function.__code__==expected[-1] and Path(function.__code__.co_filename).resolve()==path,'QUANTSTATE_METHOD_REPLACEMENT')
    return PUBLIC.copy()

def default_functions(B,roots):
    q4,d4=B.original_cpu_functions(roots['bitsandbytes'])
    import bitsandbytes.backends.default.ops as default
    path=roots['bitsandbytes']/'backends/default/ops.py'; tree=ast.parse(path.read_text()); namespace=dict(vars(default))
    def extract(alias,operator=None,name=None):
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and
               (n.name==name if name else any(isinstance(d,ast.Call) and d.args and isinstance(d.args[0],ast.Constant)
                and d.args[0].value=='bitsandbytes::'+operator for d in n.decorator_list))]
        B.require(len(nodes)==1,'DEFAULT_EXTRACTION'); node=copy.deepcopy(nodes[0]); node.decorator_list=[]; node.name=alias
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),namespace)
        return namespace[alias]
    return q4,d4,extract('oracle_q8',operator='quantize_blockwise'),extract('oracle_d8',name='_dequantize_blockwise_compute')

def binary(array):
    values=array['values']; dtype=array['dtype']
    if dtype=='uint8': return bytes(values)
    if dtype=='float32': return b''.join(struct.pack('<f',v) for v in values)
    if dtype=='bfloat16': return b''.join(struct.pack('<I',struct.unpack('<I',struct.pack('<f',v))[0])[2:] for v in values)
    raise ValueError('ARRAY_DTYPE')

def tensor_record(B,tensor):
    result=B.tensor_record(tensor); result['bytes_sha256']=hashlib.sha256(binary(result)).hexdigest(); return result

def expected(case):
    kind=case['kind']
    if kind=='block':
        n=case['length']; return {'codes':([n],'uint8'),'second_scales':([(n+255)//256],'float32'),
            'dynamic_map':([256],'float32'),'decoded':([n],'float32'),'decoded_out':([n],'float32')}
    if kind=='signed_known':
        n=case['length']; return {'codes':([n],'uint8'),'second_scales':([(n+255)//256],'float32'),
            'dynamic_map':([256],'float32'),'offset':([],'float32'),'restored_scales':([n],'float32')}
    shape=case['weight_shape']; count=math.prod(shape); scales=(count+63)//64
    outputs={'packed':([(count+1)//2,1],'uint8'),'scale_codes':([scales],'uint8'),
        'offset':([],'float32'),'second_scales':([(scales+255)//256],'float32'),
        'dynamic_map':([256],'float32'),'nf4_map':([16],'float32'),'restored_scales':([scales],'float32'),
        'decoded':(shape,case['dtype']),'decoded_out':(shape,case['dtype'])}
    if kind in ('module','positive_known'):
        outputs['y']=(case['x_shape'][:-1]+[shape[0]],case['dtype'])
    if kind=='module' and case['dtype']=='float32':
        outputs['dx']=(case['x_shape'],'float32')
        if case['bias'] is not None: outputs['db']=([shape[0]],'float32')
    if kind=='module' and case['bias'] is not None: outputs['bias']=([shape[0]],case['dtype'])
    return outputs

def metadata(case):
    nested=case['kind'] in ('nested','module','positive_known')
    return {'kind':case['kind'],'nested':nested,'quant_state_class':'bitsandbytes.functional.QuantState',
            'blocksize':64 if nested else 256,'dtype':case['dtype'],'quant_type':'nf4' if nested else None,
            'shape':case.get('weight_shape'),'state2':{'class':'bitsandbytes.functional.QuantState','blocksize':256,'dtype':'float32'} if nested else None}

def compute(B,case,data,torch,bnb,device,functions=None):
    dtype=getattr(torch,case['dtype']); dynamic=torch.tensor(data['dynamic_map'],dtype=torch.float32,device=device)
    nf4=torch.tensor(data['nf4_map'],dtype=torch.float32,device=device)
    oracle=functions is not None; details={'metadata':metadata(case)}
    if oracle: q4,d4,q8,d8=functions
    def decode8(codes,second):
        if oracle: return d8(codes,second,dynamic,256,torch.float32)
        state=bnb.functional.QuantState(absmax=second,code=dynamic,blocksize=256,dtype=torch.float32)
        return bnb.functional.dequantize_blockwise(codes,quant_state=state)
    if case['kind']=='block':
        value=torch.tensor(case['data'],dtype=torch.float32,device=device)
        if oracle:
            codes,second=q8(value,dynamic,256); decoded=decode8(codes,second); out=decoded.clone()
        else:
            codes,state=bnb.functional.quantize_blockwise(value,code=dynamic,blocksize=256)
            second=state.absmax; dynamic=state.code; decoded=bnb.functional.dequantize_blockwise(codes,state)
            target=torch.empty_like(decoded); out=bnb.functional.dequantize_blockwise(codes,state,out=target)
            B.require(out is target,'BLOCKWISE_OUT_IDENTITY')
        details['out_identity']=True
        return dict(codes=codes,second_scales=second,dynamic_map=dynamic,decoded=decoded,decoded_out=out),details
    if case['kind'] in ('signed_known','positive_known'):
        codes=torch.tensor(case['codes'],dtype=torch.uint8,device=device)
        second=torch.tensor(case['second_scales'],dtype=torch.float32,device=device)
        offset=torch.tensor(case['offset'],dtype=torch.float32,device=device)
        restored=decode8(codes,second)+offset
        if case['kind']=='signed_known':
            return dict(codes=codes,second_scales=second,dynamic_map=dynamic,offset=offset,restored_scales=restored),details
        packed=torch.full((math.prod(case['weight_shape'])//2,1),0x78,dtype=torch.uint8,device=device)
    else:
        weight=torch.tensor(case['weight'],dtype=dtype,device=device if oracle or case['kind']!='module' else 'cpu').reshape(case['weight_shape'])
        if oracle:
            packed,first=q4(weight,64,'nf4',torch.uint8); offset=first.mean(); codes,second=q8(first-offset,dynamic,256)
        elif case['kind']=='module':
            n,k=case['weight_shape']; module=bnb.nn.Linear4bit(k,n,bias=case['bias'] is not None,compute_dtype=dtype,quant_type='nf4').to(dtype=dtype)
            with torch.no_grad():
                module.weight.copy_(weight)
                if module.bias is not None: module.bias.copy_(torch.tensor(case['bias'],dtype=dtype))
            original_weight=module.weight; original_module=module; module.to(device).train()
            state=module.weight.quant_state; packed=module.weight.data
            B.require(module is original_module and module.weight is original_weight and module.quant_state is state and type(module.weight) is bnb.nn.Params4bit,'MODULE_IDENTITY')
            B.require(module.weight.compress_statistics and state.nested,'DEFAULT_STATISTICS')
            codes,second,offset=state.absmax,state.state2.absmax,state.offset; nf4,dynamic=state.code,state.state2.code
        else:
            packed,state=bnb.functional.quantize_4bit(weight,blocksize=64,compress_statistics=True,quant_type='nf4')
            codes,second,offset=state.absmax,state.state2.absmax,state.offset; nf4,dynamic=state.code,state.state2.code
        restored=decode8(codes,second)+offset
    state2=bnb.functional.QuantState(absmax=second,code=dynamic,blocksize=256,dtype=torch.float32)
    if oracle or case['kind']=='positive_known':
        state=bnb.functional.QuantState(absmax=codes,code=nf4,blocksize=64,dtype=dtype,shape=tuple(case['weight_shape']),quant_type='nf4',offset=offset,state2=state2)
    B.require(type(state) is bnb.functional.QuantState and type(state.state2) is bnb.functional.QuantState,'NESTED_STATE_CLASS')
    B.require(state.nested and state.blocksize==64 and state.state2.blocksize==256 and state.state2.dtype==torch.float32 and state.dtype==dtype and list(state.shape)==case['weight_shape'] and state.quant_type=='nf4','NESTED_STATE_FORMAT')
    if oracle:
        decoded=d4(packed.reshape(-1),restored,nf4,64,tuple(case['weight_shape']),dtype); out=decoded.clone()
    else:
        decoded=bnb.functional.dequantize_4bit(packed,state)
        target=torch.empty_like(decoded); out=bnb.functional.dequantize_4bit(packed,state,out=target)
        B.require(out is target or packed.shape[0]==1 and out._base is target,'NF4_PUBLIC_OUT_IDENTITY')
    details['out_identity']=True
    outputs=dict(packed=packed,scale_codes=codes,offset=offset,second_scales=second,dynamic_map=dynamic,nf4_map=nf4,
                 restored_scales=restored,decoded=decoded,decoded_out=out)
    if case['kind'] in ('module','positive_known'):
        x=torch.tensor(case['x'],dtype=dtype,device=device).reshape(case['x_shape'])
        gradient=case['kind']=='module' and dtype==torch.float32; x.requires_grad_(gradient)
        bias=torch.tensor(case['bias'],dtype=dtype,device=device,requires_grad=gradient) if case.get('bias') is not None else None
        before=packed.detach().clone()
        y=torch.nn.functional.linear(x,decoded,bias) if oracle else module(x) if case['kind']=='module' else bnb.matmul_4bit(x,packed.t(),quant_state=state)
        outputs['y']=y
        if gradient:
            dy=torch.tensor([((i%5)-2)/8 for i in range(y.numel())],dtype=dtype,device=device).reshape(y.shape); y.backward(dy)
            outputs['dx']=x.grad
            if bias is not None: outputs['db']=bias.grad if oracle else module.bias.grad
        if case['kind']=='module':
            if bias is not None: outputs['bias']=bias if oracle else module.bias
            details.update(gradients='EXECUTED' if gradient else 'NOT_RUN',default_compress_statistics=True,
                frozen_weight=True if oracle else not module.weight.requires_grad and module.weight.grad is None,
                module_state_alias=True if oracle else module.quant_state is module.weight.quant_state,
                parameter_module_alias=True if oracle else module.weight.module is module,
                parameter_identity=True if oracle else module.weight is original_weight,
                original_classes=True if oracle else type(module) is bnb.nn.Linear4bit and type(module.weight) is bnb.nn.Params4bit,
                bias_class=None if bias is None else 'torch.nn.parameter.Parameter' if oracle or type(module.bias) is torch.nn.Parameter else 'INVALID',
                _packed_before=before,_packed_after=packed if oracle else module.weight.data)
    return outputs,details

def serialize_details(B,details):
    result=dict(details)
    for key in ('packed_before','packed_after'):
        if '_'+key in result: result[key]=tensor_record(B,result.pop('_'+key))
    return result

def validate_arrays(B,P,case,raw,tpu):
    B.require(raw.get('metadata')==metadata(case),'NESTED_METADATA')
    B.require(raw.get('status')=='PASS','ROW_NOT_COMPLETE'); shapes=expected(case)
    B.require(shapes and set(raw.get('outputs',{}))==set(shapes),'OUTPUT_SET')
    B.require(set(raw.get('placements',{}))==set(shapes),'PLACEMENT_SET')
    for name,(shape,dtype) in shapes.items():
        array=raw['outputs'][name]; P.array_check(B,array,shape,dtype)
        encoded=binary(array)
        if dtype=='float32': B.require(all(struct.unpack('<f',encoded[i*4:i*4+4])[0]==v for i,v in enumerate(array['values'])),'FLOAT32_REPRESENTATION')
        if dtype=='bfloat16': B.require(all(struct.unpack('<f',b'\0\0'+encoded[i*2:i*2+2])[0]==v for i,v in enumerate(array['values'])),'BF16_REPRESENTATION')
        B.require(array.get('bytes_sha256')==hashlib.sha256(encoded).hexdigest(),'ARRAY_BYTE_HASH')
        place=raw['placements'][name]; B.require(isinstance(place,str) and (place.startswith('xla:') if tpu else place=='cpu'),'PLACEMENT')
    if case['kind']!='signed_known': B.require(raw.get('out_identity') is True,'PUBLIC_OUT_IDENTITY')
    if case['kind']=='module':
        B.require(all(raw.get(key) is True for key in ('default_compress_statistics','frozen_weight','module_state_alias','parameter_module_alias','parameter_identity','original_classes')),'MODULE_STATE')
        B.require(raw.get('gradients')==('EXECUTED' if case['dtype']=='float32' else 'NOT_RUN'),'GRADIENT_SCOPE')
        B.require(raw.get('packed_before')==raw['outputs']['packed']==raw.get('packed_after'),'PACKED_CHANGED')
        B.require(raw.get('bias_class')==(None if case['bias'] is None else 'torch.nn.parameter.Parameter'),'BIAS_CLASS')
        if case['bias'] is not None: B.require(raw['outputs']['bias']['values']==case['bias'],'BIAS_VALUES')

def bindings(B,args):
    profile,_=B.load_spec(); data,_,_=spec(B)
    return {'kind':KIND,'probe_sha256':B.sha(__file__),'inputs_sha256':INPUT_SHA,'nested_source_sha256':SOURCE_SHA,
            'source_variant':'nested-v1',
            'schema_sha256':SCHEMA_SHA,'helper_sha256':PINS,'patch_manifest_sha256':B.sha(args.patch_manifest),
            'profile_sha256':B.PROFILE_SHA,'runtime_lock_sha256':profile['runtime_lock_sha256'],
            'source_admission_sha256':args.admission_sha256,'case_ids':[c['id'] for c in data['cases']],
            'selected_reference':'PINNED_DEFAULT_PYTHON','default_body_sha256':DEFAULT_SHA,
            'm3_status':'DEPENDENCY_NOT_ACCEPTED_BY_THIS_PROBE','m4_status':'NOT_QUALIFIED',
            'cuda_golden':'NOT_RUN','native_cpu_golden':'NOT_RUN','hlo_scope':'PRE_OPTIMIZATION_OUTPUT_GRAPH'}

def validate_receipt(B,P,args,root,name,tpu):
    rec=B.read(root/name); _,_,schemas=spec(B)
    for key,value in bindings(B,args).items(): B.require(rec.get(key)==value,'BINDING_'+key)
    B.require(rec.get('status')=='COMPLETE','TERMINAL_INCOMPLETE')
    B.require(rec.get('source_pre')==rec.get('source_post')==args.admission_sha256,'SOURCE_CHANGED')
    B.require(rec.get('public_api')==PUBLIC and rec.get('schemas')==schemas,'PUBLIC_SCHEMA_METHOD')
    B.require(type(rec.get('pid')) is int and rec['pid']>0 and type(rec.get('pgid')) is int and rec['pgid']>0
              and isinstance(rec.get('process_token'),str) and re.fullmatch('[0-9a-f]{32}',rec['process_token']),'PROCESS_IDENTITY')
    profile,_=B.load_spec()
    if tpu:
        B.require(rec['pid']==rec['pgid'],'DEVICE_PROCESS_GROUP')
        B.require(rec.get('runtime')==profile['runtime'],'RUNTIME_PAIR'); P.validate_environment(B,rec.get('precision_environment'))
        B.require(rec.get('precision')=={'requested':'highest','readback':'highest','set_calls':1,'before_graph':True},'PRECISION')
        B.require(rec.get('device')=={'type':'xla','hardware':'TPU','pjrt':'TPU'},'DEVICE_PROOF')
        B.require(rec.get('dispatch')=={k:True for k in schemas},'DISPATCH')
    else:
        B.require(rec.get('runtime')=={'python':'3.12','torch':'2.9.0+cpu','platform':'linux','machine':'x86_64'},'CPU_ORACLE_RUNTIME')
        B.require(rec.get('oracle_method')=='EXPLICIT_PINNED_DEFAULT_BODIES_NO_NATIVE_DISPATCH','ORACLE_METHOD')
    B.check_seal(root,name,rec); return rec

def tolerance(profile,case,name):
    if name in EXACT: return {'exact':True}
    if name in ('y','dx','db'): return profile['tolerances']['fp32_forward_gradient' if case['dtype']=='float32' else 'bf16_forward_unit_scale']
    return profile['tolerances']['fp32_decode']

def scalar_witness(case,reference):
    """Independent dense equations on the sealed decoded weight, including rank three."""
    if case['kind'] not in ('module','positive_known'): return {}
    n,k=case['weight_shape']; m=math.prod(case['x_shape'][:-1]); w=reference['outputs']['decoded']['values']; x=case['x']
    bias=case.get('bias'); y=[sum(x[i*k+q]*w[j*k+q] for q in range(k))+(bias[j] if bias is not None else 0) for i in range(m) for j in range(n)]
    witness={'y':y}
    if case['kind']=='module' and case['dtype']=='float32':
        dy=[((i%5)-2)/8 for i in range(m*n)]
        witness['dx']=[sum(dy[i*n+j]*w[j*k+q] for j in range(n)) for i in range(m) for q in range(k)]
        if bias is not None: witness['db']=[sum(dy[i*n+j] for i in range(m)) for j in range(n)]
    return witness

def validate_oracle(B,P,args):
    profile,_=B.load_spec(); data,_,_=spec(B); root=Path(args.oracle)
    B.require(B.sha(root/'oracle-seal.json')==args.oracle_sha256,'ORACLE_SEAL_HASH')
    seal=validate_receipt(B,P,args,root,'oracle-seal.json',False); required=set(); references={}
    for case in data['cases']:
        path='raw/'+case['id']+'.json'; required.add(path); raw=B.read(root/path)
        B.require(raw.get('case_id')==case['id'] and raw.get('input_sha256')==B.case_sha(case)
                  and raw.get('pid')==seal['pid'] and raw.get('process_token')==seal['process_token'],'ORACLE_ROW_BINDING')
        validate_arrays(B,P,case,raw,False)
        for name,key in (('dynamic_map','dynamic_map'),('nf4_map','nf4_map')):
            if name in raw['outputs']: B.require(raw['outputs'][name]['values']==data[key],'ORACLE_CODEBOOK')
        for name,values in scalar_witness(case,raw).items():
            B.require(B.numeric(raw['outputs'][name]['values'],values,tolerance(profile,case,name))['status']=='PASS','SCALAR_REFERENCE_WITNESS')
        references[case['id']]=raw
    B.require(set(seal['artifacts'])==required,'ORACLE_ARTIFACT_MATRIX')
    return seal,references

def verify(B,R,P,A,args):
    profile,_=B.load_spec(); data,_,_=spec(B); oracle=Path(args.oracle); actual=Path(args.actual)
    gold,references=validate_oracle(B,P,args); rec=validate_receipt(B,P,args,actual,'receipt.json',True)
    B.require(rec.get('oracle_sha256')==args.oracle_sha256,'ORACLE_BINDING')
    B.require(rec['pid']!=gold['pid'] and rec['process_token']!=gold['process_token'],'SEPARATE_CPU_DEVICE_PROCESS')
    rows=[]; required=set(); oracle_required=set()
    for case in data['cases']:
        path='raw/'+case['id']; raw=B.read(actual/(path+'.json')); reference=references[case['id']]
        required.update(path+suffix for suffix in ('.json','.hlo.txt','.metrics.txt')); oracle_required.add(path+'.json')
        for record,receipt in ((raw,rec),(reference,gold)):
            B.require(record.get('case_id')==case['id'] and record.get('input_sha256')==B.case_sha(case)
                      and record.get('pid')==receipt['pid'] and record.get('process_token')==receipt['process_token'],'ROW_BINDING')
        if raw.get('status')=='ERROR':
            B.require(all(isinstance(raw.get(k),str) and raw[k] for k in ('error_type','error_message','traceback')),'ERROR_RECORD')
            rows.append({'case_id':case['id'],'status':'ERROR','gates':{}}); continue
        validate_arrays(B,P,case,raw,True); P.validate_execution(B,raw)
        hlo=(actual/(path+'.hlo.txt')).read_text(); B.require(hlo.strip() and (actual/(path+'.metrics.txt')).stat().st_size>0,'EXECUTION_DIAGNOSTIC')
        if case['kind']=='module' and case['dtype']=='float32' and len(case['x_shape'])==2:
            parsed=P.hlo_precision(hlo); B.require(raw.get('hlo_precision')==parsed,'HLO_RECORD')
            dots=[d for d in parsed if d['result_dtype']=='f32' and d['operand_dtypes']==['f32','f32']]
            for dot in dots: P.validate_dot_matrix(B,dot)
            B.require(dots and all(d['operands']==['highest','highest'] for d in dots),'HLO_HIGHEST')
            shapes={tuple(d['result_shape']) for d in dots}
            B.require(tuple(case['x_shape']) in shapes and tuple(case['x_shape'][:-1]+[case['weight_shape'][0]]) in shapes,'HLO_FORWARD_DX')
        else: B.require(raw.get('hlo_precision') is None,'HLO_OBSERVER_SCOPE')
        gates={}
        for name,array in raw['outputs'].items():
            reference_array=reference['outputs'][name]; gate=B.numeric(array['values'],reference_array['values'],tolerance(profile,case,name))
            B.require(isinstance(gate,dict) and set(gate)=={'status','max_abs','rmse','nrmse'} and gate['status'] in ('PASS','FAIL'),'GATE_SCHEMA')
            if name in EXACT and array['bytes_sha256']!=reference_array['bytes_sha256']: gate['status']='FAIL'
            B.require(all(type(v) in (int,float) and math.isfinite(v) for k,v in gate.items() if k!='status'),'NONFINITE_GATE')
            gates[name]=gate
        B.require(gates and set(gates)==set(expected(case)),'EMPTY_OR_INCOMPLETE_GATES')
        rows.append({'case_id':case['id'],'status':'PASS' if all(g['status']=='PASS' for g in gates.values()) else 'FAIL','gates':gates})
    B.require(set(rec['artifacts'])==required and set(gold['artifacts'])==oracle_required,'ARTIFACT_MATRIX')
    status='ERROR' if any(r['status']=='ERROR' for r in rows) else 'PASS' if rows and all(r['status']=='PASS' for r in rows) else 'FAIL'
    return {'record_validation':'PASS','numerical_status':status,'rows':rows,'m4_status':'NOT_QUALIFIED',
            'm3_status':'DEPENDENCY_NOT_ACCEPTED_BY_THIS_PROBE','native_cpu_golden':'NOT_RUN','cuda_golden':'NOT_RUN'}

def audit_after_execution(B,R,P,A,args,output,rec):
    args.actual=output
    try: return verify(B,R,P,A,args)
    except Exception:
        rec['status']='FAILED'; (output/'post-validation.error.log').write_text(traceback.format_exc())
        rec['artifacts']=B.inventory(output,'receipt.json'); B.write(output/'receipt.json',rec)
        raise

def case_error_record(raw,error):
    """Retain an invalid case without leaving nonfinite arrays in numeric fields."""
    raw=dict(raw); outputs=raw.pop('outputs',None); raw.pop('placements',None)
    if outputs is not None:
        nonfinite=[(name,index,repr(value)) for name,array in outputs.items()
                   for index,value in enumerate(array.get('values',[]))
                   if isinstance(value,float) and not math.isfinite(value)][:16]
        prefix=repr(nonfinite)+'; '
        raw['invalid_output_snapshot']=prefix+repr(outputs)[:max(0,4096-len(prefix))]
    raw.update(status='ERROR',error_type=type(error).__name__,error_message=str(error),traceback=traceback.format_exc())
    def safe(value):
        if isinstance(value,float) and not math.isfinite(value): return 'NONFINITE:'+repr(value)
        if isinstance(value,dict): return {key:safe(item) for key,item in value.items()}
        if isinstance(value,(tuple,list)): return [safe(item) for item in value]
        return value
    return safe(raw)

def run(B,R,P,A,args):
    prepare=args.command=='prepare'; output=Path(args.output); output.mkdir(parents=True,exist_ok=False)
    data,_,schemas=spec(B); profile,_=B.load_spec(); name='oracle-seal.json' if prepare else 'receipt.json'
    rec=dict(bindings(B,args),status='PARTIAL',pid=os.getpid(),pgid=os.getpgid(0),process_token=secrets.token_hex(16))
    admission=None; roots={}; bnb=None
    try:
        if not prepare: validate_oracle(B,P,args)
        admission,roots=admit(B,A,args); rec['source_pre']=args.admission_sha256; versions=B.runtime_check(not prepare)
        import torch
        B.require(torch.__version__=='2.9.0+cpu','LOADED_TORCH_RUNTIME')
        if prepare:
            device='cpu'; rec.update(runtime={'python':versions['python'],'torch':versions['torch'],'platform':sys.platform,'machine':__import__('platform').machine()},oracle_method='EXPLICIT_PINNED_DEFAULT_BODIES_NO_NATIVE_DISPATCH')
        else:
            import torch_xla
            import torch_xla.backends as backends
            backends.set_mat_mul_precision('highest')
            rec['precision']={'requested':'highest','readback':backends.get_mat_mul_precision(),'set_calls':1,'before_graph':True}
            import torch_xla.core.xla_model as xm
            import torch_xla.debug.metrics as metrics
            B.require(torch_xla.__version__.split('+')[0]=='2.9.0' and rec['precision']['readback']=='highest','LOADED_XLA_PRECISION')
            device=xm.xla_device(); B.require(device.type=='xla' and xm.xla_device_hw(device)=='TPU','ACTUAL_TPU')
            rec.update(runtime=versions,device={'type':'xla','hardware':'TPU','pjrt':'TPU'},oracle_sha256=args.oracle_sha256,precision_environment=P.precision_environment())
        import bitsandbytes as bnb
        rec['public_api']=public_methods(B,A,bnb,roots)
        rec['schemas']={key:str(torch._C._dispatch_find_schema_or_throw('bitsandbytes::'+key.partition('.')[0],key.partition('.')[2]).schema()) for key in schemas}
        B.require(rec['schemas']==schemas,'SCHEMA_IDENTITY')
        rec['cpu_dispatch_observed']={key:torch._C._dispatch_has_kernel_for_dispatch_key('bitsandbytes::'+key,'CPU') for key in schemas}
        if not prepare:
            rec['dispatch']={key:torch._C._dispatch_has_kernel_for_dispatch_key('bitsandbytes::'+key,'XLA') for key in schemas}; B.require(all(rec['dispatch'].values()),'NESTED_DISPATCH')
        functions=default_functions(B,roots) if prepare else None
        for case in data['cases']:
            if not prepare: B.require(time.time()<args.deadline_epoch,'OWNER_DEADLINE')
            raw={'case_id':case['id'],'input_sha256':B.case_sha(case),'pid':rec['pid'],'process_token':rec['process_token'],'status':'ERROR'}
            prefix=output/'raw'/case['id']; prefix.parent.mkdir(exist_ok=True)
            try:
                public_methods(B,A,bnb,roots)
                if not prepare: metrics.clear_all()
                tensors,details=compute(B,case,data,torch,bnb,device,functions)
                if not prepare:
                    hlo=torch_xla._XLAC._get_xla_tensors_hlo(list(tensors.values())); prefix.with_suffix('.hlo.txt').write_text(hlo)
                    raw['hlo_precision']=P.hlo_precision(hlo) if case['kind']=='module' and case['dtype']=='float32' and len(case['x_shape'])==2 else None
                    xm.mark_step(wait=True); xm.wait_device_ops()
                    raw['counters']={key:metrics.counter_value(key) for key in metrics.counter_names()}
                    raw['execution_metrics']={key:list(value) for key in profile['execution_metrics'] if (value:=metrics.metric_data(key)) is not None}
                    prefix.with_suffix('.metrics.txt').write_text(metrics.metrics_report())
                raw.update(serialize_details(B,details))
                raw.update(outputs={key:tensor_record(B,value) for key,value in tensors.items()},placements={key:str(value.device) for key,value in tensors.items()},status='PASS')
                validate_arrays(B,P,case,raw,not prepare)
            except Exception as error:
                raw=case_error_record(raw,error)
                if not prepare:
                    for suffix in ('.hlo.txt','.metrics.txt'):
                        if not prefix.with_suffix(suffix).exists(): prefix.with_suffix(suffix).write_text('NOT_RUN_CASE_ERROR\n')
            B.write(prefix.with_suffix('.json'),raw)
            if prepare: B.require(raw['status']=='PASS','CPU_ORACLE_CASE_ERROR')
        rec['status']='COMPLETE'
    except Exception:
        rec['status']='FAILED'; (output/'error.log').write_text(traceback.format_exc())
    finally:
        if admission is not None:
            try:
                A.verify_installed(B,admission,roots)
                if bnb is not None: public_methods(B,A,bnb,roots)
                rec['source_post']=args.admission_sha256
            except Exception:
                rec['status']='FAILED'; (output/'source-post.error.log').write_text(traceback.format_exc())
        rec['artifacts']=B.inventory(output,name); B.write(output/name,rec)
    if rec['status']!='COMPLETE': return {'record_validation':'FAIL','numerical_status':'ERROR','m4_status':'NOT_QUALIFIED'},1
    if prepare: return {'oracle_status':'SEALED','oracle_sha256':B.sha(output/name),'cases':79,'m4_status':'NOT_QUALIFIED'},0
    # No precomputed PASS survives an exception from this post-execution verifier.
    result=audit_after_execution(B,R,P,A,args,output,rec); return result,0 if result['numerical_status']=='PASS' else 2

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    for command in ('prepare','execute','verify'):
        cmd=sub.add_parser(command)
        for name in (*PINS,'patch_manifest','admission_sha256'): cmd.add_argument('--'+name.replace('_','-'),required=True)
        if command=='verify': cmd.add_argument('--actual',required=True)
        else:
            cmd.add_argument('--admission',required=True); cmd.add_argument('--output',required=True)
        if command!='prepare':
            cmd.add_argument('--oracle',required=True); cmd.add_argument('--oracle-sha256',required=True)
        if command=='execute': cmd.add_argument('--deadline-epoch',type=float,required=True)
    args=parser.parse_args(argv)
    try:
        B,R,P,A=helpers(args)
        if args.command=='verify':
            result=verify(B,R,P,A,args); code=0 if result['numerical_status']=='PASS' else 2
        else: result,code=run(B,R,P,A,args)
    except Exception as error:
        result={'record_validation':'FAIL','numerical_status':'ERROR','m4_status':'NOT_QUALIFIED','reason':str(error)}; code=1
    print(json.dumps(result,allow_nan=False)); return code

if __name__=='__main__': raise SystemExit(main())
