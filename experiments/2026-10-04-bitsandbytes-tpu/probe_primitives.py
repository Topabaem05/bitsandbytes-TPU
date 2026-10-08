"""Private R4 arithmetic diagnostic. A complete record does not qualify M4."""
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
import struct
import sys
import time
import traceback
import types

HERE=Path(__file__).resolve().parent
N_PIN='9cfbc7c6b9841b365b47ecfd0bf1b184fda754208aec5f214fe3d0cd9198348d'
S_PIN='1233803f173b9e0d6e864e1fa3081f66408b357ffe17d925b0a69459514fb1f1'
INPUT_SHA='7cb1be0bf806f3da98f9617f90ba99e042331ecbc07bf79799d779088919e5b5'
KERNEL_SHA='a532a70b86001f8d3cd755ddd95ed03ba78b2c5570e2e6b278e92577ec52f81a'
# Exact reviewed standard-library helper bytes.
REFERENCE_SHA='7e0bc0f7e6cfbf049609001613157ba7fd3ec123b5e29b8ed104087d642b5ee8'
HLO_SHA='60f8ac1ea1b0a7ebfce27036da02470dae6ad1bb824dc894d462d6e54ce1a39b'
TORCH_COMMIT='0fabc3ba44823f257e70ce397d989c8de5e362c1'
BUILDER_SHA='ee315ac0cfa9f052a5a437d693bc1d65aae7ea7bf7c3d0b661384bc6723a8772'
LOWERING_SOURCE_SHA='85105171d16d84dd181f364128a8098d9a02d9934c6a9865f81b012a457fcf43'
KIND='PRIVATE_R4_ARITHMETIC_DIAGNOSTIC'
PHASES=('native-view','builder')
PROTOCOL={'format':'R4_ARITHMETIC_PRIMITIVES_V1','input':'20 finite signed IEEE32 patterns and 28 original default-body first-scale arrays',
 'reference':'fresh Linux2.9 default Python first NF4 scales; source-derived AVX2 sum order and exact FP32 bit checks',
 'phases':list(PHASES),'phase_ceiling_seconds':1500,'child_limits_seconds':{'native-view':180,'builder':900},'cleanup_reserve_seconds':10,'minimum_remaining_seconds':15,'builder':'original pinned xla_builder BitcastConvert; no numerical cast or CPU fallback',
 'arithmetic':['mean','sum_div','sum_reciprocal','ordered_sum8','ordered_div8'],
 'state':'no upstream source modification; original nested-v1 classes/methods and seven schemas',
 'process':'owned parent group; sequential fresh direct children; durable launch identity; reap before next child',
 'precision':'highest once before graphs','hlo_scope':'PRE_OPTIMIZATION_OUTPUT_GRAPH',
 'numeric':'exact bytes including signed zero; differences are diagnostic observations',
 'input_guards':'upload first_scales+[2.125], slice n; upload [n,n+.5], slice first scalar; guards never enter arithmetic',
 'lowering_source_sha256':LOWERING_SOURCE_SHA,'qualification':'all milestone statuses NOT_QUALIFIED'}
COMMON=('backend_probe','route_probe','precision_probe','transfer_admission','nested_probe','nested_state_probe','patch_manifest','admission_sha256')

def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def token(B,value):B.require(isinstance(value,str) and re.fullmatch('[0-9a-f]{32}',value),'PROCESS_TOKEN')
def number(value):return type(value) in (int,float) and math.isfinite(value)

def helpers(args):
 for name,pin in [('nested_probe',N_PIN),('nested_state_probe',S_PIN)]:
  if hashlib.sha256(Path(getattr(args,name)).read_bytes()).hexdigest()!=pin:raise ValueError('HELPER_PIN_'+name)
 N=load(args.nested_probe,'primitive_nested');S=load(args.nested_state_probe,'primitive_state')
 B,R,P,A=N.helpers(args)
 for name,pin in [('primitive_kernels.py',KERNEL_SHA),('primitive_reference.py',REFERENCE_SHA),('hlo_bitcast.py',HLO_SHA),('primitive-inputs.json',INPUT_SHA)]:B.require(B.sha(HERE/name)==pin,'PRIMITIVE_SIBLING_PIN_'+name)
 return B,R,P,A,N,S

def pure():return load(HERE/'primitive_reference.py','primitive_reference'),load(HERE/'hlo_bitcast.py','primitive_hlo')

def spec(B,N):
 B.require(B.sha(HERE/'primitive-inputs.json')==INPUT_SHA,'INPUT_PIN');data=B.read(HERE/'primitive-inputs.json');Ref,_=pure()
 B.require(data.get('format')=='PRIVATE_R4_ARITHMETIC_PRIMITIVE_INPUTS_V1' and data.get('base_input_sha256')==N.INPUT_SHA,'INPUT_FORMAT')
 ints=data['bit_patterns_signed32'];B.require(len(ints)==20 and len(set(ints))==20 and all(type(v) is int and -(1<<31)<=v<(1<<31) for v in ints),'BIT_INPUT_MATRIX')
 B.require(data['bit_patterns_hex']==[struct.pack('>i',v).hex() for v in ints] and all(math.isfinite(Ref.from_bits(v)) for v in ints),'BIT_PATTERN_BINDING')
 original={c['id']:c for c in N.spec(B)[0]['cases']};seen=set()
 B.require(len(data['mean_cases'])==28,'MEAN_MATRIX')
 for c in data['mean_cases']:
  name=c['case_id'];B.require(name in original and name not in seen and original[name]['kind'] in ('nested','module'),'MEAN_CASE_ID');seen.add(name)
  scales=Ref.first_scales(original[name]);expected=Ref.array(scales,[len(scales)],'float32')
  B.require(c['count']==len(scales) and c['first_scales']==scales and c['first_scales_bytes_sha256']==expected['bytes_sha256'],'FIRST_SCALES_SOURCE_DERIVATION')
 return data

def binding(B,N,args):
 profile,_=B.load_spec();data=spec(B,N)
 return {'kind':KIND,'primitive_variant':'arithmetic-v1','source_variant':'nested-v1','probe_sha256':B.sha(__file__),
 'kernels_sha256':KERNEL_SHA,'inputs_sha256':INPUT_SHA,'reference_sha256':REFERENCE_SHA,'hlo_helper_sha256':HLO_SHA,
 'helper_sha256':dict(N.PINS,nested_probe=N_PIN,nested_state_probe=S_PIN),'protocol_sha256':digest(PROTOCOL),
 'nested_inputs_sha256':N.INPUT_SHA,'nested_source_sha256':N.SOURCE_SHA,'schema_sha256':N.SCHEMA_SHA,'default_body_sha256':N.DEFAULT_SHA,
 'profile_sha256':B.PROFILE_SHA,'runtime_lock_sha256':profile['runtime_lock_sha256'],'patch_manifest_sha256':B.sha(args.patch_manifest),
 'source_admission_sha256':args.admission_sha256,'phase_ids':list(PHASES),'mean_case_ids':[c['case_id'] for c in data['mean_cases']],
 'hlo_scope':'PRE_OPTIMIZATION_OUTPUT_GRAPH',**{k:'NOT_QUALIFIED' for k in ('api42_status','m3_status','m4_status','m5_status')}}

def validate_array(B,Ref,value,expected):
 B.require(type(value) is dict and set(value)=={'values','shape','dtype','bytes_sha256'} and value['shape']==expected['shape'] and value['dtype']==expected['dtype'],'ARRAY_SCHEMA')
 values=value['values'];B.require(type(values) is list and len(values)==math.prod(expected['shape']) and all(number(v) for v in values),'ARRAY_FINITE')
 if value['dtype']=='int32':B.require(all(type(v) is int and -(1<<31)<=v<(1<<31) for v in values),'ARRAY_INT32')
 else:B.require(all(type(v) in (int,float) and Ref.bits(Ref.f32(v))==Ref.bits(v) and Ref.f32(v)==v for v in values),'ARRAY_FP32')
 B.require(value['bytes_sha256']==Ref.array(values,value['shape'],value['dtype'])['bytes_sha256'],'ARRAY_BYTE_SEAL')

def public_record(B,N,rec):
 _,_,schemas=N.spec(B);B.require(rec.get('public_api')==N.PUBLIC and rec.get('schemas')==schemas,'PUBLIC_SCHEMA_IDENTITY')
 B.require(type(rec.get('cpu_dispatch_observed')) is dict and set(rec['cpu_dispatch_observed'])==set(schemas) and all(type(v) is bool for v in rec['cpu_dispatch_observed'].values()),'CPU_DISPATCH_OBSERVATION')

def validate_receipt(B,N,args,root,name,role):
 rec=B.read(root/name)
 for key,value in binding(B,N,args).items():B.require(rec.get(key)==value,'BINDING_'+key)
 B.require(rec.get('status')=='COMPLETE' and rec.get('source_pre')==rec.get('source_post')==args.admission_sha256,'TERMINAL_SOURCE')
 B.require(type(rec.get('pid')) is int and rec['pid']>0 and type(rec.get('pgid')) is int and rec['pgid']>0 and type(rec.get('parent_pid')) is int and rec['parent_pid']>0,'PROCESS_IDENTITY');token(B,rec.get('process_token'))
 if role in ('cpu','parent'):B.require(rec['pid']==rec['pgid'],'OWNED_GROUP_LEADER')
 if role!='parent':public_record(B,N,rec)
 if role=='cpu':
  B.require(rec.get('runtime')=={'python':'3.12','torch':'2.9.0+cpu','platform':'linux','machine':'x86_64'},'CPU_ORACLE_RUNTIME')
  B.require(rec.get('oracle_method')=='FRESH_DEFAULT_PYTHON_SCALES_PLUS_SOURCE_FIXED_FP32_BITS','ORACLE_METHOD')
 else:
  profile,_=B.load_spec();B.require(rec.get('runtime')==profile['runtime'],'TPU_RUNTIME');P=load(args.precision_probe,'primitive_receipt_precision');P.validate_environment(B,rec.get('precision_environment'))
  B.require(rec.get('oracle_sha256')==args.oracle_sha256,'ORACLE_BINDING')
  if role=='device':
   B.require(rec.get('device')=={'type':'xla','hardware':'TPU','pjrt':'TPU'} and rec.get('precision')=={'requested':'highest','readback':'highest','set_calls':1,'before_graph':True},'DEVICE_PRECISION')
   B.require(rec.get('dispatch')=={k:True for k in N.spec(B)[2]},'NESTED_DISPATCH')
 B.check_seal(root,name,rec);return rec

def verify_cpu_oracle(B,R,P,A,N,S,args,root=None,expected_sha=None):
 """Read-only host gate. No torch import, compiler, or device access."""
 root=Path(root or args.oracle);expected_sha=expected_sha or args.oracle_sha256
 B.require(B.sha(root/'oracle-seal.json')==expected_sha,'ORACLE_SEAL_HASH');rec=validate_receipt(B,N,args,root,'oracle-seal.json','cpu');data=spec(B,N);Ref,_=pure()
 B.require(set(rec['artifacts'])=={'inputs.json','outputs.json','cpu-evidence.json','source-admission.json'},'CPU_ARTIFACT_MATRIX')
 B.require(B.sha(root/'source-admission.json')==args.admission_sha256,'CPU_SOURCE_ADMISSION_BYTES');N.validate_admission(B,A,B.read(root/'source-admission.json'),A.load_manifest(B,args.patch_manifest))
 B.require(B.read(root/'inputs.json')==data,'ORACLE_INPUTS');outputs=B.read(root/'outputs.json');expected=Ref.expected(data)
 B.require(set(outputs)==set(expected),'CPU_OUTPUT_MATRIX');count=0
 for name,row in expected.items():
  B.require(set(outputs[name])==set(row),'CPU_ROW_OUTPUTS')
  for key,value in row.items():
   validate_array(B,Ref,outputs[name][key],value);B.require(outputs[name][key]['bytes_sha256']==value['bytes_sha256'],'CPU_SOURCE_FIXED_BITS_'+name+'_'+key);count+=1
 evidence=B.read(root/'cpu-evidence.json')
 B.require(evidence.get('capability') in ('AVX2','AVX512') and evidence.get('capability_api')=='torch.backends.cpu.get_cpu_capability' and evidence.get('sum_dispatch')=='PINNED_SUM_STUB_AVX2_EIGHT_LANE_FALLBACK','CPU_CAPABILITY')
 B.require(evidence.get('torch_git_version')==TORCH_COMMIT and evidence.get('loaded_torch')=='2.9.0+cpu' and evidence.get('build_api')=='torch.__config__.show' and isinstance(evidence.get('build'),str) and evidence['build'].strip(),'CPU_BUILD')
 B.require(evidence.get('build_sha256')==hashlib.sha256(evidence['build'].encode()).hexdigest(),'CPU_BUILD_SEAL')
 B.require(all(type(evidence.get(k)) is int and evidence[k]>0 for k in ('threads','interop_threads')) and evidence.get('aten_cpu_capability_override') is None,'CPU_THREAD_OVERRIDE')
 B.require(evidence.get('default_body_sha256')==N.DEFAULT_SHA and evidence.get('first_scales')=={c['case_id']:Ref.array(c['first_scales'],[c['count']],'float32') for c in data['mean_cases']},'CPU_DEFAULT_BODY_SCALES')
 B.require(evidence.get('runtime_inventory')==B.load_spec()[0]['runtime'],'CPU_RUNTIME_INVENTORY')
 B.require(evidence.get('pid')==rec['pid'] and evidence.get('pgid')==rec['pgid'] and evidence.get('parent_pid')==rec['parent_pid'] and evidence.get('process_token')==rec['process_token'],'CPU_EVIDENCE_PROCESS')
 return {'record_validation':'PASS','oracle_status':'VERIFIED','oracle_sha256':expected_sha,'row_count':len(expected),'mean_case_count':28,'output_count':count,
 'pid':rec['pid'],'pgid':rec['pgid'],'parent_pid':rec['parent_pid'],'process_token':rec['process_token'],
 'capability':evidence['capability'],'build_sha256':evidence['build_sha256'],'artifacts':rec['artifacts'],**{k:'NOT_QUALIFIED' for k in ('api42_status','m3_status','m4_status','m5_status')}}

def tensor_record(Ref,value):return Ref.array(value.detach().cpu().reshape(-1).tolist(),list(value.shape),str(value.dtype).removeprefix('torch.'))

def cpu_outputs(B,N,roots,data,torch,K):
 Ref,_=pure();original={c['id']:c for c in N.spec(B)[0]['cases']};q4=N.default_functions(B,roots)[0];fresh={}
 for c in data['mean_cases']:
  case=original[c['case_id']];weight=torch.tensor(case['weight'],dtype=getattr(torch,case['dtype'])).reshape(case['weight_shape']);_,first=q4(weight,64,'nf4',torch.uint8)
  actual=tensor_record(Ref,first);B.require(actual==Ref.array(c['first_scales'],[c['count']],'float32'),'FRESH_DEFAULT_SCALES');fresh[c['case_id']]=actual
 ints=torch.tensor(data['bit_patterns_signed32'],dtype=torch.int32);value=ints.view(torch.float32)
 result={'host-float-bits':{'bits':tensor_record(Ref,value.view(torch.int32))},'device-int-float':{'float':tensor_record(Ref,ints.view(torch.float32))},
 'native-bitcast':{'bits':tensor_record(Ref,value.view(torch.int32)),'float':tensor_record(Ref,ints.view(torch.float32))}}
 arith={'add_zero':value+torch.zeros((),dtype=torch.float32),'multiply_one':value*torch.ones((),dtype=torch.float32),'divide_one':value/torch.ones((),dtype=torch.float32),
        'absolute':value.abs(),'clamp':value.clamp_min(1e-38),'maximum':value.abs().max()}
 result['float-arithmetic']={k:tensor_record(Ref,v.view(torch.int32)) for k,v in arith.items()}
 for c in data['mean_cases']:
  v=torch.tensor(fresh[c['case_id']]['values'],dtype=torch.float32);count=torch.tensor(float(c['count']),dtype=torch.float32);row=K.arithmetic_outputs(v,count);row.update(input_bits=v,count_bits=count)
  result['mean-'+c['case_id']]={k:tensor_record(Ref,x.view(torch.int32)) for k,x in row.items()}
 return result,fresh

def record_public(B,A,N,roots,torch,bnb):
 _,_,schemas=N.spec(B)
 return {'public_api':N.public_methods(B,A,bnb,roots),'schemas':{k:str(torch._C._dispatch_find_schema_or_throw('bitsandbytes::'+k.partition('.')[0],k.partition('.')[2]).schema()) for k in schemas},
 'cpu_dispatch_observed':{k:torch._C._dispatch_has_kernel_for_dispatch_key('bitsandbytes::'+k,'CPU') for k in schemas}}

def prepare(B,R,P,A,N,S,args):
 B.require(os.getpid()==os.getpgid(0),'CPU_GROUP_LEADER');token(B,args.process_token);output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
 rec=dict(binding(B,N,args),status='PARTIAL',pid=os.getpid(),pgid=os.getpgid(0),parent_pid=os.getppid(),process_token=args.process_token)
 path=output/'oracle-seal.json'
 try:
  admission,roots=N.admit(B,A,args);rec['source_pre']=args.admission_sha256;versions=B.runtime_check(False)
  import torch
  import bitsandbytes as bnb
  B.require(torch.__version__=='2.9.0+cpu' and not os.environ.get('ATEN_CPU_CAPABILITY'),'QUALIFIED_CPU_LOADED_OVERRIDE')
  K=load(HERE/'primitive_kernels.py','primitive_cpu_kernels');data=spec(B,N);outputs,fresh=cpu_outputs(B,N,roots,data,torch,K)
  rec.update(record_public(B,A,N,roots,torch,bnb));rec.update(runtime={'python':versions['python'],'torch':versions['torch'],'platform':sys.platform,'machine':__import__('platform').machine()},oracle_method='FRESH_DEFAULT_PYTHON_SCALES_PLUS_SOURCE_FIXED_FP32_BITS')
  build=torch.__config__.show();evidence={'capability':torch.backends.cpu.get_cpu_capability(),'capability_api':'torch.backends.cpu.get_cpu_capability','sum_dispatch':'PINNED_SUM_STUB_AVX2_EIGHT_LANE_FALLBACK',
   'torch_git_version':torch.version.git_version,'loaded_torch':torch.__version__,'build_api':'torch.__config__.show','build':build,'build_sha256':hashlib.sha256(build.encode()).hexdigest(),
   'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'aten_cpu_capability_override':os.environ.get('ATEN_CPU_CAPABILITY') or None,
   'default_body_sha256':N.DEFAULT_SHA,'first_scales':fresh,'runtime_inventory':versions,**{k:rec[k] for k in ('pid','pgid','parent_pid','process_token')}}
  (output/'source-admission.json').write_bytes(Path(args.admission).read_bytes());B.write(output/'inputs.json',data);B.write(output/'outputs.json',outputs);B.write(output/'cpu-evidence.json',evidence)
  N.public_methods(B,A,bnb,roots);A.verify_installed(B,admission,roots);rec.update(source_post=args.admission_sha256,status='COMPLETE',artifacts=B.inventory(output,'oracle-seal.json'));S.durable_json(path,rec)
  result=verify_cpu_oracle(B,R,P,A,N,S,args,output,B.sha(path));return result,0
 except BaseException:
  rec['status']='FAILED';(output/'error.log').write_text(traceback.format_exc());rec['artifacts']=B.inventory(output,'oracle-seal.json');S.durable_json(path,rec);raise

def row_matrix(data):
 Ref,_=pure();expected=Ref.expected(data)
 return {'native-view':['native-bitcast'],'builder':[k for k in expected if k!='native-bitcast']},expected

def source_inputs(Ref,data,name,materialized=None):
 ints=data['bit_patterns_signed32'];values=[Ref.from_bits(v) for v in ints]
 if name=='native-bitcast':return {'float_input':Ref.array(values,[20],'float32'),'int_input':Ref.array(ints,[20],'int32')}
 if name=='host-float-bits':return {'float_input':Ref.array(values,[20],'float32')}
 if name=='device-int-float':return {'int_input':Ref.array(ints,[20],'int32')}
 if name=='float-arithmetic':return {'float_input':materialized}
 c=next(c for c in data['mean_cases'] if name=='mean-'+c['case_id'])
 return {'float_input':Ref.array(c['first_scales']+[2.125],[c['count']+1],'float32'),'count_input':Ref.array([float(c['count']),float(c['count'])+.5],[2],'float32')}

def row_graph(B,H,name,raw,hlo,expected):
 order=list(expected);B.require(raw.get('output_order')==order,'OUTPUT_ORDER');input_ids=raw.get('input_parameter_ids');source=raw.get('source_inputs');mapping=raw.get('parameter_values')
 B.require(type(input_ids) is dict and set(input_ids)==set(source) and all(type(v) is int and v>=0 for v in input_ids.values()) and len(set(input_ids.values()))==len(input_ids),'INPUT_PARAMETER_IDS')
 B.require(type(mapping) is dict and set(mapping)=={str(v) for v in input_ids.values()},'PARAMETER_VALUE_MATRIX')
 types={input_ids[k]:('s32' if v['dtype']=='int32' else 'f32',v['shape']) for k,v in source.items()}
 observed_outputs={k:('s32' if v['dtype']=='int32' else 'f32',v['shape']) for k,v in expected.items()}
 direct={'native-bitcast':{'bits':input_ids.get('float_input'),'float':input_ids.get('int_input')},'host-float-bits':{'bits':input_ids.get('float_input')},'device-int-float':{'float':input_ids.get('int_input')}}
 links=direct.get(name,{'input_bits':input_ids.get('float_input'),'count_bits':input_ids.get('count_input')} if name.startswith('mean-') else {})
 witnesses=H.bound_witnesses(hlo,order,observed_outputs,links,types)
 for key,w in witnesses.items():
  required={input_ids['float_input']} if 'float_input' in input_ids else {input_ids['int_input']}
  if key=='float' and name=='native-bitcast':required={input_ids['int_input']}
  if name.startswith('mean-') and key in ('sum_div','sum_reciprocal','ordered_div8'):required.add(input_ids['count_input'])
  if key=='count_bits':required={input_ids['count_input']}
  B.require(set(w['parameters'])==required,'HLO_ARITHMETIC_INPUT_BINDING')
 return witnesses

def make_device_outputs(B,K,torch,name,inputs,data,materialized):
 computations={}
 def bit(value,dtype,key):
  output,hlo=K.xla_bitcast(value,dtype);computations[key]=hlo;return output
 if name=='host-float-bits':out={'bits':bit(inputs['float_input'],torch.int32,'bits')}
 elif name=='device-int-float':out={'float':bit(inputs['int_input'],torch.float32,'float')}
 elif name=='float-arithmetic':
  value=materialized;one=torch.ones((),dtype=torch.float32,device=value.device);zero=torch.zeros((),dtype=torch.float32,device=value.device)
  out={k:bit(v,torch.int32,k) for k,v in {'add_zero':value+zero,'multiply_one':value*one,'divide_one':value/one,'absolute':value.abs(),'clamp':value.clamp_min(1e-38),'maximum':value.abs().max()}.items()}
 elif name=='native-bitcast':out={'bits':inputs['float_input'].view(torch.int32),'float':inputs['int_input'].view(torch.float32)}
 else:
  c=next(c for c in data['mean_cases'] if name=='mean-'+c['case_id']);value=inputs['float_input'][:c['count']];count=inputs['count_input'][:1].reshape(())
  expressions=K.arithmetic_outputs(value,count);expressions.update(input_bits=value,count_bits=count);out={k:bit(v,torch.int32,k) for k,v in expressions.items()}
 return out,computations

def metrics_record(metrics,profile):
 return {'counters':{k:metrics.counter_value(k) for k in metrics.counter_names()},'execution_metrics':{k:list(v) for k in profile['execution_metrics'] if (v:=metrics.metric_data(k)) is not None}}

def run_phase(B,R,P,A,N,S,args):
 token(B,args.process_token);token(B,args.parent_process_token);B.require(os.getppid()==args.parent_pid and os.getpgid(0)==args.parent_pid and os.getpid()!=args.parent_pid,'DIRECT_INHERITED_CHILD')
 B.require(number(args.deadline_epoch) and time.time()<args.deadline_epoch,'CHILD_DEADLINE');verify_cpu_oracle(B,R,P,A,N,S,args)
 output=Path(args.output);output.mkdir(parents=True,exist_ok=False);data=spec(B,N);matrix,expected=row_matrix(data);Ref,H=pure();profile,_=B.load_spec()
 rec=dict(binding(B,N,args),status='PARTIAL',pid=os.getpid(),pgid=os.getpgid(0),parent_pid=os.getppid(),process_token=args.process_token,parent_process_token=args.parent_process_token,
   phase=args.phase,deadline_epoch=args.deadline_epoch,oracle_sha256=args.oracle_sha256,precision_environment=P.precision_environment())
 path=output/'receipt.json'
 try:
  admission,roots=N.admit(B,A,args);rec['source_pre']=args.admission_sha256;rec['runtime']=B.runtime_check(True);P.validate_environment(B,rec['precision_environment'])
  import torch
  import torch_xla
  import torch_xla.backends as backends
  backends.set_mat_mul_precision('highest');rec['precision']={'requested':'highest','readback':backends.get_mat_mul_precision(),'set_calls':1,'before_graph':True}
  B.require(torch.__version__=='2.9.0+cpu' and torch_xla.__version__.split('+')[0]=='2.9.0' and rec['precision']['readback']=='highest','LOADED_TPU_RUNTIME')
  import torch_xla.core.xla_model as xm
  import torch_xla.core.xla_builder as xb
  import torch_xla.debug.metrics as metrics
  import bitsandbytes as bnb
  device=xm.xla_device();B.require(device.type=='xla' and xm.xla_device_hw(device)=='TPU','ACTUAL_TPU');rec['device']={'type':'xla','hardware':'TPU','pjrt':'TPU'}
  rec.update(record_public(B,A,N,roots,torch,bnb));rec['dispatch']={k:torch._C._dispatch_has_kernel_for_dispatch_key('bitsandbytes::'+k,'XLA') for k in N.spec(B)[2]};B.require(all(rec['dispatch'].values()),'NESTED_DISPATCH')
  B.require(B.sha(xb.__file__)==load(HERE/'primitive_kernels.py','primitive_builder_pin').BUILDER_SHA,'BUILDER_SOURCE');K=load(HERE/'primitive_kernels.py','primitive_device_kernels')
  rec['case_ids']=matrix[args.phase];materialized=None;materialized_record=None
  for name in matrix[args.phase]:
   B.require(time.time()<args.deadline_epoch,'ROW_DEADLINE');source=source_inputs(Ref,data,name,materialized_record)
   inputs={k:(materialized if name=='float-arithmetic' else torch.tensor(v['values'],dtype=getattr(torch,v['dtype']),device=device).reshape(v['shape'])) for k,v in source.items()}
   # Materialize uploads first. Parameter mapping is later recorded after graph execution.
   xm.mark_step(wait=True);xm.wait_device_ops();metrics.clear_all();prefix=output/'raw'/name;prefix.parent.mkdir(exist_ok=True)
   raw={'case_id':name,'pid':rec['pid'],'process_token':rec['process_token'],'status':'ERROR','source_inputs':source,'source_inputs_sha256':digest(source),'input_manifest_sha256':INPUT_SHA}
   try:
    tensors,computations=make_device_outputs(B,K,torch,name,inputs,data,materialized)
   except Exception as error:
    if args.phase!='native-view':raise
    raw.update(status='OBSERVED_UNSUPPORTED',reason='NATIVE_VIEW_EXCEPTION',error_type=type(error).__name__,error_message=str(error),traceback=traceback.format_exc(),**metrics_record(metrics,profile),outputs={},placements={})
    prefix.with_suffix('.hlo.txt').write_text('NOT_RUN_NATIVE_VIEW_EXCEPTION\n');prefix.with_suffix('.metrics.txt').write_text(metrics.metrics_report() or 'NO_EXECUTION_OBSERVED\n');B.write(prefix.with_suffix('.json'),raw);continue
   order=list(expected[name]);B.require(list(tensors)==order,'DEVICE_OUTPUT_ORDER');raw['output_order']=order
   if any(v.device.type!='xla' for v in tensors.values()):
    B.require(args.phase=='native-view','BUILDER_HOST_OUTPUT');raw.update(status='OBSERVED_UNSUPPORTED',reason='NATIVE_VIEW_HOST_OUTPUT',outputs={},placements={k:str(v.device) for k,v in tensors.items()},**metrics_record(metrics,profile))
    prefix.with_suffix('.hlo.txt').write_text('NOT_RUN_NATIVE_HOST_OUTPUT\n');prefix.with_suffix('.metrics.txt').write_text(metrics.metrics_report() or 'NO_EXECUTION_OBSERVED\n');B.write(prefix.with_suffix('.json'),raw);continue
   ctx=torch_xla._XLAC.lowering.LoweringContext('R4PrimitiveObservation');ctx.build(list(tensors.values()))
   comp=xb.computation_from_module_proto('R4PrimitiveObservation',ctx.hlo());hlo=xb.get_computation_hlo(comp);prefix.with_suffix('.hlo.txt').write_text(hlo)
   raw['input_parameter_ids']={k:ctx.tensor_parameter_id(v) for k,v in inputs.items()};parameter_tensors=ctx.device_parameter_id_tensor_mapping()
   xm.mark_step(wait=True);xm.wait_device_ops();raw.update(metrics_record(metrics,profile));raw['synchronized']=True
   prefix.with_suffix('.metrics.txt').write_text(metrics.metrics_report() or 'NO_EXECUTION_OBSERVED\n')
   if args.phase=='native-view' and any(k.startswith('aten::') and v>0 for k,v in raw['counters'].items()):
    raw.update(status='OBSERVED_UNSUPPORTED',reason='NATIVE_VIEW_ATEN_FALLBACK',outputs={},placements={k:str(v.device) for k,v in tensors.items()});B.write(prefix.with_suffix('.json'),raw);continue
   # Extraction is only an evidence boundary after synchronous execution, never kernel arithmetic.
   raw['parameter_values']={str(k):tensor_record(Ref,v) for k,v in parameter_tensors.items()};raw['outputs']={k:tensor_record(Ref,v) for k,v in tensors.items()};raw['placements']={k:str(v.device) for k,v in tensors.items()}
   raw['hlo_witnesses']=row_graph(B,H,name,raw,hlo,expected[name]);raw['builder_computations']={}
   for key,text in computations.items():
    p=prefix.parent/(name+'.'+key+'.builder.hlo.txt');p.write_text(text);value=expected[name][key];target='s32' if value['dtype']=='int32' else 'f32'
    raw['builder_computations'][key]={'artifact':p.relative_to(output).as_posix(),'sha256':B.sha(p),'witness':H.computation(text,'f32' if target=='s32' else 's32',target,value['shape'])}
   raw.update(status='OBSERVED',builder_source_sha256=K.BUILDER_SHA,parameter_mapping_api='LoweringContext.device_parameter_id_tensor_mapping+tensor_parameter_id',materialized_from='device-int-float' if name=='float-arithmetic' else None)
   P.validate_execution(B,raw);B.write(prefix.with_suffix('.json'),raw)
   if name=='device-int-float':materialized=tensors['float'];materialized_record=raw['outputs']['float'];rec['materialized_before_arithmetic']=True
  N.public_methods(B,A,bnb,roots);A.verify_installed(B,admission,roots);rec.update(source_post=args.admission_sha256,status='COMPLETE',artifacts=B.inventory(output,'receipt.json'));S.durable_json(path,rec)
  return {'record_status':'COMPLETE','phase':args.phase},2 if args.phase=='native-view' and B.read(output/'raw/native-bitcast.json')['status']=='OBSERVED_UNSUPPORTED' else 0
 except BaseException:
  rec['status']='FAILED';(output/'error.log').write_text(traceback.format_exc());rec['artifacts']=B.inventory(output,'receipt.json');S.durable_json(path,rec);raise

def bit_gate(actual,expected):
 return {'status':'PASS' if actual['bytes_sha256']==expected['bytes_sha256'] else 'FAIL','actual_bytes_sha256':actual['bytes_sha256'],'expected_bytes_sha256':expected['bytes_sha256']}

def validate_counter_observation(B,raw):
 counters=raw.get('counters');B.require(type(counters) is dict and all(type(k) is str and type(v) is int and v>=0 for k,v in counters.items()),'COUNTER_OBSERVATION')
 metrics=raw.get('execution_metrics');B.require(type(metrics) is dict and set(metrics)<={'ExecuteTime','ExecuteReplicatedTime'},'METRIC_OBSERVATION')
 for value in metrics.values():
  B.require(type(value) is list and len(value)==3 and type(value[0]) is int and value[0]>=0 and number(value[1]) and value[1]>=0,'METRIC_OBSERVATION')
  B.require(type(value[2]) is list and all(type(pair) is list and len(pair)==2 and all(number(v) and v>=0 for v in pair) for pair in value[2]) and (value[0]==0 or value[2]),'METRIC_OBSERVATION_SAMPLES')


def verify(B,R,P,A,N,S,args):
 cpu=verify_cpu_oracle(B,R,P,A,N,S,args);root=Path(args.actual);parent=validate_receipt(B,N,args,root,'receipt.json','parent');data=spec(B,N);Ref,H=pure();matrix,expected=row_matrix(data)
 launches=parent.get('children');B.require(type(launches) is list and [c.get('phase') for c in launches]==list(PHASES),'PHASE_MATRIX')
 B.require(number(parent.get('deadline_epoch')) and number(parent.get('started_epoch')) and number(parent.get('started_monotonic')) and parent['started_epoch']<parent['deadline_epoch']<=parent['started_epoch']+1500,'PARENT_DEADLINE');pids={parent['pid'],cpu['pid']};tokens={parent['process_token'],cpu['process_token']};B.require(len(pids)==len(tokens)==2,'FRESH_CPU_DEVICE');required=set();rows=[];native_status=None;materialized=None
 for phase,launch in zip(PHASES,launches):
  child=root/phase;rec=validate_receipt(B,N,args,child,'receipt.json','device');B.require(rec.get('phase')==phase and rec.get('case_ids')==matrix[phase],'CHILD_CASE_MATRIX')
  B.require(rec['pid'] not in pids and rec['process_token'] not in tokens,'FRESH_CHILD');pids.add(rec['pid']);tokens.add(rec['process_token'])
  B.require(rec['pid']==launch.get('pid')==launch.get('launched_pid') and rec['pgid']==launch.get('pgid')==parent['pid'],'ACTUAL_LAUNCH_GROUP')
  B.require(rec['parent_pid']==launch.get('parent_pid')==parent['pid'] and rec.get('parent_process_token')==launch.get('parent_process_token')==parent['process_token'] and rec['process_token']==launch.get('process_token'),'PARENT_CHILD_IDENTITY')
  B.require(launch.get('reaped') is True and launch.get('child_absent') is True and launch.get('cleanup_errors')==[] and launch.get('timeout') is False and type(launch.get('exit_code')) is int and launch['exit_code'] in (0,2),'CHILD_TERMINAL')
  B.require(launch.get('receipt')==phase+'/receipt.json' and launch.get('receipt_sha256')==B.sha(child/'receipt.json'),'CHILD_RECEIPT')
  B.require(all(number(launch.get(k)) for k in ('started_epoch','finished_epoch','started_monotonic','finished_monotonic','deadline_epoch')) and launch['started_epoch']<=launch['finished_epoch']<=launch['deadline_epoch'] and launch['started_monotonic']<=launch['finished_monotonic'] and rec.get('deadline_epoch')==launch['deadline_epoch']<=parent['deadline_epoch'],'CHILD_DEADLINE')
  B.require(launch['started_epoch']>=parent['started_epoch'] and launch['started_monotonic']>=parent['started_monotonic'] and launch['deadline_epoch']<=launch['started_epoch']+PROTOCOL['child_limits_seconds'][phase],'CHILD_LIMIT')
  argv=launch.get('argv');B.require(type(argv) is list and len(argv)>4 and all(type(v) is str for v in argv) and argv[1]=='-B' and Path(argv[2]).name=='probe_primitives.py' and argv[3]=='device-child','CHILD_ARGV')
  pairs=argv[4:];B.require(len(pairs)%2==0 and len(set(pairs[::2]))==len(pairs)//2,'CHILD_ARGV_FLAGS');flags=dict(zip(pairs[::2],pairs[1::2]))
  keys=(*COMMON,'oracle','oracle_sha256','admission','phase','output','parent_pid','parent_process_token','process_token','deadline_epoch')
  B.require(set(flags)=={'--'+k.replace('_','-') for k in keys},'CHILD_ARGV_MATRIX')
  for key,value in [('phase',phase),('parent_pid',str(parent['pid'])),('parent_process_token',parent['process_token']),('process_token',rec['process_token']),('admission_sha256',args.admission_sha256),('oracle_sha256',args.oracle_sha256)]:B.require(flags['--'+key.replace('_','-')]==value,'CHILD_ARGV_IDENTITY')
  B.require(float(flags['--deadline-epoch'])==launch['deadline_epoch'] and Path(flags['--output']).name==phase,'CHILD_ARGV_DEADLINE_OUTPUT')
  if phase=='builder':B.require(launch['exit_code']==0 and rec.get('materialized_before_arithmetic') is True,'BUILDER_COMPLETION')
  phase_artifacts=set()
  for name in matrix[phase]:
   stem='raw/'+name;phase_artifacts.update(stem+s for s in ('.json','.hlo.txt','.metrics.txt'));raw=B.read(child/(stem+'.json'));row_expected=expected[name]
   B.require(raw.get('case_id')==name and raw.get('pid')==rec['pid'] and raw.get('process_token')==rec['process_token'] and raw.get('input_manifest_sha256')==INPUT_SHA,'ROW_IDENTITY')
   B.require(raw.get('source_inputs')==source_inputs(Ref,data,name,materialized) and raw.get('source_inputs_sha256')==digest(raw['source_inputs']),'SOURCE_INPUT_BINDING')
   B.require((child/(stem+'.metrics.txt')).stat().st_size>0 and (child/(stem+'.hlo.txt')).stat().st_size>0,'OBSERVATION_ARTIFACTS')
   if raw.get('status')=='OBSERVED_UNSUPPORTED':
    B.require(phase=='native-view' and raw.get('outputs')=={} and launch['exit_code']==2,'NATIVE_UNSUPPORTED_SCOPE');validate_counter_observation(B,raw)
    reason=raw.get('reason')
    if reason=='NATIVE_VIEW_EXCEPTION':B.require(all(isinstance(raw.get(k),str) and raw[k] for k in ('error_type','error_message','traceback')),'NATIVE_EXCEPTION_EVIDENCE')
    elif reason=='NATIVE_VIEW_ATEN_FALLBACK':B.require(any(k.startswith('aten::') and v>0 for k,v in raw['counters'].items()) and raw.get('synchronized') is True,'NATIVE_FALLBACK_EVIDENCE')
    else:B.require(reason=='NATIVE_VIEW_HOST_OUTPUT' and type(raw.get('placements')) is dict and any(not str(v).startswith('xla:') for v in raw['placements'].values()),'NATIVE_HOST_EVIDENCE')
    native_status='UNSUPPORTED';rows.append({'phase':phase,'case_id':name,'status':'UNSUPPORTED','gates':{}});continue
   B.require(raw.get('status')=='OBSERVED' and not any(k in raw for k in ('error_type','error_message','traceback','reason')),'OBSERVATION_STATUS')
   B.require(type(raw.get('outputs')) is dict and set(raw['outputs'])==set(row_expected) and raw.get('synchronized') is True,'OUTPUT_MATRIX_SYNC')
   B.require(type(raw.get('placements')) is dict and set(raw['placements'])==set(row_expected) and all(type(v) is str and re.fullmatch(r'xla:\d+',v) for v in raw['placements'].values()),'XLA_OUTPUTS')
   for key,value in row_expected.items():validate_array(B,Ref,raw['outputs'][key],value)
   P.validate_execution(B,raw);hlo=(child/(stem+'.hlo.txt')).read_text();witness=row_graph(B,H,name,raw,hlo,row_expected);B.require(raw.get('hlo_witnesses')==witness,'HLO_WITNESS_BINDING')
   B.require(raw.get('parameter_mapping_api')=='LoweringContext.device_parameter_id_tensor_mapping+tensor_parameter_id','PARAMETER_MAPPING_API')
   gates={key:bit_gate(raw['outputs'][key],value) for key,value in row_expected.items()}
   for key,value in raw['source_inputs'].items():
    actual=raw['parameter_values'][str(raw['input_parameter_ids'][key])];validate_array(B,Ref,actual,value);gates['parameter_'+key]=bit_gate(actual,value)
   computations=raw.get('builder_computations');B.require(type(computations) is dict and set(computations)==(set(row_expected) if phase=='builder' else set()),'BUILDER_COMPUTATION_MATRIX')
   for key,descriptor in computations.items():
    artifact=stem+'.'+key+'.builder.hlo.txt';phase_artifacts.add(artifact);B.require(type(descriptor) is dict and set(descriptor)=={'artifact','sha256','witness'} and descriptor['artifact']==artifact and descriptor['sha256']==B.sha(child/artifact),'BUILDER_COMPUTATION_SEAL')
    value=row_expected[key];target='s32' if value['dtype']=='int32' else 'f32';parsed=H.computation((child/artifact).read_text(),'f32' if target=='s32' else 's32',target,value['shape']);B.require(descriptor['witness']==parsed,'BUILDER_COMPUTATION_BINDING')
   B.require(raw.get('builder_source_sha256')==BUILDER_SHA,'BUILDER_SOURCE')
   B.require(raw.get('materialized_from')==('device-int-float' if name=='float-arithmetic' else None),'MATERIALIZED_INPUT_LINK')
   if name=='device-int-float':materialized=raw['outputs']['float']
   if phase=='native-view':native_status='SUPPORTED';B.require(launch['exit_code']==0,'NATIVE_SUPPORTED_EXIT')
   B.require(gates and all(g['status'] in ('PASS','FAIL') for g in gates.values()),'NONEMPTY_BIT_GATES')
   rows.append({'phase':phase,'case_id':name,'status':'PASS' if all(g['status']=='PASS' for g in gates.values()) else 'FAIL','gates':gates})
  B.require(set(rec['artifacts'])==phase_artifacts,'PHASE_ARTIFACT_MATRIX');required.update(phase+'/'+p for p in phase_artifacts|{'receipt.json'});required.update((phase+'-logs/stdout.raw',phase+'-logs/stderr.raw'))
 B.require(launches[0]['finished_epoch']<=launches[1]['started_epoch'] and launches[0]['finished_monotonic']<=launches[1]['started_monotonic'],'SEQUENTIAL_REAP')
 B.require(set(parent['artifacts'])==required,'PARENT_ARTIFACT_MATRIX');builder=[r for r in rows if r['phase']=='builder'];B.require(len(rows)==32 and len(builder)==31 and sum(len(r['gates']) for r in builder)>0,'COMPLETE_RECORD_MATRIX')
 status='FAIL' if any(r['status']=='FAIL' for r in rows) else 'PASS';builder_status='FAIL' if any(r['status']=='FAIL' for r in builder) else 'PASS'
 return {'record_validation':'PASS','numerical_status':status,'builder_bit_status':builder_status,'native_view_status':native_status,'phase_ids':list(PHASES),'row_count':len(rows),'gate_count':sum(len(r['gates']) for r in rows),
 'cpu_oracle':cpu,'rows':rows,'api42_status':'NOT_QUALIFIED','m3_status':'NOT_QUALIFIED','m4_status':'NOT_QUALIFIED','m5_status':'NOT_QUALIFIED'}

def audit_after_execution(B,R,P,A,N,S,args,output,parent):
 try:return verify(B,R,P,A,N,S,args)
 except Exception:
  parent['status']='FAILED';(output/'post-validation.error.log').write_text(traceback.format_exc());parent['artifacts']=B.inventory(output,'receipt.json');S.durable_json(output/'receipt.json',parent);raise

class PhaseBudgetUnavailable(ValueError):pass

def phase_deadline(B,parent_deadline,index,now):
 remaining=parent_deadline-now-10
 if not number(parent_deadline) or remaining<=15:raise PhaseBudgetUnavailable('PRIMITIVE_PHASE_NOT_RUN_INSUFFICIENT_REMAINING_BUDGET')
 return now+min((180.,900.)[index],remaining/(2-index))

def execute(B,R,P,A,N,S,args):
 output=Path(args.output);B.require(os.getpid()==os.getpgid(0) and not output.exists(),'FRESH_OWNED_PARENT');token(B,args.process_token)
 B.require(number(args.deadline_epoch) and time.time()+15<args.deadline_epoch and args.deadline_epoch<=time.time()+1500,'PARENT_DEADLINE');verify_cpu_oracle(B,R,P,A,N,S,args)
 admission,roots=N.admit(B,A,args);P.validate_environment(B,P.precision_environment());output.mkdir(parents=True)
 parent=dict(binding(B,N,args),status='PARTIAL',pid=os.getpid(),pgid=os.getpgid(0),parent_pid=os.getppid(),process_token=args.process_token,started_epoch=time.time(),started_monotonic=time.monotonic(),
  runtime=B.runtime_check(True),oracle_sha256=args.oracle_sha256,source_pre=args.admission_sha256,precision_environment=P.precision_environment(),deadline_epoch=args.deadline_epoch,children=[])
 path=output/'receipt.json';S.durable_json(path,parent);previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)};interrupts={'launching':False,'pending':None}
 def interrupted(signum,frame):
  interrupts['pending']=signum
  if not interrupts['launching']:raise InterruptedError('PRIMITIVE_PARENT_SIGNAL:'+str(signum))
 for sig in previous:signal.signal(sig,interrupted)
 try:
  for index,phase in enumerate(PHASES):
   deadline=phase_deadline(B,args.deadline_epoch,index,time.time());child_token=secrets.token_hex(16)
   cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'device-child']
   for key in (*COMMON,'oracle','oracle_sha256','admission'):cmd+=['--'+key.replace('_','-'),str(getattr(args,key))]
   cmd+=['--phase',phase,'--output',str(output/phase),'--parent-pid',str(parent['pid']),'--parent-process-token',parent['process_token'],'--process-token',child_token,'--deadline-epoch',str(deadline)]
   logs=output/(phase+'-logs');logs.mkdir();launch=S.run_child(cmd,output=logs,descriptor={'phase':phase,'process_token':child_token},parent=parent,parent_path=path,deadline=deadline,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),interrupts=interrupts)
   launch.update(receipt=phase+'/receipt.json',receipt_sha256=B.sha(output/phase/'receipt.json'));S.durable_json(path,parent)
  A.verify_installed(B,admission,roots);parent.update(source_post=args.admission_sha256,status='COMPLETE',artifacts=B.inventory(output,'receipt.json'));S.durable_json(path,parent);args.actual=output
  return audit_after_execution(B,R,P,A,N,S,args,output,parent),0
 except BaseException as error:
  parent.update(status='FAILED',error={'type':type(error).__name__,'message':str(error)});parent['artifacts']=B.inventory(output,'receipt.json');S.durable_json(path,parent);raise
 finally:
  for sig,handler in previous.items():signal.signal(sig,handler)

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
 for command in ('prepare','execute','verify','device-child'):
  cmd=sub.add_parser(command)
  for key in COMMON:cmd.add_argument('--'+key.replace('_','-'),required=True)
  if command=='verify':cmd.add_argument('--actual',required=True)
  else:cmd.add_argument('--admission',required=True);cmd.add_argument('--output',required=True);cmd.add_argument('--process-token',required=True)
  if command!='prepare':cmd.add_argument('--oracle',required=True);cmd.add_argument('--oracle-sha256',required=True)
  if command in ('execute','device-child'):cmd.add_argument('--deadline-epoch',type=float,required=True)
  if command=='device-child':
   cmd.add_argument('--phase',choices=PHASES,required=True);cmd.add_argument('--parent-pid',type=int,required=True);cmd.add_argument('--parent-process-token',required=True)
 args=parser.parse_args(argv)
 try:
  B,R,P,A,N,S=helpers(args)
  if args.command=='verify':result=verify(B,R,P,A,N,S,args);code=0
  elif args.command=='prepare':result,code=prepare(B,R,P,A,N,S,args)
  elif args.command=='execute':result,code=execute(B,R,P,A,N,S,args)
  else:result,code=run_phase(B,R,P,A,N,S,args)
 except Exception as error:result={'record_validation':'FAIL','numerical_status':'ERROR','m4_status':'NOT_QUALIFIED','reason':str(error),'execution_status':'NOT_RUN' if isinstance(error,PhaseBudgetUnavailable) else 'INVALID'};code=1
 print(json.dumps(result,allow_nan=False));return code

if __name__=='__main__':raise SystemExit(main())
