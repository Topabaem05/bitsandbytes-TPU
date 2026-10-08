"""CPU-only source/input/oracle contract helpers. Never initializes JAX or XLA."""
import ast
import copy
import hashlib
import importlib.util
import json
import math
import platform
from pathlib import Path
import sys
import torch
import verifier as V

HERE=Path(__file__).resolve().parent
BACKEND_SHA='e8743e33e6a0454d5d05b77075d47a5c9f983cf32f3956c38f2b544f86f5ef6d'
PRECISION_SHA='e0534955507e5f91e67ecddfffc738a367ad752e69a4eea7ae8052c6d76a8852'
PROFILE_SHA='ad53f6416bdabf6440f08b313963c947a468cd39afd413bfe41bb381c4a7a166'
DEFAULT_SHA='e39afd16dca6e6f34a14305ade0ba4acbb49dc2c7a9baf937341d7e33b682819'
UTILS_SHA='dc564f2fbf13dba81388a23d4c87167dd54da04af8f17b0db52685fe42b11c84'
POST_PATCH_PYTHON_MAP_SHA='29990cb6e282b1b7c1ca4585cfdc3c723871104f47ad155ca53e168c4385b69d'
require=V.require

def sha(path):return V.digest(Path(path).read_bytes())
def read(path):return json.loads(Path(path).read_text(),object_pairs_hook=V.unique_pairs)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def spec():return read(HERE/'protocol.json')
def sources():return {p.name:sha(p) for p in (HERE/'adapter.py',HERE/'recorder.py',HERE/'verifier.py',HERE/'device_diagnostic.py',HERE/'kernel.py',HERE/'protocol.py',HERE/'protocol.json',HERE/'prepare_oracle.py',HERE/'coordinator.py',HERE/'verify_run.py',HERE/'state_child_owner.py',HERE/'ownership/lifecycle.py',HERE/'ownership/cleanup_lifecycle.py',HERE/'ownership/bootstrap.py',HERE/'graph_binding.py',HERE/'proto_contract.py')}
def tensor(value):return {'shape':list(value.shape),'dtype':str(value.dtype).removeprefix('torch.'),'values':value.detach().cpu().reshape(-1).tolist()}
def inputs(case):
    s=spec();require(case in s['case_ids'],'CASE_ID');m=129 if case=='tail-reference-fp32' else 128;n,k=256,512
    dtype=torch.bfloat16 if case.endswith('bf16') else torch.float32
    codes=((torch.arange(n*k)*7)%16).to(torch.uint8);packed=((codes[::2]<<4)|codes[1::2]).reshape(-1,1);scales=(torch.arange(n*k//64)%17).float()/8
    activation=(((torch.arange(m*k)%31)-15).float()/32).reshape(m,k).to(dtype)
    return activation,packed,scales

def original_decode(upstream):
    upstream=Path(upstream);op=upstream/'backends/default/ops.py';utils=upstream/'backends/utils.py'
    require(sha(op)==DEFAULT_SHA and sha(utils)==UTILS_SHA,'UPSTREAM_DEFAULT_SOURCE')
    nodes=ast.parse(op.read_text());function=copy.deepcopy(next(n for n in nodes.body if isinstance(n,ast.FunctionDef) and n.name=='_dequantize_4bit_compute'));function.decorator_list=[]
    namespace={'torch':torch,'prod':math.prod,'Sequence':list}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(op),'exec'),namespace)
    code_node=next(n for n in ast.parse(utils.read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_NF4_QUANT_TABLE' for t in n.targets));code=torch.tensor(ast.literal_eval(code_node.value.args[0]),dtype=torch.float32)
    require(code.shape==(16,),'NF4_CODEBOOK');return namespace['_dequantize_4bit_compute'],code

def inventory(root,exclude=()):
    root=Path(root);return {p.relative_to(root).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(root.rglob('*')) if p.is_file() and p.relative_to(root).as_posix() not in exclude}
def check_inventory(root,records,exclude=()):require(inventory(root,exclude)==records,'ARTIFACT_INVENTORY')
def oracle(root,expected_sha,*,qualified):
    root=Path(root);require(sha(root/'oracle-seal.json')==expected_sha,'ORACLE_SEAL_HASH');seal=read(root/'oracle-seal.json');s=spec()
    require(seal.get('kind')=='R6_CPU_ORACLE' and seal.get('status')=='COMPLETE' and seal.get('protocol_sha256')==sha(HERE/'protocol.json') and seal.get('protocol_source_sha256')==sha(__file__),'ORACLE_PROTOCOL_SOURCE')
    require(seal.get('source_variant')==s['source_variant'] and seal.get('plugin_python_files')==s['plugin_python_files'] and seal.get('upstream_default_sha256')==DEFAULT_SHA and seal.get('upstream_utils_sha256')==UTILS_SHA,'ORACLE_SOURCE_VARIANT')
    require(seal.get('source_pre')==seal.get('source_post')==seal.get('upstream_python_map_sha256')==POST_PATCH_PYTHON_MAP_SHA,'ORACLE_UPSTREAM_FULL_SOURCE_MAP')
    require(seal.get('sources')==sources(),'ORACLE_CANDIDATE_SOURCE')
    require(seal.get('profile_sha256')==PROFILE_SHA and seal.get('runtime_lock_sha256')==s['runtime_lock_sha256'],'ORACLE_FIXED_GATES_RUNTIME')
    require(seal.get('qualified_runtime') in (True,False),'ORACLE_QUALIFICATION')
    if qualified:require(seal['qualified_runtime'] is True and seal.get('cpu_runtime')=={'python':'3.12','torch':'2.9.0+cpu','platform':'linux','machine':'x86_64'},'ORACLE_QUALIFIED_CPU_REQUIRED')
    require(seal.get('case_ids')==s['case_ids'] and seal.get('no_xla_jax_import') is True,'ORACLE_CASES_NO_CLIENT');check_inventory(root,seal['artifacts'],('oracle-seal.json',))
    cases={}
    for case in s['case_ids']:
        row=read(root/(case+'.json'));require(row['case']==case and row['status']=='COMPLETE','ORACLE_CASE_TERMINAL');a,b,c=inputs(case);expected={'activation':tensor(a),'packed':tensor(b),'scales':tensor(c)}
        require(row.get('inputs')==expected and row.get('input_sha256')==V.digest(expected),'ORACLE_DETERMINISTIC_INPUTS')
        V.array(row['cpu_reference']);require(row['cpu_reference']['dtype']==expected['activation']['dtype'] and row['cpu_reference']['shape']==[a.shape[0],256],'ORACLE_OUTPUT_CONTRACT');cases[case]=row
    return seal,cases

def expected_case(case,row,owner):
    a,b,c=row['inputs']['activation'],row['inputs']['packed'],row['inputs']['scales'];m,k=a['shape'];n,q=256,2;source=sources()
    grouped=[a,V.transpose({'dtype':'uint8','shape':[n,q,128],'values':b['values']},[1,0,2],[q,n,128]),V.transpose({'dtype':'float32','shape':[n,q,4],'values':c['values']},[1,0,2],[q,n,4])]
    profile=read(HERE/'probe-profile.json');require(sha(HERE/'probe-profile.json')==PROFILE_SHA,'FIXED_PROFILE')
    return {'case':case,'inputs':row['inputs'],'grouped':grouped,'cpu_reference':row['cpu_reference'],'owner':owner,'recorder_sha256':source['recorder.py'],'source_pins':{'kernel':source['kernel.py'],'bridge':'c347f8fcb4844fa8849109680ad81b94e221a2c8b011462553b48ec9f67e6338','reference':spec()['plugin_python_files']['reference.py'],'backend':BACKEND_SHA,'precision':PRECISION_SHA,'adapter':source['adapter.py'],'recorder':source['recorder.py'],'diagnostic':source['device_diagnostic.py']},'operand_hlo':[{'dtype':a['dtype'],'shape':[m,k],'layout':[1,0]},{'dtype':'uint8','shape':[q,n,128],'layout':[2,1,0]},{'dtype':'float32','shape':[q,n,4],'layout':[2,1,0]}],'result_hlo':{'dtype':a['dtype'],'shape':[m,n],'layout':[1,0]},'tolerance':profile['tolerances']['fp32_forward_gradient' if a['dtype']=='float32' else 'bf16_forward_unit_scale']}
