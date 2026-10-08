"""Independent original default decode plus dense CPU linear/backward reference."""
import ast
import copy
import math
from pathlib import Path
import torch
import contract as C

DEFAULT_SHA='e39afd16dca6e6f34a14305ade0ba4acbb49dc2c7a9baf937341d7e33b682819'
UTILS_SHA='dc564f2fbf13dba81388a23d4c87167dd54da04af8f17b0db52685fe42b11c84'

def tensor(v):return {'dtype':str(v.dtype).removeprefix('torch.'),'shape':list(v.shape),'values':v.detach().cpu().reshape(-1).tolist()}
def materialize(v):return torch.tensor(v['values'],dtype=getattr(torch,v['dtype'])).reshape(v['shape'])
def decode_original(root):
    root=Path(root);op=root/'backends/default/ops.py';utils=root/'backends/utils.py'
    C.require(C.sha(op)==DEFAULT_SHA and C.sha(utils)==UTILS_SHA,'ORIGINAL_DEFAULT_BODY')
    f=copy.deepcopy(next(n for n in ast.parse(op.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_dequantize_4bit_compute'));f.decorator_list=[]
    ns={'torch':torch,'prod':math.prod,'Sequence':list};exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),str(op),'exec'),ns)
    table=next(n for n in ast.parse(utils.read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_NF4_QUANT_TABLE' for t in n.targets))
    code=torch.tensor(ast.literal_eval(table.value.args[0]),dtype=torch.float32);C.require(code.shape==(16,),'ORIGINAL_NF4_LITERAL');return ns['_dequantize_4bit_compute'],code

def inputs(case):
    shape=case['activation_shape'];m=math.prod(shape[:-1]);n,k=case['weight_shape'];dtype=getattr(torch,case['dtype'])
    a=(((torch.arange(m*k)%31)-15).float()/32).reshape(shape).to(dtype)
    if case['layout']=='transpose-copy-transpose':a=a.t().contiguous().t()
    codes=((torch.arange(n*k)*7)%16).to(torch.uint8);b=((codes[::2]<<4)|codes[1::2]).reshape(-1,1)
    s=(torch.arange(n*k//64)%17).float()/8;dy=(((torch.arange(m*n)%23)-11).float()/32).reshape(*shape[:-1],n)
    bias=((torch.arange(n)%17)-8).float()/32 if case['bias'] else None
    return a,b,s,dy,bias

def input_record(case):
    a,b,s,dy,bias=inputs(case);return {key:tensor(v) for key,v in [('activation',a),('packed',b),('scales',s),('dy',dy),('bias',bias)] if v is not None}

def expected(case,decode,code):
    a,b,s,dy,bias=inputs(case);n,k=case['weight_shape'];dense=decode(b.flatten(),s,code,64,(n,k),a.dtype)
    y=torch.nn.functional.linear(a,dense,bias);out={'y':tensor(y)}
    if case['gradients']!='NOT_RUN':
        x=a.detach().requires_grad_(True);bp=bias.detach().requires_grad_(True) if bias is not None else None
        result=torch.nn.functional.linear(x,dense,bp);result.backward(dy,retain_graph=True)
        out['dX']=tensor(x.grad);result.backward(dy);out['dX_accumulated']=tensor(x.grad)
        if bp is not None:
            # Capture first and repeated accumulation through an independent replay.
            first=dy.reshape(-1,n).sum(0);out['db']=tensor(first);out['db_accumulated']=tensor(bp.grad)
    return {'case':case['id'],'inputs_sha256':C.digest(input_record(case)),'outputs':out}

def allowed_reference_values(case,decode,code,outputs):
    a,b,s,dy,bias=inputs(case);n,k=case['weight_shape'];dense=decode(b.flatten(),s,code,64,(n,k),a.dtype)
    return [tensor(v) for v in (a,b,s,dy,bias,code,dense,dense.t().contiguous()) if v is not None]+list(outputs.values())

def byte_sha(value):
    import hashlib
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def state_contract(scales,code,dtype):
    return {'shape':[256,512],'dtype':'torch.'+dtype,'blocksize':64,'quant_type':'nf4','nested':False,'absmax_byte_sha256':byte_sha(scales),'code_byte_sha256':byte_sha(code)}
