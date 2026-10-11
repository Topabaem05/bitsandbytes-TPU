"""Fixed79 public calls. Explicit new restore composite replaces diagnostic-only float addition.

The CPU path uses original default functions. No public method is replaced.
"""
import math

def compute_public(B,N,case,data,torch,bnb,device,functions=None):
    dtype=getattr(torch,case['dtype']); dynamic=torch.tensor(data['dynamic_map'],dtype=torch.float32,device=device)
    nf4=torch.tensor(data['nf4_map'],dtype=torch.float32,device=device)
    oracle=functions is not None; details={'metadata':N.metadata(case)}
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
        restored=decode8(codes,second)+offset if oracle else torch.ops.bnb_tpu_exact.restore_scales.default(codes,second,dynamic,offset)
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
        restored=decode8(codes,second)+offset if oracle else torch.ops.bnb_tpu_exact.restore_scales.default(codes,second,dynamic,offset)
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
