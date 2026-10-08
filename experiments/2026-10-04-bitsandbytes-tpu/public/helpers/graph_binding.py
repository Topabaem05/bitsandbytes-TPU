"""Bind mapped context proto, tensor-printer proto, and HLO syntax. No executable ID claim."""
import proto_contract as C
import verifier as V


def info(raw):
    value=C.shape(raw)
    if 'tuple' in value:return {'tuple':[{'dtype':v['dtype'],'shape':v['shape'],'layout':v['minor_to_major']} for v in value['tuple']]}
    return {'dtype':value['dtype'],'shape':value['shape'],'layout':value['minor_to_major']}

def bind(context_proto,printer_proto,hlo_text,parameters,payload):
    context=C.module(context_proto);printer=C.module(printer_proto)
    V.require(context['canonical']==printer['canonical'],'MAPPED_CONTEXT_PRINTER_PROTO_CONTENT')
    V.require(C.one(context['call'],43,b'')==payload.encode(),'MAPPED_PROTO_NATIVE_PAYLOAD')
    V.require(set(parameters)=={str(k) for k in context['parameters']},'MAPPED_PROTO_PARAMETER_NUMBERS')
    for key,array in parameters.items():
        V.array(array);value=context['parameters'][int(key)];V.require(array['dtype']==value['dtype'] and array['shape']==value['shape'],'MAPPED_PROTO_PARAMETER_SHAPE_DTYPE')
    hn,hr=V.parse_entry(hlo_text)
    if hn[hr]['opcode']=='tuple':V.require(len(hn[hr]['deps'])==1,'TEXT_ROOT_TUPLE_SCOPE');hr=hn[hr]['deps'][0]
    reached=set();active=set()
    def hproject(name):
        V.require(name not in active,'TEXT_GRAPH_CYCLE');active.add(name);reached.add(name);n=hn[name]
        V.require(set(n['attrs'])<=({'dimensions','metadata'} if n['opcode']=='transpose' else {'custom_call_target','backend_config','operand_layout_constraints','metadata'} if n['opcode']=='custom-call' else {'metadata'}),'HLO_TEXT_ATTRIBUTE_SCOPE')
        if n['opcode'] in ('reshape','copy'):
            V.require(len(n['deps'])==1,'PUBLIC_UNARY_VIEW');source=hn[n['deps'][0]]['info'];target=n['info'];V.require(source['dtype']==target['dtype'] and __import__('math').prod(source['shape'])==__import__('math').prod(target['shape']),'PUBLIC_VIEW_SIZE_DTYPE');V.require(n['opcode']!='copy' or source['shape']==target['shape'],'PUBLIC_COPY_SHAPE')
        value={'opcode':n['opcode'],'signature':n['signature'],'parameter':n['parameter'],'operands':[hproject(dep) for dep in n['deps']]}
        if n['opcode']=='transpose':value['dimensions']=V.integer_list(n['attrs'].get('dimensions'))
        if n['opcode']=='custom-call':
            value.update(target=n['attrs'].get('custom_call_target'),payload=n['attrs'].get('backend_config'));constraints=n['attrs'].get('operand_layout_constraints');V.require(constraints is not None and constraints.startswith('{') and constraints.endswith('}'),'TEXT_CONSTRAINTS_REQUIRED');value['constraints']=[V.shape(s) for s in V.split(constraints[1:-1])]
        active.remove(name);return value
    def pproject(i):
        n=printer['nodes'][i];op=C.one(n,2,b'').decode();value={'opcode':op,'signature':info(C.one(n,3,b'')),'parameter':C.one(n,9,0) if op=='parameter' else None,'operands':[pproject(dep) for dep in C.ints(n,36)]}
        if op in ('reshape','copy'):
            deps=C.ints(n,36);V.require(len(deps)==1,'PUBLIC_PROTO_UNARY_VIEW');source=info(C.one(printer['nodes'][deps[0]],3,b''));target=value['signature'];V.require(source['dtype']==target['dtype'] and __import__('math').prod(source['shape'])==__import__('math').prod(target['shape']),'PUBLIC_PROTO_VIEW_SIZE_DTYPE');V.require(op!='copy' or source['shape']==target['shape'],'PUBLIC_PROTO_COPY_SHAPE')
        if op=='transpose':value['dimensions']=C.ints(n,14)
        if op=='custom-call':value.update(target='"'+C.one(n,28,b'').decode()+'"',payload=C.one(n,43,b'').decode(),constraints=[info(b) for w,b in n.get(57,[])])
        return value
    V.require(hproject(hr)==pproject(printer['root']),'PRINTER_TEXT_PROTO_CONTENT')
    extra=set(hn)-reached;V.require(not extra or (len(extra)==1 and hn[next(iter(extra))]['opcode']=='tuple'),'TEXT_DISCONNECTED_GRAPH')
    return {'status':'MAPPED_PRINTER_GRAPH_BINDING_PASS','context_proto_sha256':V.digest(context_proto),'printer_proto_sha256':V.digest(printer_proto),'printer_hlo_sha256':V.digest(hlo_text),'native_payload_sha256':V.digest(payload),'parameter_shape_dtype_binding':'PASS','parameter_values':'REQUIRES_SEPARATE_FIXED_INPUT_VERIFIER','executed_executable_link':'UNKNOWN'}
