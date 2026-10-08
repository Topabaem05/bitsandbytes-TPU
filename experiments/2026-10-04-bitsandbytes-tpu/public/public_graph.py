"""Recover actual public-output graph; never replaces a factory or upstream call."""
import base64
import json
import math
import sys
from pathlib import Path
import contract as C

# Explicit sibling directory contains admitted helper bytes; no installed-module replacement.
HELPERS=Path(__file__).resolve().parent/'helpers'
_names=('verifier','proto_contract','graph_binding','mosaic_compatibility','recorder','conversion_recovery')
for _name in _names:
    if _name in sys.modules:C.require(Path(sys.modules[_name].__file__).resolve()==(HELPERS/(_name+'.py')).resolve(),'HELPER_MODULE_COLLISION:'+_name)
_added={_name for _name in _names if _name not in sys.modules}
sys.path.insert(0,str(HELPERS))
try:
    import verifier as V
    import proto_contract as PC
    import graph_binding as G
    import recorder as E
    import conversion_recovery as CR
finally:
    sys.path.pop(0)
    for _name in _added:sys.modules.pop(_name,None)


def grouped(inputs):
    a,b,s=inputs['activation'],inputs['packed'],inputs['scales'];n,k=256,512;m=math.prod(a['shape'][:-1])
    return [{'dtype':a['dtype'],'shape':[m,k],'values':a['values']},V.transpose({'dtype':'uint8','shape':[n,2,128],'values':b['values']},[1,0,2],[2,n,128]),V.transpose({'dtype':'float32','shape':[n,2,4],'values':s['values']},[1,0,2],[2,n,4])]

def payload(hlo):
    nodes,root=V.parse_entry(hlo);reachable=set();active=set()
    def visit(name):
        C.require(name not in active,'OUTPUT_GRAPH_CYCLE')
        if name in reachable:return
        active.add(name)
        for dep in nodes[name]['deps']:visit(dep)
        active.remove(name);reachable.add(name)
    visit(root);calls=[n for n in reachable if nodes[n]['opcode']=='custom-call']
    C.require(len(calls)==1,'ONE_ROOT_REACHABLE_PUBLIC_NATIVE_CALL');call=nodes[calls[0]]
    C.require(call['attrs'].get('custom_call_target')=='"tpu_custom_call"','PUBLIC_NATIVE_TARGET')
    value=call['attrs'].get('backend_config');info=V.recover_payload(value);config=json.loads(value,object_pairs_hook=V.unique_pairs)
    C.require(config['custom_call_config'].get('needs_layout_passes') is True,'MOSAIC_LAYOUT_PASSES_REQUIRED')
    return nodes,root,calls[0],value,info

def verify_native(hlo,context_proto,printer_proto,parameters,expected):
    nodes,root,callname,config,info=payload(hlo);binding=G.bind(context_proto,printer_proto,hlo,parameters,config)
    if nodes[root]['opcode']=='tuple':C.require(len(nodes[root]['deps'])==1,'SINGLE_PUBLIC_OUTPUT');root=nodes[root]['deps'][0]
    visible=nodes[root]['info'];C.require(visible['shape']==expected['outputs']['y']['shape'] and visible['dtype']==expected['outputs']['y']['dtype'],'PUBLIC_ROOT_SHAPE_DTYPE')
    cur=root;seen=set()
    while cur!=callname:
        C.require(cur not in seen,'PUBLIC_OUTPUT_CYCLE');seen.add(cur);node=nodes[cur]
        C.require(node['opcode'] in ('reshape','copy') and len(node['deps'])==1,'PUBLIC_OUTPUT_VIEW_ONLY');cur=node['deps'][0]
    call=nodes[callname];target=grouped(expected['inputs']);dtype=target[0]['dtype']
    C.require(call['info']=={'dtype':dtype,'shape':[128,256],'layout':[1,0]},'EXACT_NATIVE_TILE_OUTPUT')
    C.require(len(call['deps'])==3,'PUBLIC_ORDERED_OPERANDS')
    constraints=call['attrs'].get('operand_layout_constraints');C.require(isinstance(constraints,str) and constraints.startswith('{') and constraints.endswith('}'),'PUBLIC_LAYOUT_CONSTRAINTS')
    layout=[V.shape(s) for s in V.split(constraints[1:-1])]
    C.require(layout==[{'dtype':v['dtype'],'shape':v['shape'],'layout':list(reversed(range(len(v['shape']))))} for v in target],'PUBLIC_FIXED_OPERAND_LAYOUT')
    cache={};active=set();used=set()
    def evaluate(name):
        if name in cache:return cache[name]
        C.require(name not in active,'OPERAND_CYCLE');active.add(name);node=nodes[name];meta=node['info'];op=node['opcode']
        if op=='parameter':
            key=str(node['parameter']);C.require(key in parameters,'ACTUAL_PARAMETER_ID');value=V.array(parameters[key]);used.add(key)
        else:
            C.require(op in ('reshape','copy','transpose') and len(node['deps'])==1,'PUBLIC_INPUT_LINEAGE')
            value=evaluate(node['deps'][0]);C.require(value['dtype']==meta['dtype'],'PUBLIC_OPERAND_DTYPE')
            if op=='transpose':value=V.transpose(value,V.integer_list(node['attrs'].get('dimensions')),meta['shape'])
            elif op=='reshape':
                C.require(math.prod(value['shape'])==math.prod(meta['shape']),'PUBLIC_OPERAND_RESHAPE');value={**value,'shape':meta['shape']}
            else:C.require(value['shape']==meta['shape'],'PUBLIC_OPERAND_COPY')
        C.require(value['shape']==meta['shape'] and value['dtype']==meta['dtype'],'PUBLIC_OPERAND_METADATA');active.remove(name);cache[name]=value;return value
    values=[evaluate(dep) for dep in call['deps']]
    C.require(values==target and used==set(parameters),'PUBLIC_FIXED_RECIPE_OPERAND_CONTENT_ORDER')
    return {'record_validation':'PASS','graph_scope':'ACTUAL_PUBLIC_OUTPUT_GRAPH','graph_binding':binding,'operand_value_sha256':[C.digest(v) for v in values],**info,'observed_call_count':1,'executed_executable_link':'UNKNOWN'}


def verify_boundary(row,payload,expected):
    """Bind retained conversion to the actual root-reachable graph payload."""
    C.require(isinstance(row,dict) and row.get('status')=='RETURNED' and row.get('native_api')=='_xla_tpu_custom_call' and row.get('calls')==1,'PUBLIC_NATIVE_RETURNED_ONCE')
    admission=C.read(C.HERE/'package-admission.json')
    C.require(row.get('factory_sha256')==admission['plugin_python_files']['native_factory.py'] and row.get('converter_sha256')==C.sha(HELPERS/'mosaic_compatibility.py'),'PUBLIC_FACTORY_CONVERTER_SOURCE')
    C.require(row.get('payload')==payload and row.get('payload_sha256')==C.sha_bytes(payload.encode()),'PUBLIC_CONVERTED_ACTUAL_GRAPH_CONFIG')
    original=row.get('original_payload');C.require(type(original)is str and row.get('original_payload_sha256')==C.sha_bytes(original.encode()),'PUBLIC_ORIGINAL_PAYLOAD_RETAINED')
    old=json.loads(original,object_pairs_hook=V.unique_pairs);new=json.loads(payload,object_pairs_hook=V.unique_pairs)
    old_body=base64.b64decode(old['custom_call_config'].pop('body'),validate=True);new_body=base64.b64decode(new['custom_call_config'].pop('body'),validate=True)
    C.require(old==new and old_body and new_body,'PUBLIC_NONBODY_CONFIG_UNCHANGED')
    a=row.get('payload_conversion');C.require(isinstance(a,dict),'PUBLIC_CONVERSION_AUDIT')
    fields={'kind':'MOSAIC_SUPPORTED_SERDE_8_TO_7_V1','status':'FRONTEND_CONVERSION_ONLY','source_version':8,'target_version':7,'original_payload_sha256':C.sha_bytes(original.encode()),'original_body_sha256':C.sha_bytes(old_body),'converted_payload_sha256':C.sha_bytes(payload.encode()),'converted_body_sha256':C.sha_bytes(new_body),'exact_current_ir_equal':True,'jax':'0.7.1','jaxlib':'0.7.1','serializer_python_sha256':E.C.TCC_SHA,'converter_sha256':C.sha(HELPERS/'mosaic_compatibility.py'),'native_backend_acceptance':'NOT_RUN','runtime_changed':False,'executable_link':'UNKNOWN','allocator_peak':'UNKNOWN'}
    C.require(set(a)==set(fields)|{'current_ir_sha256','roundtrip_ir_sha256'} and all(a.get(k)==v for k,v in fields.items()),'PUBLIC_CONVERSION_FIELDS')
    C.require(isinstance(a.get('current_ir_sha256'),str) and __import__('re').fullmatch('[0-9a-f]{64}',a['current_ir_sha256']) and a['current_ir_sha256']==a.get('roundtrip_ir_sha256'),'PUBLIC_CURRENT_IR_READBACK')
    target=grouped(expected['inputs']);C.require(row.get('metadata')==[{'shape':v['shape'],'dtype':v['dtype']}for v in target] and row.get('output_shapes')==[[128,256]] and row.get('output_dtypes')==[target[0]['dtype']],'PUBLIC_ACTUAL_CALL_METADATA')
    ids=row.get('operand_object_ids');C.require(isinstance(ids,list) and len(ids)==len(set(ids))==3 and all(type(v)is int and v>0 for v in ids),'PUBLIC_ORIGINAL_OPERAND_OBJECT_IDS')
    return row


def general_graph(proto):
    """Bind the ENTRY and each reachable reduction computation without native calls."""
    f=PC.fields(proto);entry=PC.one(f,6,0);comps={}
    for w,raw in f.get(3,[]):
        C.require(w==2,'REFERENCE_COMPUTATION_WIRE');comp=PC.fields(raw);cid=PC.one(comp,5,0);C.require(cid not in comps,'REFERENCE_COMPUTATION_ID');nodes={}
        for iw,b in comp.get(2,[]):
            n=PC.fields(b);i=PC.one(n,35,0);C.require(iw==2 and i not in nodes,'REFERENCE_NODE_ID');nodes[i]=n
        root=PC.one(comp,6,0);C.require(root in nodes,'REFERENCE_ROOT');comps[cid]=(nodes,root)
    C.require(entry in comps,'REFERENCE_ENTRY');active=set();reached=set();called=set();ops=[]
    def project(cid,i):
        key=(cid,i);nodes,_=comps[cid];C.require(key not in active and i in nodes,'REFERENCE_GRAPH_CYCLE_OR_ID');active.add(key);reached.add(key);n=nodes[i]
        C.require(not PC.ints(n,37),'REFERENCE_CONTROL_DEPENDENCIES_NOT_ADMITTED')
        op=PC.one(n,2,b'').decode();C.require(op!='custom-call','REFERENCE_NO_CUSTOM_CALL');ops.append(op)
        keep={str(k):[(w,v.hex() if isinstance(v,bytes) else v) for w,v in values] for k,values in n.items() if k not in (1,7,35,36,38)}
        keep['operands']=[project(cid,d) for d in PC.ints(n,36)]
        callees=PC.ints(n,38);C.require(not callees or op=='reduce','REFERENCE_CALLEE_OPCODE_SCOPE')
        keep['callees']=[]
        for callee in callees:
            C.require(callee in comps and callee!=cid,'REFERENCE_CALLEE_ID');called.add(callee);keep['callees'].append(project(callee,comps[callee][1]))
        active.remove(key);return keep
    nodes,root=comps[entry];canonical=project(entry,root)
    C.require(set(comps)=={entry}|called,'REFERENCE_UNREACHABLE_COMPUTATION')
    for cid,(cnodes,croot) in comps.items():C.require(all((cid,i) in reached for i in cnodes),'REFERENCE_DISCONNECTED_NODE')
    visible=root
    if PC.one(nodes[root],2,b'')==b'tuple':
        deps=PC.ints(nodes[root],36);C.require(len(deps)==1,'REFERENCE_SINGLE_OUTPUT_TUPLE');visible=deps[0]
    return {'canonical':canonical,'opcodes':ops,'shape':PC.shape(PC.one(nodes[visible],3,b'')),'parameters':{str(PC.one(n,9,0)):PC.shape(PC.one(n,3,b'')) for i,n in nodes.items() if (entry,i) in reached and PC.one(n,2,b'')==b'parameter'}}


def verify_reference(context_proto,printer_proto,expected_shape,dtype,*,bias_gradient=False,parameters=None,allowed_parameters=None):
    a,b=general_graph(context_proto),general_graph(printer_proto);C.require(a['canonical']==b['canonical'],'REFERENCE_CONTEXT_PRINTER_CONTENT')
    C.require(isinstance(parameters,dict) and set(parameters)==set(a['parameters']),'REFERENCE_PARAMETER_NUMBERS')
    C.require(isinstance(allowed_parameters,list) and allowed_parameters,'REFERENCE_FIXED_INPUT_EXPECTATIONS')
    for key,value in parameters.items():
        V.array(value);meta=a['parameters'][key];C.require(value['dtype']==meta['dtype'] and value['shape']==meta['shape'],'REFERENCE_PARAMETER_SHAPE_DTYPE')
        C.require(any(value['dtype']==v['dtype'] and value['values']==v['values'] for v in allowed_parameters),'REFERENCE_FIXED_PARAMETER_CONTENT')
    shape=a['shape'];C.require('tuple' not in shape and shape['shape']==expected_shape and shape['dtype']==dtype,'REFERENCE_ACTUAL_OUTPUT_SHAPE')
    C.require(('reduce' if bias_gradient else 'dot') in a['opcodes'],'REFERENCE_REDUCTION_REQUIRED' if bias_gradient else 'REFERENCE_DENSE_DOT_REQUIRED')
    return {'record_validation':'PASS','graph_scope':'ACTUAL_REFERENCE_OR_GRADIENT_OUTPUT','context_proto_sha256':C.sha_bytes(context_proto),'printer_proto_sha256':C.sha_bytes(printer_proto),'opcode_counts':{op:a['opcodes'].count(op) for op in sorted(set(a['opcodes']))},'executed_executable_link':'UNKNOWN'}
