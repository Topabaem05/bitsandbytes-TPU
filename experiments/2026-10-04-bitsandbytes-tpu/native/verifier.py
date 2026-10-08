"""Private bounded native-boundary verifier. No device initialization or M6 acceptance."""
import base64
import hashlib
import json
import math
import re

DTYPES={'f32':'float32','bf16':'bfloat16','u8':'uint8'}

def require(value, message):
    if not value: raise ValueError(message)
def digest(value):
    data=value if isinstance(value,bytes) else value.encode() if isinstance(value,str) else json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()
def unique_pairs(pairs):
    result={}
    for key,value in pairs:
        require(key not in result,'DUPLICATE_JSON_KEY');result[key]=value
    return result

def split(text):
    """Split HLO commas only at top level, including quotes and nested shapes."""
    parts=[];start=0;stack=[];quoted=False;escape=False
    for i,char in enumerate(text):
        if quoted:
            if escape:escape=False
            elif char=='\\':escape=True
            elif char=='"':quoted=False
        elif char=='"':quoted=True
        elif char in '([{':stack.append(char)
        elif char in ')]}':
            require(stack and stack.pop()=={')':'(',']':'[','}':'{'}[char],'HLO_BALANCE')
        elif char==',' and not stack:parts.append(text[start:i].strip());start=i+1
    require(not quoted and not stack,'HLO_BALANCE')
    if text[start:].strip():parts.append(text[start:].strip())
    return parts

def shape(text):
    match=re.fullmatch(r'(f32|bf16|u8)\[([0-9,]*)\](?:\{([0-9,]*)\})?',text.strip())
    require(match is not None,'HLO_SHAPE_UNSUPPORTED')
    dims=[int(v) for v in match[2].split(',') if v];layout=[int(v) for v in match[3].split(',') if v] if match[3] is not None else None
    require(dims and all(v>0 for v in dims) and (layout is None or sorted(layout)==list(range(len(dims)))),'HLO_SHAPE')
    return {'dtype':DTYPES[match[1]],'shape':dims,'layout':layout}

def signature(text):
    text=text.strip()
    if text.startswith('('):
        require(text.endswith(')'),'HLO_TUPLE_SHAPE');return {'tuple':[signature(v) for v in split(text[1:-1])]}
    return shape(text)

def parse_entry(text):
    lines=text.splitlines();entries=[i for i,line in enumerate(lines) if re.match(r'^ENTRY\s',line)]
    require(len(entries)==1,'HLO_ENTRY');nodes={};root=None;closed=False
    for line in lines[entries[0]+1:]:
        if line.strip()=='}':closed=True;break
        line=line.strip()
        if not line:continue
        binding=re.fullmatch(r'(ROOT\s+)?%?([\w.-]+)\s*=\s*(.+)',line);require(binding is not None,'HLO_SINGLE_LINE_NODE')
        name,body=binding[2],binding[3];require(name not in nodes,'HLO_DUPLICATE_NODE')
        operation=re.search(r'\s([a-z][a-z-]*)\(',body);require(operation is not None,'HLO_OPCODE')
        prefix=body[:operation.start()].strip();opcode=operation[1]
        # Tuple operations are only output containers. Tuple custom calls are excluded in this one-output protocol.
        info=None if prefix.startswith('(') and opcode in ('tuple','get-tuple-element') else shape(prefix)
        start=operation.end();depth=1;end=start
        while end<len(body) and depth:
            if body[end]=='(':depth+=1
            elif body[end]==')':depth-=1
            end+=1
        require(depth==0,'HLO_OPERANDS')
        arguments=split(body[start:end-1]);attrs={}
        tail=body[end:].strip().removeprefix(',').strip()
        for value in split(tail):
            pair=value.split('=',1);require(len(pair)==2 and pair[0].strip() not in attrs,'HLO_ATTRIBUTE');attrs[pair[0].strip()]=pair[1].strip()
        if opcode=='parameter':
            require(len(arguments)==1 and arguments[0].isdigit(),'HLO_PARAMETER');deps=[];parameter=int(arguments[0])
        else:
            deps=[];parameter=None
            for value in arguments:
                symbol=re.search(r'%?([\w.-]+)$',value);require(symbol is not None,'HLO_DEPENDENCY');deps.append(symbol[1])
        require(opcode in ('parameter','reshape','transpose','copy','tuple','get-tuple-element','custom-call'),'HLO_OPCODE_UNSUPPORTED')
        annotations=[]
        if opcode!='parameter':
            for value in arguments:
                symbol=re.search(r'%?([\w.-]+)$',value);annotation=value[:symbol.start()].strip();annotations.append(signature(annotation) if annotation else None)
        nodes[name]={'signature':signature(prefix),'annotations':annotations,'info':info,'opcode':opcode,'deps':deps,'attrs':attrs,'parameter':parameter}
        if binding[1]:require(root is None,'HLO_ROOT');root=name
    require(closed and root in nodes,'HLO_ROOT')
    for node in nodes.values():
        require(all(dep in nodes for dep in node['deps']),'HLO_DEPENDENCY')
        for dep,annotation in zip(node['deps'],node['annotations']):
            require(annotation is None or annotation==nodes[dep]['signature'],'HLO_OPERAND_ANNOTATION')
        if node['opcode']=='tuple':require(node['signature']=={'tuple':[nodes[dep]['signature'] for dep in node['deps']]},'HLO_TUPLE_SHAPE')
    return nodes,root

def array(value,expected=None):
    require(isinstance(value,dict) and set(value)=={'dtype','shape','values'},'ARRAY_FIELDS')
    dims=value['shape'];dtype=value['dtype'];values=value['values']
    require(isinstance(dims,list) and dims and all(type(v)is int and v>0 for v in dims) and dtype in DTYPES.values(),'ARRAY_IDENTITY')
    require(isinstance(values,list) and len(values)==math.prod(dims) and all(type(v)in(int,float) and math.isfinite(v) for v in values),'ARRAY_VALUES')
    if dtype=='uint8':require(all(type(v)is int and 0<=v<=255 for v in values),'ARRAY_UINT8')
    if expected is not None:require(value==expected,'EXACT_ARRAY_BINDING')
    return value

def transpose(value,permutation,target):
    require(sorted(permutation)==list(range(len(value['shape']))) and target==[value['shape'][i] for i in permutation],'TRANSPOSE_DIMS')
    source=value['shape'];src_strides=[math.prod(source[i+1:]) for i in range(len(source))];out=[]
    for flat in range(math.prod(target)):
        index=[];remain=flat
        for i in range(len(target)):
            stride=math.prod(target[i+1:]);index.append(remain//stride);remain%=stride
        offset=sum(index[j]*src_strides[permutation[j]] for j in range(len(index)));out.append(value['values'][offset])
    return {'dtype':value['dtype'],'shape':target,'values':out}

def integer_list(value):
    require(isinstance(value,str) and re.fullmatch(r'\{[0-9, ]*\}',value),'HLO_INTEGER_LIST')
    return [int(v.strip()) for v in value[1:-1].split(',') if v.strip()]

def recover_payload(value):
    require(isinstance(value,str) and value,'CONFIG_MISSING')
    # Pinned HLO prints JSON dictionaries verbatim; quoted non-JSON opaque payloads are outside this protocol.
    require(value.startswith('{'),'CONFIG_JSON_DICTIONARY_REQUIRED')
    config=json.loads(value,object_pairs_hook=unique_pairs);require(isinstance(config,dict) and isinstance(config.get('custom_call_config'),dict),'CONFIG_MOSAIC')
    body=config['custom_call_config'].get('body');require(isinstance(body,str) and body,'CONFIG_BODY')
    decoded=base64.b64decode(body,validate=True);require(decoded,'CONFIG_BODY')
    return {'payload_sha256':digest(value),'body_sha256':digest(decoded),'payload_bytes':len(value.encode()),'body_bytes':len(decoded)}

def verify(record,hlo_text,expected,B,P):
    """expected is an independently owned protocol/input/oracle/source/launch contract, not a child claim."""
    require(record.get('status')=='COMPLETE' and not record.get('error'),'TERMINAL')
    require(record.get('protocol')=='R6_NATIVE_BOUNDARY_V1','PROTOCOL')
    for key in ('source_pins','owner','case','inputs'):require(record.get(key)==expected.get(key),'EXPECTED_'+key.upper())
    require(record.get('hlo_sha256')==digest(hlo_text),'HLO_HASH')
    require(record.get('selected_path')=={'path':'PALLAS_CANDIDATE','calls':1,'fallback':False},'SELECTED_PATH')
    require(record.get('scope') in ('SYNTHETIC_PROTOCOL_FIXTURE','ACTUAL_NATIVE_BOUNDARY'),'SCOPE')
    boundary=record.get('native_boundary');require(isinstance(boundary,dict) and boundary.get('status')=='RETURNED' and boundary.get('native_api')=='_xla_tpu_custom_call' and boundary.get('calls')==1,'NATIVE_BOUNDARY')
    require(boundary.get('recorder_sha256')==expected['recorder_sha256'],'RECORDER_SOURCE')
    info=recover_payload(boundary.get('payload'));require(boundary.get('payload_sha256')==info['payload_sha256'],'BOUNDARY_PAYLOAD_HASH')
    nodes,root=parse_entry(hlo_text)
    # Select exactly the returned one-output graph; other roots/calls are not evidence.
    seen=set()
    def output_call(name):
        require(name not in seen,'HLO_CYCLE');seen.add(name);node=nodes[name];op=node['opcode'];deps=node['deps']
        if op=='custom-call':return name
        if op in ('copy','reshape'):
            require(len(deps)==1 and node['info']['dtype']==nodes[deps[0]]['info']['dtype'] and math.prod(node['info']['shape'])==math.prod(nodes[deps[0]]['info']['shape']),'OUTPUT_ALIAS');return output_call(deps[0])
        if op=='tuple':require(len(deps)==1,'ACTUAL_OUTPUT_ONLY');return output_call(deps[0])
        if op=='get-tuple-element':
            require(len(deps)==1 and nodes[deps[0]]['opcode']=='tuple','OUTPUT_TUPLE');index=node['attrs'].get('index');require(index is not None and index.isdigit(),'OUTPUT_TUPLE_INDEX');entries=nodes[deps[0]]['deps'];require(int(index)<len(entries),'OUTPUT_TUPLE_INDEX');return output_call(entries[int(index)])
        raise ValueError('ACTUAL_ROOT_NOT_CUSTOM_CALL')
    call_name=output_call(root);call=nodes[call_name];attrs=call['attrs']
    visible=nodes[root]['info']
    if nodes[root]['opcode']=='tuple':visible=nodes[nodes[root]['deps'][0]]['info']
    require(visible is not None and visible['shape']==expected['result_hlo']['shape'] and visible['dtype']==expected['result_hlo']['dtype'],'ACTUAL_ROOT_OUTPUT_SHAPE')
    require(attrs.get('custom_call_target')=='"tpu_custom_call"','CALL_TARGET')
    require(attrs.get('backend_config')==boundary['payload'],'EXECUTED_GRAPH_PAYLOAD_BYTES')
    require(len(call['deps'])==3 and len(boundary.get('operands',[]))==3,'OPERAND_ORDER')
    require(call['info']==expected['result_hlo'],'RESULT_CONTRACT')
    require(boundary.get('output_shapes')==[expected['result_hlo']['shape']] and boundary.get('output_dtypes')==[expected['result_hlo']['dtype']],'BOUNDARY_OUTPUT_SPEC')
    constraints=attrs.get('operand_layout_constraints');require(constraints is not None and constraints.startswith('{') and constraints.endswith('}'),'OPERAND_LAYOUT_MISSING')
    layouts=[shape(v) for v in split(constraints[1:-1])];require(layouts==expected['operand_hlo'],'OPERAND_LAYOUT')
    parameters=record.get('parameters');require(isinstance(parameters,dict),'PARAMETER_MAP');cache={};pending=set();used=set()
    def evaluate(name):
        if name in cache:return cache[name]
        require(name not in pending,'HLO_CYCLE');pending.add(name);node=nodes[name];op=node['opcode'];dims=node['info']['shape'];dtype=node['info']['dtype']
        if op=='parameter':
            pid=str(node['parameter']);require(pid in parameters,'PARAMETER_MAP');value=array(parameters[pid]);used.add(pid)
        else:
            require(op in ('reshape','transpose','copy') and len(node['deps'])==1,'OPERAND_LINEAGE_UNSUPPORTED');value=evaluate(node['deps'][0]);require(value['dtype']==dtype,'OPERAND_DTYPE')
            if op=='transpose':value=transpose(value,integer_list(node['attrs'].get('dimensions')),dims)
            elif op=='reshape':require(math.prod(value['shape'])==math.prod(dims),'RESHAPE_SIZE');value={'dtype':dtype,'shape':dims,'values':value['values']}
            else:require(value['shape']==dims,'COPY_SHAPE')
        require(value['shape']==dims and value['dtype']==dtype,'PARAMETER_DTYPE_SHAPE');pending.remove(name);cache[name]=value;return value
    canonical=expected['inputs'];grouped=expected['grouped']
    require(set(canonical)=={'activation','packed','scales'},'CANONICAL_INPUT_FIELDS')
    for value in canonical.values():array(value)
    m,k=canonical['activation']['shape'];q=k//256;n=expected['result_hlo']['shape'][1]
    require(m%128==n%128==k%256==0 and k>256 and canonical['packed']['shape']==[n*k//2,1] and canonical['packed']['dtype']=='uint8' and canonical['scales']['shape']==[n*k//64] and canonical['scales']['dtype']=='float32','FIXED_LAYOUT_CASE')
    require(expected['result_hlo']['shape']==[m,n] and expected['result_hlo']['dtype']==canonical['activation']['dtype'],'FIXED_OUTPUT_SHAPE')
    dtype=canonical['activation']['dtype'];profile,_=B.load_spec()
    require(expected['tolerance']==profile['tolerances']['fp32_forward_gradient' if dtype=='float32' else 'bf16_forward_unit_scale'],'FIXED_TOLERANCE')
    require(len(grouped)==3,'EXPECTED_OPERANDS')
    independently_grouped=[canonical['activation'],transpose({'dtype':'uint8','shape':[n,q,128],'values':canonical['packed']['values']},[1,0,2],[q,n,128]),transpose({'dtype':'float32','shape':[n,q,4],'values':canonical['scales']['values']},[1,0,2],[q,n,4])]
    require(grouped==independently_grouped,'EXPECTED_GROUPED_LAYOUT')
    for i,name in enumerate(call['deps']):
        value=evaluate(name);array(value,grouped[i]);require(nodes[name]['info']['dtype']==layouts[i]['dtype'] and nodes[name]['info']['shape']==layouts[i]['shape'],'OPERAND_GRAPH_SHAPE')
        require(boundary['operands'][i]=={'dtype':value['dtype'],'shape':value['shape'],'value_sha256':digest(value)},'BOUNDARY_OPERAND_BINDING')
    require(set(parameters)==used,'PARAMETER_MAP_EXACT')
    require(sorted(digest(array(v)) for v in parameters.values())==sorted(digest(array(v)) for v in canonical.values()),'CANONICAL_PARAMETER_VALUES')
    sync=record.get('execution');require(isinstance(sync,dict) and sync.get('api')=='_xla_sync_multi' and sync.get('targets')==['actual'] and sync.get('wait') is True and sync.get('wait_device_ops') is True and sync.get('metrics_after_sync_before_output') is True,'ACTUAL_EXECUTION_INTERVAL')
    require(sync.get('graph_hash_binding')=='PRE_SYNC_IDENTIFIER_NOT_EXECUTABLE_LINK','GRAPH_HASH_SCOPE')
    require(sync.get('other_execution_in_interval') is False and sync.get('graph_hash') and sync.get('graph_hash')==boundary.get('graph_hash'),'GRAPH_SYNC_BINDING')
    require(all(type(sync.get(k))in(int,float) and math.isfinite(sync[k]) for k in ('started_monotonic','finished_monotonic')) and sync['started_monotonic']<=sync['finished_monotonic'],'EXECUTION_TIMESTAMPS')
    P.validate_execution(B,record)
    array(record.get('output'));array(expected['cpu_reference']);require(record['output']['shape']==expected['cpu_reference']['shape'] and record['output']['dtype']==expected['cpu_reference']['dtype'],'OUTPUT_ARRAY_IDENTITY')
    gate=B.numeric(record['output']['values'],expected['cpu_reference']['values'],expected['tolerance']);require(gate['status']=='PASS','NUMERICAL_GATE')
    # These are intentionally separate, even when all bounded diagnostic evidence passes.
    return {'status':'BINDING_PASS','scope':record['scope'],'native_boundary':'PASS','root_reachable_graph':'PASS','operand_values_order_layout':'PASS','synchronized_execution':'PASS','numerical_gate':gate,**info,'actual_hardware_runtime':'REQUIRES_PARENT_ADMISSION','public_api':'NOT_QUALIFIED','optimized_executable_link':'NOT_QUALIFIED','compiler_body':'NOT_QUALIFIED','compiler_memory':'NOT_QUALIFIED','allocator_peak':'NOT_QUALIFIED','m6':'NOT_QUALIFIED'}
