"""Bounded typed HLO output/call witness. No compiler or device execution.

Accept single-line pinned HLO nodes, tuple projections, simple aliases, and
calls with a direct bitcast result. Unsupported control flow fails closed.
"""
import re

def require(value,message):
    if not value:raise ValueError(message)

def split(value):
    parts=[];start=0;depth=0
    for i,c in enumerate(value):
        if c in '[{(':depth+=1
        elif c in ']})':depth-=1
        elif c==',' and depth==0:parts.append(value[start:i].strip());start=i+1
        require(depth>=0,'HLO_ARGUMENTS')
    require(depth==0,'HLO_ARGUMENTS');parts.append(value[start:].strip())
    return parts if value.strip() else []

def parse(text):
    computations={};current=None;entry=None
    for raw in text.splitlines():
        line=re.sub(r'/\*.*?\*/','',raw).strip()
        if not line or line.startswith('HloModule'):continue
        if line=='}':require(current is not None,'HLO_COMPUTATION_END');current=None;continue
        if current is None:
            header=re.match(r'(ENTRY\s+)?%?([\w.:-]+)\s*(?:\(.*\)\s*->.*?)?\{\s*$',line)
            require(header is not None,'HLO_COMPUTATION_HEADER');name=header.group(2)
            require(name not in computations,'HLO_COMPUTATION_DUPLICATE')
            current={'nodes':{},'root':None,'name':name};computations[name]=current
            if header.group(1):require(entry is None,'HLO_ENTRY');entry=name
            continue
        match=re.match(r'(ROOT\s+)?%?([\w.:-]+)\s*=\s*(.*)',line);require(match is not None,'HLO_NODE')
        name,body=match.group(2),match.group(3);op=re.search(r'\b([a-z][a-z0-9-]*)\(',body)
        require(op is not None and name not in current['nodes'],'HLO_NODE')
        end=op.end();depth=1
        while end<len(body) and depth:
            if body[end]=='(':depth+=1
            elif body[end]==')':depth-=1
            end+=1
        require(depth==0,'HLO_NODE');opcode=op.group(1);args=split(body[op.end():end-1]);deps=[]
        shape=re.match(r'([a-z0-9]+)\[([0-9,]*)\]',body)
        if opcode not in ('constant','parameter','iota'):
            for arg in args:
                symbol=re.search(r'%?([\w.:-]+)\s*$',arg);require(symbol is not None,'HLO_DEPENDENCY');deps.append(symbol.group(1))
        require(opcode not in ('fusion','while','conditional','custom-call'),'HLO_UNSUPPORTED_COMPUTATION')
        node={'dtype':shape.group(1) if shape else None,'shape':[int(x) for x in shape.group(2).split(',')] if shape and shape.group(2) else [],
              'opcode':opcode,'deps':deps,'line':line,'body':body}
        if opcode=='parameter':require(len(args)==1 and args[0].isdigit(),'HLO_PARAMETER');node['parameter']=int(args[0])
        if opcode=='get-tuple-element':
            index=re.search(r'\bindex\s*=\s*(\d+)',body);require(index is not None and len(deps)==1,'HLO_TUPLE');node['index']=int(index.group(1))
        if opcode=='call':
            callee=re.search(r'\bto_apply\s*=\s*%?([\w.:-]+)',body);require(callee is not None,'HLO_CALL');node['callee']=callee.group(1)
        current['nodes'][name]=node
        if match.group(1):require(current['root'] is None,'HLO_ROOT');current['root']=name
    require(current is None and entry is not None,'HLO_ENTRY')
    for comp in computations.values():
        require(comp['root'] is not None,'HLO_ROOT')
        for node in comp['nodes'].values():
            require(all(dep in comp['nodes'] for dep in node['deps']),'HLO_DEPENDENCY')
            if node['opcode']=='call':require(node['callee'] in computations,'HLO_CALL_COMPUTATION')
    return computations,entry

def output_nodes(comps,entry,order):
    comp=comps[entry];root=comp['nodes'][comp['root']]
    if root['opcode']=='tuple':require(len(root['deps'])==len(order),'HLO_OUTPUT_MATRIX');names=root['deps']
    else:require(len(order)==1,'HLO_OUTPUT_MATRIX');names=[comp['root']]
    return dict(zip(order,names))

def resolve(comps,comp_name,name,projection=None,trail=()):
    require((comp_name,name,projection) not in trail,'HLO_CYCLE');trail=trail+((comp_name,name,projection),)
    comp=comps[comp_name];node=comp['nodes'][name];op=node['opcode']
    if op=='tuple':
        require(projection is not None and projection<len(node['deps']),'HLO_TUPLE')
        return resolve(comps,comp_name,node['deps'][projection],None,trail)
    if op=='get-tuple-element':return resolve(comps,comp_name,node['deps'][0],node['index'],trail)
    if op in ('copy','reshape'):
        require(projection is None and len(node['deps'])==1,'HLO_ALIAS');result=resolve(comps,comp_name,node['deps'][0],None,trail)
        require(node['dtype']==result['target_dtype'] and node['shape']==result['shape'],'HLO_ALIAS_TYPE');return result
    if op=='call':
        callee=comps[node['callee']];params=[n for n in callee['nodes'].values() if n['opcode']=='parameter']
        require(len(params)==len(node['deps'])==1 and params[0]['parameter']==0,'HLO_CALL_ARITY')
        source=comp['nodes'][node['deps'][0]]
        require(source['dtype']==params[0]['dtype'] and source['shape']==params[0]['shape'],'HLO_CALL_ARGUMENT_TYPE')
        result=resolve(comps,node['callee'],callee['root'],projection,trail)
        require(result['source_parameter']==0 and result['source_dtype']==source['dtype'] and result['shape']==source['shape'],'HLO_CALL_BITCAST_BINDING')
        result=dict(result,call_line=node['line'],callee=node['callee'],call_input=node['deps'][0],entry_output=name)
        if projection is None:require(node['dtype']==result['target_dtype'] and node['shape']==result['shape'],'HLO_CALL_RESULT_TYPE')
        return result
    require(op=='bitcast-convert' and projection is None and len(node['deps'])==1,'HLO_OUTPUT_BITCAST_REQUIRED')
    source=comp['nodes'][node['deps'][0]]
    require((source['dtype'],node['dtype']) in (('f32','s32'),('s32','f32')) and source['shape']==node['shape'],'HLO_BITCAST_TYPE')
    return {'source_dtype':source['dtype'],'target_dtype':node['dtype'],'shape':node['shape'],'line':node['line'],
            'source_parameter':source.get('parameter'),'computation':comp_name,'entry_output':name,'root_reachable':True}

def witnesses(text,order,expected):
    comps,entry=parse(text);nodes=output_nodes(comps,entry,order);result={}
    require(set(nodes)==set(expected),'HLO_OUTPUT_MATRIX')
    for key,name in nodes.items():
        witness=resolve(comps,entry,name);dtype,shape=expected[key]
        require(witness['target_dtype']==dtype and witness['shape']==shape,'HLO_OUTPUT_TYPE');result[key]=witness
    return result

def computation(text,source_dtype,target_dtype,shape):
    comps,entry=parse(text);comp=comps[entry];result=resolve(comps,entry,comp['root'])
    require(result['source_dtype']==source_dtype and result['target_dtype']==target_dtype and result['shape']==shape and result['source_parameter']==0,'HLO_BUILDER_COMPUTATION')
    return result

def parameters(comps,entry,name,trail=()):
    require(name not in trail,'HLO_CYCLE');node=comps[entry]['nodes'][name]
    if node['opcode']=='parameter':return {node['parameter']}
    result=set()
    for dep in node['deps']:result.update(parameters(comps,entry,dep,trail+(name,)))
    return result

def input_lineage(comps,entry,name):
    """Only parameter or source-preserving static slice/reshape/copy is valid."""
    nodes=comps[entry]['nodes'];node=nodes[name];path=[];seen=set()
    while node['opcode']!='parameter':
        require(name not in seen and node['opcode'] in ('copy','reshape','slice') and len(node['deps'])==1,'HLO_INPUT_LINEAGE');seen.add(name)
        source=nodes[node['deps'][0]];require(node['dtype']==source['dtype'],'HLO_INPUT_DTYPE')
        if node['opcode']=='reshape':require(__import__('math').prod(node['shape'])==__import__('math').prod(source['shape']),'HLO_INPUT_RESHAPE')
        elif node['opcode']=='copy':require(node['shape']==source['shape'],'HLO_INPUT_COPY')
        else:
            interval=re.search(r'\bslice\s*=\s*\{\s*\[0:(\d+)(?::1)?\]\s*\}',node['body'])
            require(interval is not None and len(source['shape'])==len(node['shape'])==1 and int(interval.group(1))==node['shape'][0]<=source['shape'][0],'HLO_INPUT_SLICE')
        path.append(node['line']);name=node['deps'][0];node=source
    return {'parameter':node['parameter'],'dtype':node['dtype'],'shape':node['shape'],'line':node['line'],'aliases':path}

def bound_witnesses(text,order,expected,input_outputs,input_parameters):
    comps,entry=parse(text);roots=output_nodes(comps,entry,order);observed=witnesses(text,order,expected)
    params=[n for n in comps[entry]['nodes'].values() if n['opcode']=='parameter']
    require(len(params)==len(input_parameters) and len({n['parameter'] for n in params})==len(params),'HLO_PARAMETER_MATRIX')
    require({n['parameter'] for n in params}==set(input_parameters),'HLO_PARAMETER_IDS')
    for n in params:
        dtype,shape=input_parameters[n['parameter']];require(n['dtype']==dtype and n['shape']==shape,'HLO_PARAMETER_TYPE')
    for key,witness in observed.items():
        name=roots[key];node=comps[entry]['nodes'][name]
        # Find the caller input of a call or the source of a direct bitcast.
        if 'call_input' in witness:source=witness['call_input']
        else:
            while node['opcode'] in ('copy','reshape','get-tuple-element'):
                name=node['deps'][0];node=comps[entry]['nodes'][name]
            require(node['opcode']=='bitcast-convert','HLO_INPUT_WITNESS');source=node['deps'][0]
        witness['parameters']=sorted(parameters(comps,entry,source))
        if key in input_outputs:
            lineage=input_lineage(comps,entry,source)
            require(lineage['parameter']==input_outputs[key],'HLO_INPUT_PARAMETER_BINDING');witness['input_lineage']=lineage
    return observed
