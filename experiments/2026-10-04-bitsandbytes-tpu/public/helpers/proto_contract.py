"""Bounded compiler-content analyzer. Stdlib only; never proves loaded executable identity."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def require(v,m):
    if not v:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def varint(b,p):
    value=0
    for i in range(10):
        require(p<len(b),'PROTO_TRUNCATED');v=b[p];p+=1;value|=(v&127)<<(7*i)
        if v<128:return value,p
    raise ValueError('PROTO_VARINT')
def fields(b):
    p=0;out={}
    while p<len(b):
        tag,p=varint(b,p);n,w=tag>>3,tag&7;require(n>0,'PROTO_FIELD')
        if w==0:v,p=varint(b,p)
        elif w==2:
            size,p=varint(b,p);require(size<=len(b)-p,'PROTO_TRUNCATED');v=b[p:p+size];p+=size
        elif w in (1,5):
            size=8 if w==1 else 4;require(size<=len(b)-p,'PROTO_TRUNCATED');v=b[p:p+size];p+=size
        else:raise ValueError('PROTO_WIRE_UNSUPPORTED')
        out.setdefault(n,[]).append((w,v))
    return out
def one(f,n,default=None):
    values=f.get(n,[]);require(len(values)<=1,'PROTO_DUPLICATE_SINGULAR');return values[0][1] if values else default
def ints(f,n):
    out=[]
    for w,v in f.get(n,[]):
        if w==0:out.append(v)
        else:
            require(w==2,'PROTO_PACKED_WIRE');p=0
            while p<len(v):x,p=varint(v,p);out.append(x)
    return out
def shape(b):
    f=fields(b);dtype=one(f,2,0);dims=ints(f,3)
    if dtype==13:return {'tuple':[shape(v) for w,v in f.get(4,[]) if w==2]}
    require(dtype in (6,11,16) and dims and all(x>0 for x in dims),'SHAPE_SCOPE')
    layout=fields(one(f,5,b''));minor=ints(layout,1)
    require(not minor or sorted(minor)==list(range(len(dims))),'SHAPE_LAYOUT')
    return {'dtype':{6:'uint8',11:'float32',16:'bfloat16'}[dtype],'shape':dims,'minor_to_major':minor,'full_shape_sha256':sha(b)}
def module(b,envelope=False):
    outer=fields(b);f=fields(one(outer,1,b'')) if envelope else outer
    require(3 in f,'MODULE_COMPUTATIONS');mid=one(f,5,0);entry=one(f,6,0);computations=[fields(v) for w,v in f[3] if w==2]
    require(len(computations)==1 and one(computations[0],5,0)==entry,'SINGLE_ENTRY_SCOPE')
    c=computations[0];nodes={}
    for w,v in c.get(2,[]):
        require(w==2,'INSTRUCTION_WIRE');n=fields(v);nid=one(n,35,0);require(nid not in nodes,'INSTRUCTION_ID_UNIQUE');nodes[nid]=n
    root=one(c,6,0);require(root in nodes,'ROOT_EXISTS')
    for n in nodes.values():
        require(all(i in nodes for i in ints(n,36)),'OPERAND_IDS')
        require(not ints(n,37) and not ints(n,38),'CONTROL_CALLEE_SCOPE')
        require(one(n,2,b'').decode() in ('parameter','reshape','transpose','copy','tuple','custom-call'),'OPCODE_SCOPE')
    if one(nodes[root],2,b'')==b'tuple':
        deps=ints(nodes[root],36);require(len(deps)==1,'ROOT_TUPLE_SCOPE');root=deps[0]
    reachable=set();visiting=set()
    def visit(i):
        require(i not in visiting,'GRAPH_CYCLE')
        if i in reachable:return
        visiting.add(i)
        for dep in ints(nodes[i],36):visit(dep)
        visiting.remove(i);reachable.add(i)
    visit(root)
    # A singleton output tuple is the only allowed extra container.
    extras=set(nodes)-reachable;require(not extras or (len(extras)==1 and one(nodes[next(iter(extras))],2,b'')==b'tuple'),'DISCONNECTED_GRAPH')
    cc=[i for i in reachable if one(nodes[i],2,b'')==b'custom-call'];require(len(cc)==1,'ONE_REACHABLE_CUSTOM_CALL');call=nodes[cc[0]]
    require(one(call,28,b'')==b'tpu_custom_call','NATIVE_TARGET');require(len(ints(call,36))==3,'THREE_ORDERED_OPERANDS')
    def canonical(i):
        n=nodes[i];keep={str(k):[(w,v.hex() if isinstance(v,bytes) else v) for w,v in values] for k,values in n.items() if k not in (1,7,35,36)}
        keep['operands']=[canonical(dep) for dep in ints(n,36)];return keep
    parameters={}
    for i in reachable:
        n=nodes[i]
        if one(n,2,b'')==b'parameter':
            num=one(n,9,0);require(num not in parameters,'PARAMETER_NUMBER_UNIQUE');parameters[num]=shape(one(n,3,b''))
    return {'id':mid,'nodes':nodes,'root':root,'call':call,'call_id':cc[0],'canonical':canonical(root),'parameters':parameters,'envelope':outer,'shape':shape(one(call,3,b''))}

def textproto(text):
    # Bounded protobuf text format. Comments, extension names, lists, floats,
    # and unknown syntax fail closed. Unused known fields remain parsed.
    pattern=re.compile(r'\s*("(?:\\.|[^"\\])*"|[A-Za-z_][A-Za-z_0-9]*|-?[0-9]+|[{}:])')
    tokens=[];p=0
    while p<len(text):
        if not text[p:].strip():break
        m=pattern.match(text,p);require(m is not None,'METADATA_TEXTPROTO_SCOPE');tokens.append(m[1]);p=m.end()
    pos=0
    def parse(close=False):
        nonlocal pos
        out={}
        while pos<len(tokens):
            key=tokens[pos];pos+=1
            if key=='}':require(close,'METADATA_BALANCE');return out
            require(re.fullmatch('[A-Za-z_][A-Za-z_0-9]*',key),'METADATA_FIELD')
            if tokens[pos]==':':pos+=1
            require(pos<len(tokens),'METADATA_TRUNCATED');v=tokens[pos];pos+=1
            if v=='{':value=parse(True)
            elif v.startswith('"'):value=json.loads(v)
            elif v in ('true','false'):value=v=='true'
            elif re.fullmatch('-?[0-9]+',v):value=int(v)
            else:raise ValueError('METADATA_VALUE_SCOPE')
            out.setdefault(key,[]).append(value)
        require(not close,'METADATA_BALANCE');return out
    return parse()
def scalar(f,key,default=None):
    vals=f.get(key,[]);require(len(vals)<=1,'METADATA_DUPLICATE_SINGULAR');return vals[0] if vals else default

def allocation_summary(after):
    raw=one(after['envelope'],3);require(raw is not None,'BUFFER_ASSIGNMENT_ABSENT');f=fields(raw);logical={};allocs=[];instruction_ids=set(after['nodes'])
    for w,b in f.get(1,[]):
        n=fields(b);i=one(n,1,0);size=one(n,2,0);location=fields(one(n,3,b''));require(i not in logical and size>=0 and one(location,4,0) in instruction_ids,'LOGICAL_BUFFER_BINDING');logical[i]=size
    require(logical,'LOGICAL_BUFFERS_REQUIRED');indexes=set()
    for w,b in f.get(3,[]):
        n=fields(b);index,size=one(n,1,0),one(n,2,0);require(index not in indexes and size>=0,'ALLOCATION_INDEX_SIZE');indexes.add(index);assigned=[]
        for aw,ab in n.get(9,[]):
            a=fields(ab);lid,offset,count=one(a,1,0),one(a,2,0),one(a,3,0)
            require(lid in logical and offset>=0 and count>=0 and offset+count<=size,'ALLOCATION_SLICE_BINDING');assigned.append({'logical_buffer_id':lid,'offset':offset,'bytes':count})
        allocs.append({'index':index,'bytes':size,'assigned':assigned,'parameter':bool(one(n,5,0)),'parameter_number':one(n,6,0) if one(n,5,0) else None,'live_out':bool(one(n,7,0)),'thread_local':bool(one(n,3,0)),'constant':bool(one(n,12,0)),'color':one(n,8,0)})
    require(allocs,'ALLOCATIONS_REQUIRED');return {'allocation_size_sum_bytes':sum(a['bytes'] for a in allocs),'allocations':allocs,'allocator_peak':'UNKNOWN','native_body_internal_memory':'UNKNOWN','meaning':'COMPILER_ALLOCATION_PLAN_NOT_RUNTIME_PEAK'}

def analyze(root,expected):
    root=Path(root);names=('boundary.hlo.pb','before.hlo.pb','after.hlo.pb','metadata.textproto');inventory={name:{'sha256':sha((root/name).read_bytes()),'bytes':(root/name).stat().st_size} for name in names};require(inventory==expected.get('compiler_artifact_inventory'),'INDEPENDENT_COMPILER_ARTIFACT_INVENTORY')
    b=(root/'boundary.hlo.pb').read_bytes();before_b=(root/'before.hlo.pb').read_bytes();after_b=(root/'after.hlo.pb').read_bytes();metadata=(root/'metadata.textproto').read_text()
    require(sha(b)==expected['boundary_proto_sha256'],'EXPECTED_BOUNDARY_PROTO')
    boundary=module(b);before=module(before_b,True);after=module(after_b,True)
    require(boundary['canonical']==before['canonical'],'BEFORE_CONTENT_GRAPH_BINDING')
    payload=expected['payload'].encode();require(one(boundary['call'],43,b'')==one(before['call'],43,b'')==one(after['call'],43,b'')==payload,'EXACT_PAYLOAD_CHAIN')
    params={int(k):v for k,v in expected['parameters'].items()}
    require({i:{'dtype':v['dtype'],'shape':v['shape']} for i,v in boundary['parameters'].items()}==params,'PARAMETER_POSITIONAL_CONTRACT')
    require(before['parameters']==boundary['parameters'],'BEFORE_PARAMETER_BINDING')
    out=after['root']
    while out!=after['call_id']:
        n=after['nodes'][out];deps=ints(n,36);require(one(n,2,b'')==b'copy' and len(deps)==1,'AFTER_OUTPUT_COPY_ONLY_SCOPE');out=deps[0]
    shapes=[shape(one(after['nodes'][i],3,b'')) for i in ints(after['call'],36)]
    require([{'dtype':v['dtype'],'shape':v['shape']} for v in shapes]==expected['ordered_operand_shapes'],'AFTER_ORDERED_OPERAND_SHAPES')
    require({'dtype':after['shape']['dtype'],'shape':after['shape']['shape']}==expected['result'],'AFTER_RESULT')
    meta=textproto(metadata);canonical=scalar(meta,'canonical_module_id');require(canonical==before['id']==after['id'] and not meta.get('partitioned_module_ids') and not scalar(meta,'original_module_id',0),'UNPARTITIONED_PIPELINE_ID_SCOPE')
    stage_names=expected['stage_dump_basenames'];require(set(stage_names)=={'before','after'} and len(set(stage_names.values()))==2,'INDEPENDENT_STAGE_NAMES')
    passes=meta.get('pass_metadata',[]);require(passes,'PASS_METADATA_REQUIRED');ids=[];stage={}
    for row in passes:
        pid=scalar(row,'pass_id',0);require(isinstance(pid,int) and pid not in ids and scalar(row,'module_id',0)==canonical and scalar(row,'pass_name') and scalar(row,'pipeline_name'),'PASS_ID_MODULE_PIPELINE');ids.append(pid)
        for name in row.get('dump_filenames',[]):
            basename=Path(name).name
            if basename in stage_names.values():require(basename not in stage,'DUPLICATE_STAGE_REFERENCE');stage[basename]=pid
    require(set(stage)==set(stage_names.values()) and stage[stage_names['before']]<stage[stage_names['after']],'ORDERED_PIPELINE_STAGE_CONTENT')
    return {'status':'COMPILER_CONTENT_CHAIN_PASS','before_matches_native_graph':True,'payload_sha256':sha(payload),'compiler_module_id':canonical,'stage_pass_ids':stage,'artifacts':{name:sha((root/name).read_bytes()) for name in ('boundary.hlo.pb','before.hlo.pb','after.hlo.pb','metadata.textproto')},'compiler_allocation_plan':allocation_summary(after),'actual_selected_executable_link':'UNKNOWN','actual_execution':'REQUIRES_SEPARATE_REVIEWED_NATIVE_BOUNDARY_AND_OWNERSHIP_VERIFIER','compiler_implementation_revision':'LIBTPU_INTERNAL_REVISION_UNKNOWN','body_semantics':'UNKNOWN','m6':'NOT_QUALIFIED'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifacts',required=True);p.add_argument('--expected',required=True);p.add_argument('--expected-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args();path=Path(a.expected);require(sha(path.read_bytes())==a.expected_sha256,'INDEPENDENT_EXPECTED_SHA');result=analyze(a.artifacts,json.loads(path.read_text()));Path(a.output).write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
