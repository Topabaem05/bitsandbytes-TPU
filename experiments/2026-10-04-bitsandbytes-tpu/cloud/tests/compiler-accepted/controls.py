"""Real wire-format synthetic artifacts. No compiler/device result is synthesized as actual."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent/'compiler-native'))
import argparse
import copy
import json
from pathlib import Path
import shutil
import analyzer as A


def v(x):
    b=bytearray()
    while x>=128:b.append((x&127)|128);x>>=7
    return bytes(b)+bytes([x])
def f(n,x):return v(n<<3)+v(x) if isinstance(x,int) else v((n<<3)|2)+v(len(x))+x
def packed(n,x):return f(n,b''.join(v(i) for i in x))
def shape(dtype,dims):return f(2,dtype)+packed(3,dims)+f(5,packed(1,list(reversed(range(len(dims))))))

def fixture(root,*,payload=b'{"body":"fixture-only"}',operand_order=(1,2,3),root_id=4,mid=7,allocation_size=100,slice_offset=0,logical_instruction=4,after_copy=False,rename=False):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    shapes=[shape(11,[128,512]),shape(6,[2,256,128]),shape(11,[2,256,4])];out=shape(11,[128,256]);nodes=[]
    for i,s in enumerate(shapes,1):nodes.append(f(1,('renamed' if rename else 'p').encode()+str(i).encode())+f(2,b'parameter')+f(3,s)+f(9,i-1)+f(35,i))
    call=f(1,b'cc')+f(2,b'custom-call')+f(3,out)+f(28,b'tpu_custom_call')+f(35,4)+packed(36,operand_order)+f(43,payload)
    nodes.append(call)
    if after_copy:nodes.append(f(1,b'legal-copy')+f(2,b'copy')+f(3,out)+f(35,5)+packed(36,[4]));root_id=5
    comp=f(1,b'entry')+b''.join(f(2,n) for n in nodes)+f(5,20)+f(6,root_id)
    module=f(1,b'arbitrary-module-name')+f(3,comp)+f(5,mid)+f(6,20)
    location=f(4,logical_instruction);logical=f(1,1)+f(2,100)+f(3,location);assigned=f(1,1)+f(2,slice_offset)+f(3,100);alloc=f(1,0)+f(2,allocation_size)+f(9,assigned);ba=f(1,logical)+f(3,alloc)
    return module,f(1,module)+f(3,ba)

def make(root):
    root=Path(root);boundary,_=fixture(root);before,_=fixture(root,rename=True);_,after=fixture(root,after_copy=True)
    (root/'boundary.hlo.pb').write_bytes(boundary);(root/'before.hlo.pb').write_bytes(f(1,before));(root/'after.hlo.pb').write_bytes(after)
    (root/'metadata.textproto').write_text('canonical_module_id: 7\npass_metadata { pass_id: 1 pass_name: "start" pipeline_name: "fixture-pipeline" module_id: 7 dump_filenames: "module_0007.before_optimizations.hlo.pb" }\npass_metadata { pass_id: 2 pass_name: "end" pipeline_name: "fixture-pipeline" module_id: 7 dump_filenames: "module_0007.after_optimizations.hlo.pb" }\n')
    e={'boundary_proto_sha256':A.sha(boundary),'payload':'{"body":"fixture-only"}','parameters':{'0':{'dtype':'float32','shape':[128,512]},'1':{'dtype':'uint8','shape':[2,256,128]},'2':{'dtype':'float32','shape':[2,256,4]}},'ordered_operand_shapes':[{'dtype':'float32','shape':[128,512]},{'dtype':'uint8','shape':[2,256,128]},{'dtype':'float32','shape':[2,256,4]}],'result':{'dtype':'float32','shape':[128,256]},'stage_dump_basenames':{'before':'module_0007.before_optimizations.hlo.pb','after':'module_0007.after_optimizations.hlo.pb'}}
    e['compiler_artifact_inventory']={name:{'sha256':A.sha((root/name).read_bytes()),'bytes':(root/name).stat().st_size} for name in ('boundary.hlo.pb','before.hlo.pb','after.hlo.pb','metadata.textproto')}
    (root/'expected.json').write_text(json.dumps(e,indent=2)+'\n');return e

def run(output):
    output=Path(output);output.mkdir(exist_ok=False);good=output/'good';expected=make(good);result=A.analyze(good,expected);assert result['status']=='COMPILER_CONTENT_CHAIN_PASS' and result['actual_selected_executable_link']=='UNKNOWN' and result['compiler_allocation_plan']['allocation_size_sum_bytes']==100
    (good/'report.json').write_text(json.dumps(result,indent=2)+'\n');rows=[{'name':'correct-renamed-before-legal-after-copy-content-chain','status':'PASS'}]
    def reject(name,fn,reason):
        root=output/name;shutil.copytree(good,root);e=copy.deepcopy(expected);fn(root,e)
        if name!='wrong-artifact-inventory':e['compiler_artifact_inventory']={item:{'sha256':A.sha((root/item).read_bytes()),'bytes':(root/item).stat().st_size} for item in ('boundary.hlo.pb','before.hlo.pb','after.hlo.pb','metadata.textproto')}
        try:A.analyze(root,e);raise AssertionError('FALSE_ACCEPT:'+name)
        except ValueError as error:assert str(error)==reason,(name,str(error),reason);rows.append({'name':name,'status':'PASS','rejected':reason})
    def before(root,**kw):m,_=fixture(root,**kw);(root/'before.hlo.pb').write_bytes(f(1,m))
    def after(root,**kw):_,p=fixture(root,**kw);(root/'after.hlo.pb').write_bytes(p)
    reject('same-name-different-before-payload',lambda r,e:before(r,payload=b'wrong'),'BEFORE_CONTENT_GRAPH_BINDING')
    reject('same-module-id-different-before-operands',lambda r,e:before(r,operand_order=(1,3,2)),'BEFORE_CONTENT_GRAPH_BINDING')
    reject('after-wrong-payload',lambda r,e:after(r,payload=b'wrong'),'EXACT_PAYLOAD_CHAIN')
    reject('after-wrong-operand-order',lambda r,e:after(r,operand_order=(1,3,2)),'AFTER_OPERAND_PARAMETER_LINEAGE')
    reject('disconnected-native-call',lambda r,e:after(r,root_id=1),'DISCONNECTED_GRAPH')
    reject('different-module-id',lambda r,e:after(r,mid=8),'UNPARTITIONED_PIPELINE_ID_SCOPE')
    reject('wrong-fixed-boundary-hash',lambda r,e:e.__setitem__('boundary_proto_sha256','b'*64),'EXPECTED_BOUNDARY_PROTO')
    def meta(root,old,new):p=root/'metadata.textproto';p.write_text(p.read_text().replace(old,new))
    reject('metadata-name-only-no-pass-chain',lambda r,e:(r/'metadata.textproto').write_text('canonical_module_id: 7\n'),'PASS_METADATA_REQUIRED')
    reject('metadata-wrong-pass-module-id',lambda r,e:meta(r,'module_id: 7 dump','module_id: 8 dump'),'PASS_ID_MODULE_PIPELINE')
    reject('metadata-reversed-passes',lambda r,e:meta(r,'pass_id: 1','pass_id: 3'),'ORDERED_PIPELINE_STAGE_CONTENT')
    reject('metadata-wrong-stage-name',lambda r,e:meta(r,'after_optimizations','different_optimizations'),'ORDERED_PIPELINE_STAGE_CONTENT')
    reject('metadata-partition-not-supported',lambda r,e:(r/'metadata.textproto').write_text((r/'metadata.textproto').read_text()+'partitioned_module_ids: 8\n'),'UNPARTITIONED_PIPELINE_ID_SCOPE')
    reject('allocation-missing',lambda r,e:(r/'after.hlo.pb').write_bytes(f(1,fixture(r)[0])),'BUFFER_ASSIGNMENT_ABSENT')
    reject('allocation-out-of-bounds',lambda r,e:after(r,allocation_size=90),'ALLOCATION_SLICE_BINDING')
    reject('allocation-wrong-instruction',lambda r,e:after(r,logical_instruction=999),'LOGICAL_BUFFER_BINDING')
    reject('wrong-artifact-inventory',lambda r,e:e['compiler_artifact_inventory']['after.hlo.pb'].__setitem__('sha256','b'*64),'INDEPENDENT_COMPILER_ARTIFACT_INVENTORY')
    reject('proto-truncated',lambda r,e:(r/'before.hlo.pb').write_bytes(b'\x0a\x7f'),'PROTO_TRUNCATED')
    report={'status':'PASS','scope':'SYNTHETIC_WIRE_ARTIFACTS_NO_COMPILER_OR_TPU','count':len(rows),'controls':rows,'selected_executable_link':'UNKNOWN','allocator_peak':'UNKNOWN'};(output/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'PASS','count':len(rows)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(Path(a.output))
