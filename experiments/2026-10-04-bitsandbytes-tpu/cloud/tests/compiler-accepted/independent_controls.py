"""Retained false accepts and alternative artifact layouts, independently encoded."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent/'compiler-native'))
import argparse, copy, hashlib, json, shutil
from pathlib import Path
import analyzer as A
import wire_codec as W

HERE=Path(__file__).resolve().parent
OLD=HERE/'fixtures/compiler'
OUT=HERE/'independent-controls'

def file_edit(root,name,path,operation):
    p=root/name;p.write_bytes(W.edit(p.read_bytes(),path,operation))

def row(items,field):return next(item for item in items if item[0]==field)
def set_field(field,kind,value):return lambda items:W.replace(items,field,kind,value)

def seal(root,expected):
    e=copy.deepcopy(expected);e['compiler_artifact_inventory']={name:{'sha256':hashlib.sha256((root/name).read_bytes()).hexdigest(),'bytes':(root/name).stat().st_size} for name in ('boundary.hlo.pb','before.hlo.pb','after.hlo.pb','metadata.textproto')};e['boundary_proto_sha256']=e['compiler_artifact_inventory']['boundary.hlo.pb']['sha256'];(root/'expected.json').write_text(json.dumps(e,indent=2)+'\n');return e

def after(root,path,operation):file_edit(root,'after.hlo.pb',path,operation)

def input_copy(root):
    def add(items):
        first=W.decode(row(items,2)[2]);shape=row(first,3)[2]
        node=W.encode([[1,2,b'input-copy'],[2,2,b'copy'],[3,2,shape],[35,0,6],[36,2,W.encode_number(1)]])
        items.append([2,2,node]);return items
    after(root,[(1,0),(3,0)],add)
    after(root,[(1,0),(3,0),(2,3)],set_field(36,2,b''.join(W.encode_number(n) for n in (6,2,3))))

def tuple_container(root):
    def add(items):
        copy_node=W.decode([item for item in items if item[0]==2][4][2]);child_shape=row(copy_node,3)[2]
        tuple_shape=W.encode([[2,0,13],[4,2,child_shape]])
        node=W.encode([[1,2,b'tuple-wrapper'],[2,2,b'tuple'],[3,2,tuple_shape],[35,0,6],[36,2,W.encode_number(5)]])
        items.append([2,2,node]);return W.replace(items,6,0,6)
    after(root,[(1,0),(3,0)],add)

def arbitrary_ids(root):
    mapping={1:111,2:17,3:803,4:222,5:91}
    def instructions(items):
        for item in items:
            if item[0]==2:
                n=W.decode(item[2]);n=W.replace(n,35,0,mapping[row(n,35)[2]])
                if any(r[0]==36 for r in n):
                    source=row(n,36)[2];pos=0;deps=[]
                    while pos<len(source):value,pos=W.number(source,pos);deps.append(mapping[value])
                    n=W.replace(n,36,2,b''.join(W.encode_number(value) for value in deps))
                item[2]=W.encode(n)
        return W.replace(items,6,0,mapping[row(items,6)[2]])
    for name,path in [('boundary.hlo.pb',[(3,0)]),('before.hlo.pb',[(1,0),(3,0)]),('after.hlo.pb',[(1,0),(3,0)])]:file_edit(root,name,path,instructions)
    after(root,[(3,0),(1,0),(3,0)],set_field(4,0,222))
    after(root,[(3,0),(1,0)],set_field(1,0,510))
    after(root,[(3,0),(3,0),(9,0)],set_field(1,0,510))
    for name,path in [('boundary.hlo.pb',[]),('before.hlo.pb',[(1,0)]),('after.hlo.pb',[(1,0)])]:file_edit(root,name,path,set_field(5,0,901))
    p=root/'metadata.textproto';p.write_text(p.read_text().replace('module_id: 7','module_id: 901').replace('module_0007','module_0901'))

def source_aliases(root):
    # Explicit custom-call output/operand alias 0, with equal input/result shapes.
    # This is a structural schema fixture; no Mosaic runtime or dataflow proof is asserted.
    for name,prefix in [('boundary.hlo.pb',[]),('before.hlo.pb',[(1,0)]),('after.hlo.pb',[(1,0)])]:
        raw=(root/name).read_bytes()
        module=W.decode(raw) if not prefix else W.decode(row(W.decode(raw),1)[2]);comp=W.decode(row(module,3)[2]);nodes=[W.decode(item[2]) for item in comp if item[0]==2];shape=row(nodes[3],3)[2]
        file_edit(root,name,prefix+[(3,0),(2,0)],set_field(3,2,shape))
        file_edit(root,name,prefix+[(3,0),(2,3)],lambda items:items+[[74,2,W.encode([[2,0,0]])]])
    def buffers(items):
        logical1=W.encode([[1,0,41],[2,0,100],[3,2,W.encode([[4,0,1]])]])
        logical2=W.encode([[1,0,75],[2,0,100],[3,2,W.encode([[4,0,4]])]])
        alias1=W.encode([[1,0,41],[2,2,W.encode([[4,0,4]])]])
        alias2=W.encode([[1,0,75],[2,2,W.encode([[4,0,1]])]])
        assigned1=W.encode([[1,0,41],[2,0,0],[3,0,100]]);assigned2=W.encode([[1,0,75],[2,0,0],[3,0,100]])
        allocation=W.encode([[1,0,0],[2,0,100],[9,2,assigned1],[9,2,assigned2]])
        return [[1,2,logical1],[1,2,logical2],[2,2,alias1],[2,2,alias2],[3,2,allocation]]
    after(root,[(3,0)],buffers)

def run():
    OUT.mkdir(exist_ok=False);base=json.loads((OLD/'expected.json').read_text());rows=[]
    def check(name,change,reason=None,adjust=None):
        root=OUT/name;shutil.copytree(OLD,root);e=copy.deepcopy(base);change(root)
        if adjust:adjust(e)
        e=seal(root,e)
        try:report=A.analyze(root,e)
        except ValueError as error:
            assert reason==str(error),(name,reason,str(error));report={'status':'REJECTED','error':str(error)}
        else:
            assert reason is None,(name,'FALSE_ACCEPT',reason)
            assert report['actual_selected_executable_link']=='UNKNOWN' and report['compiler_allocation_plan']['allocator_peak']=='UNKNOWN'
        (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');rows.append({'case':name,'status':'PASS','observed':report['status'],'rejected':reason})
    check('retained-correct',lambda r:None)
    check('arbitrary-instruction-logical-and-module-ids',arbitrary_ids,adjust=lambda e:e.__setitem__('stage_dump_basenames',{k:v.replace('module_0007','module_0901') for k,v in e['stage_dump_basenames'].items()}))
    check('valid-input-copy',input_copy)
    check('valid-tuple-subshape-location',lambda r:(tuple_container(r),after(r,[(3,0),(1,0),(3,0)],lambda items:W.replace(W.replace(items,4,0,6),3,2,W.encode_number(0)))))
    def alias_expected(e):
        e['parameters']['0']['shape']=[128,256];e['ordered_operand_shapes'][0]['shape']=[128,256]
    check('explicit-source-aliases-distinct-values-shared-slice',source_aliases,adjust=alias_expected)
    faults=[
        ('retained-after-parameter-number-99',[(1,0),(3,0),(2,0)],set_field(9,0,99),'AFTER_PARAMETER_INTERFACE'),
        ('retained-array-shape-index-99',[(3,0),(1,0),(3,0)],set_field(3,2,W.encode_number(99)),'LOGICAL_BUFFER_SHAPE_INDEX'),
        ('retained-duplicate-logical-assignment',[(3,0),(3,0)],lambda items:items+[copy.deepcopy(row(items,9))],'LOGICAL_BUFFER_ASSIGNMENT_UNIQUE'),
        ('after-parameter-numbers-swapped',[(1,0),(3,0)],lambda items:[item if item[0]!=2 else [2,2,W.edit(item[2],[],lambda n:W.replace(n,9,0,1-row(n,9)[2]) if row(n,2)[2]==b'parameter' and row(n,9)[2]<2 else n)] for item in items],'AFTER_PARAMETER_INTERFACE'),
        ('unassigned-logical-buffer',[(3,0),(3,0)],lambda items:[item for item in items if item[0]!=9],'ALL_LOGICAL_BUFFERS_ASSIGNED'),
        ('different-native-call-alias-contract',[(1,0),(3,0),(2,3)],lambda items:items+[[74,2,W.encode([[2,0,0]])]],'AFTER_NATIVE_CALL_SEMANTICS'),
        ('duplicate-logical-buffer-location',[(3,0)],lambda items:items+[[1,2,W.edit(row(items,1)[2],[],set_field(1,0,92))]],'LOGICAL_BUFFER_LOCATION_UNIQUE'),
        ('output-copy-wrong-shape',[(1,0),(3,0),(2,4),(3,0)],set_field(3,2,W.encode_number(129)+W.encode_number(256)),'AFTER_OUTPUT_COPY_SHAPE'),
    ]
    for name,path,operation,reason in faults:check(name,lambda r,path=path,operation=operation:after(r,path,operation),reason)
    check('tuple-index-out-of-bounds',lambda r:(tuple_container(r),after(r,[(3,0),(1,0),(3,0)],lambda items:W.replace(W.replace(items,4,0,6),3,2,W.encode_number(1)))),'LOGICAL_BUFFER_SHAPE_INDEX')
    check('tuple-index-continues-through-array',lambda r:(tuple_container(r),after(r,[(3,0),(1,0),(3,0)],lambda items:W.replace(W.replace(items,4,0,6),3,2,W.encode_number(0)+W.encode_number(0)))),'LOGICAL_BUFFER_SHAPE_INDEX')
    check('alias-unknown-source-id',lambda r:after(r,[(3,0)],lambda items:items+[[2,2,W.encode([[1,0,99],[2,2,W.encode([[4,0,4]])]])]]),'BUFFER_ALIAS_SOURCE')
    check('alias-different-shape',lambda r:after(r,[(3,0)],lambda items:items+[[2,2,W.encode([[1,0,1],[2,2,W.encode([[4,0,1]])]])]]),'BUFFER_ALIAS_LOCATION')
    # A separate nested ShapeIndex test avoids widening the graph's singleton tuple scope.
    array=W.encode([[2,0,11],[3,2,W.encode_number(2)+W.encode_number(3)]]);inner=W.encode([[2,0,13],[4,2,array]]);outer=W.encode([[2,0,13],[4,2,inner]])
    graph={'nodes':{808:{3:[(2,outer)]}}};location=W.encode([[4,0,808],[3,2,b'\x00\x00']]);resolved=A.resolve_location(graph,location);assert resolved['shape']=={'dtype':'float32','shape':[2,3]}
    rows.append({'case':'valid-nested-tuple-subshape-traversal','status':'PASS','observed':'STRUCTURAL_LOCATION_ONLY'})
    report={'status':'PASS','scope':'SYNTHETIC_SOURCE_SCHEMA_AND_RETAINED_ARTIFACTS_NO_COMPILER_DEVICE','count':len(rows),'controls':rows,'selected_executable_link':'UNKNOWN','allocator_peak':'UNKNOWN','alias_semantics':'BACKEND_DATAFLOW_UNKNOWN'};(OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'PASS','count':len(rows)}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args();OUT=args.output.resolve();run()
