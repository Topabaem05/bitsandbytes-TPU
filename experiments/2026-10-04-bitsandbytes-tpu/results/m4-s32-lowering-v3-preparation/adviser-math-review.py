import ast,copy,hashlib,importlib.util,json,math,operator,stat,sys,time,types,zipfile
from pathlib import Path
ROOT=Path('/tmp/bnb-tpu-m4-s32-lowering-v3-20261011-oj7mwf5m')
HERE=Path(__file__).resolve().parent
BASE=ROOT/'bundle/experiments/2026-10-04-bitsandbytes-tpu/nested-exact'
PLUG=BASE/'integration/plugin/src/bitsandbytes_tpu'
CANON=Path('/Users/topamini/code/bitsandbytes-TPU')
checks=[]; proof={}
def sha(b):return hashlib.sha256(b).hexdigest()
def check(name,fn):
    value=fn(); checks.append({'name':name,'status':'PASS'}); return value
def must_reject(fn,error=(ValueError,AssertionError)):
    try:fn()
    except error:return
    raise AssertionError('WRONG_CASE_ACCEPTED')
def archives():
    seal=json.loads((ROOT/'final-seal.json').read_text());raw=(ROOT/'source.zip').read_bytes()
    assert sha(raw)==seal['source_archive_sha256']=='e8837609208c3610f2140b4cee6a083d923d8aa61d9ca01bbab0b4b18b885048'
    closure=json.loads((ROOT/'source-closure.json').read_text()); assert sha((ROOT/'source-closure.json').read_bytes())==seal['source_closure_sha256']
    with zipfile.ZipFile(ROOT/'source.zip') as z:
        names=z.namelist(); assert len(names)==len(set(names))==215
        assert set(names)==set(closure['files'])|{'source-closure.json'}
        total=0
        for n in names:
            p=Path(n); assert not p.is_absolute() and '..' not in p.parts and not (ROOT/n).is_symlink()
            b=z.read(n); assert b==(ROOT/n).read_bytes(); total+=len(b)
            assert not stat.S_ISLNK(z.getinfo(n).external_attr>>16)
            if n!='source-closure.json':assert closure['files'][n]=={'bytes':len(b),'sha256':sha(b)}
        assert total==seal['source_raw_bytes']
    proof['archive']={'members':len(names),'raw_bytes':total,'sha256':sha(raw)}
check('archive_every_member_full_bytes_and_closed_inventory',archives)
def images():
    m=json.loads((ROOT/'replacement-map.json').read_text()); assert len(m['files'])==173
    physical={p.relative_to(ROOT/'bundle').as_posix() for p in (ROOT/'bundle').rglob('*') if p.is_file()}
    assert physical==set(m['files']); actions={};changed=[]
    for n,r in m['files'].items():
        b=(ROOT/'bundle'/n).read_bytes(); assert r['after']=={'bytes':len(b),'sha256':sha(b)}
        if r['before'] is None:assert not (CANON/n).exists() and r['action']=='CREATE'
        else:
            b=(CANON/n).read_bytes(); assert r['before']=={'bytes':len(b),'sha256':sha(b)},'PREIMAGE:'+n
        assert (r['action']=='KEEP')==(r['before']==r['after']);actions[r['action']]=actions.get(r['action'],0)+1
        if r['action']!='KEEP':changed.append(n)
    assert len(changed)==27
    proof['replacement']={'files':173,'actions':actions,'changed':changed,'canonical_preimages_verified':True}
check('173_source_afterimages_and_live_canonical_preimages',images)
def science():
    new=json.loads((BASE/'source/science-manifest.json').read_text());old=json.loads((BASE/'history/functional-output-v2-20261011/science-manifest.json').read_text())
    assert len(new['files'])==len(old['files'])==19 and set(new['files'])==set(old['files'])
    changed=[]
    for n,d in new['files'].items():
        b=(BASE/'source'/n).read_bytes();assert d=={'bytes':len(b),'sha256':sha(b)}
        if d!=old['files'][n]:changed.append(n)
        else:assert b==(CANON/'experiments/2026-10-04-bitsandbytes-tpu/nested-exact/source'/n).read_bytes()
    assert sorted(changed)==['exact_admission.py','generation_gate.py','integration-admission.json']
    proof['science']={'changed_identity_files':changed,'unchanged_scientific_files':16,'sha256':sha((BASE/'source/science-manifest.json').read_bytes()),'strict_gate_sha256':{n:sha((BASE/'source'/n).read_bytes()) for n in ['public_trace.py','record_validation.py','graph_witness.py']}}
    assert new['status']=='NOT_ADOPTED' and new['device_acceptance'] is None
check('science19_only3_identity_changes_and16_exact_criteria',science)
OP={'eq':ast.Eq,'ne':ast.NotEq,'lt':ast.Lt,'le':ast.LtE,'gt':ast.Gt,'ge':ast.GtE}
class UndoCompare(ast.NodeTransformer):
    def __init__(self):self.count=0
    def visit_Call(self,n):
        self.generic_visit(n);f=ast.unparse(n.func)
        if f.startswith('torch.ops.aten.') and f.endswith('.Tensor') and f.split('.')[-2] in OP:
            assert len(n.args)==2 and not n.keywords
            a=n.args[1];assert isinstance(a,ast.Call) and ast.unparse(a.func) in ('scalar_s32','C.scalar_s32') and len(a.args)==2 and not a.keywords
            self.count+=1;return ast.Compare(left=n.args[0],ops=[OP[f.split('.')[-2]]()],comparators=[a.args[1]])
        return n
class UndoShift(ast.NodeTransformer):
    def __init__(self):self.count=0
    def visit_Call(self,n):
        self.generic_visit(n)
        if isinstance(n.func,ast.Name) and n.func.id in ('left_shift','right_shift'):
            assert len(n.args)==2 and not n.keywords; self.count+=1
            return ast.BinOp(left=n.args[0],op=ast.LShift() if n.func.id=='left_shift' else ast.RShift(),right=n.args[1])
        return n
def clean_compare(t):
    t.body=[n for n in t.body if not(isinstance(n,ast.FunctionDef) and n.name=='scalar_s32')]
    for n in t.body:
        if isinstance(n,ast.ImportFrom):n.names=[v for v in n.names if v.name!='scalar_s32']
    u=UndoCompare();return u.visit(t),u.count
def ast_proof(source,name):
    t,c=clean_compare(ast.parse(source));old=ast.parse((ROOT/'test_support/shift-only'/name).read_text())
    assert ast.dump(t,include_attributes=False)==ast.dump(old,include_attributes=False),'COMPARE_PREIMAGE'
    if name=='_integer_clamp.py':t.body=[n for n in t.body if not(isinstance(n,ast.FunctionDef) and n.name in ('_shift_count','left_shift','right_shift'))]
    else:t.body=[n for n in t.body if not(isinstance(n,ast.ImportFrom) and n.module=='_integer_clamp' and [a.name for a in n.names]==['left_shift','right_shift'])]
    u=UndoShift();t=u.visit(t)
    if name=='exact.py':
        count=0
        for n in ast.walk(t):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='amax' and ast.unparse(n.args[0])=='remaining & 2147483647':
                assert len(n.keywords)==1 and n.keywords[0].arg=='dim' and ast.literal_eval(n.keywords[0].value)==0;n.keywords=[];count+=1
        assert count==1
    parent=ast.parse((BASE/'history/functional-output-v2-20261011'/name).read_text())
    assert ast.dump(t,include_attributes=False)==ast.dump(parent,include_attributes=False),'SHIFT_PREIMAGE'
    return {'typed_comparisons':c,'typed_shifts':u.count}
def ast_all():
    proof['ast']={name:ast_proof((PLUG/name).read_text(),name) for name in ['_integer_clamp.py','_integer_addsub.py','_integer_muldiv.py','exact.py']}
    assert sum(v['typed_comparisons'] for v in proof['ast'].values())==79
    assert sum(v['typed_shifts'] for v in proof['ast'].values())==44
check('79_comparison_and44_shift_reverse_AST_exact_preimages',ast_all)
src=(PLUG/'_integer_clamp.py').read_text()
wrong=src.replace('torch.ops.aten.lt.Tensor(words, scalar_s32(words, 0))','torch.ops.aten.gt.Tensor(words, scalar_s32(words, 0))',1);assert wrong!=src
check('wrong_swapped_direction_rejected_by_preimage',lambda:must_reject(lambda:ast_proof(wrong,'_integer_clamp.py')))
wrong=src.replace('return SIGN-word if word<0 else word','return SIGN-word if word>0 else word',1);assert wrong!=src
check('wrong_static_metadata_change_rejected_by_preimage',lambda:must_reject(lambda:ast_proof(wrong,'_integer_clamp.py')))
# Tensor execution is bounded local preparation, not qualified Linux or TPU.
import torch
proof['runtime']={'torch':torch.__version__,'python':sys.version.split()[0],'platform':sys.platform,'qualified':False,'TPU':'NOT_RUN'}
ns={'__name__':'_adviser_clamp'};exec(compile(src,str(PLUG/'_integer_clamp.py'),'exec'),ns);scalar=ns['scalar_s32']
fs={};exec(compile((PLUG/'functionalization.py').read_text(),str(PLUG/'functionalization.py'),'exec'),fs)
vals=[-(1<<31),-123,-1,0,1,123,(1<<31)-1];bounds=[-(1<<31),-1,0,1,(1<<31)-1]
def scalar_cases():
    refs=[torch.tensor(0,dtype=torch.int32),torch.tensor(vals,dtype=torch.int32),torch.arange(12,dtype=torch.int32).reshape(3,4).t(),torch._to_functional_tensor(torch.tensor([1],dtype=torch.int32)),torch.empty(4,dtype=torch.int32,device='meta')]
    for ref in refs:
        for value in bounds:
            y=scalar(ref,value);assert y.dtype==torch.int32 and y.shape==torch.Size([]) and y.device==ref.device
            if y.device.type!='meta':assert y.item()==value
check('typed_scalar_bounds_scalar_vector_noncontiguous_functional_meta',scalar_cases)
for label,ref,value in [('bool',torch.tensor(0,dtype=torch.int32),False),('float',torch.tensor(0,dtype=torch.int32),1.0),('underflow',torch.tensor(0,dtype=torch.int32),-(1<<31)-1),('overflow',torch.tensor(0,dtype=torch.int32),1<<31),('int64_reference',torch.tensor(0,dtype=torch.int64),1),('float_reference',torch.tensor(0,dtype=torch.float32),1),('non_tensor_reference',None,1)]:
    check('reject_'+label,lambda ref=ref,value=value:must_reject(lambda:scalar(ref,value)))
OPS={'eq':operator.eq,'ne':operator.ne,'lt':operator.lt,'le':operator.le,'gt':operator.gt,'ge':operator.ge}
def compare_cases():
    x=torch.tensor(vals,dtype=torch.int32)
    for name,op in OPS.items():
        for value in bounds:
            expected=[op(v,value) for v in vals]
            got=getattr(torch.ops.aten,name).Tensor(x,scalar(x,value));assert got.dtype==torch.bool and got.tolist()==expected
check('six_comparisons_independent_signed_python_integer_oracle',compare_cases)
def tls():return (str(torch._C._dispatch_tls_local_include_set()),str(torch._C._dispatch_tls_local_exclude_set()))
def functional_case():
    before=tls();x=torch.tensor([-2,-1,0],dtype=torch.int32)
    def body(a,b):
        assert a is b and torch._is_functional_tensor(a);a.add_(1)
        return torch.ops.aten.lt.Tensor(a,scalar(a,0))
    f=fs['functionalized_kernel'](body);out=f(x,x)
    assert x.tolist()==[-1,0,1] and out.tolist()==[True,False,False] and not torch._is_functional_tensor(out)
    assert tls()==before
    must_reject(lambda:f(torch._to_functional_tensor(x),x),error=(RuntimeError,))
    assert tls()==before
check('native_functional_pending_update_alias_and_TLS_preserved',functional_case)
def shift_cases():
    x=torch.tensor(vals,dtype=torch.int32).unsqueeze(1);counts=[0,1,23,30,31];c=torch.tensor(counts,dtype=torch.int32)
    right=ns['right_shift'](x,c).tolist();left=ns['left_shift'](x,c).tolist()
    s32=lambda n:((n+(1<<31))%(1<<32))-(1<<31)
    assert right==[[v>>n for n in counts] for v in vals]
    assert left==[[s32(v<<n) for n in counts] for v in vals]
check('S32_tensor_shifts_independent_signed_word_oracle',shift_cases)
def mutant_constructor(replacement):
    key='return torch.tensor(value, dtype=torch.int32, device=reference.device)';assert src.count(key)==1
    bad={'__name__':'_adviser_mutant'};exec(compile(src.replace(key,replacement),'<memory-mutant>','exec'),bad)
    ref=torch.tensor(0,dtype=torch.int32);out=bad['scalar_s32'](ref,0)
    assert out.dtype==torch.int32 and out.shape==torch.Size([]) and out.device==ref.device and out.item()==0
for name,body in [('int64','return torch.tensor(value, dtype=torch.int64, device=reference.device)'),('wrong_device',"return torch.tensor(value, dtype=torch.int32, device='meta')"),('wrong_value','return torch.tensor(value+1, dtype=torch.int32, device=reference.device)'),('vector_shape','return torch.tensor([value], dtype=torch.int32, device=reference.device)')]:
    check('detect_wrong_constructor_'+name,lambda body=body:must_reject(lambda:mutant_constructor(body)))
proof.update(kind='ASTRA_M4_S32_SOURCE_INDEPENDENT_REVIEW',status='PASS',checks=checks,check_count=len(checks),source_variant='nested-exact-v3-s32-tensor-lowering',scope='SOURCE_PREPARATION_ONLY',m4_status='NOT_QUALIFIED',provider_calls=0)
(HERE/'review-result.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({'status':'PASS','controls':len(checks),'scope':proof['scope']}))
