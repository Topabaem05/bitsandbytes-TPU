"""Replay the frozen17 scope controls with freshly generated primary references."""
import importlib.util,json,struct,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'))
import admission as A,training as T,scope_rational_reference as R
C=T.setup(local=True);C.refs=T.references(C);torch=C.torch

def package(name,path):
 p=types.ModuleType(name);p.__path__=[str(path)];sys.modules[name]=p;s=importlib.util.spec_from_file_location(name+'.exact',path/'exact.py');m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
E=package('m5_scope_candidate',ROOT/'integration/plugin/src/bitsandbytes_tpu');old=package('m4_frozen_scope',ROOT/'tests/frozen-m4-plugin')
A.require(A.sha(ROOT/'tests/frozen-m4-plugin/exact.py')=='e5064072609753d3a5f1e1920b46db48ccce5751402fe7a5ba5c5b7c0ddf2831','FROZEN_M4_GUARD_SOURCE')
def word(x):return R.A.signed(struct.unpack('<I',struct.pack('<f',x))[0])
def tensor(v):return torch.tensor(v,dtype=torch.int32)
def words(v):return v.reshape(-1).view(torch.int32).tolist()
rows=[];dynamic=words(C.refs[0]['dynamic'])
def record(name,**kw):rows.append({'name':name,'status':'PASS',**kw})
def reject(name,fn):
 try:fn()
 except ValueError as e:record(name,rejected=str(e));return
 raise AssertionError('INCORRECT_ACCEPTANCE:'+name)
def check(name,first):
 r=R.nested(first,dynamic);codes,offset,second=E.compress_words(tensor(first),tensor(dynamic))
 A.require(codes.tolist()==r['codes']and offset.item()==r['offset']and second.tolist()==r['second']and E.S.cpu_sum8_bits(tensor(first),E.A.add_bits).item()==r['sum']and E.A.subtract_bits(tensor(first),offset).tolist()==r['centered']and E.restore_words(codes,second,tensor(dynamic),offset).tolist()==r['restored'],'EXACT1024:'+name)
 record(name,first_count=1024,second_count=4,reference_sha256=T.M.digest(r))
for i,ref in enumerate(C.refs):check('fixed_layer'+str(i),words(ref['first']))
for name,first in [('zeros',[0]*1024),('positive_ramp',[word((i%17)/256)for i in range(1024)]),('block_boundaries',[word(1 if i in(255,256,511,512,767,768,1023)else 0)for i in range(1024)]),('subnormal_first',[(i%3)+1 for i in range(1024)]),('order_sensitive',[word(2**24)]+[word(1)]*1023)]:check(name,first)
order=[word(2**24)]+[word(1)]*1023;A.require(R.serial_mean(order)!=R.mean_center(order)[1],'INDEPENDENT_ORDER_NEGATIVE');record('incorrect_serial_reduction_detected')
wrong=list(R.nested(order,dynamic)['codes']);wrong[256]=(wrong[256]+1)%256;A.require(wrong!=E.compress_words(tensor(order),tensor(dynamic))[0].tolist(),'CHANGED_CODE_NEGATIVE');record('incorrect_nested_code_detected')
reject('frozen_m4_still_rejects1024',lambda:old.compress_words(tensor(order),tensor(dynamic)))
for count in [0,514,1023,1025]:reject('unsupported_count'+str(count),lambda count=count:E.compress_words(torch.zeros(count,dtype=torch.int32),tensor(dynamic)))
reject('rank2_rejected',lambda:E.compress_words(torch.zeros((32,32),dtype=torch.int32),tensor(dynamic)));reject('wrong_dtype_rejected',lambda:E.compress_words(torch.zeros(1024),tensor(dynamic)))
small=tensor([word((i%7)/256)for i in range(513)]);A.require(all(torch.equal(x,y)for x,y in zip(E.compress_words(small,tensor(dynamic)),old.compress_words(small,tensor(dynamic)))),'OLD513_BYTE_EQUAL');record('existing513_unchanged')
A.require(len(rows)==17,'FIXED17_CONTROLS');out=Path(sys.argv[1]);A.write(out,{'status':'PASS','count':17,'controls':rows,'scope':'LOCAL_UNQUALIFIED_ONLY','fresh_primary_reference':True,'historical_checkpoint_used':False,'m5_status':'NOT_QUALIFIED','provider_calls':0});print(json.dumps({'status':'PASS','controls':17}))
