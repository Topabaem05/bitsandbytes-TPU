"""Offline actual-graph replay and in-memory mutation controls; no device execution."""
import argparse,copy,hashlib,json,sys
from pathlib import Path


def run(source,graphs,upstream,fixtures,output):
 sys.path.insert(0,str(source.resolve()));import public_graph as G,oracle_math as M,reference_observer_v3 as R
 PC=G.PC;rows=[];case=next(c for c in G.C.read(source/'public-protocol.json')['cases']if c['id']=='public-train-rank2');decode,code=M.decode_original(upstream)
 allowed=G.allowed_reference_values(M.allowed_reference_values(case,decode,code,{}),case);fixed=M.input_record(case);prefix=graphs/'public-train-rank2-dX'
 context=prefix.with_suffix('.context.pb').read_bytes();printer=prefix.with_suffix('.printer.pb').read_bytes();parameters=G.C.read(prefix.with_suffix('.parameters.json'))
 before={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in graphs.iterdir()if p.is_file()}
 def test(name,c=context,p=printer,params=parameters,expected=None,shape=(128,512),dtype='float32',allowed_values=allowed,inputs=fixed):
  try:proof=G.verify_reference(c,p,list(shape),dtype,parameters=params,allowed_parameters=allowed_values,reference_case=case,fixed_inputs=inputs)
  except ValueError as error:
   assert expected is not None and any(code in str(error) for code in (expected if isinstance(expected,tuple) else (expected,))),(name,str(error),expected)
   rows.append({'case':name,'status':'PASS','rejected':str(error)});return
  assert expected is None,name
  assert proof['reference_observer_version']==3 and proof['record_validation']=='PASS'
  rows.append({'case':name,'status':'PASS','scalar_roles':[r['role']for r in proof['scalar_role_bindings']]});return proof
 def varint(value):
  out=bytearray()
  while value>127:out.append((value&127)|128);value>>=7
  out.append(value);return bytes(out)
 def encode(fields):
  out=bytearray()
  for number,values in fields.items():
   for wire,value in values:
    out+=varint((number<<3)|wire)
    if wire==0:out+=varint(value)
    elif wire==2:out+=varint(len(value))+value
    elif wire in(1,5):out+=value
    else:raise AssertionError(wire)
  return bytes(out)
 def mutate(raw,edit):
  outer=PC.fields(raw);entry=PC.one(outer,6,0);comps=[]
  for wire,comp_raw in outer[3]:
   comp=PC.fields(comp_raw)
   if PC.one(comp,5,0)==entry:
    nodes=[PC.fields(v)for w,v in comp[2]];edit(nodes)
    comp[2]=[(2,encode(n))for n in nodes]
   comps.append((wire,encode(comp)))
  outer[3]=comps;return encode(outer)
 def pair(edit):return mutate(context,edit),mutate(printer,edit)
 def node(nodes,pid):return next(n for n in nodes if PC.one(n,2,b'')==b'parameter' and PC.one(n,9,0)==pid)
 proof=test('actual-dX-pre-sync-observer-replay-not-execution')
 # The original native-only helper still rejects each newly observed scalar.
 for key in ('0','1','3','5'):
  sf=next(PC.one(n,3)for n in G.general_graph(context)['entry_nodes'].values()if PC.one(n,2,b'')==b'parameter'and str(PC.one(n,9,0))==key)
  try:PC.shape(sf)
  except ValueError as error:assert str(error)=='SHAPE_SCOPE'
  else:raise AssertionError('native helper unexpectedly widened')
  rows.append({'case':'frozen-native-scalar-rejection-'+key,'status':'PASS'})
 for key in ('0','1','3','5'):
  bad=copy.deepcopy(parameters);bad[key]['values'][0]+=1;test('wrong-scalar-value-'+key,params=bad,expected='REFERENCE_FIXED_PARAMETER_CONTENT')
  bad=copy.deepcopy(parameters);bad[key]['values'][0]=float(bad[key]['values'][0]);test('float-integer-scalar-'+key,params=bad,expected='REFERENCE_SCALAR_EXACT_INTEGER')
  bad=copy.deepcopy(parameters);bad[key]['shape']=[1];test('wrong-record-scalar-shape-'+key,params=bad,expected=('REFERENCE_PARAMETER_SHAPE_DTYPE','REFERENCE_ARRAY_DTYPE'))
  newdtype='uint8'if bad[key]['dtype']=='int64'else'int64';enum=6 if newdtype=='uint8'else 5
  bad=copy.deepcopy(parameters);bad[key]['dtype']=newdtype
  def edit(nodes,key=key,enum=enum):
   n=node(nodes,int(key));sf=PC.fields(PC.one(n,3));sf[2]=[(0,enum)];n[3]=[(2,encode(sf))]
  c,p=pair(edit);test('coherent-wrong-scalar-dtype-'+key,c,p,bad,expected=('REFERENCE_FIXED_PARAMETER_CONTENT','REFERENCE_SCALAR_EXACT_INTEGER'))
  def edit(nodes,key=key):
   n=node(nodes,int(key));sf=PC.fields(PC.one(n,3));sf[3]=[(0,1)];n[3]=[(2,encode(sf))]
  c,p=pair(edit);bad=copy.deepcopy(parameters);bad[key]['shape']=[1]
  test('coherent-wrong-scalar-vector-'+key,c,p,bad,expected='SHAPE_SCOPE'if key!='3'else'REFERENCE_FIXED_PARAMETER_CONTENT')
 for left,right in [('0','1'),('1','5'),('0','5')]:
  bad=copy.deepcopy(parameters);bad[left]['values'],bad[right]['values']=bad[right]['values'],bad[left]['values']
  test('same-allowed-scalar-multiset-wrong-role-'+left+'-'+right,params=bad,expected='REFERENCE_SCALAR_ROLE_USE')
 for opcode,replacement in [('divide','remainder'),('and','or'),('add','subtract')]:
  def edit(nodes,opcode=opcode,replacement=replacement):
   # Choose the first matching scalar-reachable opcode, not a fixed node name.
   candidates=[n for n in nodes if PC.one(n,2,b'')==opcode.encode()]
   if opcode=='divide':n=next(n for n in candidates if PC.one(PC.fields(PC.one(n,3)),2)==11)
   else:n=candidates[0]
   n[2]=[(2,replacement.encode())]
  c,p=pair(edit);test('unsupported-scalar-consumer-'+replacement,c,p,expected=('REFERENCE_SCALAR_ROLE_USE','REFERENCE_SCALAR_INDEX_LINEAGE_SCOPE'))
 def edit(nodes):
  next(n for n in nodes if PC.one(n,2,b'')==b'floor')[2]=[(2,b'tanh')]
 c,p=pair(edit);test('outside-scalar-index-lineage-opcode',c,p,expected='REFERENCE_SCALAR_INDEX_LINEAGE_SCOPE')
 def edit(nodes):node(nodes,1)[9]=[(0,0)]
 c,p=pair(edit);test('duplicate-parameter-number',c,p,expected='REFERENCE_PARAMETER_NUMBER_UNIQUE')
 bad=copy.deepcopy(parameters);bad.pop('0');test('missing-required-parameter',params=bad,expected='REFERENCE_PARAMETER_NUMBERS')
 bad=copy.deepcopy(parameters);bad['99']=copy.deepcopy(bad['0']);test('extra-unbound-parameter',params=bad,expected='REFERENCE_PARAMETER_NUMBERS')
 bad=copy.deepcopy(parameters);bad['0']['values']=[1<<63];test('int64-out-of-range',params=bad,expected='REFERENCE_SCALAR_EXACT_INTEGER')
 bad=copy.deepcopy(parameters);bad['3']['values']=[256];test('uint8-out-of-range',params=bad,expected='REFERENCE_SCALAR_EXACT_INTEGER')
 def edit(nodes):node(nodes,0)[9]=[(0,99)]
 c,p=pair(edit);test('context-printer-parameter-mismatch',c,printer,expected='REFERENCE_CONTEXT_PRINTER_CONTENT')
 test('incorrect-dX-expected-shape',shape=(128,256),expected='REFERENCE_ACTUAL_OUTPUT_SHAPE')
 test('incorrect-dX-expected-dtype',dtype='bfloat16',expected='REFERENCE_ACTUAL_OUTPUT_SHAPE')
 def edit(nodes):next(n for n in nodes if PC.one(n,2,b'')==b'dot')[2]=[(2,b'multiply')]
 c,p=pair(edit);test('required-dense-dot-not-removed',c,p,expected='REFERENCE_DENSE_DOT_REQUIRED')
 def edit(nodes):next(n for n in nodes if PC.one(n,2,b'')==b'dot')[2]=[(2,b'custom-call')]
 c,p=pair(edit);test('custom-call-in-reference-rejected',c,p,expected='REFERENCE_NO_CUSTOM_CALL')
 bad=copy.deepcopy(fixed);bad['scales']['values'][0]+=1;test('incorrect-fixed-scales-payload',inputs=bad,expected='REFERENCE_SCALAR_ROLE_USE')
 # Valid changes preserve semantics without freezing parameter IDs or spelling.
 def edit(nodes):
  for n in nodes:
   if PC.one(n,2,b'')==b'parameter':n[9]=[(0,PC.one(n,9,0)+100)]
 c,p=pair(edit);renumbered={str(int(k)+100):v for k,v in parameters.items()};test('alternative-parameter-numbering',c,p,renumbered)
 def edit(nodes):
  for n in nodes:
   if PC.one(n,2,b'')in(b'add',b'and'):n[36]=[(0,d)for d in reversed(PC.ints(n,36))]
 c,p=pair(edit);test('alternative-commutative-consumer-operand-order',c,p)
 def edit(nodes):
  original=node(nodes,0);pid=PC.one(original,35);new_id=max(PC.one(n,35,0)for n in nodes)+1
  for n in nodes:
   if 36 in n:n[36]=[(0,new_id if d==pid else d)for d in PC.ints(n,36)]
  nodes.append({1:[(2,b'allowed-scalar-copy')],2:[(2,b'copy')],3:copy.deepcopy(original[3]),35:[(0,new_id)],36:[(0,pid)]})
 c,p=pair(edit);test('alternative-benign-scalar-copy',c,p)
 # Bound semantics must be exactly signed negative-index wrapping.
 for position in (0,1):
  def compare(nodes,position=position):return [n for n in nodes if PC.one(n,2,b'')==b'compare'][position]
  def zero_node(nodes,position=position):
   pred=compare(nodes);broadcast_id=PC.ints(pred,36)[1]
   broadcast=next(n for n in nodes if PC.one(n,35,0)==broadcast_id)
   constant_id=PC.ints(broadcast,36)[0]
   return broadcast,next(n for n in nodes if PC.one(n,35,0)==constant_id)
  for direction in (b'GT',b'LE'):
   def edit(nodes,direction=direction):compare(nodes)[63]=[(2,direction)]
   c,p=pair(edit);test('bound-wrong-direction-'+str(position)+'-'+direction.decode(),c,p,expected='REFERENCE_SCALAR_ROLE_USE')
  def edit(nodes):
   _,constant=zero_node(nodes);literal=PC.fields(PC.one(constant,8));literal[5]=[(0,1)];constant[8]=[(2,encode(literal))]
  c,p=pair(edit);test('bound-nonzero-predicate-'+str(position),c,p,expected='REFERENCE_SCALAR_ROLE_USE')
  def edit(nodes):
   broadcast,constant=zero_node(nodes)
   for n in (broadcast,constant):
    sf=PC.fields(PC.one(n,3));sf[2]=[(0,4)];n[3]=[(2,encode(sf))]
   literal=PC.fields(PC.one(constant,8));literal[1]=constant[3];literal.pop(5);literal[4]=[(0,0)];constant[8]=[(2,encode(literal))]
  c,p=pair(edit);test('bound-wrong-typed-zero-'+str(position),c,p,expected='REFERENCE_SCALAR_ROLE_USE')
  def edit(nodes):
   pred=compare(nodes);broadcast,constant=zero_node(nodes)
   broadcast[2]=[(2,b'copy')];broadcast[36]=[(0,PC.ints(pred,36)[0])];nodes.remove(constant)
  c,p=pair(edit);test('bound-unrelated-other-predicate-operand-'+str(position),c,p,expected='REFERENCE_SCALAR_ROLE_USE')
  def edit(nodes):
   pred_id=PC.one(compare(nodes),35);select=next(n for n in nodes if PC.one(n,2,b'')==b'select'and PC.ints(n,36)[0]==pred_id)
   operands=PC.ints(select,36);select[36]=[(0,operands[0]),(0,operands[2]),(0,operands[1])]
  c,p=pair(edit);test('bound-swapped-select-branches-'+str(position),c,p,expected='REFERENCE_SCALAR_ROLE_USE')
  def edit(nodes):compare(nodes)[72]=[(2,b'UNSIGNED')]
  c,p=pair(edit);test('bound-unsigned-predicate-'+str(position),c,p,expected='REFERENCE_SCALAR_ROLE_USE')
  def edit(nodes):
   pred=compare(nodes);pred[36]=[(0,d)for d in reversed(PC.ints(pred,36))];pred[63]=[(2,b'GT')]
  c,p=pair(edit);test('equivalent-zero-greater-than-index-'+str(position),c,p)
  def edit(nodes):
   broadcast,constant=zero_node(nodes);sf=PC.one(broadcast,3);count=__import__('math').prod(PC.ints(PC.fields(sf),3))
   broadcast[2]=[(2,b'constant')];broadcast.pop(36);broadcast[8]=[(2,encode({1:[(2,sf)],5:[(2,b'\x00'*count)]}))];nodes.remove(constant)
  c,p=pair(edit);test('equivalent-materialized-typed-zero-'+str(position),c,p)
 # Equal literal values can have distinct backend handles. Both complete
 # bound/gather paths are valid; minimum of equal gathers preserves values.
 def duplicate_bound(nodes):
  by_id={PC.one(n,35,0):n for n in nodes};start=PC.one(node(nodes,1),35);chosen={start}
  admitted={b'reshape',b'copy',b'broadcast',b'add',b'select',b'concatenate',b'gather'}
  for _ in nodes:
   for i,n in by_id.items():
    if PC.one(n,2,b'')in admitted and any(d in chosen for d in PC.ints(n,36)):chosen.add(i)
  gather=next(i for i in chosen if PC.one(by_id[i],2,b'')==b'gather')
  first=max(by_id)+1;mapping={old:first+offset for offset,old in enumerate(sorted(chosen))}
  for old,new in mapping.items():
   n=copy.deepcopy(by_id[old]);n[35]=[(0,new)];n[36]=[(0,mapping.get(d,d))for d in PC.ints(n,36)]
   if old==start:n[9]=[(0,99)]
   nodes.append(n)
  merge=max(mapping.values())+1
  for n in nodes:
   if PC.one(n,35,0)not in mapping.values()and 36 in n:n[36]=[(0,merge if d==gather else d)for d in PC.ints(n,36)]
  nodes.append({2:[(2,b'minimum')],3:copy.deepcopy(by_id[gather][3]),35:[(0,merge)],36:[(0,gather),(0,mapping[gather])]})
  return merge
 c,p=pair(duplicate_bound);duplicate_params=copy.deepcopy(parameters);duplicate_params['99']=copy.deepcopy(parameters['1'])
 test('valid-distinct-handles-duplicate-bound-role',c,p,duplicate_params)
 bad=copy.deepcopy(duplicate_params);bad['99']=copy.deepcopy(parameters['5'])
 test('duplicate-handle-wrong-role-rejected',c,p,bad,expected='REFERENCE_SCALAR_ROLE_USE')
 def edit(nodes):
  merge=duplicate_bound(nodes);param=PC.one(node(nodes,99),35);last=max(PC.one(n,35,0)for n in nodes)
  # Add a second use of the correctly bound scalar outside indexing.
  scalar_shape=PC.fields(PC.one(node(nodes,99),3));scalar_shape[2]=[(0,11)]
  nodes.append({2:[(2,b'convert')],3:[(2,encode(scalar_shape))],35:[(0,last+1)],36:[(0,param)]})
  merge_node=next(n for n in nodes if PC.one(n,35,0)==merge);vector_shape=copy.deepcopy(merge_node[3])
  nodes.append({2:[(2,b'broadcast')],3:vector_shape,35:[(0,last+2)],36:[(0,last+1)]})
  for n in nodes:
   if PC.one(n,35,0)!=merge and 36 in n:n[36]=[(0,last+3 if d==merge else d)for d in PC.ints(n,36)]
  nodes.append({2:[(2,b'multiply')],3:vector_shape,35:[(0,last+3)],36:[(0,merge),(0,last+2)]})
 c,p=pair(edit);test('duplicate-handle-correct-role-with-extra-consumer-rejected',c,p,duplicate_params,expected='REFERENCE_SCALAR_ROLE_USE')
 # Correct wrapping must feed the actual gather without later arithmetic.
 for position in (0,1):
  def edit(nodes,position=position):
   n=[n for n in nodes if PC.one(n,2,b'')==b'gather'][position]
   index=next(v for v in nodes if PC.one(v,35,0)==PC.ints(n,36)[1]);operand=PC.ints(index,36)[0]
   index[2]=[(2,b'add')];index[36]=[(0,operand),(0,operand)]
  c,p=pair(edit);test('bound-doubled-after-valid-wrap-'+str(position),c,p,expected='REFERENCE_SCALAR_ROLE_USE')
 # Retain all old no-scalar reference shapes and reduction paths.
 dense=M.tensor(decode(M.materialize(fixed['packed']).flatten(),M.materialize(fixed['scales']),code,64,(256,512),__import__('torch').float32))
 for rank in (2,3):
  rcase=next(c for c in G.C.read(source/'public-protocol.json')['cases']if c['id']=='public-train-rank'+str(rank));inp=M.input_record(rcase);base_allowed=G.allowed_reference_values(M.allowed_reference_values(rcase,decode,code,{}),rcase)
  for kind in ('dx','dx2','db','db2'):
   raw=(fixtures/(kind+'-rank'+str(rank)+'.pb')).read_bytes();ps={'0':inp['dy']}
   if kind.startswith('dx'):ps['1']=dense
   shape=rcase['activation_shape']if kind.startswith('dx')else[256]
   value=G.verify_reference(raw,raw,shape,'float32',bias_gradient=kind.startswith('db'),parameters=ps,allowed_parameters=base_allowed)
   assert value['scalar_role_bindings']==[]
   rows.append({'case':'existing-reference-'+kind+'-rank'+str(rank),'status':'PASS'})
 after={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in graphs.iterdir()if p.is_file()};assert before==after
 result={'status':'PASS','controls_count':len(rows),'controls':rows,'scope':'OFFLINE_PRE_SYNC_OBSERVER_AND_ISOLATED_MEMORY_MUTATIONS','actual_input_hashes_unchanged':True,'actual_input_hashes':before,'actual_original_result':'4PASS_1ERROR_10NOT_RUN_UNCHANGED','actual_dX_synchronization':'NOT_RUN','numerical_result':'NOT_EVALUATED','actual_repaired_device':'NOT_RUN','provider_calls':0,'fixture_files_written':0,'observer_sha256':G.C.sha(source/'reference_observer_v3.py'),'proof':proof}
 with output.open('x')as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps({'status':'PASS','controls':len(rows),'actual_dX':'OFFLINE_PRE_SYNC_OBSERVER_ONLY','provider_calls':0}))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('public-source','actual-graphs','upstream-root','original-fixtures','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();run(a.public_source,a.actual_graphs,a.upstream_root,a.original_fixtures,a.output)
