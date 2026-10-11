"""Bounded source controls for the draft S32 shift generation."""
import argparse,ast,hashlib,json,shutil,sys,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'source'))
import exact_admission as A,generation_gate as G

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(original):
 old=original/'experiments/2026-10-04-bitsandbytes-tpu/nested-exact';rows=[]
 def test(name,fn):fn();rows.append({'name':name,'status':'PASS'})
 def reject(fn,error):
  try:fn()
  except ValueError as e:assert error in str(e),(error,str(e));return
  raise AssertionError('INCORRECT_GENERATION_ACCEPTED')
 admission=HERE/'integration/source-admission.json';plugin=HERE/'integration/plugin'
 test('correct_complete_source_generation',lambda:G.verify(admission,A.INTEGRATION_ADMISSION_SHA,plugin=plugin,upstream=HERE/'integration/upstream/bitsandbytes'))
 test('parent_admission_sha_rejected',lambda:reject(lambda:A.admission(old/'integration/source-admission.json',sha(old/'integration/source-admission.json')),'EXACT_ADMISSION_LITERAL'))
 test('parent_variant_rejected_even_with_self_hash',lambda:reject(lambda:G.verify(old/'integration/source-admission.json',sha(old/'integration/source-admission.json')),'EXACT_SOURCE_VARIANT'))
 def unchanged():
  before=json.loads((old/'source/science-manifest.json').read_text());after=json.loads((HERE/'source/science-manifest.json').read_text());assert len(before['files'])==len(after['files'])==19
  changed={n for n in before['files']if before['files'][n]!=after['files'][n]};assert changed=={'exact_admission.py','generation_gate.py','integration-admission.json'}
  for name in after['files']:
   p=HERE/'source'/name;assert {'sha256':sha(p),'bytes':p.stat().st_size}==after['files'][name]
   if name not in changed:assert p.read_bytes()==(old/'source'/name).read_bytes()
 test('sixteen_scientific_criteria_and_inputs_unchanged',unchanged)
 def math():
  class Normalize(ast.NodeTransformer):
   def visit_Call(self,n):
    self.generic_visit(n)
    if isinstance(n.func,ast.Attribute) and n.func.attr=='Tensor' and isinstance(n.func.value,ast.Attribute) and n.func.value.attr in ('eq','ne','lt','le','gt','ge') and len(n.args)==2:
     right=n.args[1]
     if isinstance(right,ast.Call) and ((isinstance(right.func,ast.Name) and right.func.id=='scalar_s32') or (isinstance(right.func,ast.Attribute) and right.func.attr=='scalar_s32')):
      operators={'eq':ast.Eq,'ne':ast.NotEq,'lt':ast.Lt,'le':ast.LtE,'gt':ast.Gt,'ge':ast.GtE}
      return ast.Compare(left=n.args[0],ops=[operators[n.func.value.attr]()],comparators=[right.args[1]])
    if isinstance(n.func,ast.Name)and n.func.id in('left_shift','right_shift'):return ast.BinOp(left=n.args[0],op=ast.LShift()if n.func.id=='left_shift'else ast.RShift(),right=n.args[1])
    if isinstance(n.func,ast.Attribute)and n.func.attr=='amax'and n.keywords:n.keywords=[k for k in n.keywords if not(k.arg=='dim'and isinstance(k.value,ast.Constant)and k.value.value==0)]
    return n
  pkg='integration/plugin/src/bitsandbytes_tpu'
  for module in('_integer_addsub.py','_integer_muldiv.py','_integer_clamp.py','exact.py'):
   before={n.name:ast.dump(n,include_attributes=False)for n in ast.parse((old/pkg/module).read_text()).body if isinstance(n,ast.FunctionDef)}
   after={n.name:ast.dump(Normalize().visit(n),include_attributes=False)for n in ast.parse((HERE/pkg/module).read_text()).body if isinstance(n,ast.FunctionDef)and n.name not in('_shift_count','left_shift','right_shift','scalar_s32')};assert before==after,module
  for module in('_source_order.py','functionalization.py','registration.py','blockwise.py','reference.py'):assert(old/pkg/module).read_bytes()==(HERE/pkg/module).read_bytes()
 test('all_existing_math_asts_differ_only_in_shifts_typed_comparisons_tail_axis',math)
 def history():
  h=HERE/'history/functional-output-v2-20261011';m=json.loads((h/'source-map.json').read_text());s=json.loads((h/'science-manifest.json').read_text());assert sha(h/'science-manifest.json')=='29beb28f4320acc5e07ce6822f2120eaff846809bfc54bd123ac37f501700c06';assert len(m['files'])==19
  for name,r in m['files'].items():p=HERE/r['path'];assert sha(p)==r['sha256']==s['files'][name]['sha256']and p.stat().st_size==r['bytes']==s['files'][name]['bytes']
 test('parent_science19_reconstructed_exactly',history)
 with tempfile.TemporaryDirectory(prefix='m4-shift-generation-',dir='/tmp')as tmp:
  fixture=Path(tmp)/'plugin';shutil.copytree(plugin,fixture);p=fixture/'src/bitsandbytes_tpu/_integer_clamp.py';original_bytes=p.read_bytes();p.write_bytes(original_bytes+b'\n');test('altered_helper_rejected',lambda:reject(lambda:G.verify(admission,A.INTEGRATION_ADMISSION_SHA,plugin=fixture),'PLUGIN_PACKAGE_BYTES'));p.unlink();p.symlink_to(plugin/'src/bitsandbytes_tpu/_integer_clamp.py');test('symlink_helper_rejected',lambda:reject(lambda:G.verify(admission,A.INTEGRATION_ADMISSION_SHA,plugin=fixture),'SOURCE_SYMLINK'))
 assert not Path(tmp).exists()
 return {'kind':'M4_S32_SHIFT_GENERATION_CONTROLS_V1','status':'PASS','controls':rows,'control_count':len(rows),'source_variant':A.VARIANT,'fixtures':'DELETED_AFTER_TEST','provider_calls':0,'TPU':'NOT_RUN','m4_status':'NOT_QUALIFIED'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--original-repository',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.original_repository);a.output.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'PASS','controls':len(r['controls'])}))
