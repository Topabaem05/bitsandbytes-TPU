"""Create isolated diagnostic variants. Original plugin bytes remain unchanged."""
import ast,hashlib,json,shutil
from pathlib import Path
MODULES=('exact.py','functionalization.py','_integer_clamp.py','_integer_addsub.py','_integer_muldiv.py','_source_order.py')
class Rewrite(ast.NodeTransformer):
 def __init__(self,amax=False):self.events=[];self.amax=amax
 def visit_BinOp(self,node):
  node=self.generic_visit(node)
  if isinstance(node.op,(ast.LShift,ast.RShift)):
   kind='left'if isinstance(node.op,ast.LShift)else'right';self.events.append({'kind':'runtime_tensor_'+kind+'_shift','line':node.lineno,'expression':ast.unparse(node)})
   return ast.copy_location(ast.Call(func=ast.Name(id='_diagnostic_tensor_shift',ctx=ast.Load()),args=[node.left,node.right,ast.Constant(kind)],keywords=[]),node)
  return node
 def visit_Call(self,node):
  node=self.generic_visit(node)
  if self.amax and isinstance(node.func,ast.Attribute)and isinstance(node.func.value,ast.Name)and node.func.value.id=='torch'and node.func.attr=='amax'and len(node.args)==1 and not node.keywords:
   self.events.append({'kind':'explicit_flat_amax_dim0','line':node.lineno,'expression':ast.unparse(node)});node.keywords=[ast.keyword(arg='dim',value=ast.Constant(0))]
  return node

def generate(packet):
 packet=Path(packet);root=packet/'variants';root.mkdir();records={}
 for name,shift,amax in [('original',False,False),('true_shift',True,False),('true_shift_amax',True,True)]:
  target=root/name;target.mkdir();records[name]={}
  for module in MODULES:
   src=packet/'original'/module;text=src.read_text();events=[]
   if shift and module not in ('functionalization.py','_source_order.py'):
    rw=Rewrite(amax);tree=rw.visit(ast.parse(text));ast.fix_missing_locations(tree);events=rw.events
    if events:
     text=ast.unparse(tree)+'\n'
     if any(e['kind'].startswith('runtime_tensor_') for e in events):text+='\nfrom diagnostic_ops import tensor_shift as _diagnostic_tensor_shift\n'
   dest=target/module;dest.write_text(text)
   records[name][module]={'preimage_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'postimage_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'rewrites':events}
 (packet/'source/rewrite-records.json').write_text(json.dumps(records,sort_keys=True,indent=2)+'\n');return records
if __name__=='__main__':generate(Path(__file__).resolve().parents[1])
