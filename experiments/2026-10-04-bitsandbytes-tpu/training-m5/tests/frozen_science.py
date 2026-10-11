"""Verify unchanged original M5 protocol, generators and reference function syntax trees."""
import ast,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'));import admission as A,math_model as M
old=ast.parse((ROOT/'historical/cpu_prototype.py').read_text());new=ast.parse((ROOT/'source/math_model.py').read_text());old_functions={n.name:n for n in old.body if isinstance(n,ast.FunctionDef)};rows=[]
for n in new.body:
 if isinstance(n,ast.FunctionDef):
  A.require(n.name in old_functions and ast.dump(n,include_attributes=False)==ast.dump(old_functions[n.name],include_attributes=False),'FROZEN_FUNCTION:'+n.name);rows.append(n.name)
A.require(M.PROTOCOL_SHA=='bd221d5dcd8a06a3080ff30345f80b8b99d8ca609fadcb813ea8da09964d007f','FROZEN_PROTOCOL')
for name in ['PROTOCOL','GENERATOR','ADAPTER_NAMES']:
 def value(tree):return ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id==name for t in n.targets)))
 A.require(value(old)==value(new),'FROZEN_CONSTANT:'+name)
result={'status':'PASS','unchanged_functions':rows,'function_count':len(rows),'unchanged_constants':['PROTOCOL','GENERATOR','ADAPTER_NAMES'],'protocol_sha256':M.PROTOCOL_SHA,'original_prototype_sha256':A.sha(ROOT/'historical/cpu_prototype.py'),'m5_status':'NOT_QUALIFIED','provider_calls':0};A.write(Path(sys.argv[1]),result);print(json.dumps({'status':'PASS','unchanged_functions':len(rows)}))
