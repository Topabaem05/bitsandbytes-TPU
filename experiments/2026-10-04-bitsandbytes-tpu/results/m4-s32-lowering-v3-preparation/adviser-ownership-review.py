import sys,json,hashlib,importlib.util,tempfile,difflib
from pathlib import Path
root=Path('/tmp/bnb-tpu-m4-s32-lowering-v3-20261011-oj7mwf5m')
out=Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('fixture',root/'browser_ownership_controls.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.main(out/'baseline.json')
old=root/'test_support/pre-browser-ownership-seal/driver.py';new=root/'bundle/experiments/2026-10-04-bitsandbytes-tpu/cloud/exact-nested/driver.py'
diff=list(difflib.unified_diff(old.read_text().splitlines(),new.read_text().splitlines()))
extra=[]
for name,repl in [('hardware_prefix',"hardware='V5E1-extra'if case=='wrong_hardware'else'V5E1'"),('variant_prefix',None)]:
 source=(root/'browser_ownership_controls.py').read_text()
 if repl: source=source.replace("hardware='V6E1'if case=='wrong_hardware'else'V5E1'",repl)
 else: source=source.replace("| Variant: TPU'","| Variant: TPU-extra'")
 ns={'__file__':str(root/'browser_ownership_controls.py'),'__name__':'fixture_'+name};exec(compile(source,str(root/'browser_ownership_controls.py'),'exec'),ns)
 with tempfile.TemporaryDirectory(prefix='m4-owned-prefix-',dir='/tmp') as tmp:
  try:
   result=ns['run_case'](Path(tmp)/'case','wrong_hardware' if repl else 'valid_marker')
   outcome='returned';calls=result['calls']
  except AssertionError:
   outcome='no_stop_assertion_failed';r=json.loads((Path(tmp)/'case/actual-driver-output/driver.json').read_text());calls=r['commands']
  extra.append({'case':name,'outcome':outcome,'calls':calls})
report={'kind':'INDEPENDENT_M4_TRANSPORT_REVIEW','baseline':json.loads((out/'baseline.json').read_text()),'driver_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),'diff':diff,'prefix_controls':extra}
(out/'summary.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps({'baseline':report['baseline']['control_count'],'driver_sha256':report['driver_sha256'],'prefix_controls':[(x['case'],x['outcome'])for x in extra]}))
