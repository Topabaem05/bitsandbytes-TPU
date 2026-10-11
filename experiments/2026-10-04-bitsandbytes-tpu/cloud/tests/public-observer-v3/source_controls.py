"""Versioned observer source closure and frozen-math controls; no provider calls."""
import argparse, ast, copy, hashlib, importlib.util, json, shutil, sys, tempfile
from pathlib import Path


def run(source, cloud_contract, baseline_seals, original_source, output):
 source=Path(source).resolve();sys.path.insert(0,str(source));import contract as C
 spec=importlib.util.spec_from_file_location('observer_cloud_source_controls',cloud_contract);U=importlib.util.module_from_spec(spec);spec.loader.exec_module(U)
 rows=[]
 def check(name,fn,error=None):
  try:fn()
  except ValueError as e:
   assert error is not None and error in str(e),(name,str(e),error)
   rows.append({'case':name,'status':'PASS','rejected':str(e)});return
  assert error is None,name
  rows.append({'case':name,'status':'PASS'})
 s=C.read(source/'public-device-manifest.json');sources=C.scientific_sources()
 check('complete-new-scientific24-map',lambda:C.require(len(sources)==24 and s['source_files']==sources and 'reference_observer_v3.py' in sources,'SCIENTIFIC_MAP'))
 check('cloud-new-generation-and-manifest-pin',lambda:C.require(s['scientific_variant']==C.VARIANT==U.GENERATION and C.sha(source/'public-device-manifest.json')==U.MANIFEST_SHA,'GENERATION_PIN'))
 m={'experiment':U.MODE,'public_generation':U.GENERATION,'public_source_variant':U.VARIANT,'public_scope':U.SCOPE,'public_result_max_bytes':U.RESULT_MAX_BYTES,'public_result_max_members':U.RESULT_MAX_MEMBERS,'precision':'highest','m6_status':'NOT_QUALIFIED','files':{}}
 for field,pin in zip(U.BINDINGS,(U.MANIFEST_SHA,U.ADMISSION_SHA,U.PROTOCOL_SHA,U.NATIVE_ACCEPTANCE_SHA)):
  m[field]=pin;m['files'][U.BINDINGS[field]]={'sha256':pin}
 check('new-generation-manifest-admission',lambda:U.verify_manifest(m))
 for field,value,error in [('public_generation','public-pallas-device-mosaic7-gather-bf16-fp32-dyadic-v2','PUBLIC_EXACT_MODE_SCOPE'),('public_manifest_sha256','72e0f202ae8c274853b0a2e1e29c4dd21aeaf5b5b9464839025926b641563b5d','PUBLIC_LITERAL_BINDINGS'),('public_result_max_bytes',U.RESULT_MAX_BYTES+1,'PUBLIC_EXACT_RESULT_LIMITS'),('m6_status','QUALIFIED','PUBLIC_CLAIM_SCOPE')]:
  bad=copy.deepcopy(m);bad[field]=value;check('wrong-'+field,lambda bad=bad:U.verify_manifest(bad),error)
 baseline=C.read(baseline_seals);changed={'contract.py','public_graph.py','probe_public.py','verify_public.py','public-device-manifest.json'}
 unchanged={n:r for n,r in baseline.items()if n not in changed}
 check('all19-frozen-primary-native-helper-and-protocol-files',lambda:C.require(all({'sha256':C.sha(source/n),'bytes':(source/n).stat().st_size}==r for n,r in unchanged.items())and len(unchanged)==19,'FROZEN_SOURCE_BYTES'))
 def functions(path):
  text=Path(path).read_text();lines=text.splitlines(keepends=True)
  return {n.name:''.join(lines[n.lineno-1:n.end_lineno])for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
 old=functions(Path(original_source)/'public_graph.py');new=functions(source/'public_graph.py')
 check('native-graph-functions-byte-identical',lambda:C.require(all(new[n]==old[n]for n in ('grouped','payload','verify_native','verify_boundary')),'NATIVE_FUNCTION_BYTES'))
 with tempfile.TemporaryDirectory(prefix='bnb-public-observer-source-controls-')as tmp:
  fixture=Path(tmp).resolve()/'source';fixture.mkdir()
  for n in sources:
   target=fixture/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/n,target)
  check('complete-source-body-map',lambda:C.source_files(fixture,sources))
  module=fixture/'reference_observer_v3.py';module.unlink()
  check('missing-observer-rejected',lambda:C.source_files(fixture,sources),'PYTHON_SOURCE_INVENTORY')
  module.write_bytes((source/'reference_observer_v3.py').read_bytes()+b'\n# wrong source\n')
  check('changed-observer-rejected',lambda:C.source_files(fixture,sources),'SOURCE_BYTES:reference_observer_v3.py')
 result={'status':'PASS','control_count':len(rows),'controls':rows,'scientific_files':sources,'manifest_sha256':U.MANIFEST_SHA,'cloud_contract_sha256':C.sha(cloud_contract),'unchanged_original_files':unchanged,'provider_calls':0,'full_packet_build':'NOT_RUN','fixture_cleanup':'TEMPORARY_DIRECTORY_REMOVED'}
 Path(output).write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps({'status':'PASS','controls':len(rows),'provider_calls':0}))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('public-source','cloud-contract','baseline-seals','original-source','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();run(a.public_source,a.cloud_contract,a.baseline_seals,a.original_source,a.output)
