"""V3 real interactive-source regression plus isolated marker/readback negatives."""
import argparse,contextlib,copy,hashlib,importlib.util,io,json,os,shutil,sys,tempfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE));import contract as C,kernel_protocol as K,build_packet as B
sys.path.insert(0,str(HERE/'ownership'));from cleanup_lifecycle import Ownership


def main(output):
 output=Path(output).resolve();C.require(not output.exists(),'FRESH_KERNEL_CONTROL_OUTPUT');output.mkdir();rows=[];disposed=[]
 def check(name,fn,error=None):
  try:value=fn()
  except (ValueError,FileNotFoundError)as e:
   assert error is not None and error in str(e),(name,str(e),error);rows.append({'case':name,'status':'PASS','rejected':str(e)});return
  assert error is None,name;rows.append({'case':name,'status':'PASS'});return value
 with tempfile.TemporaryDirectory(prefix='kernel-v3-fixtures-',dir=output)as temp:
  root=Path(temp).resolve();review=root/'synthetic-review.json';C.write(review,{'kind':'ROOT_NESTED_EXACT_SOURCE_REVIEW','status':'SOURCE_REVIEW_ACCEPTED_FOR_EXPERIMENT','integration_map_sha256':C.MAP_SHA,'source_admission_sha256':C.ADMISSION_SHA,'science_map_sha256':C.SCIENCE_SHA,'device_acceptance':None,'fixture_only':True});packet=root/'packet';built=B.build(HERE.parents[1],review,C.sha(review),packet);packet_sha=built['packet_sha256'];manifest=C.read(packet/'manifest.json');birth=time.time()
  def interactive(name,prefix=True,stale=False):
   base=root/name;base.mkdir();shutil.copytree(packet/'control',base/'control');shutil.copyfile(built['archive'],base/'payload.zip');sources={str(base/n):r['sha256']for n,r in manifest['files'].items()if n.startswith('control/')and n.endswith('.py')};modules={Path(n).stem:(str(base/n),r['sha256'])for n,r in manifest['files'].items()if n.startswith('control/')and n.endswith('.py')and not Path(n).stem.startswith('__')}
   for module in ('generation_gate','exact_admission'):
    relative='nested-exact/source/'+module+'.py';modules[module]=(str(base/'payload'/relative),manifest['files'][relative]['sha256'])
   unpack='import runpy,sys\nfrom pathlib import Path\n'+f'b=Path({str(base)!r});sys.path.insert(0,str(b/"control"));sys.argv=["remote.py","unpack","--base",str(b),"--packet-sha256",{packet_sha!r},"--allocation-epoch",{str(birth)!r}]\nrunpy.run_path(str(b/"control/remote.py"),run_name="__main__")\n'
   readmit='import contract as C\nC.admit(b/"payload")\n'
   child='import subprocess\nassert subprocess.check_output([sys.executable,"-c",'+repr('import sys;sys.path.insert(0,'+repr(str(base/'payload/nested-exact/source'))+');import generation_gate;print(sys.dont_write_bytecode)')+'],text=True).strip()=="True"\nC.admit(b/"payload")\n'
   codes=[];requests=[]
   for phase,body in [('unpack',unpack),('readmit',readmit),('childinherit',child)]:
    req=K.request(body,phase,base,packet_sha,birth,sources,artifact=base/'records/allocation.json',artifact_expected={'birth_epoch':birth,'packet_sha256':packet_sha,'original_deadline_epoch':birth+3600});requests.append(req);codes.append(K.render(body,req,sources,modules)if prefix else body)
   runner=base/'runner.py';result=base/'result.json';foreign=base/'foreign.py';foreign.write_text('# foreign fixture\n')
   preface='import sys,os,io,json,contextlib\nfrom pathlib import Path\nsys.dont_write_bytecode=False;os.environ.pop("PYTHONDONTWRITEBYTECODE",None)\nlogs=[];failure=None\n'
   if stale:preface+='import types\nsys.modules["contract"]=types.SimpleNamespace(__file__='+repr(str(foreign))+')\n'
   runner.write_text(preface+'codes='+repr(codes)+'\nfor code in codes:\n stream=io.StringIO()\n try:\n  with contextlib.redirect_stdout(stream):exec(compile(code,"<interactive-kernel-fixture>","exec"),globals())\n except BaseException as e:failure={"type":type(e).__name__,"message":str(e)}\n logs.append(stream.getvalue())\n if failure:break\n'+f'Path({str(result)!r}).write_text(json.dumps({{"initial_bytecode_disabled":False,"logs":logs,"error":failure,"pyc":[str(p.relative_to(Path({str(base)!r})))for p in Path({str(base)!r}).rglob("*.pyc")]}}))\n')
   (base/'owned').mkdir();owner=Ownership(base/'owned');streams=[(base/n).open('wb')for n in('stdout.raw','stderr.raw')]
   try:
    with owner.guard(60):
     process=owner.launch([sys.executable,str(runner)],record=base/'owned/ownership.json',cwd=base,env={k:v for k,v in os.environ.items()if k not in{'PYTHONDONTWRITEBYTECODE','PYTHONPATH','PYTHONHOME'}},stdout=streams[0],stderr=streams[1]);process.wait()
   finally:
    for f in streams:f.close()
   closure=owner.summary();assert not closure['errors'];assert process.returncode==0,(name,(base/'stderr.raw').read_text());value=C.read(result)
   if prefix and not stale:
    assert value['error']is None and value['pyc']==[],value
    for text,req in zip(value['logs'],requests):K.verify(text,req,req['completion_path'],base/'records/allocation.json')
   elif stale:assert value['error']and'KERNEL_STALE_MODULE:contract'in value['error']['message'],value
   else:assert value['error']and'BYTECODE'in value['error']['message']and value['pyc'],value
   return {'case':name,'status':'PASS','observed_error':value['error'],'sealed_pyc_count':len(value['pyc']),'initial_bytecode_disabled':False,'phase_count':len(value['logs']),'owned_closure':closure,'fixture_only':True}
  rows.append(interactive('old-kernel-reproduces-bytecode',False));rows.append(interactive('v3-unpack-readmit-childinherit-no-bytecode'));rows.append(interactive('v3-stale-module-rejected',stale=True))
  browser_base=root/'browser-fresh-generation';namespace={}
  browser_marker=root/'browser-marker.txt';browser_marker.write_text('root fixture marker')
  for phase,body in [('browser-marker','from pathlib import Path\n'+f'p=Path({str(browser_marker)!r});assert p.is_file()\nprint("EXACT_ROOT_MARKER_MATCH")\n'),('readiness','from pathlib import Path\n'+f'b=Path({str(browser_base)!r});assert not b.exists()and not b.is_symlink();(b/"control/ownership").mkdir(parents=True)\n')]:
   req=K.request(body,phase,browser_base,packet_sha,birth,{});assert not Path(req['completion_path']).is_relative_to(browser_base)
   stream=io.StringIO()
   with contextlib.redirect_stdout(stream):exec(compile(K.render(body,req,{},{}),'<browser-bootstrap-fixture>','exec'),namespace)
   K.verify(stream.getvalue(),req,req['completion_path'])
   if phase=='browser-marker':assert not browser_base.exists()
  assert(browser_base/'control/ownership').is_dir();rows.append({'case':'browser-marker-completion-preserves-fresh-readiness-base','status':'PASS','fixture_only':True})
  req=K.request(body,'readiness',browser_base,packet_sha,birth,{});stream=io.StringIO()
  try:
   with contextlib.redirect_stdout(stream):exec(compile(K.render(body,req,{},{}),'<occupied-readiness-fixture>','exec'),namespace)
  except AssertionError:assert not Path(req['completion_path']).exists();rows.append({'case':'readiness-rejects-preexisting-experiment-base-without-success-marker','status':'PASS','fixture_only':True})
  else:raise AssertionError('occupied readiness base accepted')
  body='print("fixture successful body")\n';base=root/'marker';base.mkdir();artifact=base/'phase.json';C.write(artifact,{'phase':'install','packet_sha256':packet_sha});req=K.request(body,'install',base,packet_sha,birth,{},artifact=artifact,artifact_expected={'phase':'install','packet_sha256':packet_sha});receipt=base/'completion.json';value={**req,'status':'COMPLETED','artifact':K.row(artifact)};C.write(receipt,value)
  marker=K.VERSION+' '+json.dumps({'nonce':req['nonce'],'phase':req['phase'],'admitted_source_sha256':req['admitted_source_sha256'],'readback_sha256':C.sha(receipt)},sort_keys=True)+'\n'
  check('correct-marker-and-independent-artifact-readback',lambda:K.verify(marker,req,receipt,artifact))
  for name,text,error in [('cli-zero-traceback-without-marker','ValueError: kernel failed\n','KERNEL_COMPLETION_LINE'),('duplicate-marker',marker+marker,'KERNEL_COMPLETION_LINE'),('marker-followed-error',marker+'ValueError: late failure\n','KERNEL_COMPLETION_LINE'),('marker-followed-output',marker+'unexpected output\n','KERNEL_COMPLETION_LINE'),('malformed-marker-json',K.VERSION+' not-json\n','KERNEL_MARKER_JSON')]:check(name,lambda text=text:K.verify(text,req,receipt,artifact),error)
  for field in ('nonce','phase','admitted_source_sha256'):
   mark=json.loads(marker.split(' ',1)[1]);mark[field]='f'*32 if field=='nonce'else'wrong';text=K.VERSION+' '+json.dumps(mark)+'\n';check('wrong-marker-'+field,lambda text=text:K.verify(text,req,receipt,artifact),'KERNEL_COMPLETION_BINDING')
  old=copy.deepcopy(req);old['nonce']='e'*32;check('stale-prior-request-marker',lambda:K.verify(marker,old,receipt,artifact),'KERNEL_COMPLETION_BINDING')
  missing=base/'missing.json';check('missing-independent-completion-readback',lambda:K.verify(marker,req,missing,artifact),'KERNEL_READBACK_REGULAR')
  receipt.write_text('{}');check('altered-completion-readback',lambda:K.verify(marker,req,receipt,artifact),'KERNEL_COMPLETION_READBACK_SHA');C.write(receipt,value)
  original_artifact=artifact.read_bytes();artifact.write_text('{}');check('altered-independent-artifact',lambda:K.verify(marker,req,receipt,artifact),'KERNEL_ARTIFACT_READBACK_SHA');artifact.write_bytes(original_artifact)
  check('missing-independent-artifact',lambda:K.verify(marker,req,receipt),'KERNEL_ARTIFACT_READBACK_SHA')
  try:K.render(body+'raise ValueError("wrong source")\n',req,{},{} )
  except ValueError as e:assert str(e)=='KERNEL_BODY_BINDING';rows.append({'case':'wrong-body-source-before-submission','status':'PASS'})
  else:raise AssertionError('wrong body accepted')
  fixture_module=base/'cache-fixture.py'
  admitted='VALUE=1\ndef run(value=3,*,option=4):return value+option\nclass Example:\n def method(self):return 2\n @staticmethod\n def static():return 5\n @classmethod\n def cls(cls):return 6\n @property\n def prop(self):return 7\n'
  def cached(old,new=admitted):
   fixture_module.write_text(old);spec=importlib.util.spec_from_file_location('sealed_cache_fixture',fixture_module);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);fixture_module.write_text(new);return module
  module=cached(admitted);check('correct-cached-function-class-and-descriptor-code',lambda:K.validate_cached_module(module,fixture_module))
  for name,old in [('function',admitted.replace('return value+option','return value-option')),('class-method',admitted.replace('return 2','return 9')),('literal-global',admitted.replace('VALUE=1','VALUE=0')),('positional-default',admitted.replace('value=3','value=9')),('keyword-default',admitted.replace('option=4','option=9'))]:
   module=cached(old);check('same-path-stale-'+name+'-after-admitted-bytes-replaced',lambda module=module:K.validate_cached_module(module,fixture_module),'KERNEL_STALE_MODULE_CODE')
  def altered_receipt(name,change,error):
   bad=copy.deepcopy(value);change(bad);C.write(receipt,bad);mark=json.loads(marker.split(' ',1)[1]);mark['readback_sha256']=C.sha(receipt);text=K.VERSION+' '+json.dumps(mark)+'\n';check(name,lambda:K.verify(text,req,receipt,artifact),error);C.write(receipt,value)
  altered_receipt('correct-hash-wrong-completion-phase',lambda bad:bad.update(phase='other'),'KERNEL_COMPLETION_READBACK_BINDING')
  altered_receipt('correct-hash-noncompleted-status',lambda bad:bad.update(status='ERROR'),'KERNEL_COMPLETION_READBACK_BINDING')
  raw=receipt.read_text().rstrip();receipt.write_text(raw[:-1]+',"phase":"install"}');mark=json.loads(marker.split(' ',1)[1]);mark['readback_sha256']=C.sha(receipt);text=K.VERSION+' '+json.dumps(mark)+'\n';check('duplicate-readback-json-field',lambda:K.verify(text,req,receipt,artifact),'KERNEL_READBACK_DUPLICATE_JSON');C.write(receipt,value)
  inventory=C.inventory(root)if not any(root.rglob('*.pyc'))else{p.relative_to(root).as_posix():{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}for p in root.rglob('*')if p.is_file()}
  disposed.append({'root':str(root),'files':len(inventory),'bytes':sum(r['bytes']for r in inventory.values()),'inventory_sha256':K.digest(inventory)})
 assert all(not Path(v['root']).exists()for v in disposed)
 C.write(output/'review.json',{'status':'PASS','control_count':len(rows),'controls':rows,'disposable_fixture_inventory':disposed,'disposable_roots_absent':True,'provider_calls':0,'TPU':'NOT_RUN','m4_status':'NOT_QUALIFIED','science_manifest_sha256':C.SCIENCE_SHA,'kernel_protocol_sha256':C.sha(HERE/'kernel_protocol.py'),'driver_sha256':C.sha(HERE/'driver.py')});print(json.dumps({'status':'PASS','controls':len(rows),'fixture_roots_absent':True,'provider_calls':0}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
