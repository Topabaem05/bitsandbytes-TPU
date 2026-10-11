"""Real bytecode-enabled interactive regression; synthetic packet dependency only."""
import argparse,ast,contextlib,copy,hashlib,importlib.util,io,json,os,shutil,sys,tempfile,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'control'),str(ROOT/'source')]
import contract as C,kernel_protocol as K,build_packet as B,admission as A,root_records as R
sys.path.insert(0,str(ROOT/'control/ownership'));from cleanup_lifecycle import Ownership

def synthetic_m4():
 tree=ast.parse((ROOT/'tests/cloud_controls.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='m4');space={'A':A,'hashlib':hashlib,'time':time};exec(compile(ast.Module(body=[node],type_ignores=[]),'<synthetic-m4-fixture>','exec'),space);return space['m4']()
def packet(root):
 accepted=root/'synthetic-m4.json';C.write(accepted,synthetic_m4());review=root/'synthetic-review.json';R.source_review(ROOT,review,'ACCEPT_SOURCE_AFTER_ROOT_REVIEW','6553e22ca52bee23f2378eac14998c19e857219f');built=B.build(ROOT,review,C.sha(review),accepted,C.sha(accepted),root/'packet');return Path(built['packet']),Path(built['archive']),built['packet_sha256']
def main(output):
 output=Path(output);C.require(not output.exists(),'FRESH_KERNEL_CONTROLS');rows=[]
 def check(name,fn,error=None):
  try:fn()
  except (ValueError,FileNotFoundError)as e:
   assert error is not None and error in str(e),(name,str(e),error);rows.append({'name':name,'status':'PASS','rejected':str(e)});return
  assert error is None,name;rows.append({'name':name,'status':'PASS'})
 with tempfile.TemporaryDirectory(prefix='bnb-m5-v2-kernel-',dir='/tmp')as td:
  root=Path(td);pkt,z,pktsha=packet(root);manifest=C.read(pkt/'manifest.json');birth=time.time()
  def interactive(name,prefix=True,stale=False):
   base=root/name;base.mkdir();shutil.copytree(pkt/'control',base/'control');shutil.copyfile(z,base/'payload.zip');sources={str(base/n):r['sha256']for n,r in manifest['files'].items()if n.startswith('control/')and n.endswith('.py')};modules={Path(n).stem:(str(base/n),r['sha256'])for n,r in manifest['files'].items()if n.startswith('control/')and n.endswith('.py')and not Path(n).stem.startswith('__')};modules['m5_packet_admission']=(str(base/'payload/source/admission.py'),manifest['files']['source/admission.py']['sha256'])
   unpack='import runpy,sys\nfrom pathlib import Path\n'+f'b=Path({str(base)!r});sys.path.insert(0,str(b/"control"));sys.argv=["remote.py","unpack","--base",str(b),"--packet-sha256",{pktsha!r},"--allocation-epoch",{str(birth)!r}]\nrunpy.run_path(str(b/"control/remote.py"),run_name="__main__")\n'
   readmit='import contract as C\nC.admit(b/"payload")\n'
   child='import subprocess\nassert subprocess.check_output([sys.executable,"-c",'+repr('import sys;sys.path.insert(0,'+repr(str(base/'payload/source'))+');import admission;admission.source();print(sys.dont_write_bytecode)')+'],text=True).strip()=="True"\nC.admit(b/"payload")\n'
   codes=[];requests=[]
   for phase,body in [('unpack',unpack),('readmit',readmit),('childinherit',child)]:
    req=K.request(body,phase,base,pktsha,birth,sources,artifact=base/'records/allocation.json',artifact_expected={'birth_epoch':birth,'packet_sha256':pktsha,'original_deadline_epoch':birth+3600});requests.append(req);codes.append(K.render(body,req,sources,modules)if prefix else body)
   runner=base/'runner.py';result=base/'result.json';foreign=base/'foreign.py';foreign.write_text('# foreign fixture\n')
   preface='import sys,os,io,json,contextlib\nfrom pathlib import Path\nsys.dont_write_bytecode=False;os.environ.pop("PYTHONDONTWRITEBYTECODE",None)\nlogs=[];failure=None\n'
   if stale:preface+='import types\nsys.modules["contract"]=types.SimpleNamespace(__file__='+repr(str(foreign))+')\n'
   runner.write_text(preface+'codes='+repr(codes)+'\nfor code in codes:\n stream=io.StringIO()\n try:\n  with contextlib.redirect_stdout(stream):exec(compile(code,"<interactive-kernel-fixture>","exec"),globals())\n except BaseException as e:failure={"type":type(e).__name__,"message":str(e)}\n logs.append(stream.getvalue())\n if failure:break\n'+f'Path({str(result)!r}).write_text(json.dumps({{"logs":logs,"error":failure,"pyc":[str(p.relative_to(Path({str(base)!r})))for p in Path({str(base)!r}).rglob("*.pyc")]}}))\n')
   (base/'owned').mkdir();owner=Ownership(base/'owned');streams=[(base/n).open('wb')for n in('stdout.raw','stderr.raw')]
   try:
    with owner.guard(60):
     process=owner.launch([sys.executable,str(runner)],record=base/'owned/ownership.json',cwd=base,env={k:v for k,v in os.environ.items()if k not in{'PYTHONDONTWRITEBYTECODE','PYTHONPATH','PYTHONHOME'}},stdout=streams[0],stderr=streams[1]);process.wait()
   finally:
    for f in streams:f.close()
   closure=owner.summary();assert not closure['errors'];assert process.returncode==0,(base/'stderr.raw').read_text();value=C.read(result)
   if prefix and not stale:
    assert value['error']is None and value['pyc']==[],value
    for text,req in zip(value['logs'],requests):K.verify(text,req,req['completion_path'],base/'records/allocation.json')
   elif stale:assert value['error']and'KERNEL_STALE_MODULE:contract'in value['error']['message'],value
   else:assert value['error']and value['error']['message']in('BYTECODE','EXACT_SCIENCE_CLOSURE')and value['pyc'],value
   rows.append({'name':name,'status':'PASS','observed_error':value['error'],'pyc_count':len(value['pyc']),'initial_dont_write_bytecode':False,'owned_closure':closure})
  interactive('old-unprefixed-interactive-bytecode-regression',False);interactive('correct-interactive-unpack-readmit-child-inheritance');interactive('foreign-cached-module',stale=True)
  base=root/'marker';base.mkdir();artifact=base/'phase.json';C.write(artifact,{'phase':'install','packet_sha256':pktsha});body='print("fixture body complete")\n';req=K.request(body,'install',base,pktsha,birth,{},artifact=artifact,artifact_expected={'phase':'install','packet_sha256':pktsha});receipt=base/'completion.json';value={**req,'status':'COMPLETED','artifact':K.row(artifact)};C.write(receipt,value);marker=K.VERSION+' '+json.dumps({'nonce':req['nonce'],'phase':req['phase'],'admitted_source_sha256':req['admitted_source_sha256'],'readback_sha256':C.sha(receipt)})+'\n'
  check('correct-independent-completion-and-artifact',lambda:K.verify(marker,req,receipt,artifact))
  for name,text,error in [('missing-marker-cli-zero','ValueError: failed\n','KERNEL_COMPLETION_LINE'),('duplicate-marker',marker+marker,'KERNEL_COMPLETION_LINE'),('post-marker-error',marker+'ValueError: late error\n','KERNEL_COMPLETION_LINE'),('post-marker-output',marker+'unexpected late output\n','KERNEL_COMPLETION_LINE'),('malformed-marker',K.VERSION+' not-json\n','KERNEL_MARKER_JSON')]:check(name,lambda text=text:K.verify(text,req,receipt,artifact),error)
  for field in ['nonce','phase','admitted_source_sha256']:
   r=json.loads(marker.split(' ',1)[1]);r[field]='wrong';check('wrong-'+field,lambda r=r:K.verify(K.VERSION+' '+json.dumps(r)+'\n',req,receipt,artifact),'KERNEL_COMPLETION_BINDING')
  check('missing-independent-completion',lambda:K.verify(marker,req,base/'missing',artifact),'KERNEL_READBACK_REGULAR')
  receipt.write_text('{}');check('wrong-independent-completion-sha',lambda:K.verify(marker,req,receipt,artifact),'KERNEL_COMPLETION_READBACK_SHA');C.write(receipt,value)
  original=artifact.read_bytes();artifact.write_text('{}');check('wrong-independent-artifact-sha',lambda:K.verify(marker,req,receipt,artifact),'KERNEL_ARTIFACT_READBACK_SHA');artifact.write_bytes(original)
  check('missing-independent-artifact',lambda:K.verify(marker,req,receipt),'KERNEL_ARTIFACT_READBACK_SHA')
  check('wrong-submitted-body-sha',lambda:K.render(body+'raise ValueError()\n',req,{},{}),'KERNEL_BODY_BINDING')
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
  oldreq={**req,'nonce':'wrong'};check('stale-prior-request-marker',lambda:K.verify(marker,oldreq,receipt,artifact),'KERNEL_COMPLETION_BINDING')
  def bootstrap(name,legacy_store=False):
   base=root/name;scope={};body='print("EXACT_ROOT_MARKER_MATCH")\n';req=K.request(body,'browser-marker',base,pktsha,birth,{})
   if legacy_store:req['completion_path']=str(base/'kernel-completions'/'marker.json')
   stream=io.StringIO()
   with contextlib.redirect_stdout(stream):exec(compile(K.render(body,req,{},{}),'<interactive-marker>','exec'),scope)
   K.verify(stream.getvalue(),req,req['completion_path']);ready='from pathlib import Path\n'+f'b=Path({str(base)!r});assert not b.exists()and not b.is_symlink();(b/"control/ownership").mkdir(parents=True)\n';request=K.request(ready,'readiness',base,pktsha,birth,{})
   if legacy_store:
    try:exec(compile(K.render(ready,request,{},{}),'<interactive-readiness>','exec'),scope)
    except AssertionError:rows.append({'name':'legacy-completion-store-creates-base-before-readiness','status':'PASS','observed':'CORRECT_MARKER_THEN_READINESS_FAILS'})
    else:raise AssertionError('legacy readiness unexpectedly passed')
   else:
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):exec(compile(K.render(ready,request,{},{}),'<interactive-readiness>','exec'),scope)
    K.verify(stream.getvalue(),request,request['completion_path']);assert(base/'control/ownership').is_dir();rows.append({'name':'actual-browser-marker-then-fresh-readiness-success','status':'PASS'})
  bootstrap('legacy-bootstrap-store',True);bootstrap('correct-bootstrap-store')
  disposed_bytes=sum(q.stat().st_size for q in root.rglob('*')if q.is_file());disposed_files=sum(q.is_file()for q in root.rglob('*'))
 C.write(output,{'status':'PASS','controls':rows,'count':len(rows),'fixtures_deleted':not Path(td).exists(),'disposed_bytes':disposed_bytes,'disposed_files':disposed_files,'synthetic_dependency_fixture_only':True,'provider_calls':0,'m5_status':'NOT_QUALIFIED','kernel_protocol_sha256':C.sha(ROOT/'control/kernel_protocol.py'),'science_manifest_sha256':C.science(ROOT)});print(json.dumps({'status':'PASS','controls':len(rows),'fixtures_deleted':True}))
if __name__=='__main__':main(Path(sys.argv[1]))
