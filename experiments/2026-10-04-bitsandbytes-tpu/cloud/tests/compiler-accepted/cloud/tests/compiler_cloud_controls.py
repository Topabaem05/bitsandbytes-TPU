"""Portable isolated cloud controls. No provider, compiler, or qualified device science."""
import argparse,copy,hashlib,importlib.util,json,os,shutil,subprocess,sys,tarfile,time,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;CLOUD=HERE.parent;PREP=CLOUD.parent
sys.path.insert(0,str(CLOUD));import remote,owner,compiler_cloud_contract as CC,compiler_recovery as CR
from unittest.mock import patch

def run(packet,output):
 assert sys.version_info[:3]==(3,12,14),'USE_PYTHON_3_12_14'
 packet=Path(packet).resolve();output=Path(output).resolve();output.mkdir(exist_ok=False);rows=[]
 def yes(name,fn):
  value=fn();rows.append({'case':name,'status':'PASS'});return value
 def no(name,fn,reason=None):
  try:fn()
  except (ValueError,AssertionError,PermissionError,FileNotFoundError)as e:
   if reason is not None:assert reason in str(e),(name,str(e),reason)
   rows.append({'case':name,'status':'PASS','rejected':type(e).__name__+': '+str(e)})
  else:raise AssertionError('FALSE_ACCEPT:'+name)
 m=yes('exact34-production-shaped-preflight',lambda:owner.preflight(packet,CC.sha(packet/'payload.zip')))
 assert m['experiment']==CC.MODE and len(CC.SOURCES)==34
 def wrong_manifest(key,val):
  bad=copy.deepcopy(m);bad[key]=val;return remote.verify_experiment(bad)
 no('native-mode-cannot-admit-compiler34',lambda:wrong_manifest('experiment','m6-native-boundary'),'NATIVE_REVIEWED_SCOPE')
 no('policy-hash-unknown',lambda:wrong_manifest('compiler_policy_sha256','0'*64),'COMPILER_EXACT_POLICY')
 no('compiler-source-unknown',lambda:wrong_manifest('compiler_manifest_sha256','0'*64),'NATIVE_REVIEWED_SCOPE')
 no('wrong-source-variant',lambda:wrong_manifest('compiler_source_variant','nested-v1'),'NATIVE_REVIEWED_SCOPE')
 bad=copy.deepcopy(m);bad['files']['native/kernel.py']['sha256']='0'*64
 no('known-kernel-only',lambda:remote.verify_experiment(bad),'COMPILER_EXACT_34_SOURCES')
 bad=copy.deepcopy(m);bad['precision']='high';no('fixed-highest',lambda:remote.verify_experiment(bad),'TRANSFER_REVIEWED_SCOPE')
 # Actual empty-module-cache runpy import, unrelated cwd and no cloud/native PYTHONPATH.
 script=output/'fresh-runpy.py';script.write_text('import runpy,sys,json\nsys.path=[p for p in sys.path if "r6-compiler" not in p]\nr=runpy.run_path(sys.argv[1],run_name="isolated_remote")\nm=json.load(open(sys.argv[2]));assert r["verify_experiment"](m)=="m6-compiler-content"\nassert "torch" not in sys.modules and "jax" not in sys.modules\nprint("PASS_FRESH_RUNPY_NO_CLIENT")\n')
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None)
 rec=subprocess.run([sys.executable,'-B',str(script),str(CLOUD/'remote.py'),str(packet/'manifest.json')],cwd=output,env=env,capture_output=True,text=True,timeout=30)
 (output/'fresh-runpy.stdout').write_text(rec.stdout);(output/'fresh-runpy.stderr').write_text(rec.stderr);assert rec.returncode==0,rec.stderr;yes('fresh-runpy-exact-sibling-no-client',lambda:True)
 # real group leader → direct child → same-PID exec. Only fixture writer runs.
 parent=PREP/'fixtures/topology_parent.py'
 groups=[]
 def os_case(name,mode,seconds=5):
  out=output/name;out.mkdir();deadline=time.time()+seconds
  argv=[sys.executable,'-B',str(parent),'--output',str(out/'native'),'--mode',mode,'--process-token','1'*32,'--deadline-epoch',str(deadline),'--compiler-dumps','request','--compiler-policy-sha256','e'*64]
  rec=remote.run_step(out,'12-native-parent',argv,deadline+3,seconds+3,cwd=out)
  outer=CC.read(out/'steps/12-native-parent/ownership.json');CR.closed(outer,live=True);groups.append({'name':name,'pid':outer['pid'],'pgid':outer['pgid'],'cleanup':outer['cleanup']})
  return out,outer,rec
 out,outer,rec=os_case('correct-owned-hook','correct');assert rec['status']=='PASS'
 # Hook uses real installed archive/source contract, but fixture writes no compiler/device work.
 command=[sys.executable,'-B',str(CLOUD/'compiler_postclosure.py'),'--payload',str(packet),'--out',str(out),'--outer-sha256',CC.sha(out/'steps/12-native-parent/ownership.json')]
 r=remote.run_step(out,'12b-compiler-postclosure',command,time.time()+30,30,cwd=output)
 assert r['status']=='PASS',CC.read(out/'steps/12b-compiler-postclosure/result.json')
 hook=CC.read(out/'compiler-capture.json');assert hook['compiler_capture_status']=='SEALED_BOUNDED_RAW_CAPTURE' and hook['status']=='POSTCLOSURE_SEALED'
 yes('real-owner-closure-before-hook',lambda:True)
 def recovered(name,receipt=hook,mutate=None):
  dest=output/name;dest.mkdir();shutil.copyfile(out/'compiler-inventory.json',dest/'inventory.json');shutil.copyfile(out/'compiler-evidence.zip',dest/'evidence.zip')
  sealed=dict(receipt)
  if mutate:mutate(dest,sealed)
  return CR.recover(dest,sealed)
 recovered_root,inv=yes('separate-bounded-archive-exact-recovery',lambda:recovered('correct-recovery'))
 assert inv['selected_executable_link']==inv['allocator_peak']=='UNKNOWN'
 assert CC.read(recovered_root/'monitor.json')['terminal_snapshot']==CC.read(recovered_root/'postclosure.json')['snapshot']
 no('archive-byte-alteration',lambda:recovered('bad-archive',mutate=lambda d,r:(d/'evidence.zip').write_bytes((d/'evidence.zip').read_bytes()+b'altered')),'COMPILER_RECOVERY_SEAL')
 no('inventory-byte-alteration',lambda:recovered('bad-inventory',mutate=lambda d,r:(d/'inventory.json').write_text('{}')),'COMPILER_RECOVERY_SEAL')
 def inventory_edit(d,r,fn):
  inv=CC.read(d/'inventory.json');fn(inv);CC.write(d/'inventory.json',inv);r['compiler_inventory_sha256']=CC.sha(d/'inventory.json')
 no('false-hard-total-quota',lambda:recovered('bad-hard',mutate=lambda d,r:inventory_edit(d,r,lambda v:v.__setitem__('hard_aggregate_quota','ENFORCED'))),'COMPILER_RECOVERY_STATUS')
 no('unsafe-directory-path',lambda:recovered('bad-path',mutate=lambda d,r:inventory_edit(d,r,lambda v:v.__setitem__('directories',['raw-private/../../escape']))),'COMPILER_RECOVERY_DIRECTORY_PATH')
 no('raw-under-metadata-only',lambda:recovered('bad-metadata',mutate=lambda d,r:(inventory_edit(d,r,lambda v:v.update(status='METADATA_ONLY_CAPTURE_FAILED',error='OVERFLOW')),r.update(compiler_capture_status='METADATA_ONLY_CAPTURE_FAILED',compiler_capture_error='OVERFLOW'))),'COMPILER_RECOVERY_RAW_SCOPE')
 # Live process cannot authorize postclosure. A genuinely registered group remains present here.
 own=remote.Ownership(output/'live-owner');(output/'live-owner').mkdir()
 with own.guard(10):
  child=own.launch([sys.executable,'-B','-c','import time;time.sleep(3)'],record=output/'live-owner/ownership.json')
  fake=copy.deepcopy(outer);fake['pid']=fake['pgid']=child.pid
  no('live-group-cannot-authorize-hook',lambda:CR.closed(fake,live=True),'COMPILER_GROUP_STILL_PRESENT');child.terminate();child.wait()
 live_record=CC.read(output/'live-owner/ownership.json');CR.closed(live_record,live=True);groups.append({'name':'live-owner','pid':child.pid,'pgid':child.pid,'cleanup':live_record['cleanup']})
 bad=copy.deepcopy(outer);bad['cleanup']['group_absence']='UNVERIFIED';no('unclosed-group-no-hook',lambda:CR.closed(bad),'COMPILER_EXTERNAL_GROUP_CLOSURE')
 # Empty directories must survive exact recovery; snapshots count entries, not only files.
 raw=out/'native.compiler-private/raw-private';(raw/'empty').mkdir()
 with patch.object(sys,'path',[str(packet/'native'),*sys.path]):
  import compiler_verify as V
 post=V.postclosure(out/'native.compiler-private',outer);CC.write(out/'native.compiler-private/postclosure.json',post)
 sealed=CR.seal(out,outer,{'status':'POSTCLOSURE_SEALED','error':None})
 root,inv=yes('empty-directory-exact-retention',lambda:recovered('empty-directory-recovery',receipt=sealed));assert(root/'raw-private/empty').is_dir()
 # Closed overflow record is preserved; raw omission never authorizes supplemental verifier.
 fail_out,fail_outer,fail_rec=os_case('count-overflow','count-overflow')
 # Overshoot may return under policy limits after shutdown; label child failure regardless of current raw size.
 fail_hook={'status':'POSTCLOSURE_FAILED','error':'FIXTURE_OBSERVED_OVERFLOW_UNQUALIFIED'}
 fail_receipt=CR.seal(fail_out,fail_outer,fail_hook)
 target=output/'overflow-recovery';target.mkdir();shutil.copyfile(fail_out/'compiler-evidence.zip',target/'evidence.zip');shutil.copyfile(fail_out/'compiler-inventory.json',target/'inventory.json')
 _,failure_inv=yes('overflow-metadata-retained-no-acceptance',lambda:CR.recover(target,fail_receipt));assert failure_inv['raw_recovery']=='OMITTED_UNQUALIFIED'
 assert CR.readback_missing(fail_receipt,output), 'OVERFLOW_FALSE_READBACK_READY'
 assert CC.read(fail_out/'native.compiler-private/monitor.json')['overflow_observation'] is not None
 # External run_step remains the final owner when direct child TERM cleanup cannot finish.
 timeout_out=output/'external-timeout';timeout_out.mkdir()
 code='import signal,subprocess,sys,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);subprocess.Popen([sys.executable,"-B","-c","import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)"]);time.sleep(30)'
 timeout_record=remote.run_step(timeout_out,'12-native-parent',[sys.executable,'-B','-c',code],time.time()+.3,.3,cwd=timeout_out)
 timeout_owner=CC.read(timeout_out/'steps/12-native-parent/ownership.json');CR.closed(timeout_owner,live=True)
 assert timeout_record['status']=='BLOCKED' and timeout_record['error']['type']=='DeadlineExceeded'
 groups.append({'name':'external-timeout','pid':timeout_owner['pid'],'pgid':timeout_owner['pgid'],'cleanup':timeout_owner['cleanup']})
 yes('external-deadline-complete-group-closure',lambda:True)
 assert any(any(s.get('signal')==9 and s.get('result')=='SENT' for s in g['signals']) for g in timeout_record['cleanup']['groups'])
 yes('ignored-TERM-descendant-external-KILL',lambda:True)
 # Main archive excludes sibling raw ONLY in explicit compiler mode.
 CC.write(out/'receipt.json',{'experiment':CC.MODE,'status':'FIXTURE_ONLY'})
 packed=remote.package(out);names=CC.read(out/'archive-members.json');assert not any(n.startswith('native.compiler-private/') or n in {'compiler-evidence.zip','compiler-inventory.json'}for n in names)
 yes('ordinary-science-archive-separated-from-compiler',lambda:True)
 CC.write(out/'receipt.json',{'experiment':'m6-native-boundary','status':'FIXTURE_ONLY'});remote.package(out);names=CC.read(out/'archive-members.json');assert any(n.startswith('native.compiler-private/')for n in names)
 yes('native-v1-package-behavior-unchanged',lambda:True)
 # Complete retained fields do not make an incomplete native receipt acceptable.
 fakeowner={'recovered_oracle_sha256':'a'*64};fakereceipt={'status':'BLOCKED','error':'ModuleNotFoundError: primary'}
 missing=owner.native_readback_missing(fakeowner,fakereceipt,output,None);assert missing and fakereceipt['error']=='ModuleNotFoundError: primary';yes('early-failure-primary-retained-verifier-skipped',lambda:True)
 review={'status':'PASS','controls':rows,'registered_groups':groups,'provider_calls':0,'actual_compiler':'NOT_RUN','actual_TPU':'NOT_RUN','qualified_CPU':'NOT_RUN','native_verifier_bypass':'NONE','scope':'ISOLATED_SOURCE_ARCHIVE_OS_ONLY','m6':'NOT_QUALIFIED'}
 CC.write(output/'results.json',review);print(json.dumps({'status':'PASS','controls':len(rows),'groups':len(groups)}));return review
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--packet',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.packet,a.output)
