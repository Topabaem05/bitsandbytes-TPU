"""Pure post-closure hook. The owner must first reap the native leader and its group."""
import argparse,sys,os,json
from pathlib import Path
import compiler_cloud_contract as CC
import compiler_recovery as CR

def capture(payload,out,outer_sha):
 payload=Path(payload);out=Path(out);m=CC.read(payload/'manifest.json');CC.payload(payload,m)
 outer_path=out/'steps/12-native-parent/ownership.json';CC.require(CC.sha(outer_path)==outer_sha,'COMPILER_OUTER_INDEPENDENT_HASH');outer=CC.read(outer_path);CR.closed(outer,live=True)
 sys.path.insert(0,str(payload/'native'));import compiler_verify as V
 CC.require(Path(V.__file__).resolve()==(payload/'native/compiler_verify.py').resolve(),'COMPILER_POSTCLOSURE_HELPER_PATH')
 root=out/'native.compiler-private';hook={'status':'POSTCLOSURE_FAILED','error':None,'raw_recovery':'OMITTED_UNQUALIFIED'}
 try:
  p=V.postclosure(root,outer);CC.write(root/'postclosure.json',p);hook={'status':'POSTCLOSURE_SEALED','error':None,'compiler_expected_sha256':CC.sha(root/'expected.json'),'compiler_postclosure_sha256':CC.sha(root/'postclosure.json')}
 except Exception as e:hook['error']=type(e).__name__+': '+str(e)
 CC.write(out/'compiler-capture.json',hook);result={**hook,**CR.seal(out,outer,hook)};CC.write(out/'compiler-capture.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ('payload','out','outer-sha256'):p.add_argument('--'+n,required=True)
 a=p.parse_args();r=capture(a.payload,a.out,a.outer_sha256);print(json.dumps(r,sort_keys=True));raise SystemExit(0 if r['status']=='POSTCLOSURE_SEALED' else 2)
