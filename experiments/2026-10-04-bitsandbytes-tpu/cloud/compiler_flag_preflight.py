"""Owned exact-library flag preflight, separate from native science and compiler inventory."""
import importlib.util,os,secrets,sys,time,shutil
from pathlib import Path
_spec=importlib.util.spec_from_file_location('_flag_preflight_cc',Path(__file__).resolve().parent/'compiler_cloud_contract.py');CC=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(CC)

def execute(base,payload,out,python,deadline,*,run_step):
 payload=Path(payload);out=Path(out);m=CC.read(payload/'manifest.json');CC.payload(payload,m);sys.path.insert(0,str(payload/'native'))
 import compiler_flags as F
 CC.require(Path(F.__file__).resolve()==(payload/'native/compiler_flags.py').resolve(),'FLAG_PREFLIGHT_HELPER_PATH')
 spec=importlib.util.spec_from_file_location('_flag_dump_observation',payload/'native/dump_observation.py');D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
 dump=Path(base)/'compiler-flag-preflight-private';dump.mkdir(mode=0o700,exist_ok=False);s=dump.lstat();identity=(s.st_dev,s.st_ino);policy=CC.read(payload/'native/compiler-policy.json');token=secrets.token_hex(16)
 CC.require(shutil.disk_usage(dump).free>=policy['minimum_free_bytes'],'FLAG_PREFLIGHT_AVAILABLE_SPACE')
 def observe():return D.observe(dump,identity,policy['limits'])
 argv=[str(python),'-B',str(payload/'native/compiler_flag_probe.py'),'--output',str(out/'compiler-flag-preflight.json'),'--dump-directory',str(dump),'--policy',str(payload/'native/compiler-policy.json'),'--policy-sha256',CC.POLICY_SHA,'--process-token',token]
 result=run_step(out,'11d-compiler-flags',argv,deadline,90,cwd=payload,tpu=True,flag_dump_observer=observe)
 CC.require(result['status']=='PASS'and result['exit_code']==0 and not result['cleanup']['errors'],'FLAG_PREFLIGHT_STEP_PASS')
 record=CC.read(out/'compiler-flag-preflight.json');ownership=CC.read(out/'steps/11d-compiler-flags/ownership.json');F.owned(record,ownership,result,manifest_sha=CC.MANIFEST_SHA,policy_sha=CC.POLICY_SHA,dump_directory=dump)
 CC.require(record.get('process_token')==token,'FLAG_PREFLIGHT_TOKEN_BINDING')
 return {'compiler_flag_preflight_sha256':CC.sha(out/'compiler-flag-preflight.json'),'compiler_flag_preflight_status':record['status'],'compiler_flag_preflight_library_sha256':record['library_sha256'],'compiler_flag_preflight_outer_sha256':CC.sha(out/'steps/11d-compiler-flags/ownership.json')}
