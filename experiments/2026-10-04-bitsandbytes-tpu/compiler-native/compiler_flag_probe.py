"""Fresh fixed-libTPU parser/backend probe. No tensors, native kernel, or compiler memory claim."""
import argparse,importlib.metadata,json,os,platform,re,resource,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'ownership'));from lifecycle import durable_json
import compiler_contract as C,compiler_flags as F

def execute(a):
 output=Path(a.output);dump=Path(a.dump_directory);F.require(not output.exists()and not output.is_symlink(),'FLAG_PREFLIGHT_FRESH_OUTPUT');F.require(dump.is_dir()and not dump.is_symlink()and not any(dump.iterdir()),'FLAG_PREFLIGHT_FRESH_DUMP')
 F.require(not os.environ.get('XLA_FLAGS')and os.environ.get('PJRT_DEVICE')=='TPU'and not os.environ.get('TPU_LIBRARY_PATH')and not os.environ.get('PTXLA_TPU_LIBRARY_PATH')and not os.environ.get('LIBTPU_INIT_ARGS'),'FLAG_PREFLIGHT_ENVIRONMENT')
 F.require(re.fullmatch('[0-9a-f]{32}',a.process_token)and os.getpid()==os.getpgrp(),'FLAG_PREFLIGHT_OWNED_TOKEN');policy=C.admit(a.policy,a.policy_sha256)
 resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_FSIZE,(policy['limits']['file_bytes'],)*2)
 request=F.flags(dump);os.environ['XLA_FLAGS']=request;os.environ.pop('JAX_PLATFORMS',None)
 record={'kind':'R6_LIBTPU_DUMP_FLAG_PREFLIGHT_V1','status':'PARTIAL_BEFORE_BACKEND_INITIALIZATION','system':platform.system(),'machine':platform.machine(),'python':platform.python_version(),'requested_backend':os.environ.get('PJRT_DEVICE'),'pid':os.getpid(),'pgid':os.getpgrp(),'parent_pid':os.getppid(),'process_token':a.process_token,'started_epoch':time.time(),'manifest_sha256':C.sha(HERE/'manifest.json'),'policy_sha256':a.policy_sha256,'dump_directory':str(dump),'xla_flags':request,'native_case_count':0,'tensor_computation_count':0,'scope':'PARSER_AND_BACKEND_INITIALIZATION_ONLY_NOT_NATIVE_SCIENCE'};durable_json(output,record)
 try:
  F.require(record['system']=='Linux'and record['machine']=='x86_64'and record['python']=='3.12.14','FLAG_PREFLIGHT_RUNTIME')
  record['versions']={name:importlib.metadata.version(name)for name in F.VERSIONS};F.require(record['versions']==F.VERSIONS,'FLAG_PREFLIGHT_FIXED_LIBRARY')
  library=Path(importlib.metadata.distribution('libtpu').locate_file('libtpu/libtpu.so'));F.require(library.is_file()and not library.is_symlink(),'FLAG_PREFLIGHT_LIBRARY_ORIGIN');record['library_sha256']=F.sha(library);F.require(record['library_sha256']==F.LIBTPU_SHA,'FLAG_PREFLIGHT_FIXED_LIBRARY');durable_json(output,record)
  import torch_xla
  import torch_xla.runtime as xr
  # This initializes the actual parser/client. It constructs no tensors or graphs.
  record['devices']=torch_xla._XLAC._xla_get_devices();record['observed_backend']=xr.device_type();record['torch_xla_init_sha256']=F.sha(torch_xla.__file__);record['effective_xla_flags']=os.environ.get('XLA_FLAGS');record['effective_libtpu_init_args']=os.environ.get('LIBTPU_INIT_ARGS')
  F.require(os.environ.get('PTXLA_TPU_LIBRARY_PATH')==str(library),'FLAG_PREFLIGHT_LIBRARY_ORIGIN')
  record.update(status='LIBTPU_FLAGS_RECOGNIZED_BACKEND_INIT_ONLY',finished_epoch=time.time());F.admit(record,manifest_sha=C.sha(HERE/'manifest.json'),policy_sha=a.policy_sha256,dump_directory=dump)
 except Exception as e:record.update(status='BLOCKED',error={'type':type(e).__name__,'message':str(e)},finished_epoch=time.time())
 durable_json(output,record);print(json.dumps({'status':record['status'],'scope':record['scope']}));return 0 if record['status']=='LIBTPU_FLAGS_RECOGNIZED_BACKEND_INIT_ONLY'else 2
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ('output','dump-directory','policy','policy-sha256','process-token'):p.add_argument('--'+name,required=True)
 raise SystemExit(execute(p.parse_args()))
