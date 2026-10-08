"""Prepare independent CPU records before a device child. No installed source or class changes."""
import argparse
import os
import platform
from pathlib import Path
import re
import sys
import traceback
import protocol as S
import torch


def prepare(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    seal={'kind':'R6_CPU_ORACLE','status':'PARTIAL','protocol_sha256':S.sha(S.HERE/'protocol.json'),'protocol_source_sha256':S.sha(S.__file__),'sources':S.sources(),'pid':os.getpid(),'pgid':os.getpgid(0),'parent_pid':os.getppid(),'process_token':args.process_token,'source_variant':S.spec()['source_variant'],'plugin_python_files':S.spec()['plugin_python_files'],'profile_sha256':S.PROFILE_SHA,'runtime_lock_sha256':S.spec()['runtime_lock_sha256'],'upstream_default_sha256':S.DEFAULT_SHA,'upstream_utils_sha256':S.UTILS_SHA,'cpu_runtime':{'python':'.'.join(map(str,sys.version_info[:2])),'torch':torch.__version__,'platform':sys.platform,'machine':platform.machine()},'qualified_runtime':False,'no_xla_jax_import':False,'case_ids':S.spec()['case_ids']}
    S.write(root/'oracle-seal.json',seal)
    try:
        S.require(re.fullmatch('[0-9a-f]{32}',args.process_token),'CPU_PROCESS_TOKEN')
        S.require(not any(name in sys.modules for name in ('torch_xla','jax','bitsandbytes','bitsandbytes_tpu')),'CPU_ORACLE_NO_CLIENT_PACKAGE_IMPORT')
        qualified=seal['cpu_runtime']=={'python':'3.12','torch':'2.9.0+cpu','platform':'linux','machine':'x86_64'}
        S.require(qualified or args.unqualified_local,'QUALIFIED_CPU_RUNTIME_REQUIRED');seal['qualified_runtime']=qualified
        B=S.load(S.HERE/'probe_backend.py','r6oracleB');A=S.load(S.HERE/'transfer_admission.py','r6oracleA');manifest=A.load_manifest(B,args.patch_manifest)
        source=Path(args.upstream_source);B.verify_source_tree(source,A.python_files(manifest['post_patch_package_files']));seal['upstream_python_map_sha256']=S.V.digest(A.python_files(manifest['post_patch_package_files']));S.require(seal['upstream_python_map_sha256']==S.POST_PATCH_PYTHON_MAP_SHA,'ORACLE_UPSTREAM_FULL_SOURCE_MAP')
        decode,code=S.original_decode(source);torch.set_num_threads(1)
        for case in seal['case_ids']:
            a,b,c=S.inputs(case);dense=decode(b.reshape(-1),c,code,64,(256,512),a.dtype);reference=torch.nn.functional.linear(a,dense)
            values={'activation':S.tensor(a),'packed':S.tensor(b),'scales':S.tensor(c)};output=S.tensor(reference);S.V.array(output)
            S.write(root/(case+'.json'),{'case':case,'status':'COMPLETE','inputs':values,'input_sha256':S.V.digest(values),'cpu_reference':output,'oracle_method':S.spec()['cpu_oracle']})
        B.verify_source_tree(source,A.python_files(manifest['post_patch_package_files']));S.require(not any(name in sys.modules for name in ('torch_xla','jax','bitsandbytes','bitsandbytes_tpu')),'CPU_ORACLE_NO_CLIENT_PACKAGE_IMPORT')
        seal.update(status='COMPLETE',no_xla_jax_import=True,source_pre=seal['upstream_python_map_sha256'],source_post=seal['upstream_python_map_sha256'])
    except Exception as error:seal.update(status='ERROR',error={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
    seal['artifacts']=S.inventory(root,('oracle-seal.json',));S.write(root/'oracle-seal.json',seal)
    return 0 if seal['status']=='COMPLETE' else 2
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('upstream-source','patch-manifest','output','process-token'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--unqualified-local',action='store_true');raise SystemExit(prepare(parser.parse_args()))
