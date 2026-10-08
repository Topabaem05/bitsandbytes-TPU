"""Fresh owned qualified CPU oracle. No candidate package or device-client import."""
import argparse
import os
import platform
import sys
import time
import traceback
from pathlib import Path
import torch
import contract as C
import oracle_math as M


def prepare(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    started=time.time();runtime={'platform':platform.system(),'machine':platform.machine(),'python':platform.python_version(),'torch':str(torch.__version__)}
    qualified=runtime=={'platform':'Linux','machine':'x86_64','python':'3.12.14','torch':'2.9.0+cpu'}
    seal={'kind':'PUBLIC_PALLAS_CPU_ORACLE','status':'PARTIAL','scientific_variant':C.VARIANT,'source_variant':'pallas-forward-mosaic7-gather-bf16-fp32-v1','pid':os.getpid(),'pgid':os.getpgid(0),'parent_pid':os.getppid(),'process_token':args.process_token,'deadline_epoch':args.deadline_epoch,'started_epoch':started,'runtime':runtime,'qualified_runtime':qualified,'admission_sha256':C.ADMISSION_SHA,'protocol_sha256':C.PROTOCOL_SHA,'scope':'QUALIFIED_LINUX_CPU' if qualified else 'UNQUALIFIED_LOCAL_CPU_REHEARSAL'}
    C.write(root/'oracle-seal.json',seal)
    try:
        C.owner(os.getpid(),os.getpgid(0),os.getppid(),args.process_token,args.deadline_epoch)
        C.require(started<args.deadline_epoch,'CPU_DEADLINE')
        C.require(qualified or args.unqualified_local,'QUALIFIED_CPU_RUNTIME_REQUIRED')
        forbidden=('jax','torch_xla','bitsandbytes','bitsandbytes_tpu','bitsandbytes_tpu_pallas')
        C.require(not any(n in sys.modules for n in forbidden),'CPU_NO_PACKAGE_DEVICE_IMPORT')
        a,p,upstream=C.admission(args.admission,args.protocol,args.package_root,args.upstream_source,args.repo_root)
        seal.update(runtime_wheel_records=C.runtime_wheels(a,args.repo_root,args.runtime_wheels,qualified=qualified),installed_runtime_metadata=C.installed_runtime_metadata(args.repo_root) if qualified else None,scientific_sources=C.scientific_sources(),wheel_records=C.wheels(a,args.plugin_wheel,args.upstream_wheel),upstream_python_map_sha256=C.digest(a['upstream_python_files']),source_pre=C.ADMISSION_SHA)
        torch.set_num_threads(1);decode,code=M.decode_original(upstream)
        for case in p['cases']:
            C.require(time.time()<args.deadline_epoch,'CPU_DEADLINE')
            C.write(root/'rows'/(case['id']+'.json'),M.expected(case,decode,code))
        C.admission(args.admission,args.protocol,args.package_root,args.upstream_source,args.repo_root)
        C.require(not any(n in sys.modules for n in forbidden),'CPU_NO_PACKAGE_DEVICE_IMPORT')
        seal.update(status='COMPLETE',source_post=C.ADMISSION_SHA,no_package_device_import=True,thread_count=torch.get_num_threads(),case_ids=[c['id'] for c in p['cases']])
    except Exception as error:
        seal.update(status='ERROR',error={'type':type(error).__name__,'message':str(error)});(root/'error.log').write_text(traceback.format_exc())
    seal['finished_epoch']=time.time();seal['artifacts']=C.inventory(root,('oracle-seal.json',));C.write(root/'oracle-seal.json',seal)
    return 0 if seal['status']=='COMPLETE' else 2

def source_args(p):
    p.add_argument('--runtime-wheels')
    for name in ('admission','protocol','package-root','upstream-source','repo-root','plugin-wheel','upstream-wheel'):p.add_argument('--'+name,required=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);source_args(p)
    for name in ('output','process-token'):p.add_argument('--'+name,required=True)
    p.add_argument('--deadline-epoch',type=float,required=True);p.add_argument('--unqualified-local',action='store_true');raise SystemExit(prepare(p.parse_args()))
