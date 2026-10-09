"""Private R6 coordinator. Reviewed outer Ownership must own this parent's entire group."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'ownership'))
from lifecycle import durable_json
from state_child_owner import run_child
import protocol as S
import compiler_contract as C
import compiler_flags as F


def admission(args):
    B=S.load(HERE/'probe_backend.py','r6parentB');A=S.load(HERE/'transfer_admission.py','r6parentA')
    admitted,roots=A.admit(B,args.admission,args.admission_sha256,args.patch_manifest)
    S.require(admitted['bitsandbytes_tpu']['files']==S.spec()['plugin_python_files'],'M2_CANONICAL_NON_NESTED_SOURCE_VARIANT')
    return B,A,admitted,roots

def execute(args):
    output=Path(args.output);S.require(not output.exists() and os.getpid()==os.getpgid(0),'R6_FRESH_OWNED_PARENT')
    S.require(re.fullmatch('[0-9a-f]{32}',args.process_token) and math.isfinite(args.deadline_epoch) and time.time()+15<args.deadline_epoch,'R6_PARENT_TOKEN_DEADLINE')
    S.require(S.sha(args.manifest)==args.manifest_sha256,'R6_MANIFEST_HASH');manifest=S.read(args.manifest);S.require(manifest['kind']=='R6_DEVICE_CANDIDATE' and all(not (HERE/name).is_symlink() and (HERE/name).is_file() and S.sha(HERE/name)==value['sha256'] and (HERE/name).stat().st_size==value['bytes'] for name,value in manifest['sources'].items()),'R6_EXACT_PAYLOAD')
    compiler_source_pre=C.verify_manifest(args.manifest,args.manifest_sha256)
    policy=C.admit(args.compiler_policy,args.compiler_policy_sha256)
    S.require(args.compiler_dumps=='request','EXPLICIT_COMPILER_MODE_ONLY')
    F.require(S.sha(args.flag_preflight)==args.flag_preflight_sha256,'FLAG_PREFLIGHT_INDEPENDENT_HASH');flag_record=S.read(args.flag_preflight);F.admit(flag_record,manifest_sha=args.manifest_sha256,policy_sha=args.compiler_policy_sha256,dump_directory=flag_record['dump_directory'])
    oracle,cases=S.oracle(args.oracle,args.oracle_sha256,qualified=True);B,A,admitted,roots=admission(args)
    output.mkdir(parents=True);parent_path=output/'parent.json';child_token=secrets.token_hex(16);deadline=args.deadline_epoch-10
    parent={'kind':'R6_NATIVE_BOUNDARY_PARENT','status':'PARTIAL','pid':os.getpid(),'pgid':os.getpgid(0),'process_token':args.process_token,'deadline_epoch':args.deadline_epoch,'source_variant':S.spec()['source_variant'],'protocol_sha256':S.sha(HERE/'protocol.json'),'manifest_sha256':args.manifest_sha256,'source_pre':args.admission_sha256,'source_post':None,'source_admission_sha256':args.admission_sha256,'oracle_sha256':args.oracle_sha256,'oracle_remote_provenance':str(Path(args.oracle).resolve()),'compiler_policy_sha256':args.compiler_policy_sha256,'compiler_source_pre':compiler_source_pre,'compiler_source_post':None,'children':[],'m6_status':'NOT_QUALIFIED','compiler_body_memory_allocator':'NOT_QUALIFIED','optimized_executable_link':'NOT_QUALIFIED'}
    parent['compiler_flag_preflight']={'sha256':args.flag_preflight_sha256,'record':flag_record}
    expected={'kind':'R6_EXPECTED_PRELAUNCH','source_variant':S.spec()['source_variant'],'sources':S.sources(),'protocol_sha256':S.sha(HERE/'protocol.json'),'oracle_sha256':args.oracle_sha256,'oracle_artifacts':oracle['artifacts'],'input_sha256':{name:row['input_sha256'] for name,row in cases.items()},'source_admission_sha256':args.admission_sha256,'launch':{'parent_pid':parent['pid'],'parent_pgid':parent['pgid'],'parent_process_token':parent['process_token'],'process_token':child_token,'deadline_epoch':deadline}}
    expected_path=output/'expected.json';durable_json(expected_path,expected);parent['expected_sha256']=S.sha(expected_path);durable_json(parent_path,parent)
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)};interrupts={'launching':False,'pending':None}
    def interrupted(signum,frame):
        interrupts['pending']=signum
        if not interrupts['launching']:raise InterruptedError('R6_PARENT_SIGNAL:'+str(signum))
    for sig in previous:signal.signal(sig,interrupted)
    exit_code=2
    try:
        env=dict(os.environ);env['PJRT_DEVICE']='TPU';env.pop('JAX_PLATFORMS',None);env['PYTHONDONTWRITEBYTECODE']='1'
        S.require(not env.get('XLA_FLAGS'),'COMPILER_NO_INHERITED_FLAGS')
        compiler_root=output.absolute().with_name(output.name+'.compiler-private')
        S.require(compiler_root.resolve()==compiler_root and not compiler_root.exists(),'FRESH_COMPILER_SIBLING')
        dump=compiler_root/'raw-private'
        env['XLA_FLAGS']=' '.join(['--xla_dump_to='+str(dump)]+C.DUMP_FLAGS)
        P=S.load(HERE/'probe_precision.py','r6parentP');P.validate_environment(B,{key:env.get(key) for key in P.ENV_KEYS})
        parent['compiler_dump_request']={'mode':args.compiler_dumps,'flags':env.get('XLA_FLAGS'),'status':'REQUESTED_UNVERIFIED' if dump else 'NOT_REQUESTED','executable_link':'UNKNOWN','policy_sha256':args.compiler_policy_sha256,'sibling_evidence':str(compiler_root)};durable_json(parent_path,parent)
        argv=[sys.executable,'-B',str(HERE/'device_diagnostic.py'),'--output',str(output/'device'),'--kernel',str(HERE/'kernel.py'),'--source-pins',str(HERE/'source-pins.json'),'--backend-probe',str(HERE/'probe_backend.py'),'--admission-helper',str(HERE/'transfer_admission.py'),'--precision-probe',str(HERE/'probe_precision.py'),'--admission',str(args.admission),'--admission-sha256',args.admission_sha256,'--patch-manifest',str(args.patch_manifest),'--oracle',str(args.oracle),'--oracle-sha256',args.oracle_sha256,'--expected',str(expected_path),'--expected-sha256',parent['expected_sha256'],'--parent-pid',str(parent['pid']),'--parent-process-token',parent['process_token'],'--process-token',child_token,'--deadline-epoch',str(deadline)]
        compiler_expected=C.seal(policy,args.compiler_policy_sha256,compiler_root,parent,argv,child_token,deadline,parent['expected_sha256'])
        compiler_expected['xla_flags']=env['XLA_FLAGS']
        compiler_expected_path=compiler_root/'expected.json';durable_json(compiler_expected_path,compiler_expected)
        compiler_expected_sha=S.sha(compiler_expected_path)
        parent['compiler_expected_sha256']=compiler_expected_sha;durable_json(parent_path,parent)
        argv=[sys.executable,'-B',str(HERE/'compiler_exec.py'),'--contract',str(compiler_expected_path),'--contract-sha256',compiler_expected_sha,'--']+argv
        logs=output/'device-logs';logs.mkdir();descriptor=run_child(argv,output=logs,descriptor={'phase':'native-boundary','process_token':child_token},parent=parent,parent_path=parent_path,deadline=deadline,env=env,interrupts=interrupts,compiler=(compiler_expected,compiler_expected_path,compiler_expected_sha))
        descriptor.update(receipt='device/receipt.json',receipt_sha256=S.sha(output/'device/receipt.json'));A.verify_installed(B,admitted,roots)
        parent.update(status='COMPLETE',source_post=args.admission_sha256,child_science_exit_code=descriptor['exit_code'],artifacts=S.inventory(output,('parent.json',)));durable_json(parent_path,parent)
        exit_code=0 if descriptor['exit_code']==0 else 2
    except BaseException as error:
        parent.update(status='ERROR',error={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()},artifacts=S.inventory(output,('parent.json',)));durable_json(parent_path,parent);exit_code=2
    finally:
        try:
            A.verify_installed(B,admitted,roots);parent['source_post']=args.admission_sha256
            parent['compiler_source_post']=C.verify_manifest(args.manifest,args.manifest_sha256)
        except Exception as error:
            parent.update(status='ERROR',source_post=None,compiler_source_post=None,source_post_error={'type':type(error).__name__,'message':str(error)});exit_code=2
        parent['artifacts']=S.inventory(output,('parent.json',));durable_json(parent_path,parent)
        for sig,handler in previous.items():signal.signal(sig,handler)
    return exit_code

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','manifest','manifest-sha256','admission','admission-sha256','patch-manifest','oracle','oracle-sha256','process-token','compiler-policy','compiler-policy-sha256','flag-preflight','flag-preflight-sha256'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--deadline-epoch',type=float,required=True);parser.add_argument('--compiler-dumps',choices=['request'],required=True);raise SystemExit(execute(parser.parse_args()))
