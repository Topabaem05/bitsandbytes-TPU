"""Read-only independent recovery. Host Torch performs CPU arithmetic, not device work."""
import argparse
import math
from pathlib import Path
import torch
import contract as C
import oracle_math as M
from prepare_public_oracle import source_args


def verify(args,*,qualified=True):
    root=Path(args.oracle);C.require(C.sha(root/'oracle-seal.json')==args.oracle_sha256,'CPU_ORACLE_SEAL_SHA')
    s=C.read(root/'oracle-seal.json');a,p,upstream=C.admission(args.admission,args.protocol,args.package_root,args.upstream_source,args.repo_root)
    C.require(s.get('kind')=='PUBLIC_PALLAS_CPU_ORACLE' and s.get('status')=='COMPLETE' and not s.get('error'),'CPU_TERMINAL')
    C.require(s.get('scientific_variant')==C.VARIANT and s.get('source_variant')=='pallas-forward-mosaic7-gather-bf16-fp32-v1','CPU_VARIANT')
    C.require(s.get('admission_sha256')==s.get('source_pre')==s.get('source_post')==C.ADMISSION_SHA and s.get('protocol_sha256')==C.PROTOCOL_SHA,'CPU_SOURCE_PROTOCOL')
    C.require(s.get('scientific_sources')==C.scientific_sources() and s.get('upstream_python_map_sha256')==C.digest(a['upstream_python_files']),'CPU_FULL_SCIENTIFIC_SOURCE')
    C.owner(s.get('pid'),s.get('pgid'),s.get('parent_pid'),s.get('process_token'),s.get('deadline_epoch'))
    C.require(all(type(s.get(n))in(int,float) and math.isfinite(s[n]) for n in ('started_epoch','finished_epoch')) and s['started_epoch']<=s['finished_epoch']<=s['deadline_epoch'],'CPU_FINITE_TIME')
    C.require(s.get('qualified_runtime') in (True,False),'CPU_QUALIFICATION_TYPED')
    if qualified:
        C.require(s.get('qualified_runtime') is True and s.get('scope')=='QUALIFIED_LINUX_CPU' and s.get('runtime')=={'platform':'Linux','machine':'x86_64','python':'3.12.14','torch':'2.9.0+cpu'},'CPU_QUALIFIED_RUNTIME')
    C.require(s.get('no_package_device_import') is True and s.get('thread_count')==1,'CPU_IMPORT_THREADS')
    C.require(s.get('case_ids')==[c['id'] for c in p['cases']],'CPU_CASE_MATRIX')
    C.require(set(s['artifacts'])=={'rows/'+c['id']+'.json' for c in p['cases']} and C.inventory(root,('oracle-seal.json',))==s['artifacts'],'CPU_COMPLETE_ARTIFACT_INVENTORY')
    C.require(s.get('wheel_records')==C.wheels(a,args.plugin_wheel,args.upstream_wheel),'CPU_BUILT_WHEEL_BINDING')
    C.require(s.get('runtime_wheel_records')==C.runtime_wheels(a,args.repo_root,args.runtime_wheels,qualified=qualified),'CPU_RUNTIME_WHEEL_BODIES')
    if qualified:C.require(s.get('installed_runtime_metadata')=={w['name']:{'version':w['version'],'metadata_sha256':w['metadata_sha256']} for w in C.read(Path(args.repo_root)/'experiments/2026-10-04-bitsandbytes-tpu/runtime/requirements.lock.json')['wheels']},'CPU_INSTALLED_RUNTIME_METADATA')
    torch.set_num_threads(1);decode,code=M.decode_original(upstream);rows={}
    for case in p['cases']:
        row=C.read(root/'rows'/(case['id']+'.json'));expected=M.expected(case,decode,code)
        C.require(row==expected,'CPU_INDEPENDENT_REPLAY:'+case['id']);rows[case['id']]=row
    if qualified:verify_owned_cpu(args,s)
    return {'record_validation':'PASS','oracle_status':'INDEPENDENTLY_REPLAYED','qualified_runtime':s['qualified_runtime'],'case_count':15,'oracle_sha256':args.oracle_sha256,'native_acceptance':'REQUIRED_ROOT_ACCEPTED_BOUNDED_NATIVE_RECORDS','device_qualification':'NOT_QUALIFIED'},rows

def verify_owned_cpu(args,seal):
    C.require(getattr(args,'cpu_ownership',None) is not None and getattr(args,'cpu_ownership_sha256',None) is not None,'CPU_EXTERNAL_OWNERSHIP_REQUIRED')
    C.require(C.sha(args.cpu_ownership)==args.cpu_ownership_sha256,'CPU_OUTER_OWNERSHIP_SHA')
    C.outer_owned(C.read(args.cpu_ownership),seal,'prepare_public_oracle.py')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);source_args(p);p.add_argument('--oracle',required=True);p.add_argument('--oracle-sha256',required=True);p.add_argument('--allow-unqualified-fixture',action='store_true');p.add_argument('--cpu-ownership');p.add_argument('--cpu-ownership-sha256');args=p.parse_args();result,_=verify(args,qualified=not args.allow_unqualified_fixture);
    print(__import__('json').dumps(result,sort_keys=True))
