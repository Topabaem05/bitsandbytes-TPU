"""Explicit public-device generation. Stdlib source and artifact admission only."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
ADMISSION_SHA='e467e9d711b61140bf5be91f7ecf1da518986683a2bb530b3341194dd28fa518'
PROTOCOL_SHA='dc22c7a3891786947d1127dc45e6d82fcd993f6b6a8d64ba804ee5d835a915c6'
VARIANT='public-pallas-device-mosaic7-gather-bf16-fp32-v1'

def require(value,message):
    if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def pairs(values):
    out={}
    for k,v in values:require(k not in out,'DUPLICATE_JSON_KEY');out[k]=v
    return out
def read(path):return json.loads(Path(path).read_text(),object_pairs_hook=pairs)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def inventory(root,exclude=()):
    root=Path(root);out={}
    for p in sorted(root.rglob('*')):
        require(not p.is_symlink(),'ARTIFACT_SYMLINK')
        if p.is_file() and p.relative_to(root).as_posix() not in exclude:out[p.relative_to(root).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size}
    return out

def source_files(root,expected):
    root=Path(root);require(root.is_dir() and not root.is_symlink(),'SOURCE_ROOT')
    require({p.relative_to(root).as_posix() for p in root.rglob('*.py')}=={n for n in expected if n.endswith('.py')},'PYTHON_SOURCE_INVENTORY')
    for name,item in expected.items():
        p=root/name;require(not Path(name).is_absolute() and '..' not in Path(name).parts,'SOURCE_RELATIVE_PATH')
        require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'SOURCE_PATH')
        h=item['sha256'] if isinstance(item,dict) else item
        require(sha(p)==h,'SOURCE_BYTES:'+name)
        if isinstance(item,dict):require(p.stat().st_size==item['bytes'],'SOURCE_SIZE:'+name)

def admission(admission_path,protocol_path,package_root,upstream_source,repo_root):
    require(sha(admission_path)==ADMISSION_SHA and sha(protocol_path)==PROTOCOL_SHA,'FROZEN_PUBLIC_GENERATION')
    a=read(admission_path);p=read(protocol_path)
    require(a['source_variant']==p['source_variant']=='pallas-forward-mosaic7-gather-bf16-fp32-v1','PUBLIC_PACKAGE_VARIANT')
    require(a['native_device_dependency']['accepted_result_sha256']==p['native_acceptance_sha256']=='c64fca47f57374fab691d9870005cf614fc6eebbc8a02706ed70781f9dd1042a' and a['native_device_dependency']['status']=='REQUIRED_ROOT_ACCEPTED_BOUNDED_NATIVE_RECORDS','EXACT_BOUNDED_NATIVE_DEPENDENCY')
    require(a['public_protocol_sha256']==PROTOCOL_SHA,'ADMISSION_PROTOCOL_LINK')
    native_sources(a,repo_root)
    require(p['case_count']==15 and len(p['cases'])==15 and len({c['id'] for c in p['cases']})==15,'FIXED_CASE_MATRIX')
    package=Path(package_root);upstream=Path(upstream_source)
    if (upstream/'bitsandbytes').is_dir():upstream=upstream/'bitsandbytes'
    source_files(package,a['plugin_python_files'])
    require(sha(package/'source_manifest.json')==a['plugin_manifest_sha256'],'PLUGIN_MANIFEST_BYTES')
    source_files(upstream,a['upstream_package_files'])
    require(read(package/'source_manifest.json')['python_files']==a['plugin_python_files'],'PLUGIN_MANIFEST_FULL_MAP')
    repo=Path(repo_root)
    for name,item in a['fixed_files'].items():require(sha(repo/name)==item['sha256'] and (repo/name).stat().st_size==item['bytes'],'FIXED_GENERATION_SOURCE:'+name)
    require(sha(repo/'patches/params4bit-xla-v1.json')==a['patch_manifest_sha256'],'PATCH_MANIFEST')
    return a,p,upstream

def native_sources(a,repo_root):
    """Presence and content are both required, even after an outer reseal."""
    prefix='experiments/2026-10-04-bitsandbytes-tpu/'
    required={prefix+n for n in ('native/manifest.json','native/protocol.json','native/kernel.py','native/adapter.py','native/device_diagnostic.py','probe-profile.json','runtime/requirements.lock.json','results/native-bf16-acceptance.json','results/native-bf16-colab.json')}
    require(set(a['fixed_files'])==required,'COMPLETE_FIXED_GENERATION_PINS')
    repo=Path(repo_root);native=read(repo/(prefix+'native/manifest.json'))
    require(a['native_generation']==native['generation']=='mosaic-serde7-gather-bf16-fp32-v1' and len(native['sources'])==27,'EXACT_NATIVE27_GENERATION')
    require(sha(repo/(prefix+'native/manifest.json'))==a['fixed_files'][prefix+'native/manifest.json']['sha256']=='30b820b90a4404a89da6e6c7b6fc91dbec47afbe06ae322eb5835b9eb0e70901','EXACT_NATIVE27_MANIFEST')
    for n,target in [('kernel.py','pallas_kernel.py'),('adapter.py','adapter.py'),('device_diagnostic.py',None)]:
        require(a['fixed_files'][prefix+'native/'+n]==native['sources'][n],'NATIVE_TRANSITIVE_SOURCE_PIN:'+n)
        if target is not None:require(a['plugin_python_files'][target]==native['sources'][n]['sha256'],'PACKAGE_NATIVE_SOURCE_BINDING:'+n)
    require(a['fixed_files'][prefix+'results/native-bf16-acceptance.json']['sha256']=='c64fca47f57374fab691d9870005cf614fc6eebbc8a02706ed70781f9dd1042a' and a['fixed_files'][prefix+'results/native-bf16-colab.json']['sha256']=='fd15cb724a398b9079b0b97c19f66bd858fe3f7027e9ae9e1a0ff8983eb1a291','EXACT_NATIVE_ACCEPTANCE_PINS')

def scientific_sources():
    names=['contract.py','oracle_math.py','prepare_public_oracle.py','verify_public_oracle.py','public_graph.py','probe_public.py','verify_public.py','public-protocol.json','package-admission.json']
    names+=['helpers/'+n for n in ('verifier.py','graph_binding.py','proto_contract.py','recorder.py','probe_backend.py','probe_precision.py','probe-profile.json','probe-inputs.json','mosaic_compatibility.py','mosaic_audit.py','conversion_recovery.py')]
    return {n:{'sha256':sha(HERE/n),'bytes':(HERE/n).stat().st_size} for n in names}

def owner(pid,pgid,parent_pid,token,deadline):
    require(type(pid)is int and pid>0 and pid==pgid and type(parent_pid)is int and parent_pid>0 and parent_pid!=pid,'OWNED_GROUP_LEADER')
    require(re.fullmatch('[0-9a-f]{32}',token) is not None,'PROCESS_TOKEN')
    require(type(deadline)in(int,float) and math.isfinite(deadline),'FINITE_DEADLINE')


def accepted_native(path,expected_sha,a):
    """This separate dependency is supplied only by root after actual review."""
    require(path is not None and expected_sha is not None,'NATIVE_DEPENDENCY_PENDING')
    require(sha(path)==expected_sha,'NATIVE_DEPENDENCY_SHA')
    r=read(path)
    require(r.get('format')=='bnb-tpu.root-accepted-native-dependency.v1' and r.get('status')=='ROOT_ACCEPTED_NATIVE_DEVICE_RECORDS','NATIVE_ACCEPTANCE_STATUS')
    require(r.get('native_manifest_sha256')==a['fixed_files']['experiments/2026-10-04-bitsandbytes-tpu/native/manifest.json']['sha256'],'NATIVE_ACCEPTANCE_GENERATION')
    require(r.get('runtime_lock_sha256')==a['fixed_files']['experiments/2026-10-04-bitsandbytes-tpu/runtime/requirements.lock.json']['sha256'] and r.get('profile_sha256')==a['fixed_files']['experiments/2026-10-04-bitsandbytes-tpu/probe-profile.json']['sha256'],'NATIVE_ACCEPTANCE_RUNTIME_GATES')
    require(r.get('native_generation')=='mosaic-serde7-gather-bf16-fp32-v1','NATIVE_ACCEPTANCE_MOSAIC_GENERATION')
    require(r.get('device_case_count')==5 and r.get('record_validation')=='PASS' and r.get('numerical_status')=='PASS' and r.get('closure_status')=='CLEANUP_VERIFIED','NATIVE_ACCEPTANCE_OUTCOME')
    require(re.fullmatch('[0-9a-f]{64}',r.get('accepted_result_sha256','')) is not None,'NATIVE_ACCEPTED_RESULT_HASH')
    require(expected_sha==a['native_device_dependency']['accepted_result_sha256'],'EXACT_NATIVE_DEPENDENCY_DOCUMENT')
    require(r.get('accepted_result_sha256')==a['native_device_dependency']['accepted_science_result_sha256'] and r.get('native_adoption_revision')==a['native_device_dependency']['native_adoption_revision'],'EXACT_NATIVE_DEPENDENCY_RESULT')
    return r

def wheel_source(path,namespace,expected,entry=None):
    import zipfile
    with zipfile.ZipFile(path) as z:
        names=z.namelist();require(len(names)==len(set(names)) and all(not n.startswith('/') and '..' not in Path(n).parts for n in names),'WHEEL_PATHS_DUPLICATES')
        prefix=namespace+'/'
        actual={n[len(prefix):]:hashlib.sha256(z.read(n)).hexdigest() for n in names if n.startswith(prefix) and (n.endswith('.py') or n.endswith('source_manifest.json'))}
        require(actual==expected,'WHEEL_COMPLETE_SOURCE_MAP')
        if entry is not None:
            eps=[n for n in names if n.endswith('.dist-info/entry_points.txt')];require(len(eps)==1 and z.read(eps[0]).decode().strip()==entry,'WHEEL_ENTRY_POINT')
    return {'sha256':sha(path),'bytes':Path(path).stat().st_size,'source_map_sha256':digest(actual)}

def wheels(a,plugin_wheel,upstream_wheel):
    mapping=dict(a['plugin_python_files']);mapping['source_manifest.json']=a['plugin_manifest_sha256']
    return {'plugin':wheel_source(plugin_wheel,a['import_name'],mapping,'[bitsandbytes.backends]\ntpu_pallas_forward = bitsandbytes_tpu_pallas:register'),'upstream':wheel_source(upstream_wheel,'bitsandbytes',a['upstream_python_files'])}

def sha_bytes(value):return hashlib.sha256(value).hexdigest()
def outer_owned(record,receipt,script):
    require(record.get('pid')==receipt['pid'] and record.get('pgid')==receipt['pgid'],'OUTER_ACTUAL_IDENTITY')
    cleanup=record.get('cleanup',{});require(cleanup.get('status')=='CLEANUP_VERIFIED' and cleanup.get('leader_reaped') is True and cleanup.get('group_absence')=='OBSERVED_NO_SUCH_GROUP' and cleanup.get('errors')==[],'OUTER_CLOSURE_REQUIRED')
    argv=record.get('argv');require(isinstance(argv,list) and any(Path(v).name==script for v in argv),'OUTER_ADMITTED_SCRIPT')
    for flag,val in [('process-token',receipt['process_token']),('deadline-epoch',str(receipt['deadline_epoch']))]:
        require(argv.count('--'+flag)==1 and argv[argv.index('--'+flag)+1]==val,'OUTER_ARGV_'+flag)

def runtime_wheels(a,repo_root,directory,*,qualified):
    lock_path=Path(repo_root)/'experiments/2026-10-04-bitsandbytes-tpu/runtime/requirements.lock.json';lock=read(lock_path)
    if directory is None:
        require(not qualified,'QUALIFIED_RUNTIME_WHEEL_BODIES_REQUIRED');return {'scope':'UNQUALIFIED_LOCAL_CPU','wheel_source_check':'NOT_RUN'}
    root=Path(directory);records={}
    for wheel in lock['wheels']:
        p=root/wheel['filename'];require(p.is_file() and not p.is_symlink() and p.stat().st_size==wheel['size'] and sha(p)==wheel['sha256'],'EXACT_RUNTIME_WHEEL_BODY:'+wheel['filename']);records[wheel['filename']]={'sha256':wheel['sha256'],'bytes':wheel['size']}
    return {'scope':'FIXED_RUNTIME_LOCK_WHEEL_BODIES','wheel_source_check':'PASS','records':records}

def installed_runtime_metadata(repo_root):
    import importlib.metadata
    lock=read(Path(repo_root)/'experiments/2026-10-04-bitsandbytes-tpu/runtime/requirements.lock.json');records={}
    for wheel in lock['wheels']:
        dist=importlib.metadata.distribution(wheel['name']);require(dist.version==wheel['version'],'INSTALLED_RUNTIME_VERSION:'+wheel['name']);meta=dist.read_text('METADATA');require(isinstance(meta,str) and hashlib.sha256(meta.encode()).hexdigest()==wheel['metadata_sha256'],'INSTALLED_RUNTIME_METADATA:'+wheel['name']);records[wheel['name']]={'version':dist.version,'metadata_sha256':wheel['metadata_sha256']}
    return records
