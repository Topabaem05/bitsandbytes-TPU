"""Complete static source/wheel admission. No tensor/runtime import."""
import hashlib,json,zipfile
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def files(root):
    require(root.is_dir() and not root.is_symlink(),'SOURCE_ROOT')
    result={}
    for p in sorted(root.rglob('*')):
        require(not p.is_symlink(),'SOURCE_SYMLINK')
        if p.is_file():
            require(p.suffix!='.pyc','SOURCE_BYTECODE')
            result[p.relative_to(root).as_posix()]=sha(p)
    return result
def wheel_files(wheel,prefix):
    with zipfile.ZipFile(wheel) as z:
        names=z.namelist();require(len(names)==len(set(names)),'WHEEL_DUPLICATE')
        for name in names:require(not name.startswith('/') and '..' not in Path(name).parts,'WHEEL_PATH')
        return {n[len(prefix):]:hashlib.sha256(z.read(n)).hexdigest() for n in names if n.startswith(prefix) and not n.endswith('/')}
def verify(admission_path,expected_sha,*,plugin=None,upstream=None,plugin_wheel=None,upstream_wheel=None):
    require(not admission_path.is_symlink() and sha(admission_path)==expected_sha,'ADMISSION_BYTE_SHA')
    a=json.loads(admission_path.read_text())
    require(a.get('generation')=='nested-exact-v2-functional-output' and a.get('status')=='NOT_ADOPTED','EXACT_SOURCE_VARIANT')
    require(a.get('device_acceptance') is None and a.get('m4_status')=='NOT_QUALIFIED','NO_DEVICE_ACCEPTANCE')
    require(a.get('inputs_sha256')=='b7490f4c622aef793d14bd9a7a38727754cb1fb8e6e1dec7ff91ef8c45fbafbc','ORIGINAL79_INPUTS')
    require(a.get('profile_sha256')=='ad53f6416bdabf6440f08b313963c947a468cd39afd413bfe41bb381c4a7a166','UNCHANGED_PROFILE')
    require(a.get('cpu_oracle')=='FRESH_QUALIFIED_LINUX_REQUIRED','OLD_ORACLE_NOT_EXECUTION_ADMISSION')
    require(a['base_revision']=='833649043474794b8fe7a4136e0c40faf077b2e0','UPSTREAM_BASE_REVISION')
    require(len(a['upstream_package_files'])==48 and len(a['upstream_python_files'])==47,'UPSTREAM_COMPLETE_MAP')
    require(a['upstream_package_files']['nn/modules.py']=='690987fda1219e91b8b72930402c424ea1875b66b853a9b5472e00de18314e4a','M2_UNCHANGED')
    require(a['plugin_python_files']=={p:h for p,h in a['plugin_package_files'].items() if p.endswith('.py')},'PLUGIN_COMPLETE_PYTHON_MAP')
    fixed={'_integer_clamp.py':'77edb6a7dc30caa49cd6a4260a5663f4c5a0fa146b6f084e207a6aaae9804b59',
           '_integer_addsub.py':'0294486d6cad96d0ceb88ecd492bd0aeaa106788e289735167cd8ef3d3b9fe66',
           '_source_order.py':'700bac513eb69eb22c5d11391b488367b4f56fbeedd741a58e841c6a5fe911a0',
           '_integer_muldiv.py':'89cc079e3dee6a2ff3b79cf8b176ea62c5bd2183a14427bbc0c235de02a27187'}
    require(all(a['plugin_python_files'].get(p)==h for p,h in fixed.items()),'FROZEN_HELPER_BYTES')
    require(a['runtime']=={'python':'3.12.14','torch':'2.9.0+cpu','torch_xla':'2.9.0','libtpu':'0.0.21','jax':'0.7.1','jaxlib':'0.7.1','platform':'Linux','machine':'x86_64','precision':'highest'},'FIXED_RUNTIME')
    if plugin is not None:
        observed=files(plugin);require(observed==a['plugin_files'],'PLUGIN_PACKAGE_BYTES')
    if upstream is not None:require(files(upstream)==a['upstream_package_files'],'UPSTREAM_PACKAGE_BYTES')
    if plugin_wheel is not None:require(wheel_files(plugin_wheel,'bitsandbytes_tpu/')==a['plugin_package_files'],'PLUGIN_WHEEL_SOURCE')
    if upstream_wheel is not None:require(wheel_files(upstream_wheel,'bitsandbytes/')==a['upstream_package_files'],'UPSTREAM_WHEEL_SOURCE')
    return {'record_validation':'PASS','scope':'SOURCE_AND_WHEEL_NOT_TPU','generation':'nested-exact-v2-functional-output','upstream_files':48,
            'plugin_python_files':len(a['plugin_python_files']),'TPU':'NOT_RUN','m4_status':'NOT_QUALIFIED'}
