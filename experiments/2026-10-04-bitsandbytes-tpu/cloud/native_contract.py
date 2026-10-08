"""Explicit recorder23 packet, independent CPU gate, and recovery contract. No provider API."""
import hashlib,json,math,re,sys,zipfile
from pathlib import Path
MODE='m6-native-boundary'
VARIANT='M2_CANONICAL_NON_NESTED_R2_PATCH'
MANIFEST_SHA='b25c5d2bf128b5164538c06267ec3a2f55a001a1a7f86eb63142153842eb3909'
SOURCES={'adapter.py': {'bytes': 5992, 'sha256': '7a293cf6f1db9123dda06a10e8c23b00459d0188bf8f07dc72db0f42471b627a'}, 'coordinator.py': {'bytes': 6686, 'sha256': '2c83baf3e4b8eaa91cbdc5602f633670614b5d70ef0a93abebbd9fc4ee51ae63'}, 'device_diagnostic.py': {'bytes': 14846, 'sha256': 'e2d4c301617749485b5e13c238018b072533bd3814298fa6721af35ab31219d0'}, 'graph_binding.py': {'bytes': 3367, 'sha256': '79f446e39a8f04a89197df28897176b2b8f207b031e04da7dccad02c38a5d4e6'}, 'kernel.py': {'bytes': 4569, 'sha256': 'fd2232f0795382d4637b361caefaa8760cd593bdad89e74fa32e37be9a82cced'}, 'ownership/bootstrap.py': {'bytes': 244, 'sha256': '7ed51ad613dbd6c0cbaa76575ae1da5d68b205e8b88c9e702cf206c09bb2150d'}, 'ownership/cleanup_lifecycle.py': {'bytes': 9410, 'sha256': '6557b3522ec68ddcd9106027d08e89e872eb90587eae8cbff2859ac5668cbf76'}, 'ownership/lifecycle.py': {'bytes': 4558, 'sha256': '4ef29a2c673ab7dadce5a965bff18e51f4850134af2dfb5f44b3d3153949f986'}, 'prepare_oracle.py': {'bytes': 3669, 'sha256': '7b25b6d1f025d029a99d7e2502d64dc704fa9262f226f3ab83f3eb325d6d442d'}, 'probe-inputs.json': {'bytes': 132064, 'sha256': 'd0c9cdf6a82449a56923499698f19bfd3cc8d13c042f1f5ad2bdf587701f5e54'}, 'probe-profile.json': {'bytes': 3695, 'sha256': 'ad53f6416bdabf6440f08b313963c947a468cd39afd413bfe41bb381c4a7a166'}, 'probe_backend.py': {'bytes': 24512, 'sha256': 'e8743e33e6a0454d5d05b77075d47a5c9f983cf32f3956c38f2b544f86f5ef6d'}, 'probe_precision.py': {'bytes': 35743, 'sha256': 'e0534955507e5f91e67ecddfffc738a367ad752e69a4eea7ae8052c6d76a8852'}, 'proto_contract.py': {'bytes': 11007, 'sha256': '4341ecdaa273f449f42e6ffadad345c4abe7259d6a666067918528a70fc6aa8a'}, 'protocol.json': {'bytes': 1804, 'sha256': '9206a1007c2895b6215085f6104fa2c8f1091a38238561010c49ce78185690bc'}, 'protocol.py': {'bytes': 7041, 'sha256': '7fa666e54f7e6b82b7388745e548f7b1847f2bb8573adaab6998202e79f9d27d'}, 'recorder.py': {'bytes': 4313, 'sha256': 'ce1c2e88a8992dc481606a344af38fa3d314f47af7b5b3022e0546f4409e7a01'}, 'source-pins.json': {'bytes': 1740, 'sha256': '01563e5ae4d00b6bc6d56bc61847ec6fa5628e5573fc51b21e5ad8d149ca3836'}, 'state_child_owner.py': {'bytes': 3753, 'sha256': '36b10c2fc62e210fd5fb0d194213b660a44687137f8ab4c9d80dc0587f8f8ad5'}, 'transfer_admission.py': {'bytes': 4358, 'sha256': '6a7c925211b14aba990318360c8f0ca2d4d803e4528c634135c8bde86326816b'}, 'verifier.py': {'bytes': 15447, 'sha256': '4c0e60e602f84c6d54bec02d08f71f61c5171e68e485dde0d6ef2cb9673a8bb3'}, 'verify_record.py': {'bytes': 1892, 'sha256': '212f25c69fcb4669e517175cdb847664d446e5f8825e849b587372c3633f26be'}, 'verify_run.py': {'bytes': 11083, 'sha256': '5ccb93d0a680b38734e0e642f611822880bd2b7dddbf279f2daf84a09da9d44a'}}
BINDINGS={'native_manifest_sha256':'native/manifest.json','m2_dependency_sha256':'m2-dependency.json'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,m):
    if not v:raise ValueError(m)
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n')
def verify_manifest(m):
    native=m.get('experiment')==MODE
    if not native:
        require(not any(k in m for k in (*BINDINGS,'native_source_variant','native_scope')) and not any(n.startswith('native/') or n=='m2-dependency.json' for n in m['files']),'NATIVE_NOT_REQUESTED');return
    require(m.get('native_manifest_sha256')==MANIFEST_SHA and m.get('native_source_variant')==VARIANT and m.get('native_scope')=='BOUNDED_NATIVE_BOUNDARY_ONLY','NATIVE_REVIEWED_SCOPE')
    require(m.get('precision')=='highest' and not m.get('diagnostic_only'),'NATIVE_PRECISION')
    for k,n in BINDINGS.items():require(m.get(k)==m['files'].get(n,{}).get('sha256'),'NATIVE_BINDING:'+k)
    require({n.removeprefix('native/'):r for n,r in m['files'].items() if n.startswith('native/')}=={**SOURCES,'manifest.json':{'sha256':MANIFEST_SHA,'bytes': 3471}},'NATIVE_EXACT_23_SOURCES')
def payload(root,m):
    verify_manifest(m);root=Path(root)
    require(sha(root/'native/manifest.json')==MANIFEST_SHA and read(root/'native/manifest.json')['sources']==SOURCES,'NATIVE_MANIFEST_BYTES')
    for n,r in SOURCES.items():require(not (root/'native'/n).is_symlink() and sha(root/'native'/n)==r['sha256'] and (root/'native'/n).stat().st_size==r['bytes'],'NATIVE_SOURCE_BYTES:'+n)
    spec=read(root/'native/protocol.json');admission=read(root/'source-admission.json')
    require(admission['bitsandbytes_tpu']['files']==spec['plugin_python_files'] and 'nested_source_variant' not in admission,'NATIVE_M2_PLUGIN_MAP')
    dependency=read(root/'m2-dependency.json')
    require(dependency=={'kind':'ROOT_M2_DEPENDENCY_ACCEPTANCE','status':'ACCEPTED_ACTUAL_M2','source_variant':VARIANT,'runtime_lock_sha256':m['runtime_lock_sha256'],'patch_manifest_sha256':m['patch_manifest_sha256'],'source_admission_sha256':m['source_admission_sha256'],'accepted_result_sha256':dependency.get('accepted_result_sha256')} and re.fullmatch('[0-9a-f]{64}',dependency.get('accepted_result_sha256','')),'NATIVE_M2_DEPENDENCY')
def cpu_archive(out):
    out=Path(out);names=[p.relative_to(out).as_posix() for p in sorted((out/'cpu-oracle').rglob('*')) if p.is_file()]
    names+=['steps/11-cpu-oracle/ownership.json','steps/11-cpu-oracle/result.json','source-controls.json','installed-source.json','built-source.json']
    inventory={n:{'sha256':sha(out/n),'bytes':(out/n).stat().st_size} for n in sorted(names)};write(out/'native-cpu-inventory.json',inventory)
    with zipfile.ZipFile(out/'native-cpu-evidence.zip','w',zipfile.ZIP_DEFLATED)as z:
        for n in sorted(names):
            i=zipfile.ZipInfo(n,date_time=(2026,10,8,0,0,0));i.create_system=3;i.external_attr=0o100644<<16;i.compress_type=zipfile.ZIP_DEFLATED;z.writestr(i,(out/n).read_bytes())
    return {'native_cpu_inventory_sha256':sha(out/'native-cpu-inventory.json'),'native_cpu_evidence_sha256':sha(out/'native-cpu-evidence.zip'),'native_cpu_evidence_bytes':(out/'native-cpu-evidence.zip').stat().st_size}
def recover_cpu(directory,receipt):
    root=Path(directory);require(sha(root/'inventory.json')==receipt['native_cpu_inventory_sha256'] and sha(root/'evidence.zip')==receipt['native_cpu_evidence_sha256'] and (root/'evidence.zip').stat().st_size==receipt['native_cpu_evidence_bytes'],'NATIVE_CPU_ARCHIVE_SEAL');inventory=read(root/'inventory.json');target=root/'recovered';target.mkdir(exist_ok=False)
    with zipfile.ZipFile(root/'evidence.zip')as z:
        require(len(z.infolist())==len(inventory) and set(z.namelist())==set(inventory),'NATIVE_CPU_MEMBER_SET')
        for i in z.infolist():
            p=Path(i.filename);require(not p.is_absolute() and '..' not in p.parts and '\\' not in i.filename and i.create_system==3 and i.external_attr==0o100644<<16,'NATIVE_CPU_MEMBER_TYPE');b=z.read(i);r=inventory[i.filename];require(len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'],'NATIVE_CPU_MEMBER_BYTES');q=target/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b)
    return target

def module(root):
    # The module directory is the admitted native payload. No installed classes change.
    sys.path.insert(0,str(Path(root)/'native'));import protocol as S;require(S.HERE.resolve()==(Path(root)/'native').resolve(),'NATIVE_HELPER_IMPORT_PATH');return S

def cpu_gate(packet,recovered,receipt):
    packet=Path(packet);root=Path(recovered);m=read(packet/'manifest.json');payload(packet,m);S=module(packet)
    require(receipt.get('runtime_status')=='PASS_TPU_RUNTIME_PROBE_ONLY' and receipt.get('cpu_status')=='PASS' and receipt.get('built_source_status')=='POST_PATCH_WHEEL_PYTHON_SOURCE_PASS' and receipt.get('installed_source_status')=='POST_PATCH_PYTHON_SOURCE_PASS' and receipt.get('source_controls_status')=='PASS_QUALIFIED_LINUX_SOURCE_CONTROLS','NATIVE_CPU_PHASE_PROOFS')
    seal,cases=S.oracle(root/'cpu-oracle',receipt['oracle_sha256'],qualified=True)
    ownership=read(root/'steps/11-cpu-oracle/ownership.json');result=read(root/'steps/11-cpu-oracle/result.json');argv=ownership['argv']
    require(ownership.get('pid')==ownership.get('pgid')==seal.get('pid')==seal.get('pgid') and seal.get('parent_pid')==ownership.get('owner_pid'),'NATIVE_CPU_ACTUAL_IDENTITY')
    require(argv.count('--process-token')==1 and argv[argv.index('--process-token')+1]==seal['process_token'] and re.fullmatch('[0-9a-f]{32}',seal['process_token']) and '--unqualified-local' not in argv,'NATIVE_CPU_TOKEN_QUALIFICATION')
    require(len(argv)==11 and argv[2]=='/content/bnb-tpu-first/payload/native/prepare_oracle.py' and argv[4]=='/content/bnb-tpu-first/upstream/bitsandbytes' and argv[6]=='/content/bnb-tpu-first/payload/patches/params4bit-xla-v1.json' and argv[8]=='/content/bnb-tpu-first/records/cpu-oracle' and argv[1]=='-B' and argv[3]=='--upstream-source' and argv[5]=='--patch-manifest' and argv[7]=='--output' and argv[9]=='--process-token' and Path(argv[8]).name=='cpu-oracle','NATIVE_CPU_ARGV')
    cleanup=ownership.get('cleanup',{});require(cleanup.get('status')=='CLEANUP_VERIFIED' and cleanup.get('leader_reaped')is True and cleanup.get('group_absence')=='OBSERVED_NO_SUCH_GROUP' and cleanup.get('errors')==[] and cleanup.get('exit_status')==0,'NATIVE_CPU_GROUP_CLOSED')
    require(result.get('status')=='PASS' and result.get('exit_code')==0 and result.get('cleanup',{}).get('errors')==[],'NATIVE_CPU_STEP_PASS')
    from remote import validate_source_controls
    validate_source_controls(read(root/'source-controls.json'));admission=read(packet/'source-admission.json');proof=read(root/'installed-source.json');built=read(root/'built-source.json')
    require(proof.get('source_admission_sha256')==m['source_admission_sha256'] and proof.get('patch_manifest_sha256')==m['patch_manifest_sha256'] and proof.get('status')=='POST_PATCH_PYTHON_SOURCE_PASS' and proof.get('installed_python_files')=={k:admission[k]['files'] for k in ('bitsandbytes','bitsandbytes_tpu')},'NATIVE_CPU_INSTALLED_SOURCE')
    require(built.get('status')=='POST_PATCH_WHEEL_PYTHON_SOURCE_PASS' and built.get('patch_manifest_sha256')==m['patch_manifest_sha256'] and built.get('python_files')==admission['bitsandbytes']['files'] and any(r['name']==built.get('wheel_name') and r['sha256']==built.get('wheel_sha256') for r in receipt['built_wheels']),'NATIVE_CPU_BUILT_SOURCE')
    return {'kind':'ROOT_NATIVE_CPU_ORACLE_GATE','status':'QUALIFIED_LINUX_CPU_ORACLE_VERIFIED','oracle_sha256':receipt['oracle_sha256'],'native_manifest_sha256':MANIFEST_SHA,'source_admission_sha256':m['source_admission_sha256'],'native_cpu_inventory_sha256':receipt['native_cpu_inventory_sha256'],'native_cpu_evidence_sha256':receipt['native_cpu_evidence_sha256'],'source_variant':VARIANT,'m6':'NOT_QUALIFIED'}
