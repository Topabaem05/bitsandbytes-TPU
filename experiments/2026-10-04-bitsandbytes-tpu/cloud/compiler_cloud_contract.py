"""Explicit compiler34 packet, independent CPU gate, and recovery contract. No provider API."""
import hashlib,json,math,re,sys,zipfile
from pathlib import Path
MODE='m6-compiler-content'
VARIANT='M2_CANONICAL_NON_NESTED_R2_PATCH'
MANIFEST_SHA='f2a3e62f3a8fc7790fdeff12973af71a2be831f2a057199812b8f2f4c6676a88'
BASE_NATIVE_MANIFEST_SHA='30b820b90a4404a89da6e6c7b6fc91dbec47afbe06ae322eb5835b9eb0e70901'
SOURCES={'adapter.py': {'sha256': '08869e04047837972e7bba9a35a0951206974f4b506debad2bba08af7f1001a4', 'bytes': 5992}, 'analyzer.py': {'sha256': 'c44ecb15696031f326a7abe64ea3d8407b9bc21f527cb94f8b5f149c456fcd8d', 'bytes': 15044}, 'audit_preflight.py': {'sha256': '4189666b35d5b638399eb3f40a9174554bca8cbd0cc564b46b3b534bba2bcc77', 'bytes': 734}, 'compiler-policy.json': {'sha256': '131637cedccf2ed86c4d800a42782bb082329c64a62e06ef4f1c8c713709b932', 'bytes': 1780}, 'compiler_contract.py': {'sha256': '4264e693e0707dc608951527a1293f212724e4940dbe4d7b58ea0e38521992d0', 'bytes': 5491}, 'compiler_exec.py': {'sha256': '8a59f324954216f6156fcf09c97dcc16375dd99f30be042b67cdf0be7497e46d', 'bytes': 2812}, 'compiler_flag_probe.py': {'sha256': '1377eeebb6e6724a483e0e07e1453b43eff3fd0b49ed4e75c1d434994290e5e0', 'bytes': 3746}, 'compiler_flags.py': {'sha256': '67210fbadd872b9a4c9309ed467103e1d16322871c5d0900a92589fbfb7c0ca5', 'bytes': 6065}, 'compiler_verify.py': {'sha256': '97239a4eefa49ac09840687856e582fc986082418cc20194f10a414783745483', 'bytes': 18261}, 'conversion_recovery.py': {'sha256': '6a6ce33279a48ed22f525c04d1e995066e7b39e2516233288c673463b8ec63ff', 'bytes': 3698}, 'coordinator.py': {'sha256': '656c1f8e0073d46577d0ce533149da3cc47bd0323c4abf0cc2d089b4b3000c51', 'bytes': 8610}, 'device_diagnostic.py': {'sha256': '93f09e39e2db21ac4a3954a691bdf75caecd140484e12496fbf8f5eca5b62e20', 'bytes': 16142}, 'dump_inventory.py': {'sha256': 'd9827e99467714099a921c9d06abc744355cea352d38c996a37288304f4bab7b', 'bytes': 4423}, 'dump_observation.py': {'sha256': 'f46af314ed5488560471ad04a4887d4939d92fb76cf2dfdd012ffbfd7224720e', 'bytes': 1990}, 'graph_binding.py': {'sha256': '79f446e39a8f04a89197df28897176b2b8f207b031e04da7dccad02c38a5d4e6', 'bytes': 3367}, 'kernel.py': {'sha256': '68b872c4695b630ed83f5e02614f53bb01508bd4e317dd0d09d62ab0b3034294', 'bytes': 4871}, 'mosaic_audit.py': {'sha256': 'baf3afd93f5fd7d069cd863699ba2aeefb47de72b23c1659a3e50eb7e86357bc', 'bytes': 7067}, 'mosaic_compatibility.py': {'sha256': 'cce7a54521b3ccb51a35b4e26332c52b6e0e2dfbcb44db5c2cd290d016120542', 'bytes': 4015}, 'ownership/bootstrap.py': {'sha256': '7ed51ad613dbd6c0cbaa76575ae1da5d68b205e8b88c9e702cf206c09bb2150d', 'bytes': 244}, 'ownership/cleanup_lifecycle.py': {'sha256': '6557b3522ec68ddcd9106027d08e89e872eb90587eae8cbff2859ac5668cbf76', 'bytes': 9410}, 'ownership/lifecycle.py': {'sha256': '4ef29a2c673ab7dadce5a965bff18e51f4850134af2dfb5f44b3d3153949f986', 'bytes': 4558}, 'prepare_oracle.py': {'sha256': '7b25b6d1f025d029a99d7e2502d64dc704fa9262f226f3ab83f3eb325d6d442d', 'bytes': 3669}, 'probe-inputs.json': {'sha256': 'd0c9cdf6a82449a56923499698f19bfd3cc8d13c042f1f5ad2bdf587701f5e54', 'bytes': 132064}, 'probe-profile.json': {'sha256': 'ad53f6416bdabf6440f08b313963c947a468cd39afd413bfe41bb381c4a7a166', 'bytes': 3695}, 'probe_backend.py': {'sha256': 'e8743e33e6a0454d5d05b77075d47a5c9f983cf32f3956c38f2b544f86f5ef6d', 'bytes': 24512}, 'probe_precision.py': {'sha256': 'e0534955507e5f91e67ecddfffc738a367ad752e69a4eea7ae8052c6d76a8852', 'bytes': 35743}, 'proto_contract.py': {'sha256': '4341ecdaa273f449f42e6ffadad345c4abe7259d6a666067918528a70fc6aa8a', 'bytes': 11007}, 'protocol.json': {'sha256': '048d4a4a745ba953cfc505df2ca41047e5e39cdbda492992e6e42b1423b6383c', 'bytes': 2123}, 'protocol.py': {'sha256': 'e5a5d23e2e2d40295a3df00700273f1316e250ffb676418a2eb19b1f813e7fd7', 'bytes': 7204}, 'recorder.py': {'sha256': 'f70c5fcb3f9cb325af03258e1b27d97d3a66934d4c34f1a4a6997af1f80fe79a', 'bytes': 5122}, 'source-pins.json': {'sha256': '56db733219c667bf77c54ee9e1c4baf2b5894bc7290fc30a3ac4c045d540f036', 'bytes': 1886}, 'state_child_owner.py': {'sha256': '7d1edccb3494996a55e2023990b2faab069f1c09459172290227864aa9df2414', 'bytes': 7438}, 'transfer_admission.py': {'sha256': '6a7c925211b14aba990318360c8f0ca2d4d803e4528c634135c8bde86326816b', 'bytes': 4358}, 'verifier.py': {'sha256': '4c0e60e602f84c6d54bec02d08f71f61c5171e68e485dde0d6ef2cb9673a8bb3', 'bytes': 15447}, 'verify_record.py': {'sha256': '212f25c69fcb4669e517175cdb847664d446e5f8825e849b587372c3633f26be', 'bytes': 1892}, 'verify_run.py': {'sha256': 'e9a71bc60c0fa40aa4263bb5c024690f079f0d4bec6cbeb356746cd1765a9628', 'bytes': 11834}}
POLICY_SHA='131637cedccf2ed86c4d800a42782bb082329c64a62e06ef4f1c8c713709b932'
BINDINGS={'compiler_manifest_sha256':'native/manifest.json','compiler_policy_sha256':'native/compiler-policy.json','m2_dependency_sha256':'m2-dependency.json','native_dependency_sha256':'native-dependency.json','native_accepted_result_sha256':'accepted-native-result.json'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,m):
    if not v:raise ValueError(m)
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n')

NATIVE_DEPENDENCY_BINDINGS={'native_dependency_sha256': 'c64fca47f57374fab691d9870005cf614fc6eebbc8a02706ed70781f9dd1042a', 'native_accepted_result_sha256': 'fd15cb724a398b9079b0b97c19f66bd858fe3f7027e9ae9e1a0ff8983eb1a291', 'native_adoption_revision': 'a084a4d578362f9733453595ab208385cf70afba', 'actual_native_acceptance': 'c64fca47f57374fab691d9870005cf614fc6eebbc8a02706ed70781f9dd1042a', 'base_native_manifest_sha256': '30b820b90a4404a89da6e6c7b6fc91dbec47afbe06ae322eb5835b9eb0e70901', 'base_native_generation': 'mosaic-serde7-gather-bf16-fp32-v1'}
GENERATION='compiler-mosaic-serde7-gather-bf16-fp32-libtpu021-flags-v1'
def dependency_fields(record):
    require(all(record.get(k)==v for k,v in NATIVE_DEPENDENCY_BINDINGS.items()),'COMPILER_ACCEPTED_NATIVE_BINDING')
    return dict(NATIVE_DEPENDENCY_BINDINGS)
def reject_compiler_gate_fields(gate):
    require(not any(k in gate for k in (*NATIVE_DEPENDENCY_BINDINGS,'compiler_generation','compiler_manifest_sha256','compiler_policy_sha256','compiler_scope','compiler_source_variant')),'COMPILER_GATE_NOT_REQUESTED')
def dependency_records(dependency,result):
    require(dependency.get('format')=='bnb-tpu.root-accepted-native-dependency.v1' and dependency.get('status')=='ROOT_ACCEPTED_NATIVE_DEVICE_RECORDS','COMPILER_NATIVE_ACCEPTANCE_OUTCOME')
    for key,value in {'native_manifest_sha256':BASE_NATIVE_MANIFEST_SHA,'native_generation':NATIVE_DEPENDENCY_BINDINGS['base_native_generation'],'native_adoption_revision':NATIVE_DEPENDENCY_BINDINGS['native_adoption_revision'],'accepted_result_sha256':NATIVE_DEPENDENCY_BINDINGS['native_accepted_result_sha256'],'runtime_lock_sha256':'323371ff61c5fbcc4f79fd6a358cf2ba17cb72b907382a5ceaac07f91dc66ed6'}.items():
        require(dependency.get(key)==value,'COMPILER_NATIVE_DEPENDENCY_IDENTITY:'+key)
    require(dependency.get('record_validation')=='PASS' and dependency.get('numerical_status')=='PASS' and dependency.get('closure_status')=='CLEANUP_VERIFIED' and dependency.get('device_case_count')==5 and dependency.get('m6_status')=='NOT_QUALIFIED','COMPILER_NATIVE_DEPENDENCY_PROOFS')
    require(result.get('format')=='bnb-tpu.selected-native-bf16-colab-result.v1' and result.get('status')=='ACTUAL_BF16_OPERAND_FIVE_CASES_AND_CLOSURE_VERIFIED','COMPILER_NATIVE_RESULT_OUTCOME')
    for key in ('native_manifest_sha256','native_generation','native_adoption_revision','runtime_lock_sha256'):require(result.get(key)==dependency[key],'COMPILER_NATIVE_RESULT_IDENTITY:'+key)
    require(result.get('record_validation')=='PASS' and result.get('numerical_status')=='PASS' and result.get('cpu_gate_status')=='QUALIFIED_LINUX_CPU_ORACLE_VERIFIED' and result.get('independent_conversion_replay')=='PASS' and result.get('resource_closure')=='PASS' and result.get('actual_device_attempted')is True and result.get('server_empty')is True and result.get('active_usage_zero')is True and result.get('browser_tab_closed')is True,'COMPILER_NATIVE_RESULT_PROOFS')
    require(result.get('device_case_count')==5 and result.get('native_pass_cases')==4 and result.get('reference_pass_cases')==1 and result.get('device_error_cases')==0 and result.get('not_run_cases')==0,'COMPILER_NATIVE_RESULT_COUNTS')
    cases=result.get('cases');require(type(cases)is list and len(cases)==5 and {r.get('case')for r in cases}=={'direct-fp32','functional-fp32','direct-bf16','functional-bf16','tail-reference-fp32'} and all(r.get('status')=='PASS'for r in cases),'COMPILER_NATIVE_RESULT_CASES')
def dependency_payload(root,m):
    dependency_fields(m);root=Path(root)
    for key,name in [('native_dependency_sha256','native-dependency.json'),('native_accepted_result_sha256','accepted-native-result.json')]:require(not(root/name).is_symlink() and sha(root/name)==NATIVE_DEPENDENCY_BINDINGS[key],'COMPILER_NATIVE_DEPENDENCY_FILE:'+name)
    dependency_records(read(root/'native-dependency.json'),read(root/'accepted-native-result.json'))

def verify_manifest(m):
    native=m.get('experiment')==MODE
    if not native:
        require(not any(k in m for k in (*BINDINGS,'compiler_source_variant','compiler_scope','compiler_generation')) and not any(n.startswith('native/') or n=='m2-dependency.json' for n in m['files']),'NATIVE_NOT_REQUESTED');return
    require(m.get('compiler_manifest_sha256')==MANIFEST_SHA and m.get('compiler_source_variant')==VARIANT and m.get('compiler_scope')=='OWNED_COMPILER_CAPTURE_ONLY','NATIVE_REVIEWED_SCOPE')
    require(m.get('compiler_generation')==GENERATION,'NATIVE_MOSAIC_GENERATION')
    require(m.get('base_native_manifest_sha256')==BASE_NATIVE_MANIFEST_SHA,'COMPILER_BASE_NATIVE_MANIFEST')
    dependency_fields(m)
    require(m.get('compiler_policy_sha256')==POLICY_SHA,'COMPILER_EXACT_POLICY')
    require(m.get('precision')=='highest' and not m.get('diagnostic_only'),'NATIVE_PRECISION')
    for k,n in BINDINGS.items():require(m.get(k)==m['files'].get(n,{}).get('sha256'),'NATIVE_BINDING:'+k)
    require({n.removeprefix('native/'):r for n,r in m['files'].items() if n.startswith('native/')}=={**SOURCES,'manifest.json':{'sha256':MANIFEST_SHA,'bytes': 6203}},'COMPILER_EXACT_36_SOURCES')
def payload(root,m):
    verify_manifest(m);root=Path(root);dependency_payload(root,m)
    require(sha(root/'native/manifest.json')==MANIFEST_SHA and read(root/'native/manifest.json')['sources']==SOURCES,'NATIVE_MANIFEST_BYTES')
    for n,r in SOURCES.items():require(not (root/'native'/n).is_symlink() and sha(root/'native'/n)==r['sha256'] and (root/'native'/n).stat().st_size==r['bytes'],'NATIVE_SOURCE_BYTES:'+n)
    spec=read(root/'native/protocol.json');admission=read(root/'source-admission.json')
    require(admission['bitsandbytes_tpu']['files']==spec['plugin_python_files'] and 'nested_source_variant' not in admission,'NATIVE_M2_PLUGIN_MAP')
    dependency=read(root/'m2-dependency.json')
    require(dependency=={'kind':'ROOT_M2_DEPENDENCY_ACCEPTANCE','status':'ACCEPTED_ACTUAL_M2','source_variant':VARIANT,'runtime_lock_sha256':m['runtime_lock_sha256'],'patch_manifest_sha256':m['patch_manifest_sha256'],'source_admission_sha256':m['source_admission_sha256'],'accepted_result_sha256':dependency.get('accepted_result_sha256')} and re.fullmatch('[0-9a-f]{64}',dependency.get('accepted_result_sha256','')),'NATIVE_M2_DEPENDENCY')
def cpu_archive(out):
    out=Path(out);names=[p.relative_to(out).as_posix() for p in sorted((out/'cpu-oracle').rglob('*')) if p.is_file()]
    names+=['steps/11-cpu-oracle/ownership.json','steps/11-cpu-oracle/result.json','source-controls.json','installed-source.json','built-source.json']
    inventory={n:{'sha256':sha(out/n),'bytes':(out/n).stat().st_size} for n in sorted(names)};write(out/'compiler-cpu-inventory.json',inventory)
    with zipfile.ZipFile(out/'compiler-cpu-evidence.zip','w',zipfile.ZIP_DEFLATED)as z:
        for n in sorted(names):
            i=zipfile.ZipInfo(n,date_time=(2026,10,8,0,0,0));i.create_system=3;i.external_attr=0o100644<<16;i.compress_type=zipfile.ZIP_DEFLATED;z.writestr(i,(out/n).read_bytes())
    return {'compiler_cpu_inventory_sha256':sha(out/'compiler-cpu-inventory.json'),'compiler_cpu_evidence_sha256':sha(out/'compiler-cpu-evidence.zip'),'compiler_cpu_evidence_bytes':(out/'compiler-cpu-evidence.zip').stat().st_size}
def recover_cpu(directory,receipt):
    dependency_fields(receipt);root=Path(directory);require(sha(root/'inventory.json')==receipt['compiler_cpu_inventory_sha256'] and sha(root/'evidence.zip')==receipt['compiler_cpu_evidence_sha256'] and (root/'evidence.zip').stat().st_size==receipt['compiler_cpu_evidence_bytes'],'NATIVE_CPU_ARCHIVE_SEAL');inventory=read(root/'inventory.json');target=root/'recovered';target.mkdir(exist_ok=False)
    with zipfile.ZipFile(root/'evidence.zip')as z:
        require(len(z.infolist())==len(inventory) and set(z.namelist())==set(inventory),'NATIVE_CPU_MEMBER_SET')
        for i in z.infolist():
            p=Path(i.filename);require(not p.is_absolute() and '..' not in p.parts and '\\' not in i.filename and i.create_system==3 and i.external_attr==0o100644<<16,'NATIVE_CPU_MEMBER_TYPE');b=z.read(i);r=inventory[i.filename];require(len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'],'NATIVE_CPU_MEMBER_BYTES');q=target/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b)
    return target

def module(root):
    # The module directory is the admitted native payload. No installed classes change.
    sys.path.insert(0,str(Path(root)/'native'));import protocol as S;require(S.HERE.resolve()==(Path(root)/'native').resolve(),'NATIVE_HELPER_IMPORT_PATH');return S

def cpu_gate(packet,recovered,receipt):
    packet=Path(packet);root=Path(recovered);m=read(packet/'manifest.json');payload(packet,m);dependency_fields(receipt);require(receipt.get('compiler_manifest_sha256')==MANIFEST_SHA and receipt.get('compiler_generation')==GENERATION,'COMPILER_CPU_RECEIPT_GENERATION');S=module(packet)
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
    return {'kind':'ROOT_COMPILER_CPU_ORACLE_GATE','status':'QUALIFIED_LINUX_CPU_ORACLE_VERIFIED','oracle_sha256':receipt['oracle_sha256'],'compiler_manifest_sha256':MANIFEST_SHA,'source_admission_sha256':m['source_admission_sha256'],'compiler_cpu_inventory_sha256':receipt['compiler_cpu_inventory_sha256'],'compiler_cpu_evidence_sha256':receipt['compiler_cpu_evidence_sha256'],'compiler_policy_sha256':POLICY_SHA,'source_variant':VARIANT,'compiler_generation':GENERATION,'m6':'NOT_QUALIFIED',**dependency_fields(m)}

def reject_unrequested(m):
    if m.get("experiment")!=MODE:
        require(not any(k in m for k in ("compiler_manifest_sha256","compiler_policy_sha256","compiler_source_variant","compiler_scope","compiler_generation",*NATIVE_DEPENDENCY_BINDINGS)),"COMPILER_NOT_REQUESTED")
        require(not any("native/"+n in m["files"] for n in ("compiler_contract.py","compiler_exec.py","compiler_verify.py","compiler-policy.json","dump_inventory.py","dump_observation.py","analyzer.py")),"COMPILER_FILES_NOT_REQUESTED")
        require(not any(n in m["files"] for n in ("native-dependency.json","accepted-native-result.json")),"COMPILER_NATIVE_DEPENDENCY_NOT_REQUESTED")
