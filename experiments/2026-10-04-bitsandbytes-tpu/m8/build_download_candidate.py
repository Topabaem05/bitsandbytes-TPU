"""Render an explicit download mode from the unchanged canonical builder."""
import argparse,hashlib,importlib.util,json,math,re,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
TRANSPORT_HELPER_SHA='18e3b711c3a82f592892b06eaa3933907d444df39b21ab46387a8b7f53e459c7'
CANONICAL_HASHES={'build_candidate.py': '7fc7fc5aa5ff0fb5893d5a9f59c1bbc8b58b774e15d0a3e33db5147e039fa1d3', 'wrapper_template.py': '3b70c23597cd83f9f7016d1336c098d03343753ef7d195ca51b871cac785e3cd', 'batch.py': '30af975218b7283d76a9129d76336671c7100f3bf12a78fd7a2256501db28305', 'runner.py': 'b77029fa8b6cf191e3e0cecf70a30d431abce93d05e28bd7cddb3570953e6e45', 'recovery.py': '715b609fe46cfa8377d7c4437501c755549f5ee5f5f95135136ce979da0aa73e', 'process_provider.py': 'b53cf34ab127fa17c250e89ff966226a1c0087e3830372dcbf57d1d4bf608c84', 'sdk_worker.py': 'c33fc21a6b2405edb56486cfcc8881e683432bb0b2bd686ed66ab2c4eb49541b', 'supervisor.py': '69941df7721a7b246de780b43347daee21571ed6bb992a3965e9cb7eea1e00f1', 'http_error_retention.py': '5d57d9bf2ed6538b629c46a67996572c6caf28ca9019b9b1a9716cc8491845e0', 'installed-source-pins.json': '2a4b414f644bd8920deb6ab89f03b86aa9bb81e381d258b1de7cb5b4a92304b2', 'payload-source-pins.json': 'f7b39c3b1647d05a7d48f10c166c723a021fd7780c0fcc14366a5e98cd32f617'}

ADMITTED_TEMPLATE=b'"""Private M2 repeat. Submission requires root review of these exact bytes."""\nimport base64,hashlib,importlib.util,io,json,os,tempfile,zipfile\nfrom pathlib import Path,PurePosixPath\nPAYLOAD_B64=__SEALED_PAYLOAD_B64__\nBINDING=__SEALED_BINDING__\nDEADLINE=__SEALED_DEADLINE__\n\ndef entry(*,provider_factory=None,workspace=None,output=None):\n    data=base64.b64decode(PAYLOAD_B64,validate=True)\n    if hashlib.sha256(data).hexdigest()!=BINDING[\'packet_sha256\']:raise ValueError(\'EMBEDDED_PACKET_SHA\')\n    base=Path(workspace) if workspace is not None else Path(tempfile.mkdtemp(prefix=\'m8-m2-owned-\'))\n    base.mkdir(parents=True,exist_ok=True)\n    if any(base.iterdir()):raise ValueError(\'FRESH_WORKSPACE\')\n    (base/\'payload.zip\').write_bytes(data);payload=base/\'payload\'\n    with zipfile.ZipFile(io.BytesIO(data)) as z:\n        rows=z.infolist();names=[r.filename for r in rows]\n        if len(names)!=len(set(names)) or len(names)>1000 or sum(r.file_size for r in rows)>50*1024*1024:raise ValueError(\'PACKET_LIMIT\')\n        m=json.loads(z.read(\'manifest.json\'))\n        if set(names)!=set(m[\'files\'])|{\'manifest.json\'}:raise ValueError(\'PACKET_INVENTORY\')\n        for r in rows:\n            p=PurePosixPath(r.filename)\n            if p.is_absolute() or any(s in (\'\',\'.\',\'..\') for s in r.filename.split(\'/\')) or \'\\\\\' in r.filename or r.create_system!=3 or r.external_attr!=0o100644<<16:raise ValueError(\'PACKET_PATH_TYPE\')\n            raw=z.read(r.filename)\n            if r.filename!=\'manifest.json\' and m[\'files\'][r.filename]!={\'bytes\':len(raw),\'sha256\':hashlib.sha256(raw).hexdigest()}:raise ValueError(\'PACKET_MEMBER_SHA\')\n            path=payload/r.filename;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)\n    spec=importlib.util.spec_from_file_location(\'m8_sealed_batch\',payload/\'batch.py\');batch=importlib.util.module_from_spec(spec);spec.loader.exec_module(batch)\n    provider=provider_factory(base,m) if provider_factory else None\n    return batch.execute(base,Path(output) if output else Path(\'/kaggle/working\'),BINDING,DEADLINE,provider=provider)\n\nif __name__==\'__main__\':\n    result=entry();print(json.dumps({\'kind\':result[\'kind\'],\'status\':result[\'status\'],\'m8_status\':\'NOT_QUALIFIED\'}))\n    raise SystemExit(0 if result[\'status\']==\'COMPLETE\' else 2)\n'
SCIENTIFIC_NAMES_SHA='3c705734c18d86e551e9ce143387d0311733e98f17f464f333240936b2803312'
FIXED_BINDING={'input_sha256': 'd0c9cdf6a82449a56923499698f19bfd3cc8d13c042f1f5ad2bdf587701f5e54', 'packet_sha256': '654641d70a43bb1154097301b8ba41dfe5d2f95d16a2dd0e01c181e69eead80a', 'profile_sha256': 'ad53f6416bdabf6440f08b313963c947a468cd39afd413bfe41bb381c4a7a166', 'runtime_lock_sha256': '323371ff61c5fbcc4f79fd6a358cf2ba17cb72b907382a5ceaac07f91dc66ed6', 'source_manifest_sha256': '18af6ca3152b9b9012dda369daf7a50ecc16023f6161ac70421ebf8f81b6f5e6'}

def validate_host(root):
    root=Path(root).resolve()
    for name,digest in CANONICAL_HASHES.items():
        path=root/name
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('CANONICAL_HOST_SOURCE_CHANGED:'+name)
    return root

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def require(condition,code):
    if not condition:raise ValueError(code)

def render_wrapper(binding,commit,deadline,scientific_names,*,helper,template=ADMITTED_TEMPLATE):
    """Render the complete admitted wrapper without file or provider operations."""
    require(type(helper)is bytes and hashlib.sha256(helper).hexdigest()==TRANSPORT_HELPER_SHA,'REVIEWED_TRANSPORT_HELPER')
    require(type(template)is bytes and hashlib.sha256(template).hexdigest()==CANONICAL_HASHES['wrapper_template.py'],'REVIEWED_CANONICAL_TEMPLATE')
    require(type(binding)is dict and set(binding)==set(FIXED_BINDING)|{'nonce'},'RENDER_BINDING_SCHEMA')
    require(all(binding[k]==v for k,v in FIXED_BINDING.items()),'RENDER_FIXED_BINDING')
    require(type(binding['nonce'])is str and re.fullmatch('[0-9a-f]{32}',binding['nonce']),'RENDER_NONCE')
    require(type(commit)is str and re.fullmatch('[0-9a-f]{40}',commit),'FULL_COMMIT_REQUIRED')
    require(type(deadline)in(int,float) and math.isfinite(deadline),'RENDER_DEADLINE')
    require(type(scientific_names)in(list,tuple) and len(scientific_names)==298
        and hashlib.sha256(json.dumps(list(scientific_names),separators=(',',':')).encode()).hexdigest()==SCIENTIFIC_NAMES_SHA,'REVIEWED_SCIENTIFIC_NAMES')
    helper=helper.decode();template=template.decode()
    start=template.index('    data=base64.b64decode(')
    end=template.index("    (base/'payload.zip').write_bytes(data)")
    prefix="""    base=Path(workspace) if workspace is not None else Path(tempfile.mkdtemp(prefix='m8-m2-owned-'))
    base.mkdir(parents=True,exist_ok=True)
    if any(base.iterdir()):raise ValueError('FRESH_WORKSPACE')
    data,transport_record=fetch_packet(base/'payload-transport',PAYLOAD_COMMIT,DEADLINE,worker_command=fetch_child_command)
    if hashlib.sha256(data).hexdigest()!=BINDING['packet_sha256']:raise ValueError('EMBEDDED_PACKET_SHA')
"""
    template=template[:start]+prefix+template[end:]
    template=template.replace('def entry(*,provider_factory=None,workspace=None,output=None):','def entry(*,provider_factory=None,workspace=None,output=None,fetch_child_command=None):')
    before="return batch.execute(base,Path(output) if output else Path('/kaggle/working'),BINDING,DEADLINE,provider=provider)"
    after="""result=batch.execute(base,Path(output) if output else Path('/kaggle/working'),BINDING,DEADLINE,provider=provider)
    append_transport(Path(output) if output else Path('/kaggle/working'),BINDING,transport_record,SCIENTIFIC_MEMBERS)
    return result"""
    require(template.count(before)==1,'UNCHANGED_SCIENCE_CALL')
    template=template.replace(before,after)
    ordered_binding={k:binding[k] for k in sorted(binding)}
    template=template.replace('__SEALED_PAYLOAD_B64__','None').replace('__SEALED_BINDING__',repr(ordered_binding)).replace('__SEALED_DEADLINE__',repr(deadline))
    return (helper+'\nPAYLOAD_COMMIT='+repr(commit)+'\nSCIENTIFIC_MEMBERS='+repr(tuple(scientific_names))+'\n'+template).encode()

def build(out,owner,slug,title,nonce,deadline,*,canonical_host,project_source,base_archive,commit):
    canonical_host=validate_host(canonical_host)
    helper_path=HERE/'payload_transport.py'
    if helper_path.is_symlink() or not helper_path.is_file() or hashlib.sha256(helper_path.read_bytes()).hexdigest()!=TRANSPORT_HELPER_SHA:
        raise ValueError('REVIEWED_TRANSPORT_HELPER')
    sys.path.insert(0,str(canonical_host))
    import runner
    canonical=load(canonical_host/'build_candidate.py','canonical_embedded_builder')
    transport=load(HERE/'payload_transport.py','explicit_download_transport')
    url=transport.packet_url(commit)
    plan=canonical.build(out,owner,slug,title,nonce,deadline,project_source=project_source,base_archive=base_archive)
    out=Path(out);packet=(out/'payload.zip').read_bytes()
    transport.need(len(packet)==transport.PACKET_BYTES and transport.digest(packet)==transport.PACKET_SHA,'EXACT_REVIEWED_ZIP')
    original=(out/'submission/wrapper.py').read_bytes();(out/'original-embedded-wrapper.py').write_bytes(original)
    template=(canonical_host/'wrapper_template.py').read_text()
    original_names=list(plan['expected_member_paths'])
    helper=(HERE/'payload_transport.py').read_text()
    script=render_wrapper(plan['binding'],commit,deadline,original_names,helper=helper.encode(),template=template.encode())
    (out/'submission/wrapper.py').write_bytes(script)
    plan['wrapper_sha256']=transport.digest(script)
    plan['expected_member_paths']=sorted(original_names+[transport.TRANSPORT_MEMBER])
    runner.validate_plan(plan);canonical.write(out/'plan.json',plan)
    mode={'mode':transport.TRANSPORT_MODE,'commit':commit,'url':url,'packet_bytes':len(packet),
        'packet_sha256':transport.PACKET_SHA,'deadline_epoch':deadline,'binding':plan['binding'],
        'work_reserve_seconds':transport.WORK_RESERVE,'fetch_limit_seconds':transport.FETCH_LIMIT,
        'cleanup_limit_seconds':transport.CLEANUP_LIMIT,'scientific_members':original_names,
        'transport_member':transport.TRANSPORT_MEMBER,'wrapper_sha256':plan['wrapper_sha256'],
        'helper_sha256':transport.digest(helper.encode()),'original_embedded_wrapper_sha256':transport.digest(original)}
    canonical.write(out/'transport-plan.json',mode)
    derived=json.loads((out/'derived-maps.json').read_text());derived.update(wrapper_sha256=plan['wrapper_sha256'],
        expected_output_paths=plan['expected_member_paths'],transport_mode=transport.TRANSPORT_MODE,
        transport_plan_sha256=canonical.sha(out/'transport-plan.json'))
    canonical.write(out/'derived-maps.json',derived)
    return plan

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--canonical-host',type=Path,required=True);parser.add_argument('--project-source',type=Path,required=True)
    parser.add_argument('--base-archive',type=Path,required=True);parser.add_argument('--commit',required=True)
    for key in ['owner','slug','title','nonce']:parser.add_argument('--'+key,required=True)
    parser.add_argument('--deadline-epoch',type=float,required=True);a=parser.parse_args()
    print(json.dumps(build(a.out,a.owner,a.slug,a.title,a.nonce,a.deadline_epoch,canonical_host=a.canonical_host,
        project_source=a.project_source,base_archive=a.base_archive,commit=a.commit),sort_keys=True))
