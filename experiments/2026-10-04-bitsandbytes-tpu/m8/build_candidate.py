"""Build one sealed script from explicit immutable source inputs. No provider calls."""
import argparse,base64,copy,hashlib,io,json,math,re,shutil,subprocess,tarfile,tempfile,zipfile
from pathlib import Path
import batch
import runner
HERE=Path(__file__).resolve().parent
PINS_SHA='f7b39c3b1647d05a7d48f10c166c723a021fd7780c0fcc14366a5e98cd32f617'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n')
def source_pins():
    p=HERE/'payload-source-pins.json'
    if p.is_symlink() or sha(p)!=PINS_SHA:raise ValueError('REVIEWED_PAYLOAD_SOURCE_PINS')
    return json.loads(p.read_text())
def tree(root):
    result={}
    for p in sorted(root.rglob('*')):
        if p.is_symlink() or not(p.is_file() or p.is_dir()):raise ValueError('PATCH_SOURCE_NONREGULAR')
        if p.is_file():result[p.relative_to(root).as_posix()]=sha(p)
    return result

def patch_archive(base,output,manifest,patch,remote):
    if sha(base)!=manifest['base_archive_sha256'] or sha(patch)!=manifest['patch_sha256']:raise ValueError('EXACT_PATCH_INPUT')
    with tempfile.TemporaryDirectory(prefix='m8-fixed-patch-') as temporary:
        root=Path(temporary)/'source';remote.unpack_source(base,root)
        if tree(root/'bitsandbytes')!=manifest['base_package_files']:raise ValueError('PATCH_BASE_MAP')
        before=tree(root)
        for args in [['git','apply','--check',str(patch.resolve())],['git','apply',str(patch.resolve())]]:
            subprocess.run(args,cwd=root,check=True,capture_output=True)
        after=tree(root)
        if set(before)!=set(after) or {n for n in before if before[n]!=after[n]}!={'bitsandbytes/nn/modules.py'}:raise ValueError('PATCH_CHANGED_SCOPE')
        if tree(root/'bitsandbytes')!=manifest['post_patch_package_files']:raise ValueError('PATCH_POST_MAP')
        with tarfile.open(base) as original,tarfile.open(output,'w') as target:
            for old in original.getmembers():
                member=copy.copy(old)
                if old.isfile():
                    p=root/old.name;member.size=p.stat().st_size
                    with p.open('rb') as stream:target.addfile(member,stream)
                else:target.addfile(member)
    return {n:h for n,h in manifest['post_patch_package_files'].items() if n.endswith('.py')}

def output_paths(packet):
    # Derive all scientific rows from the immutable original protocol.
    T=batch.load(packet/'probe_transfer.py','m8_inventory_original_T')
    from types import SimpleNamespace
    B,R,P,A=T.helpers(SimpleNamespace(backend_probe=packet/'probe_backend.py',route_probe=packet/'probe_routes.py',precision_probe=packet/'probe_precision.py',patch_manifest=packet/'patches/params4bit-xla-v1.json'))
    profile,inputs=B.load_spec();cases=P.selection(B,R)
    if len(inputs['cases'])!=42 or [c['id'] for c in inputs['cases']]!=profile['case_ids'] or len(cases)!=2 or T.ROUTES!=['params_to_xla','module_to_xla']:raise ValueError('FIXED_46_ROW_PROTOCOL')
    paths={'records/'+n for n in ['built-source.json','cpu-oracle/oracle-seal.json','installed-metadata.json','installed-source.json','receipt.json','runtime-probe.json','source-controls.json','transfer/receipt.json','transfer/references.json',
        'batch.json','batch-cpu-gate.json','batch-science-verify.json','built-plugin-source.json','bootstrap/bootstrap-result.json','bootstrap/archive-inventory.json',
        'built-wheels/bitsandbytes-0.50.3.dev0-cp312-cp312-linux_x86_64.whl','built-wheels/bitsandbytes_tpu-0.1.0-py3-none-any.whl']}
    paths.update('records/cpu-oracle/raw/'+c['id']+'.json' for c in inputs['cases'])
    for group,route,case in [('matrix','api42',c) for c in inputs['cases']]+[('transfer',r,c) for c in cases for r in T.ROUTES]:
        paths.update('records/transfer/'+group+'/'+route+'-'+case['id']+suffix for suffix in ['.json','.hlo.txt','.metrics.txt'])
    for label in batch.LABELS:
        paths.update('records/steps/'+label+'/'+n for n in ['cleanup.json','lifecycle.jsonl','ownership.json','result.json','stderr.raw','stdout.raw'])
    paths.update('records/phase-'+label+'.stdout.raw' for label in ['install','cpu','runtime','transfer','verify'])
    if len(paths)!=298:raise ValueError('COMPLETE_OUTPUT_PROTOCOL')
    return sorted(paths)

def build(out,owner,slug,title,nonce,deadline,*,project_source,base_archive):
    project_source=Path(project_source).resolve();base_archive=Path(base_archive).resolve();out=Path(out)
    if out.exists() or out.is_symlink():raise ValueError('FRESH_CANDIDATE')
    if not re.fullmatch('[0-9a-f]{32}',nonce) or not math.isfinite(deadline):raise ValueError('NONCE_DEADLINE')
    pins=source_pins()
    if not base_archive.is_file() or base_archive.is_symlink() or sha(base_archive)!=pins['base_archive_sha256']:raise ValueError('GENUINE_BASE_ARCHIVE')
    for name,row in pins['files'].items():
        p=project_source/row['project_path']
        if p.is_symlink() or not p.is_file() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('REVIEWED_PROJECT_SOURCE:'+row['project_path'])
    out.mkdir();packet=out/'packet';packet.mkdir()
    for name,row in pins['files'].items():
        dst=packet/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(project_source/row['project_path'],dst)
    shutil.copyfile(base_archive,packet/'upstream.tar');shutil.copyfile(HERE/'batch.py',packet/'batch.py')
    remote=batch.load(packet/'cloud/remote.py','m8_canonical_remote_builder')
    patch=json.loads((packet/'patches/params4bit-xla-v1.json').read_text())
    bnb=patch_archive(base_archive,packet/'patched-upstream.tar',patch,packet/'patches/params4bit-xla-v1.patch',remote)
    if sha(packet/'patched-upstream.tar')!=pins['patched_archive_sha256']:raise ValueError('EXACT_PATCHED_ARCHIVE')
    admission={'format':'bnb-tpu.probe-source-admission.v1','runtime_lock_sha256':pins['runtime_lock_sha256'],
        'bitsandbytes':{'commit':pins['upstream_commit'],'files':bnb,'patch_manifest_sha256':pins['patch_manifest_sha256']},
        'bitsandbytes_tpu':{'files':json.loads((packet/'plugin/source-manifest.json').read_text())['installed_python_files']}}
    write(packet/'source-admission.json',admission)
    if sha(packet/'source-admission.json')!=pins['source_admission_sha256']:raise ValueError('EXACT_SOURCE_ADMISSION')
    manifest={k:pins[k] for k in ['runtime_lock_sha256','source_admission_sha256','plugin_source_manifest_sha256','upstream_commit','base_archive_sha256','patched_archive_sha256','patch_manifest_sha256']}
    manifest.update(format='bnb-tpu.first-packet.v1',experiment='transfer-api42',precision='highest',
        upstream_source_url='https://github.com/bitsandbytes-foundation/bitsandbytes/tree/'+pins['upstream_commit'],
        budget={'total':3600,'install':1200,'science':1800,'retrieval':600,'cleanup':60},patch_sha256=patch['patch_sha256'])
    for field,name in [('route_probe_sha256','probe_routes.py'),('precision_probe_sha256','probe_precision.py'),('transfer_probe_sha256','probe_transfer.py'),('transfer_admission_sha256','transfer_admission.py'),('source_controls_sha256','tests/test_transfer_source.py')]:manifest[field]=sha(packet/name)
    manifest['files']={p.relative_to(packet).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(packet.rglob('*')) if p.is_file()}
    write(packet/'manifest.json',manifest);batch.validate_packet(packet,manifest)
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for name in sorted([*manifest['files'],'manifest.json']):
            info=zipfile.ZipInfo(name,(2026,10,4,0,0,0));info.create_system=3;info.external_attr=0o100644<<16;info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,(packet/name).read_bytes())
    data=buffer.getvalue();(out/'payload.zip').write_bytes(data)
    binding={'nonce':nonce,'packet_sha256':hashlib.sha256(data).hexdigest(),
        'runtime_lock_sha256':sha(packet/'runtime/requirements.lock.json'),'profile_sha256':sha(packet/'probe-profile.json'),
        'input_sha256':sha(packet/'probe-inputs.json'),'source_manifest_sha256':sha(packet/'plugin/source-manifest.json')}
    script=(HERE/'wrapper_template.py').read_text().replace('__SEALED_PAYLOAD_B64__',repr(base64.b64encode(data).decode())).replace('__SEALED_BINDING__',repr(binding)).replace('__SEALED_DEADLINE__',repr(deadline))
    folder=out/'submission';folder.mkdir();(folder/'wrapper.py').write_text(script)
    write(folder/'kernel-metadata.json',{'id':f'{owner}/{slug}','title':title,'code_file':'wrapper.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':False,'enable_tpu':True,'enable_internet':True,'dataset_sources':[],'kernel_sources':[],'competition_sources':[],'model_sources':[]})
    plan={'owner':owner,'slug':slug,'title':title,'folder':str(folder.resolve()),'wrapper_sha256':sha(folder/'wrapper.py'),
        'binding':binding,'machine_shape':'TpuV6E8','session_timeout_seconds':3600,'deadline_epoch':deadline,
        'max_archive_bytes':100*1024*1024,'max_members':2000,'expected_member_paths':output_paths(packet)}
    runner.validate_plan(plan);write(out/'plan.json',plan)
    write(out/'derived-maps.json',{'accepted_admission_sha256':manifest['source_admission_sha256'],'source_admission':admission,'packet_files':manifest['files'],
        'wrapper_sha256':plan['wrapper_sha256'],'packet_sha256':binding['packet_sha256'],'manifest_sha256':sha(packet/'manifest.json'),'expected_output_paths':plan['expected_member_paths'],
        'payload_source_pins_sha256':PINS_SHA,'base_archive_sha256':sha(base_archive),'science_source':'EXPLICIT_REVIEWED_PROJECT_SOURCE'})
    return plan

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--project-source',type=Path,required=True);p.add_argument('--base-archive',type=Path,required=True)
    for k in ['owner','slug','title','nonce']:p.add_argument('--'+k,required=True)
    p.add_argument('--deadline-epoch',type=float,required=True);a=p.parse_args();print(json.dumps(build(a.out,a.owner,a.slug,a.title,a.nonce,a.deadline_epoch,project_source=a.project_source,base_archive=a.base_archive),sort_keys=True))
