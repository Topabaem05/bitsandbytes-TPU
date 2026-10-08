"""Portable fresh-runpy and early-recovery controls. No runtime, TPU, or service call."""
import argparse, hashlib, json, shutil, sys, tarfile, time, zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;CLOUD=HERE.parent;sys.path.insert(0,str(CLOUD));import remote,native_contract as NC
FIXTURES=HERE/'fixtures/native'

def baseline_sources(destination):
    shutil.copytree(CLOUD,destination,ignore=shutil.ignore_patterns('tests','__pycache__','*.pyc'))
    inventory=NC.read(FIXTURES/'pre-transport-fix-source.json');archive=FIXTURES/'pre-transport-fix-source.tar.gz'
    assert archive.stat().st_size==inventory['archive_bytes'] and NC.sha(archive)==inventory['archive_sha256']
    with tarfile.open(archive,'r:gz')as tar:
        assert len(tar.getmembers())==2 and {m.name for m in tar.getmembers()}=={'remote.py','owner.py'}
        for member in tar.getmembers():
            assert member.isfile();raw=tar.extractfile(member).read();r=inventory['members'][member.name];assert len(raw)==r['bytes']and hashlib.sha256(raw).hexdigest()==r['sha256'];(destination/member.name).write_bytes(raw)

def prepare_cpu(root,packet,archive,cloud):
    root.mkdir();shutil.copytree(packet,root/'payload');shutil.copytree(cloud,root/'control',ignore=shutil.ignore_patterns('tests','__pycache__','*.pyc'));records=root/'records';records.mkdir()
    manifest=NC.read(root/'payload/manifest.json')
    for name in ('remote.py','owner.py'):
        target=root/'payload/cloud'/name;shutil.copyfile(cloud/name,target);manifest['files']['cloud/'+name]={'sha256':NC.sha(target),'bytes':target.stat().st_size}
    NC.write(root/'payload/manifest.json',manifest)
    with zipfile.ZipFile(root/'payload.zip','w',zipfile.ZIP_DEFLATED)as z:
        for name in sorted([*manifest['files'],'manifest.json']):
            i=zipfile.ZipInfo(name,date_time=(2026,10,8,0,0,0));i.create_system=3;i.external_attr=0o100644<<16;i.compress_type=zipfile.ZIP_DEFLATED;z.writestr(i,(root/'payload'/name).read_bytes())
    shutil.copyfile(root/'payload.zip',root/'payload/payload.zip')
    inv=NC.read(archive.with_suffix('').with_suffix('.inventory.json'))
    assert archive.stat().st_size==inv['archive_bytes'] and NC.sha(archive)==inv['archive_sha256']
    with tarfile.open(archive,'r:gz')as tar:
        for member in tar.getmembers():
            selected=member.name.startswith('records/cpu-oracle/')or member.name.startswith('records/steps/11-cpu-oracle/')or member.name in('records/source-controls.json','records/installed-source.json','records/built-source.json','records/receipt.json')
            if not selected:continue
            assert member.isfile()and'..'not in Path(member.name).parts and not Path(member.name).is_absolute();raw=tar.extractfile(member).read();r=inv['members'][member.name];assert len(raw)==r['bytes']and hashlib.sha256(raw).hexdigest()==r['sha256'];target=root/member.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    receipt=NC.read(records/'receipt.json');receipt.update(status='INSTALLED_NOT_QUALIFIED',installation_status='PASS',runtime_status='NOT_RUN',cpu_status='NOT_RUN',tpu_status='NOT_RUN',steps=[],error=None,allocation_epoch=time.time(),packet_sha256=NC.sha(root/'payload.zip'),manifest_sha256=NC.sha(root/'payload/manifest.json'));NC.write(records/'receipt.json',receipt);(root/'neutral').mkdir();shutil.copyfile(HERE/'native_runpy_child.py',root/'neutral/child.py')

def owned(output,label,argv,cwd):
    result=remote.run_step(output,label,argv,time.time()+90,85,cwd=cwd,tpu=False)
    assert result['status']=='PASS'and result['cleanup']['errors']==[]and all(row['status']=='CLEANUP_VERIFIED'for row in result['cleanup']['groups']),result
    return result

def run(packet,seed,output):
    output.mkdir(exist_ok=False);baseline=output/'original-cloud';baseline_sources(baseline);cpu=output/'cpu';cpu.mkdir();rows=[]
    for name,cloud in [('original',baseline),('fixed',CLOUD),('fixed-cached-transport',CLOUD)]:
        base=cpu/name;prepare_cpu(base,packet,seed,cloud);argv=[sys.executable,'-B',str(base/'neutral/child.py'),'--base',str(base)]
        if name=='fixed-cached-transport':argv.append('--cached-transport')
        process=owned(cpu,'fresh-'+name,argv,base/'neutral');report=NC.read(base/'fresh-runpy-result.json')
        if name=='original':assert report['remote_cli_exit_code']==1 and report['receipt_error']=={'type':'ModuleNotFoundError','message':"No module named 'transport'"}and not report['transport_parts_present']
        else:assert report['remote_cli_exit_code']==0 and report['receipt_error']is None and report['transport_parts_present']
        rows.append({'case':'fresh-runpy-'+name,'status':'PASS','pid':report['pid'],'pgid':report['pgid'],'scope':report['scope'],'cleanup':process['cleanup']})
    owners=output/'owners';owners.mkdir()
    cases=[('original-early',baseline,'original','cpu-import-failure'),('original-blocked-receipt',baseline,'original','cpu-blocked-receipt'),('fixed-early',CLOUD,'fixed','cpu-import-failure'),('fixed-blocked-receipt',CLOUD,'fixed','cpu-blocked-receipt'),('fixed-early-cleanup',CLOUD,'fixed','cpu-import-failure-cleanup'),('fixed-parent-missing',CLOUD,'fixed','native-parent-missing'),('fixed-correct',CLOUD,'fixed','correct'),('fixed-cpu-part-bytes',CLOUD,'fixed','cpu-part-bytes'),('fixed-native-proto',CLOUD,'fixed','native-proto')]
    for name,cloud,cpu_name,case in cases:
        target=owners/name;argv=[sys.executable,'-B',str(HERE/'native_owner_child.py'),'--cloud',str(cloud),'--packet',str(cpu/cpu_name/'payload'),'--output',str(target),'--case',case,'--fixture-driver',str(HERE/'native_controls.py')];owned(owners,'owner-'+name,argv,owners);row=NC.read(target/'result.json');rows.append(row);print(json.dumps({'case':name,'status':'PASS'}),flush=True)
    report={'status':'PASS','controls':rows,'count':len(rows),'provider_calls':0,'actual_TPU':'NOT_RUN','scope':'PORTABLE_FRESH_PROCESS_AND_SYNTHETIC_OWNER_RECORDS','native23':'UNCHANGED'};NC.write(output/'results.json',report);print(json.dumps({'status':'PASS','count':len(rows)}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--packet',type=Path,required=True);parser.add_argument('--seed-archive',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();run(a.packet.resolve(),a.seed_archive.resolve(),a.output.resolve())
