"""Bounded CPU archive controls. These files do not authorize scientific execution."""
import argparse,hashlib,importlib.util,os,shutil,json,zipfile
from pathlib import Path
p=argparse.ArgumentParser()
for n in ('cloud','cpu-fixture','output'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();root=a.output.resolve();root.mkdir(exist_ok=False);source=a.cpu_fixture.resolve()
s=importlib.util.spec_from_file_location('dyadic_archive_contract',a.cloud/'public_contract.py');U=importlib.util.module_from_spec(s);s.loader.exec_module(U);rows=[]
def good(name,**r):rows.append({'name':name,'status':'PASS',**r})
def reject(name,fn,label):
 try:fn()
 except ValueError as e:assert label in str(e);good(name,rejected=str(e));return
 raise AssertionError('incorrect archive fixture accepted:'+name)
base=root/'remote';out=base/'records';out.mkdir(parents=True)
shutil.copytree(source/'oracle',out/'cpu-oracle',copy_function=os.link)
for n in ('source-controls.json','installed-source.json','built-source.json','built-plugin-source.json','steps/11-cpu-oracle/result.json'):U.write(out/n,{'scope':'SYNTHETIC_ARCHIVE_ONLY_NOT_CPU_ACCEPTANCE'})
U.write(out/'steps/11-cpu-oracle/ownership.json',{'scope':'SYNTHETIC_ARCHIVE_ONLY_NOT_CPU_ACCEPTANCE'})
for kind in ('plugin','upstream'):
 target=base/('built-'+kind);target.mkdir();shutil.copyfile(source/(kind+'-synthetic-source-only.whl'),target/(kind+'.whl'))
receipt=U.cpu_archive(out,base);inventory=U.read(out/'public-cpu-inventory.json');size=sum(r['bytes']for r in inventory.values());assert size<64*1024*1024 and set('cpu-oracle/supplements/'+p.name for p in(source/'oracle/supplements').glob('*.json')).issubset(inventory);good('new30_artifact_cpu_evidence_with_original64mib_bound',uncompressed_bytes=size,members=len(inventory))
download=root/'download';download.mkdir();shutil.copyfile(out/'public-cpu-inventory.json',download/'inventory.json');shutil.copyfile(out/'public-cpu-evidence.zip',download/'evidence.zip');target=U.recover_cpu(download,receipt)
assert {p.relative_to(target).as_posix():U.row(p)for p in target.rglob('*')if p.is_file()}==inventory;good('exact_supplement_and_primary_archive_readback')
# Physical aggregate overflow is rejected before archive production.
extra=out/'cpu-oracle/extra.bin';extra.write_bytes(b'0'*(64*1024*1024-size+1))
try:reject('actual_uncompressed_cpu_aggregate_over64mib_rejected',lambda:U.cpu_archive(out,base),'PUBLIC_CPU_ARCHIVE_BOUND')
finally:extra.unlink()
assert U.sha(out/'public-cpu-evidence.zip')==receipt['public_cpu_evidence_sha256'];good('previous_closed_cpu_archive_unchanged_after_overflow')
# A resealed forged inventory still cannot bypass the recovery byte limit.
bad=root/'bad-size';bad.mkdir();inv=dict(inventory);key=next(iter(inv));inv[key]=dict(inv[key],bytes=64*1024*1024+1);U.write(bad/'inventory.json',inv);shutil.copyfile(out/'public-cpu-evidence.zip',bad/'evidence.zip');r=dict(receipt,public_cpu_inventory_sha256=U.sha(bad/'inventory.json'));reject('forged_recovered_aggregate_rejected',lambda:U.recover_cpu(bad,r),'PUBLIC_CPU_RECOVERY_BOUND')
# Exact hashes and regular member names bind both reference sets.
bad=root/'bad-body';bad.mkdir();shutil.copyfile(out/'public-cpu-inventory.json',bad/'inventory.json')
with zipfile.ZipFile(out/'public-cpu-evidence.zip')as old,zipfile.ZipFile(bad/'evidence.zip','w',zipfile.ZIP_DEFLATED)as z:
 for i in old.infolist():
  data=old.read(i);data=data+b' 'if i.filename=='cpu-oracle/supplements/direct-float32.json'else data;z.writestr(i,data)
r=dict(receipt,public_cpu_evidence_sha256=U.sha(bad/'evidence.zip'),public_cpu_evidence_bytes=(bad/'evidence.zip').stat().st_size);reject('altered_supplement_member_rejected_after_outer_zip_reseal',lambda:U.recover_cpu(bad,r),'PUBLIC_CPU_MEMBER_BYTES')
U.write(root/'results.json',{'status':'PASS','count':len(rows),'controls':rows,'scope':'CPU_ARCHIVE_TRANSPORT_ONLY','qualified_linux_cpu':'NOT_RUN','actual_device':'NOT_RUN','provider_calls':0});print(json.dumps({'status':'PASS','count':len(rows)}))
