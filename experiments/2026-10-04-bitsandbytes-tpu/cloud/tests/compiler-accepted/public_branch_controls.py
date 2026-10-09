"""Build PUBLIC15 with the private merged cloud; byte-prove unchanged public science and branch."""
import argparse,json,sys,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=next(p for p in HERE.parents if(p/'packages/bitsandbytes-tpu/pyproject.toml').is_file());sys.path.insert(0,str(HERE/'cloud'));import build_public_packet as B,public_contract as U,owner
p=argparse.ArgumentParser();p.add_argument('--upstream',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output).resolve();out.mkdir(exist_ok=False);SCI=ROOT/'experiments/2026-10-04-bitsandbytes-tpu';r=B.build(a.upstream,out/'packet',ROOT,SCI/'public',U.MANIFEST_SHA);m=owner.preflight(out/'packet',r['payload_sha256']);rows=[{'case':'PUBLIC15_genuine_packet_with_merged_cloud_preflight','status':'PASS'}]
with zipfile.ZipFile(out/'packet/payload.zip')as z:
 assert len(z.infolist())==len(m['files'])+1 and set(z.namelist())=={*m['files'],'manifest.json'}
 for i in z.infolist():assert i.create_system==3 and i.external_attr==0o100644<<16 and z.read(i)==(out/'packet'/i.filename).read_bytes()
spec=U.read(SCI/'public/public-device-manifest.json')
for n,v in spec['source_files'].items():assert U.row(SCI/'public'/n)==v and (out/'packet/public'/n).read_bytes()==(SCI/'public'/n).read_bytes()
rows.append({'case':'all_PUBLIC15_scientific_sources_exact_adopted_bytes','status':'PASS'})
private=(HERE/'cloud/remote.py').read_text();adopted=(SCI/'cloud/remote.py').read_text()
def public_branch(s):return s[s.index("        elif phase == 'public':"):s.index("        elif phase == 'native':")]
assert public_branch(private)==public_branch(adopted);rows.append({'case':'PUBLIC15_remote_phase_exact_adopted_bytes','status':'PASS'})
U.write(out/'results.json',{'status':'PASS','count':len(rows),'controls':rows,'packet':r,'provider_calls':0,'actual_public_TPU':'NOT_RUN','M6':'NOT_QUALIFIED'});print(json.dumps({'status':'PASS','count':len(rows),'packet_members':r['members']}))
