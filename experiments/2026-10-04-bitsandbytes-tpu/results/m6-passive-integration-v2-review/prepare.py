import sys,json,hashlib,zipfile,shutil,stat
from pathlib import Path,PurePosixPath
kit=Path(sys.argv[1]).resolve();archive=Path(sys.argv[2]);expected=sys.argv[3];out=Path(sys.argv[4]).resolve()
def row(b):return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def confined(n):
 p=PurePosixPath(n);assert not p.is_absolute()and '..'not in p.parts and '\\'not in n and p.as_posix()==n
manifest=json.loads((kit/'source-manifest.json').read_text());files=manifest['files'];assert row(archive.read_bytes())['sha256']==expected
with zipfile.ZipFile(archive)as z:
 assert z.testzip()is None and len(z.infolist())==len(files)+1 and set(z.namelist())=={*files,'source-manifest.json'}
 assert z.read('source-manifest.json')==(kit/'source-manifest.json').read_bytes()
 for info in z.infolist():
  confined(info.filename);assert info.create_system==3 and info.external_attr==0o100644<<16
  b=z.read(info);assert b==(kit/info.filename).read_bytes()
  if info.filename!='source-manifest.json':assert row(b)==files[info.filename]
physical={p.relative_to(kit).as_posix()for p in kit.rglob('*')if p.is_file()or p.is_symlink()};assert physical=={*files,'source-manifest.json'};assert not any(p.is_symlink()for p in kit.rglob('*'))
inputs=json.loads((kit/'input-source-map.json').read_text())
assert len(inputs['files'])==426
for n,r in inputs['files'].items():assert row((kit/'inputs'/n).read_bytes())=={k:r[k]for k in ('bytes','sha256')}
packets=out/'packets';packets.mkdir();prs={}
for phase in ('compiler5','public15'):
 source=kit/'packets'/(phase+'-payload.zip');dst=packets/phase;dst.mkdir()
 with zipfile.ZipFile(source)as z:
  m=json.loads(z.read('manifest.json'));assert z.testzip()is None and len(z.infolist())==len(m['files'])+1 and set(z.namelist())=={*m['files'],'manifest.json'}
  for info in z.infolist():
   confined(info.filename);assert info.create_system==3 and info.external_attr==0o100644<<16
   b=z.read(info)
   if info.filename!='manifest.json':assert row(b)==m['files'][info.filename]
   target=dst/info.filename;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
  prs[phase]={'archive':row(source.read_bytes()),'members':len(z.infolist()),'integration_admission':row(z.read('m6-integration/integration-admission.json'))}
 shutil.copyfile(source,dst/'payload.zip')
proof={'status':'PASS_INDEPENDENT_FULL_SOURCE_AND_PACKET_READBACK','source_archive':row(archive.read_bytes()),'source_manifest':row((kit/'source-manifest.json').read_bytes()),'source_members':len(files)+1,'input_members':len(inputs['files']),'packets':prs,'source_root':str(kit),'M6':'NOT_QUALIFIED','provider_calls':0}
(out/'archive-review.json').write_text(json.dumps(proof,sort_keys=True,indent=2)+'\n');print(json.dumps(proof))
