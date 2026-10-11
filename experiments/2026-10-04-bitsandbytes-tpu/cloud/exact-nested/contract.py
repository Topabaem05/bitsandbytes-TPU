"""Dedicated exact-nested admission and bounded regular-file recovery. Stdlib only."""
import hashlib,json,math,os,re,stat,zipfile,time
from pathlib import Path,PurePosixPath
SCIENCE_SHA='9e71efc73b0f2b3a210f8131c10972569a5e2d04cf298fcf5a8da59772cf4f5a'
ADMISSION_SHA='75790f1d4613e20daa06d99111288463ef24b05aaa29922a7580512078f9912f'
MAP_SHA='cc237461aa97a4b33ebca3d794cba0b158139c137ce2323cbb9b2ebda0cd50b6'
PATCH_SHA='947b679cf5833dcef021245117a53f49d0c04eace0b32e911b492303ae6149aa'
MAX_BYTES=100*1024*1024;INNER_BYTES=96*1024*1024;MAX_FILES=2000
CLOSURE_RESERVE=120;STATE_RESERVE=780

def require(ok,why):
 if not ok:raise ValueError(why)
def sha(p):
 p=Path(p);require(stat.S_ISREG(p.lstat().st_mode),'REGULAR_FILE');h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('x')as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def name(n):
 require(type(n)is str and n and '\\'not in n and not PurePosixPath(n).is_absolute() and '..'not in PurePosixPath(n).parts and PurePosixPath(n).as_posix()==n,'CONFINED_NAME');return n
def inventory(root):
 root=Path(root);require(root.is_dir()and not root.is_symlink(),'REAL_ROOT');out={}
 for p in sorted(root.rglob('*')):
  require(not p.is_symlink(),'SOURCE_SYMLINK');require(p.is_dir()or stat.S_ISREG(p.lstat().st_mode),'SOURCE_NONREGULAR')
  if p.is_file():require(p.suffix!='.pyc','BYTECODE');out[p.relative_to(root).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size}
 return out
def limits(files,*,inner=True):
 require(type(files)is dict and len(files)<=MAX_FILES,'MEMBER_LIMIT')
 for n,v in files.items():name(n);require(set(v)=={'sha256','bytes'}and type(v['bytes'])is int and v['bytes']>=0 and isinstance(v['sha256'],str)and re.fullmatch('[0-9a-f]{64}',v['sha256'])is not None,'MEMBER_DESCRIPTOR')
 require(sum(v['bytes']for v in files.values())<=(INNER_BYTES if inner else MAX_BYTES),'PHYSICAL_BYTES_LIMIT')
def verify_files(root,files):
 require(inventory(root)==files,'EXACT_PHYSICAL_INVENTORY')
def recover(archive,expected,target,*,packet=False):
 archive=Path(archive);target=Path(target);require(sha(archive)==expected and archive.stat().st_size<=MAX_BYTES,'WHOLE_ARCHIVE_SHA_SIZE');require(not target.exists()and not target.is_symlink(),'FRESH_RECOVERY_ROOT')
 with zipfile.ZipFile(archive)as z:
  infos=z.infolist();require(len(infos)<=MAX_FILES and len({i.filename for i in infos})==len(infos),'ARCHIVE_MEMBER_COUNT')
  require(sum(i.file_size for i in infos)<=INNER_BYTES,'ARCHIVE_PHYSICAL_BYTES')
  for i in infos: name(i.filename);require(not i.is_dir()and ((i.external_attr>>16)&0o170000)in (0,0o100000),'ARCHIVE_REGULAR_MEMBERS')
  target.mkdir(mode=0o700)
  try:
   for i in infos:
    p=target/i.filename;p.parent.mkdir(parents=True,exist_ok=True)
    with z.open(i)as src,p.open('xb')as dst:
     total=0
     for b in iter(lambda:src.read(1024*1024),b''):total+=len(b);require(total<=i.file_size,'ARCHIVE_LENGTH');dst.write(b)
     require(total==i.file_size,'ARCHIVE_TRUNCATION')
   m=read(target/('manifest.json'if packet else'archive-inventory.json'));files=m['files'];limits(files)
   verify_files(target,{**files,('manifest.json'if packet else'archive-inventory.json'):{'sha256':sha(target/('manifest.json'if packet else'archive-inventory.json')),'bytes':(target/('manifest.json'if packet else'archive-inventory.json')).stat().st_size}})
   if packet:admit(target,m)
   return m
  except BaseException:
   import shutil;shutil.rmtree(target);raise

def packet_preflight(packet,archive,expected):
 import tempfile
 with tempfile.TemporaryDirectory(prefix='bnb-m4-packet-preflight-')as temporary:
  recovered=Path(temporary)/'payload';m=recover(archive,expected,recovered,packet=True)
  require(inventory(packet)==inventory(recovered),'LOOSE_PACKET_DIFFERS_FROM_ADMITTED_ARCHIVE')
 return m

def admit(root,m=None):
 root=Path(root);m=read(root/'manifest.json')if m is None else m
 require(m.get('kind')=='M4_EXACT_NESTED_PACKET_V1'and m.get('science_manifest_sha256')==SCIENCE_SHA and m.get('allocation_seconds')==3600 and m.get('closure_reserve_seconds')==CLOSURE_RESERVE and m.get('state_start_reserve_seconds')==STATE_RESERVE,'PACKET_PROFILE')
 limits(m['files']);verify_files(root,{**m['files'],'manifest.json':{'sha256':sha(root/'manifest.json'),'bytes':(root/'manifest.json').stat().st_size}})
 n=root/'nested-exact';require(sha(n/'source/science-manifest.json')==SCIENCE_SHA and sha(n/'integration/source-admission.json')==ADMISSION_SHA and sha(n/'integration/adoption-map.json')==MAP_SHA and sha(n/'integration/patch-manifest.json')==PATCH_SHA,'PINNED_SOURCE_DOCUMENTS')
 import sys;sys.path.insert(0,str(n/'source'));import generation_gate as G;import exact_admission as A
 G.verify(n/'integration/source-admission.json',ADMISSION_SHA,plugin=n/'integration/plugin',upstream=n/'integration/upstream/bitsandbytes')
 s=read(n/'source/science-manifest.json')
 for key,rec in s['files'].items():require(sha(n/'source'/name(key))==rec['sha256']and(n/'source'/key).stat().st_size==rec['bytes'],'SCIENCE_SOURCE_BYTES')
 A.reviewer_dependency(read(root/'source-review.json'),sha(root/'source-review.json'),record_path=root/'source-review.json')
 require(read(root/'source-review.json')['science_map_sha256']==SCIENCE_SHA,'ROOT_SOURCE_REVIEW_MANIFEST')
 return m

def budget(birth,requested,reserve=CLOSURE_RESERVE,now=None):
 now=time.time()if now is None else now
 require(all(type(x)in(int,float)and math.isfinite(x)for x in(birth,requested,reserve,now))and 0<=reserve<3600 and requested>0 and birth<=now,'FINITE_ORIGINAL_ALLOCATION_BUDGET')
 value=min(requested,birth+3600-now-reserve)
 if value<=0:raise TimeoutError('ORIGINAL_ALLOCATION_DEADLINE_RESERVE')
 return value

def accepted79(path,expected,*,packet_sha,receipt_sha,oracle_sha,completed_epoch,birth,now=None):
 import sys;sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'nested-exact/source'))
 # The remote extracted layout is control/ beside nested-exact/.
 if not(Path(__file__).resolve().parents[2]/'nested-exact/source').exists():sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'nested-exact/source'))
 import exact_admission as A
 require(sha(path)==expected,'ROOT_ACCEPTANCE_BYTES');r=read(path);A.state_dependency(r,path,expected,SCIENCE_SHA)
 require(r['science_map_sha256']==SCIENCE_SHA and r['packet_sha256']==packet_sha and r['actual_receipt_sha256']==receipt_sha and r['fresh_cpu_oracle_sha256']==oracle_sha,'ROOT_ACCEPTANCE_RECOVERED_DEPENDENCIES')
 now=time.time()if now is None else now
 require(type(completed_epoch)in(int,float)and math.isfinite(completed_epoch)and birth<=completed_epoch<=now,'ACCEPTED79_COMPLETION_TIMELINE');budget(birth,1,STATE_RESERVE,now=now);return r

def export(root,destination):
 root=Path(root);destination=Path(destination);require(not destination.exists()and root.resolve()not in destination.resolve().parents,'EXPORT_OUTSIDE_RECORDS')
 files=inventory(root);require('archive-inventory.json'not in files,'FRESH_ARCHIVE_INVENTORY');limits(files)
 try:
  write(root/'archive-inventory.json',{'kind':'M4_EXACT_NESTED_PHYSICAL_ENVELOPE_V1','files':files})
  complete=inventory(root);limits(complete)
  with zipfile.ZipFile(destination,'x',zipfile.ZIP_DEFLATED)as z:
   for n in sorted(complete):z.write(root/n,n)
  require(destination.stat().st_size<=MAX_BYTES,'OUTER_ARCHIVE_LIMIT');return {'sha256':sha(destination),'bytes':destination.stat().st_size,'physical_bytes':sum(x['bytes']for x in complete.values()),'members':len(complete)}
 except BaseException:
  destination.unlink(missing_ok=True);raise
 finally:(root/'archive-inventory.json').unlink(missing_ok=True)
