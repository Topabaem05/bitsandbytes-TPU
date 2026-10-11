"""Dedicated exact-nested admission and bounded regular-file recovery. Stdlib only."""
import hashlib,json,math,os,re,stat,zipfile,time
from pathlib import Path,PurePosixPath
MAX_BYTES=100*1024*1024;INNER_BYTES=96*1024*1024;MAX_FILES=2000
CPU_BYTES=64*1024*1024
CLOSURE_RESERVE=660
CPU_SECONDS=300;DEVICE_SECONDS=1500;INSTALL_SECONDS=1200

def require(ok,why):
 if not ok:raise ValueError(why)
def sha(p):
 p=Path(p);require(stat.S_ISREG(p.lstat().st_mode),'REGULAR_FILE');h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def json_pairs(rows):
 result={}
 for k,v in rows:require(k not in result,'DUPLICATE_JSON_KEY');result[k]=v
 return result
def read(p):return json.loads(Path(p).read_text(),object_pairs_hook=json_pairs)
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
 with tempfile.TemporaryDirectory(prefix='bnb-m5-packet-preflight-')as temporary:
  recovered=Path(temporary)/'payload';m=recover(archive,expected,recovered,packet=True)
  require(inventory(packet)==inventory(recovered),'LOOSE_PACKET_DIFFERS_FROM_ADMITTED_ARCHIVE')
 return m

def source(root):
 import importlib.util
 p=Path(root)/'source/admission.py';spec=importlib.util.spec_from_file_location('m5_packet_admission',p);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
 a.source();return a

def science(root):return sha(Path(root)/'source/science-manifest.json')

def source_review(root,path):
 r=read(path);require(r.get('kind')=='ROOT_M5_SOURCE_REVIEW'and r.get('status')=='SOURCE_REVIEW_ACCEPTED_FOR_EXPERIMENT'and r.get('science_manifest_sha256')==science(root)and r.get('source_admission_sha256')==sha(Path(root)/'integration/source-admission.json')and r.get('device_acceptance')is None,'EXPLICIT_ROOT_M5_SOURCE_REVIEW')
 require(re.fullmatch('[0-9a-f]{40}',r.get('reviewed_base_commit',''))is not None and r.get('m4_candidate_source_tar_sha256')=='dae4003539f09e3c784dc77234ecc817503059655d2b282f2210df8585297020','REVIEWED_BASE_SOURCE')
 return r

def admit(root,m=None):
 root=Path(root);m=read(root/'manifest.json')if m is None else m
 require(m.get('kind')=='M5_EXACT1024_TRAINING_PACKET_V1'and m.get('science_manifest_sha256')==science(root)and m.get('allocation_seconds')==3600 and m.get('closure_reserve_seconds')==660 and m.get('budget')=={'total':3600,'install':1200,'science':1800,'retrieval':600,'cleanup':60}and m.get('scope')=={'steps':20,'checkpoint_after_update':10,'resume_steps':[11,20],'first_scales':1024},'PACKET_PROFILE')
 limits(m['files']);verify_files(root,{**m['files'],'manifest.json':{'sha256':sha(root/'manifest.json'),'bytes':(root/'manifest.json').stat().st_size}})
 a=source(root);source_review(root,root/'source-review.json');a.m4_dependency(root/'accepted-m4.json',m['m4_acceptance_sha256'])
 require(sha(root/'runtime/requirements.lock.json')==a.RUNTIME_LOCK,'PINNED_RUNTIME_LOCK')
 return m


def budget(birth,requested,reserve=CLOSURE_RESERVE,now=None):
 now=time.time()if now is None else now
 require(all(type(x)in(int,float)and math.isfinite(x)for x in(birth,requested,reserve,now))and 0<=reserve<3600 and requested>0 and birth<=now,'FINITE_ORIGINAL_ALLOCATION_BUDGET')
 value=min(requested,birth+3600-now-reserve)
 if value<=0:raise TimeoutError('ORIGINAL_ALLOCATION_DEADLINE_RESERVE')
 return value


def export(root,destination):
 root=Path(root);destination=Path(destination);require(not destination.exists()and root.resolve()not in destination.resolve().parents,'EXPORT_OUTSIDE_RECORDS')
 files=inventory(root);
 if not(root/'device').exists():require(sum(r['bytes']for r in files.values())<=CPU_BYTES,'CPU_ARCHIVE_64MIB_BOUND')
 require('archive-inventory.json'not in files,'FRESH_ARCHIVE_INVENTORY');limits(files)
 try:
  write(root/'archive-inventory.json',{'kind':'M5_EXACT1024_PHYSICAL_ENVELOPE_V1','files':files})
  complete=inventory(root);limits(complete)
  with zipfile.ZipFile(destination,'x',zipfile.ZIP_DEFLATED)as z:
   for n in sorted(complete):z.write(root/n,n)
  require(destination.stat().st_size<=MAX_BYTES,'OUTER_ARCHIVE_LIMIT');return {'sha256':sha(destination),'bytes':destination.stat().st_size,'physical_bytes':sum(x['bytes']for x in complete.values()),'members':len(complete)}
 except BaseException:
  destination.unlink(missing_ok=True);raise
 finally:(root/'archive-inventory.json').unlink(missing_ok=True)

def runtime_cache(packet,cache):
 cache=Path(cache);lock=read(Path(packet)/'runtime/requirements.lock.json');require(cache.is_dir()and not cache.is_symlink(),'EXISTING_RUNTIME_CACHE')
 require(len(lock['wheels'])==35 and sum(r['size']for r in lock['wheels'])==572327598,'FIXED_RUNTIME_CACHE_PROFILE')
 for r in lock['wheels']:
  require(Path(r['filename']).name==r['filename'],'WHEEL_NAME');p=cache/r['filename'];require(p.is_file()and not p.is_symlink()and p.stat().st_size==r['size']and sha(p)==r['sha256'],'MISSING_OR_WRONG_LOCKED_RUNTIME:'+r['filename'])
 return lock['wheels']

def closed(summary):
 require(summary.get('errors')==[]and summary.get('status')=='NO_CLEANUP_ERRORS','OWNED_CLEANUP_ERRORS')
 for g in summary.get('groups',[]):require(g.get('leader_reaped')is True and g.get('status')=='CLEANUP_VERIFIED'and g.get('group_absence')=='OBSERVED_NO_SUCH_GROUP'and g.get('errors')==[],'OWNED_GROUP_CLOSURE')
 return summary
