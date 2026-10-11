"""Separate diagnostic admission; canonical confined archive functions are retained."""
import hashlib,json,math,os,re,stat,zipfile,time
from pathlib import Path,PurePosixPath
MAX_BYTES=32*1024*1024;INNER_BYTES=30*1024*1024;MAX_FILES=1000
CLOSURE_RESERVE=60
GENERATION="m4-word-lowering-diagnostic-v2-raw-input-boundary"
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


def admit(root,m=None):
 root=Path(root);m=read(root/'manifest.json')if m is None else m
 require(m.get('kind')=='M4_WORD_DIAGNOSTIC_PACKET_V2'and m.get('generation')==GENERATION and m.get('allocation_seconds')==3600 and m.get('diagnostic_seconds')==600 and m.get('closure_reserve_seconds')==60 and m.get('m4_status')=='NOT_QUALIFIED','DIAGNOSTIC_PACKET_PROFILE')
 limits(m['files']);verify_files(root,{**m['files'],'manifest.json':{'sha256':sha(root/'manifest.json'),'bytes':(root/'manifest.json').stat().st_size}})
 require(sha(root/'runtime/requirements.lock.json')=='323371ff61c5fbcc4f79fd6a358cf2ba17cb72b907382a5ceaac07f91dc66ed6','DIAGNOSTIC_RUNTIME_PIN')
 require(sha(root/'original/parent-science-manifest.json')=='29beb28f4320acc5e07ce6822f2120eaff846809bfc54bd123ac37f501700c06'and sha(root/'original/parent-source-admission.json')=='8b5aa5f0c86346db982664cfad3b1e242dc8a67a4e49865584ccc01c8620f4bf'and sha(root/'original/parent-adoption-map.json')=='bfd7ce45a900fee3bc5ecfed54524c17931e2964ef5b7295fca6e6038883729a','CURRENT_ORIGINAL_SOURCE_GENERATION')
 require(sha(root/'source/selected-inputs.json')=='cb67e828e7201ffe80c5eeaad6bfdef99f8bc1caaa508901b511eb929d096390','FROZEN_SELECTED_MAP_BYTES')
 parent=read(root/'original/parent-adoption-map.json')['plugin_files']
 for path in(root/'original').glob('*.py'):require(sha(path)==parent['src/bitsandbytes_tpu/'+path.name],'ORIGINAL_PLUGIN_PARENT_MAP_BYTES')
 require(read(root/'source/selected-inputs.json')['parent_inputs_sha256']=='b7490f4c622aef793d14bd9a7a38727754cb1fb8e6e1dec7ff91ef8c45fbafbc','FROZEN_MAP_EXTRACTION_PARENT')
 for variant,files in read(root/'source/rewrite-records.json').items():
  for module,record in files.items():require(sha(root/'original'/module)==record['preimage_sha256']and sha(root/'variants'/variant/module)==record['postimage_sha256'],'DIAGNOSTIC_VARIANT_PREIMAGE_POSTIMAGE')
 require(sha(root/'control/kernel_protocol.py')=='e9c7a5b7b943cea08dfb8986205e47e008ceac37553a2f8576687527a58f9d5f','UNCHANGED_KERNEL_COMPLETION_HELPER')
 return m

def budget(birth,requested,reserve=CLOSURE_RESERVE,now=None):
 now=time.time()if now is None else now
 require(all(type(x)in(int,float)and math.isfinite(x)for x in(birth,requested,reserve,now))and 0<=reserve<600 and requested>0 and birth<=now,'FINITE_ORIGINAL_DIAGNOSTIC_BUDGET')
 value=min(requested,birth+600-now-reserve)
 if value<=0:raise TimeoutError('ORIGINAL_600_SECOND_DIAGNOSTIC_DEADLINE')
 return value

def closed(summary):
 require(summary.get('status')=='NO_CLEANUP_ERRORS'and summary.get('errors')==[],'OWNED_CLEANUP_ERRORS')
 for group in summary.get('groups',[]):require(group.get('status')=='CLEANUP_VERIFIED'and group.get('leader_reaped')is True and group.get('group_absence')=='OBSERVED_NO_SUCH_GROUP'and group.get('errors')==[],'OWNED_GROUP_CLOSURE')
