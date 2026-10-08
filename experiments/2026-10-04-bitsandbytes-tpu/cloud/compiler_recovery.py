"""Separate post-closure archive. No executable identity or allocator claim."""
import hashlib,json,os,stat,zipfile
from pathlib import Path,PurePosixPath
import importlib.util
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_compiler_archive_contract',HERE/'compiler_cloud_contract.py')
CC=importlib.util.module_from_spec(spec);spec.loader.exec_module(CC)
METADATA=('expected.json','exec.json','monitor.json','postclosure.json')
METADATA_CAP=8*1024*1024
TOTAL_CAP=128*1024*1024+METADATA_CAP

def closed(outer,*,live=False):
 c=outer.get('cleanup',{})
 CC.require(c.get('status')=='CLEANUP_VERIFIED' and c.get('leader_reaped') is True and c.get('group_absence')=='OBSERVED_NO_SUCH_GROUP' and c.get('errors')==[],'COMPILER_EXTERNAL_GROUP_CLOSURE')
 CC.require(type(outer.get('pid')) is int and outer['pid']>1 and outer['pid']==outer.get('pgid'),'COMPILER_OUTER_LEADER')
 if live:
  try:os.killpg(outer['pgid'],0)
  except ProcessLookupError:pass
  else:raise ValueError('COMPILER_GROUP_STILL_PRESENT')

def regular(p):
 s=p.lstat();CC.require(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and not p.is_symlink(),'COMPILER_ARCHIVE_FILE_TYPE');return s

def record(p):return {'bytes':regular(p).st_size,'sha256':CC.sha(p)}

def seal(out,outer,hook):
 """Metadata remains available when bounded raw capture fails; never delete raw evidence."""
 out=Path(out);root=out/'native.compiler-private';closed(outer,live=True)
 CC.require(root.resolve()==root.absolute() and not root.is_symlink(),'COMPILER_ARCHIVE_ROOT_TYPE')
 files={};metadata_bytes=0;directories=[]
 for n in METADATA:
  if (root/n).exists():
   r=record(root/n);metadata_bytes+=r['bytes'];CC.require(metadata_bytes<=METADATA_CAP,'COMPILER_METADATA_LIMIT');files[n]=r
 status='METADATA_ONLY_CAPTURE_FAILED';reason=hook.get('error','POSTCLOSURE_UNAVAILABLE')
 if hook.get('status')=='POSTCLOSURE_SEALED':
  post=CC.read(root/'postclosure.json');snapshot=post['snapshot']
  for n,r in snapshot['files'].items():
   p=root/'raw-private'/n;CC.require(record(p)=={'bytes':r['bytes'],'sha256':r['sha256']},'COMPILER_POSTCLOSURE_CHANGED');files['raw-private/'+n]={'bytes':r['bytes'],'sha256':r['sha256']}
  directories=sorted(p.relative_to(root).as_posix() for p in (root/'raw-private').rglob('*') if p.is_dir())
  CC.require(len(directories)+len(snapshot['files'])==snapshot['entries'],'COMPILER_DIRECTORY_INVENTORY')
  status='SEALED_BOUNDED_RAW_CAPTURE';reason=None
 archive=out/'compiler-evidence.zip';inventory={**CC.NATIVE_DEPENDENCY_BINDINGS,'kind':'R6_SEPARATE_COMPILER_ARCHIVE_V1','status':status,'error':reason,'manifest_sha256':CC.MANIFEST_SHA,'policy_sha256':CC.POLICY_SHA,'outer_ownership_sha256':CC.sha(out/'steps/12-native-parent/ownership.json'),'files':files,'directories':directories,'raw_recovery':'COMPLETE_BOUNDED' if status=='SEALED_BOUNDED_RAW_CAPTURE' else 'OMITTED_UNQUALIFIED','hard_aggregate_quota':'UNAVAILABLE','polling_overshoot':'POSSIBLE_UNBOUNDED_BY_WRITER_RATE','selected_executable_link':'UNKNOWN','allocator_peak':'UNKNOWN'}
 CC.write(out/'compiler-inventory.json',inventory)
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for n in sorted(files):
   i=zipfile.ZipInfo(n,date_time=(2026,10,9,0,0,0));i.create_system=3;i.external_attr=0o100644<<16;i.compress_type=zipfile.ZIP_DEFLATED;z.writestr(i,(root/n).read_bytes())
 return {**CC.NATIVE_DEPENDENCY_BINDINGS,'compiler_capture_status':status,'compiler_capture_error':reason,'compiler_evidence_sha256':CC.sha(archive),'compiler_evidence_bytes':archive.stat().st_size,'compiler_inventory_sha256':CC.sha(out/'compiler-inventory.json')}

def recover(root,receipt):
 CC.dependency_fields(receipt);root=Path(root);CC.require(CC.sha(root/'inventory.json')==receipt['compiler_inventory_sha256'] and CC.sha(root/'evidence.zip')==receipt['compiler_evidence_sha256'] and regular(root/'evidence.zip').st_size==receipt['compiler_evidence_bytes'],'COMPILER_RECOVERY_SEAL')
 inv=CC.read(root/'inventory.json');CC.dependency_fields(inv);CC.require(inv['kind']=='R6_SEPARATE_COMPILER_ARCHIVE_V1' and inv['manifest_sha256']==CC.MANIFEST_SHA and inv['policy_sha256']==CC.POLICY_SHA and inv['status']==receipt['compiler_capture_status'] and inv['error']==receipt['compiler_capture_error'],'COMPILER_RECOVERY_SCOPE')
 CC.require(inv['status'] in {'SEALED_BOUNDED_RAW_CAPTURE','METADATA_ONLY_CAPTURE_FAILED'} and (inv['error'] is None)==(inv['status']=='SEALED_BOUNDED_RAW_CAPTURE') and inv['hard_aggregate_quota']=='UNAVAILABLE' and inv['polling_overshoot']=='POSSIBLE_UNBOUNDED_BY_WRITER_RATE','COMPILER_RECOVERY_STATUS')
 files=inv['files'];CC.require(type(files)is dict and len(files)<=1028 and sum(r['bytes'] for r in files.values())<=TOTAL_CAP,'COMPILER_RECOVERY_LIMIT')
 metadata=0;rawtotal=0
 for n,r in files.items():
  p=PurePosixPath(n);CC.require(not p.is_absolute() and '..' not in p.parts and '\\' not in n and str(p)==n and all(x not in ('','.') for x in p.parts),'COMPILER_RECOVERY_PATH')
  CC.require(type(r.get('bytes'))is int and r['bytes']>=0 and set(r)=={'bytes','sha256'},'COMPILER_RECOVERY_RECORD')
  if n in METADATA:metadata+=r['bytes']
  else:
   CC.require(n.startswith('raw-private/') and r['bytes']<=16*1024*1024 and inv['status']=='SEALED_BOUNDED_RAW_CAPTURE','COMPILER_RECOVERY_RAW_SCOPE');rawtotal+=r['bytes']
 CC.require(metadata<=METADATA_CAP and rawtotal<=128*1024*1024,'COMPILER_RECOVERY_BYTE_LIMIT')
 dirs=inv['directories'];CC.require(type(dirs)is list and dirs==sorted(set(dirs)) and len(dirs)+len([n for n in files if n.startswith('raw-private/')])<=1024,'COMPILER_RECOVERY_DIRECTORIES')
 for n in dirs:
  p=PurePosixPath(n);CC.require(str(p)==n and not p.is_absolute() and '..' not in p.parts and '\\' not in n and n.startswith('raw-private/') and n not in files and inv['status']=='SEALED_BOUNDED_RAW_CAPTURE','COMPILER_RECOVERY_DIRECTORY_PATH')
 target=root/'recovered';target.mkdir(exist_ok=False)
 for n in dirs:(target/n).mkdir(parents=True,exist_ok=True)
 if inv['status']=='SEALED_BOUNDED_RAW_CAPTURE':(target/'raw-private').mkdir(exist_ok=True)
 with zipfile.ZipFile(root/'evidence.zip')as z:
  CC.require(len(z.infolist())==len(files) and set(z.namelist())==set(files),'COMPILER_RECOVERY_MEMBER_SET')
  for i in z.infolist():
   CC.require(i.create_system==3 and i.external_attr==0o100644<<16 and i.file_size==files[i.filename]['bytes'],'COMPILER_RECOVERY_MEMBER_TYPE')
   b=z.read(i);CC.require(len(b)==files[i.filename]['bytes'] and hashlib.sha256(b).hexdigest()==files[i.filename]['sha256'],'COMPILER_RECOVERY_MEMBER_BYTES');q=target/i.filename;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b)
 if inv['status']=='SEALED_BOUNDED_RAW_CAPTURE':
  post=CC.read(target/'postclosure.json');CC.require(post['snapshot']['entries']==len(dirs)+len([n for n in files if n.startswith('raw-private/')]),'COMPILER_RECOVERY_DIRECTORY_COUNT');expected={'raw-private/'+n:{'sha256':r['sha256'],'bytes':r['bytes']}for n,r in post['snapshot']['files'].items()}
  CC.require({n:r for n,r in files.items()if n.startswith('raw-private/')}==expected,'COMPILER_RECOVERY_POSTCLOSURE_INVENTORY')
 return target,inv

def readback_missing(receipt,output):
 missing=[]
 if receipt.get('compiler_capture_status')!='SEALED_BOUNDED_RAW_CAPTURE':missing.append('complete_bounded_raw_capture')
 for k in ('compiler_expected_sha256','compiler_postclosure_sha256','compiler_inventory_sha256','compiler_evidence_sha256'):
  if k not in receipt:missing.append(k)
 for n in ('expected.json','postclosure.json','exec.json','monitor.json'):
  if not (Path(output)/'compiler-recovery/recovered'/n).is_file():missing.append(n)
 return missing
