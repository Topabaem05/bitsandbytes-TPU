"""Download the locked official wheel for static evidence only. Never install or load it."""
import argparse,hashlib,json,time,urllib.request,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();records=json.loads(a.lock.read_text())
# The lock schema is discovered from this exact current file.
def find(x):
 if isinstance(x,dict):
  if x.get('name')=='libtpu'and x.get('version')=='0.0.21'and'filename'in x:return x
  for v in x.values():
   r=find(v)
   if r:return r
 if isinstance(x,list):
  for v in x:
   r=find(v)
   if r:return r
r=find(records);assert r and r['size']==149753420 and r['sha256']=='0a5aaf71f45f8d72f1298b175d076ac2ae806ce7ab4b7d81c02315bd3adc5021';a.output.mkdir(exist_ok=False);target=a.output/r['filename'];started=time.monotonic();h=hashlib.sha256();size=0
with urllib.request.urlopen(r['url'],timeout=30)as src,target.open('xb')as dst:
 while chunk:=src.read(4*1024*1024):
  size+=len(chunk);assert size<=r['size'] and time.monotonic()-started<100;h.update(chunk);dst.write(chunk)
assert size==r['size']and h.hexdigest()==r['sha256']
with zipfile.ZipFile(target)as z:
 members=[{'name':i.filename,'bytes':i.file_size,'compressed_bytes':i.compress_size}for i in z.infolist()];assert len(members)==r['zip_members']
 metadata=z.read('libtpu-0.0.21.dist-info/METADATA');assert hashlib.sha256(metadata).hexdigest()==r['metadata_sha256'];(a.output/'wheel-metadata.txt').write_bytes(metadata)
(a.output/'wheel-readback.json').write_text(json.dumps({'status':'LOCKED_OFFICIAL_WHEEL_EXACT','lock_record':r,'members':members,'elapsed_seconds':time.monotonic()-started,'installation':'NOT_RUN','library_execution':'NOT_RUN'},sort_keys=True,indent=2)+'\n');print(json.dumps({'status':'LOCKED_OFFICIAL_WHEEL_EXACT','members':members}))
