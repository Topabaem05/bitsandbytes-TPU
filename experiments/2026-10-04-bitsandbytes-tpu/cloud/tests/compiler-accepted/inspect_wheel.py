"""Read the locked library bytes and exact flag-name strings without executing a library."""
import argparse,hashlib,json,mmap,shutil,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--wheel',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
with zipfile.ZipFile(a.wheel)as z:
 assert hashlib.sha256(a.wheel.read_bytes()).hexdigest()=='0a5aaf71f45f8d72f1298b175d076ac2ae806ce7ab4b7d81c02315bd3adc5021'
 assert z.getinfo('libtpu/libtpu.so').file_size==376021592
 with z.open('libtpu/libtpu.so')as src,(a.output/'libtpu.so').open('xb')as dst:shutil.copyfileobj(src,dst,4*1024*1024)
 (a.output/'libtpu-init.py').write_bytes(z.read('libtpu/__init__.py'))
lib=a.output/'libtpu.so';h=hashlib.sha256()
with lib.open('rb')as f:
 for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
assert h.hexdigest()=='cdb7980d4332097b8e16576568e138ef54cf9fb8ed5aae2aeefbd0c5a425e94a'
flags=['xla_dump_to','xla_dump_hlo_as_text','xla_dump_hlo_as_proto','xla_dump_module_metadata','xla_dump_hlo_pass_re','xla_dump_include_timestamp','xla_dump_max_hlo_modules','xla_dump_compress_protos','xla_dump_hlo_snapshots','xla_dump_hlo_unoptimized_snapshots','xla_dump_full_hlo_config','xla_dump_large_constants','xla_dump_hlo_as_dot','xla_dump_hlo_as_html','xla_dump_hlo_as_url'];rows={}
with lib.open('rb')as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)as data:
 for name in flags:
  needle=name.encode()+b'\0';offsets=[];start=0
  while (pos:=data.find(needle,start))!=-1:
   if pos==0 or data[pos-1]==0:offsets.append(pos)
   start=pos+len(needle)
  rows[name]={'exact_null_terminated_name_offsets':offsets,'present':bool(offsets)}
 (a.output/'flag-strings.json').write_text(json.dumps({'status':'FIXED_WHEEL_STATIC_NAMES_ONLY','library_bytes':lib.stat().st_size,'library_sha256':h.hexdigest(),'actual_runtime_library_sha256_match':True,'complete_runtime_registry':'NOT_OBTAINED','parser_execution':'NOT_RUN_MACOS_HOST','flags':rows},sort_keys=True,indent=2)+'\n')
print(json.dumps({'status':'FIXED_WHEEL_STATIC_NAMES_ONLY','flags':rows}))
