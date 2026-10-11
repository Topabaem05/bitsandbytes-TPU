"""Complete source archive correct and inconsistent member fixtures."""
import io,json,sys,tarfile,tempfile
from pathlib import Path
from bundle_readback import verify,sha
rows=[]
with tempfile.TemporaryDirectory(prefix='bnb-m5-v4-bundle-fixture-',dir='/tmp')as td:
 p=Path(td);(p/'a').write_bytes(b'correct');manifest={'files':{'a':{'sha256':sha(p/'a'),'bytes':7}}};(p/'source-closure.json').write_text(json.dumps(manifest))
 def bundle(path,mode):
  with tarfile.open(path,'w:gz')as t:
   names=['a','source-closure.json']
   if mode=='missing':names.remove('a')
   if mode=='extra':names.append('extra')
   if mode=='duplicate':names.append('a')
   for name in names:
    b=b'incorrect'if mode=='wrong'and name=='a'else b'extra'if name=='extra'else(p/name).read_bytes();i=tarfile.TarInfo(name);i.size=len(b)
    if mode=='symlink'and name=='a':i.type=tarfile.SYMTYPE;i.linkname='/outside';i.size=0;t.addfile(i)
    else:t.addfile(i,io.BytesIO(b))
 for mode in ['correct','wrong','missing','extra','duplicate','symlink']:
  archive=p/(mode+'.tar.gz');bundle(archive,mode)
  try:r=verify(p,archive,sha(archive))
  except ValueError as e:
   assert mode!='correct';rows.append({'name':mode,'status':'PASS','rejected':str(e)})
  else:assert mode=='correct';rows.append({'name':mode,'status':'PASS','evidence':r})
result={'status':'PASS','controls':len(rows),'cases':rows,'fixtures_deleted':not Path(td).exists(),'provider_calls':0}
Path(sys.argv[1]).write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps({'status':'PASS','controls':len(rows),'fixtures_deleted':True,'provider_calls':0}))
