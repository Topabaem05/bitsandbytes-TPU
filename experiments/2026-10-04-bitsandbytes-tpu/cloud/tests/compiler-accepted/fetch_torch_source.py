"""Read primary release source matching the actual installed frontend source bytes."""
import hashlib,json,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;out=HERE/'evidence/torch-xla-source';out.mkdir(exist_ok=False);url='https://raw.githubusercontent.com/pytorch/xla/5fab7053df86c8d503b98d9e7202ca8b8d4978c7/torch_xla/__init__.py'
with urllib.request.urlopen(url,timeout=15)as response:data=response.read(1024*1024+1)
assert len(data)<=1024*1024;(out/'torch_xla-init.py').write_bytes(data);digest=hashlib.sha256(data).hexdigest();assert digest=='c3e11a63b8ebe9dcf9ca94d4f7ceebba4983d49e98f57ceddeb79880063eb29c'
(out/'readback.json').write_text(json.dumps({'url':url,'sha256':digest,'bytes':len(data),'status':'MATCHES_ACTUAL_FRONTEND_SOURCE','libtpu_internal_revision':'UNKNOWN'},sort_keys=True,indent=2)+'\n');print(json.dumps({'status':'MATCHES_ACTUAL_FRONTEND_SOURCE'}))
