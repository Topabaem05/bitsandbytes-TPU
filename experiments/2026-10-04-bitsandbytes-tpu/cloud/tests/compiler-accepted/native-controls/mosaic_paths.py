"""Admit exactly the frozen compiler34 source before control imports."""
import hashlib,json,sys
from pathlib import Path
NATIVE_MANIFEST_SHA='b46a21cf9d5fa8133df456c22b31ac31ec03984fd22bc61cb93f910ef87e3dbd'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def admit(native):
    native=Path(native).resolve();require(sha(native/'manifest.json')==NATIVE_MANIFEST_SHA,'CONTROL_COMPILER34_MANIFEST')
    manifest=json.loads((native/'manifest.json').read_text());require(len(manifest['sources'])==34 and manifest['generation']=='compiler-mosaic-serde7-v1','CONTROL_COMPILER34_SCOPE')
    for name,row in manifest['sources'].items():
        path=native/name;require(not path.is_symlink() and path.is_file() and sha(path)==row['sha256'] and path.stat().st_size==row['bytes'],'CONTROL_NATIVE_SOURCE:'+name)
    sys.path.insert(0,str(native));return native
