"""Admit exactly the frozen native27 source before control imports."""
import hashlib,json,sys
from pathlib import Path
NATIVE_MANIFEST_SHA='d294b189011f8200e039761f054c5cc96a15275b679efeb7bdde1ef67b414ffa'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def admit(native):
    native=Path(native).resolve();require(sha(native/'manifest.json')==NATIVE_MANIFEST_SHA,'CONTROL_NATIVE27_MANIFEST')
    manifest=json.loads((native/'manifest.json').read_text());require(len(manifest['sources'])==27 and manifest['generation']=='mosaic-serde7-v1','CONTROL_NATIVE27_SCOPE')
    for name,row in manifest['sources'].items():
        path=native/name;require(not path.is_symlink() and path.is_file() and sha(path)==row['sha256'] and path.stat().st_size==row['bytes'],'CONTROL_NATIVE_SOURCE:'+name)
    sys.path.insert(0,str(native));return native
