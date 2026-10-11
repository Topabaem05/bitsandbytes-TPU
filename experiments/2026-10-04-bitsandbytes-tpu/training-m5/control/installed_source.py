"""Fresh installed source readback, before importing public packages."""
import argparse,importlib.util,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.packet/'source'));import admission as A
admission=A.source();roots={name:Path(importlib.util.find_spec(name).origin).resolve().parent for name in ['bitsandbytes','bitsandbytes_tpu']};A.installed(admission,roots)
r={'kind':'M5_INSTALLED_SOURCE_READBACK','status':'SOURCE_BYTES_PASS','source_admission_sha256':A.sha(a.packet/'integration/source-admission.json'),'upstream_python_files':admission['upstream_python_files'],'plugin_python_files':admission['plugin_python_files'],'py_typed_sha256':admission['upstream_package_files']['py.typed']};A.write(a.output,r)
