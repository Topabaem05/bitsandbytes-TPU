"""Read all installed admitted bytes in a fresh process before public imports."""
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.packet/'nested-exact/source'));import exact_admission as A
admission,roots=A.admit(a.packet/'nested-exact/integration/source-admission.json',A.INTEGRATION_ADMISSION_SHA,a.packet/'nested-exact/integration/patch-manifest.json')
r={'kind':'NESTED_EXACT_INSTALLED_SOURCE_READBACK','status':'SOURCE_BYTES_PASS','source_admission_sha256':A.INTEGRATION_ADMISSION_SHA,'upstream_python_files':admission['upstream_python_files'],'plugin_python_files':admission['plugin_python_files'],'py_typed_sha256':admission['upstream_package_files']['py.typed']}
a.output.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
