"""Save fixed primary source inputs. No library load or provider operation."""
import hashlib,json,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;out=HERE/'evidence/pinned-openxla';out.mkdir(exist_ok=False);urls={'jax-xla-revision.bzl':'https://raw.githubusercontent.com/jax-ml/jax/jax-v0.7.1/third_party/xla/revision.bzl','debug_options_flags.cc':'https://raw.githubusercontent.com/openxla/xla/31eb6029e8973337ef6c99810c27e1d807791c11/xla/debug_options_flags.cc','parse_flags_from_env.cc':'https://raw.githubusercontent.com/openxla/xla/31eb6029e8973337ef6c99810c27e1d807791c11/xla/parse_flags_from_env.cc'};rows={}
for name,url in urls.items():
 with urllib.request.urlopen(url,timeout=15)as response:data=response.read(1024*1024+1)
 assert 0<len(data)<=1024*1024;(out/name).write_bytes(data);rows[name]={'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
(out/'source-records.json').write_text(json.dumps({'status':'PINNED_JAXLIB071_UPSTREAM_CONTEXT_NOT_LIBTPU_BUILD_ID','xla_commit':'31eb6029e8973337ef6c99810c27e1d807791c11','libtpu_internal_commit':'UNKNOWN','files':rows},sort_keys=True,indent=2)+'\n');print(json.dumps({'status':'PINNED_PRIMARY_SOURCES_SAVED','files':len(rows)}))
