"""Build the new private compiler generation. Keep all scientific bodies unchanged."""
import ast,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;N=HERE/'compiler-native';sys.path.insert(0,str(N));import compiler_contract as C
GEN='compiler-mosaic-serde7-gather-bf16-fp32-libtpu021-flags-v1'
def write(p,x):p.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')
def pin(p,name,value):
 s=p.read_text();lines=s.splitlines(keepends=True);node=next(n for n in ast.parse(s).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id==name for t in n.targets));lines[node.lineno-1:node.end_lineno]=[name+'='+repr(value)+'\n'];p.write_text(''.join(lines))
p= C.read(N/'compiler-policy.json');p['sources']=C.sources();write(N/'compiler-policy.json',p)
m=C.read(N/'manifest.json');m['generation']=GEN;m['compiler_policy_sha256']=C.sha(N/'compiler-policy.json');names=set(m['sources'])|{'compiler_flags.py','compiler_flag_probe.py'};m['sources']={n:{'sha256':C.sha(N/n),'bytes':(N/n).stat().st_size}for n in sorted(names)};write(N/'manifest.json',m)
cc=HERE/'cloud/compiler_cloud_contract.py';pin(cc,'GENERATION',GEN);pin(cc,'SOURCES',m['sources']);pin(cc,'MANIFEST_SHA',C.sha(N/'manifest.json'));pin(cc,'POLICY_SHA',m['compiler_policy_sha256'])
s=cc.read_text().replace("m.get('compiler_generation')=='compiler-mosaic-serde7-gather-bf16-fp32-accepted-native-fix-v1'","m.get('compiler_generation')==GENERATION").replace("'bytes': 5926","'bytes': "+str((N/'manifest.json').stat().st_size)).replace('COMPILER_EXACT_34_SOURCES','COMPILER_EXACT_36_SOURCES');cc.write_text(s)
b=HERE/'cloud/build_compiler_packet.py';s=b.read_text().replace("compiler_generation='compiler-mosaic-serde7-gather-bf16-fp32-accepted-native-fix-v1'",'compiler_generation=NC.GENERATION').replace('REVIEWED_COMPILER34_HASH','REVIEWED_COMPILER36_HASH').replace('COMPILER34_FILE','COMPILER36_FILE');b.write_text(s)
C.admit(N/'compiler-policy.json',C.sha(N/'compiler-policy.json'));C.verify_manifest(N/'manifest.json',C.sha(N/'manifest.json'));print(json.dumps({'generation':GEN,'sources':len(m['sources']),'manifest_sha256':C.sha(N/'manifest.json'),'manifest_bytes':(N/'manifest.json').stat().st_size,'policy_sha256':m['compiler_policy_sha256'],'policy_sources':len(p['sources'])}))
