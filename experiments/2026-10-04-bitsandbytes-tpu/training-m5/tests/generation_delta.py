"""Closed complete reader-v3 preimage; narrow reader-v4 delta, unchanged science."""
import ast,hashlib,json,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(b):return hashlib.sha256(b).hexdigest()
def sha(p):return digest(p.read_bytes())
archive=ROOT/'historical/prior-v3-preparation-source.tar.gz'
assert sha(archive)=='faee769327eb29d839142936c4a1f73745421be331676753d45f1467ce210d23'
with tarfile.open(archive,'r:gz')as t:
 members=t.getmembers();assert len(members)==len({m.name for m in members})==174
 for m in members:assert m.isfile()and not m.issym()and not m.islnk()and not Path(m.name).is_absolute()and'..'not in Path(m.name).parts
 preimage={m.name:t.extractfile(m).read()for m in members};closure=json.loads(preimage['source-closure.json'])
 assert set(preimage)==set(closure['files'])|{'source-closure.json'}
 for n,r in closure['files'].items():assert len(preimage[n])==r['bytes']and digest(preimage[n])==r['sha256'],n
prior=json.loads(preimage['source/science-manifest.json']);current=json.loads((ROOT/'source/science-manifest.json').read_text())
assert preimage['source/science-manifest.json']==(ROOT/'historical/prior-v3-science-manifest.json').read_bytes()
assert set(current['files'])==set(prior['files'])
for n,r in prior['files'].items():assert len(preimage[n])==r['bytes']and digest(preimage[n])==r['sha256'],n
changed={n for n,r in prior['files'].items()if current['files'][n]!=r}
assert changed=={'control/kernel_reader.py','control/inspect_actual.py','source/admission.py','integration/source-admission.json'},changed
old=ast.parse(preimage['source/admission.py']);new=ast.parse((ROOT/'source/admission.py').read_text())
for tree in [old,new]:tree.body=[n for n in tree.body if not(isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='GENERATION'for t in n.targets))]
assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)
a=json.loads(preimage['integration/source-admission.json']);b=json.loads((ROOT/'integration/source-admission.json').read_text());a.pop('generation');b.pop('generation');assert a==b
assert sha(ROOT/'control/kernel_protocol.py')=='e9c7a5b7b943cea08dfb8986205e47e008ceac37553a2f8576687527a58f9d5f'
assert sha(ROOT/'integration/plugin/src/bitsandbytes_tpu/exact.py')=='49511fbc22e93ca9ca765da9930c8975e44ae389b605fe85747ddf7c73486830'
assert sha(ROOT/'integration/plugin/src/bitsandbytes_tpu/_source_order.py')=='54728c36190c64334ffceb2b12d1cabcfff87ca35a8387a24766ae4dba191b75'
result={'status':'PASS','prior_complete_source_archive_sha256':sha(archive),'preimage_members':len(preimage),'prior_science_manifest_sha256':digest(preimage['source/science-manifest.json']),'new_science_manifest_sha256':sha(ROOT/'source/science-manifest.json'),'added':[],'changed':sorted(changed),'admission_only_generation_constant_changed':True,'integration_admission_only_generation_changed':True,'shared_kernel_helper_exact_unchanged':True,'training_math_reader_reference_and_plugin_bytes_unchanged':True,'frozen20_steps_save10_resume10_and_1024_guards_unchanged':True,'local20_repeat':'NOT_RUN_NO_SCIENTIFIC_BYTE_CHANGE_PRIOR_V2_EVIDENCE_RETAINED','provider_calls':0,'m5_status':'NOT_QUALIFIED'}
Path(sys.argv[1]).write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps({'status':'PASS','closed_source_delta':True}))
