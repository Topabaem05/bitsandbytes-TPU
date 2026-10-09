"""Resolve the reviewed cloud merge with exact unchanged-block admission, no fuzzy patch."""
import ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;E=HERE/'evidence/adopted-public-cloud';public=(E/'remote.py').read_text();private=(HERE/'packet-v4/cloud/remote.py').read_text()
(E/'compose-first-failure.json').write_text(json.dumps({'status':'PRESERVED_PREPARATION_FAILURE','operation':'git apply --check private compiler delta onto adopted public cloud','reason':'First import hunk has different surrounding context after the public imports. No patch was applied.','repair':'Use exact unchanged run_step and compiler-native branch byte admission plus a unique CC import anchor.'},sort_keys=True,indent=2)+'\n')
def func(text,name):
 node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name==name);lines=text.splitlines(True);return ''.join(lines[node.lineno-1:node.end_lineno])
def branch(text):return text[text.index("        elif phase == 'compiler-native':"):text.index("        elif phase == 'primitives':")]
baseimport="_compiler_spec.loader.exec_module(CC)";assert public.count(baseimport)==1
flagimport="\n_flag_spec=importlib.util.spec_from_file_location('_compiler_flag_cloud',HERE/'compiler_flag_preflight.py');FP=importlib.util.module_from_spec(_flag_spec);_flag_spec.loader.exec_module(FP)"
# Original pre-public frozen source is the independent baseline for both changed blocks.
old=(HERE.parent/'r6-compiler-bf16-accepted-fix/cloud/remote.py').read_text()
for get in (lambda t:func(t,'run_step'),branch):
 before=get(old);assert before==get(public),'PUBLIC_BLOCK_OVERLAP';assert public.count(before)==1;public=public.replace(before,get(private))
public=public.replace(baseimport,baseimport+flagimport);ast.parse(public);(HERE/'cloud/remote.py').write_text(public)
(E/'composition-resume.json').write_text(json.dumps({'status':'EXACT_TWO_UNCHANGED_BASE_BLOCKS_COMPOSED','run_step_public_base_identical_to_frozen':True,'compiler_native_public_base_identical_to_frozen':True,'merged_remote_sha256':hashlib.sha256(public.encode()).hexdigest(),'public_and_ordinary_native_branch_changes':'NONE','provider_calls':0},sort_keys=True,indent=2)+'\n');print('EXACT_PRIVATE_CLOUD_COMPOSITION_PASS')
