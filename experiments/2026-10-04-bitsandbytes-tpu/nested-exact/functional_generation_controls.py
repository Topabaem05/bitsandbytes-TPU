"""Source identity controls for the explicit functional-output generation."""
import argparse,ast,hashlib,importlib.util,json,shutil,sys,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source'))
import generation_gate as G,exact_admission as A

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(original):
    rows=[];old=original/'experiments/2026-10-04-bitsandbytes-tpu/nested-exact'
    def check(name,callback):callback();rows.append({'name':name,'status':'PASS'})
    def reject(callback,text):
        try:callback()
        except ValueError as exc:assert text in str(exc),(str(exc),text)
        else:raise AssertionError('INCORRECT_SOURCE_ACCEPTED')
    admission=HERE/'integration/source-admission.json';plugin=HERE/'integration/plugin';upstream=HERE/'integration/upstream/bitsandbytes'
    check('correct_complete_source_generation',lambda:G.verify(admission,A.INTEGRATION_ADMISSION_SHA,plugin=plugin,upstream=upstream))
    check('old_admission_sha_rejected',lambda:reject(lambda:A.admission(old/'integration/source-admission.json',sha(old/'integration/source-admission.json')),'EXACT_ADMISSION_LITERAL'))
    check('old_generation_rejected_even_with_self_hash',lambda:reject(lambda:G.verify(old/'integration/source-admission.json',sha(old/'integration/source-admission.json')),'EXACT_SOURCE_VARIANT'))
    def science_delta():
        before=json.loads((old/'source/science-manifest.json').read_text());after=json.loads((HERE/'source/science-manifest.json').read_text())
        changed={n for n in before['files'] if before['files'][n]!=after['files'][n]}
        assert changed=={'exact_admission.py','generation_gate.py','integration-admission.json'}
        for name in before['files']:
            assert sha(HERE/'source'/name)==after['files'][name]['sha256']
            if name not in changed:assert (old/'source'/name).read_bytes()==(HERE/'source'/name).read_bytes()
    check('science19_exactly_three_identity_changes_sixteen_unchanged',science_delta)
    def math_delta():
        name='integration/plugin/src/bitsandbytes_tpu/exact.py'
        before={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((old/name).read_text()).body if isinstance(n,ast.FunctionDef)}
        after={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((HERE/name).read_text()).body if isinstance(n,ast.FunctionDef)}
        assert before.keys()==after.keys();assert {n for n in before if before[n]!=after[n]}=={'bitcast'}
        for name in ['_integer_clamp.py','_integer_addsub.py','_integer_muldiv.py','_source_order.py']:
            assert (old/'integration/plugin/src/bitsandbytes_tpu'/name).read_bytes()==(HERE/'integration/plugin/src/bitsandbytes_tpu'/name).read_bytes()
    check('all_math_function_asts_and_integer_helper_bytes_unchanged',math_delta)
    def historical():
        h=HERE/'history/functional-output-v1-20261011';m=json.loads((h/'source-map.json').read_text())
        assert sha(h/'science-manifest.json')=='9e71efc73b0f2b3a210f8131c10972569a5e2d04cf298fcf5a8da59772cf4f5a'
        s=json.loads((h/'science-manifest.json').read_text());assert set(m['files'])==set(s['files']) and len(s['files'])==19
        for n,r in m['files'].items():
            p=HERE/r['path'];assert sha(p)==r['sha256']==s['files'][n]['sha256'];assert p.stat().st_size==r['bytes']==s['files'][n]['bytes']
    check('historical19_reconstructed_from_three_preimages_and_sixteen_current_files',historical)
    with tempfile.TemporaryDirectory(prefix='m4-generation-controls-',dir='/tmp') as temporary:
        root=Path(temporary);shutil.copytree(plugin,root/'plugin')
        p=root/'plugin/src/bitsandbytes_tpu/exact.py';p.write_bytes(p.read_bytes()+b'\n')
        check('altered_plugin_rejected',lambda:reject(lambda:G.verify(admission,A.INTEGRATION_ADMISSION_SHA,plugin=root/'plugin'),'PLUGIN_PACKAGE_BYTES'))
        p.unlink();p.symlink_to(plugin/'src/bitsandbytes_tpu/exact.py')
        check('plugin_symlink_rejected',lambda:reject(lambda:G.verify(admission,A.INTEGRATION_ADMISSION_SHA,plugin=root/'plugin'),'SOURCE_SYMLINK'))
    assert not Path(temporary).exists()
    return {'kind':'M4_FUNCTIONAL_OUTPUT_GENERATION_CONTROLS_V1','status':'PASS','controls':rows,'control_count':len(rows),'source_variant':A.VARIANT,'temporary_fixture_removed':True,'provider_calls':0,'TPU':'NOT_RUN','m4_status':'NOT_QUALIFIED'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original-repository',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.original_repository);a.output.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'PASS','controls':r['control_count']}))
