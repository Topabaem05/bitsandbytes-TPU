"""Exact supplemental function with isolated dependency doubles. No production module replacement."""
import argparse,ast,json,time,types
from pathlib import Path
HERE=Path(__file__).resolve().parent

def run(output):
    output=Path(output);output.mkdir(exist_ok=False);rows=[]
    source=HERE/'compiler-native/compiler_verify.py';tree=ast.parse(source.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='verify')
    calls=[];captures=[];expected_python='/explicit/accepted/jax071/venv/bin/python';expected_deadline=1234.5
    def replay(*args,**kwargs):
        calls.append((args,kwargs))
        if kwargs.get('mosaic_audit_python')!=expected_python:raise ValueError('FIXTURE_EXPLICIT_INTERPRETER_REJECTED')
        if kwargs.get('audit_deadline_epoch')!=expected_deadline:raise ValueError('FIXTURE_ORIGINAL_AUDIT_DEADLINE_REJECTED')
        return {'status':'ISOLATED_NATIVE_DOUBLE_ONLY'}
    native=types.SimpleNamespace(verify=replay)
    def imported(name,*args,**kwargs):
        assert name=='verify_run';return native
    def capture(*args):captures.append(args);return {'status':'ISOLATED_CAPTURE_DOUBLE_ONLY','selected_executable_link':'UNKNOWN'}
    namespace={'__builtins__':{**vars(__import__('builtins')),'__import__':imported},'Path':Path,'C':types.SimpleNamespace(read=lambda p:{}),'capture_verified':capture}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(source),'exec'),namespace)
    verify=namespace['verify'];args=['actual','evidence','oracle','o'*64,'a'*64,'e'*64,'c'*64,'p'*64,{'pid':12}]
    value=verify(*args,mosaic_audit_python=expected_python,audit_deadline_epoch=expected_deadline)
    assert value['native']['status']=='ISOLATED_NATIVE_DOUBLE_ONLY'and len(captures)==1
    assert calls[-1][1]=={'mosaic_audit_python':expected_python,'audit_deadline_epoch':expected_deadline}
    rows.append({'case':'exact-supplement-forwards-explicit-interpreter-and-deadline','status':'PASS'})
    for name,kwargs,typ,label in [
        ('missing-interpreter',{'audit_deadline_epoch':expected_deadline},TypeError,None),
        ('missing-audit-deadline',{'mosaic_audit_python':expected_python},TypeError,None),
        ('incorrect-interpreter-rejection-propagates',{'mosaic_audit_python':'wrong','audit_deadline_epoch':expected_deadline},ValueError,'FIXTURE_EXPLICIT_INTERPRETER_REJECTED'),
        ('incorrect-deadline-rejection-propagates',{'mosaic_audit_python':expected_python,'audit_deadline_epoch':-1},ValueError,'FIXTURE_ORIGINAL_AUDIT_DEADLINE_REJECTED')]:
        before=len(captures)
        try:verify(*args,**kwargs)
        except typ as error:
            if label:assert str(error)==label
            assert len(captures)==before;rows.append({'case':name,'status':'PASS','rejected':type(error).__name__+':'+str(error)})
        else:raise AssertionError('FALSE_ACCEPT:'+name)
    report={'status':'PASS','count':len(rows),'controls':rows,'scope':'EXACT_FUNCTION_AST_WITH_ISOLATED_DEPENDENCY_DOUBLES','production_module_changes':'NONE','actual_compiler':'NOT_RUN','actual_TPU':'NOT_RUN'}
    (output/'results.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');print(json.dumps({'status':'PASS','count':len(rows)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
