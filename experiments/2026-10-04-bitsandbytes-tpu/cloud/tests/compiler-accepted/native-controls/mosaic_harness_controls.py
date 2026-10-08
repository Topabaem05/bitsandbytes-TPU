"""A fake-success counterexample for the rejection harness; no production helper replacement."""
import ast,json,time,types
from pathlib import Path
def run(output):
    output.mkdir(exist_ok=False)
    tree=ast.parse((Path(__file__).resolve().parent/'mosaic_audit_controls.py').read_text())
    check=next(node for node in ast.walk(tree)if isinstance(node,ast.FunctionDef)and node.name=='check')
    rows=[];namespace={'C':types.SimpleNamespace(replay=lambda *args:{}),'python':'FAKE_ONLY','time':time,'out':output,'rows':rows,'json':json}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[check],type_ignores=[])),'rejection-harness-control','exec'),namespace)
    try:namespace['check']('forced-success',{},False)
    except AssertionError as error:assert str(error)=='incorrect record accepted'
    else:raise AssertionError('harness mislabeled accepted incorrect fixture as rejection')
    assert rows==[]and not(output/'forced-success.json').exists()
    report={'status':'PASS','count':1,'controls':[{'case':'successful-fake-replay-cannot-be-counted-as-rejection','status':'PASS'}],'scope':'ISOLATED_CONTROL_FUNCTION_WITH_FAKE_SUCCESS_OBJECT','production_helper_changes':'NONE'};(output/'results.json').write_text(json.dumps(report,indent=2)+'\n');return report
