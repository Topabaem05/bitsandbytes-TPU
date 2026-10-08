"""Exact admitted case loop with a fake first runtime failure; no XLA import."""
import ast,json,os,time,traceback,types
from pathlib import Path
def run(output,native):
    output.mkdir(exist_ok=False);source=ast.parse((native/'device_diagnostic.py').read_text())
    function=next(n for n in source.body if isinstance(n,ast.FunctionDef)and n.name=='run')
    loop=next(n for n in ast.walk(function)if isinstance(n,ast.For)and ast.unparse(n.target)=='(name, dtype, use_wrapper, rows)')
    calls=[]
    def fail(*args):calls.append(args);raise ValueError('FIXTURE_FIRST_RUNTIME_FAILURE')
    def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')
    names=['direct-fp32','functional-fp32','direct-bf16','functional-bf16','tail-reference-fp32']
    ns={'first_failure':None,'settings':[(name,None,False,128)for name in names],'native_calls':[],'output':output,'os':os,'args':types.SimpleNamespace(process_token='2'*32,deadline_epoch=time.time()+30),'D':types.SimpleNamespace(require=lambda v,m:None),'time':time,'torch_xla':types.SimpleNamespace(_XLAC=types.SimpleNamespace(_set_current_graph_name=fail)),'receipt':{'cases':[]},'traceback':traceback,'write':write}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),str(native/'device_diagnostic.py'),'exec'),ns)
    assert len(calls)==1
    assert ns['receipt']['cases']==[{'name':name,'status':'ERROR'if i==0 else'NOT_RUN'}for i,name in enumerate(names)]
    first=json.loads((output/'raw/direct-fp32.json').read_text());assert first['error_type']=='ValueError'and first['error_message']=='FIXTURE_FIRST_RUNTIME_FAILURE'
    for name in names[1:]:
        row=json.loads((output/'raw'/name).with_suffix('.json').read_text());assert row['reason']=='SHARED_XLA_STATE_AFTER_FIRST_FAILURE'and row['first_failure']==ns['receipt']['first_failure']and not row.get('error_type')
    report={'status':'PASS','count':3,'controls':[{'case':'first-runtime-error-retained','status':'PASS'},{'case':'subsequent-notrun-linked-to-first-error','status':'PASS'},{'case':'no-later-XLA-operation-or-independent-error','status':'PASS'}],'scope':'EXACT_SOURCE_LOOP_WITH_FAKE_RUNTIME_FAILURE','actual_TPU':'NOT_RUN'};write(output/'results.json',report);return report
