"""Run the actual owner CLI entry point with an isolated, no-service drive result."""
import json
from pathlib import Path
import runpy
import socket
import sys

import pytest

OWNER=Path(__file__).resolve().parents[1]/'owner.py'


def run_cli(path, status, tmp_path, monkeypatch, capsys):
    """Replace only drive at the completed module's entry boundary; execute CLI bytes."""
    monkeypatch.syspath_prepend(str(OWNER.parent))
    source=path.read_text().splitlines()
    boundary=next(i for i,line in enumerate(source,1) if line=="if __name__ == '__main__':")
    calls=[];installed=[]
    def no_network(*args,**kwargs):raise AssertionError('PROVIDER_CALL_FORBIDDEN_IN_CLI_CONTROL')
    monkeypatch.setattr(socket.socket,'connect',no_network)
    monkeypatch.setattr(sys,'argv',[str(path),'--packet',str(tmp_path/'packet'),'--packet-sha256','a'*64,
        '--output',str(tmp_path/'output'),'--root-acceptance',str(tmp_path/'acceptance'),
        '--root-acceptance-sha256','b'*64,'--cli-python',str(tmp_path/'python'),'--cli-identity',str(tmp_path/'identity')])
    def drive(*args):
        calls.append(args)
        return {'status':status,'mode':'SYNTHETIC_CLI_RESULT_NO_SERVICE'}
    def trace(frame,event,arg):
        if event=='line' and frame.f_code.co_filename==str(path) and frame.f_lineno==boundary:
            assert frame.f_globals['drive'] is not drive
            frame.f_globals['drive']=drive;installed.append(True)
        return trace
    prior=sys.gettrace()
    try:
        sys.settrace(trace)
        with pytest.raises(SystemExit) as stopped:runpy.run_path(str(path),run_name='__main__')
    finally:sys.settrace(prior)
    assert installed==[True] and len(calls)==1
    assert calls[0][0]==(tmp_path/'packet').resolve() and calls[0][1]==(tmp_path/'output').resolve()
    result=json.loads(capsys.readouterr().out)
    assert result=={'status':status,'mode':'SYNTHETIC_CLI_RESULT_NO_SERVICE'}
    return stopped.value.code


@pytest.mark.parametrize('status,expected',[
    ('PASS_TPU_API_PROBE',0),('PASS_DEVICE_ROUTE_RECORDS',0),('PASS_PRECISION_RECORDS',0),
    ('PASS_TPU_TRANSFER_API42',0),('PASS_TPU_STATE_RECORDS',0),('PASS_TPU_NESTED_RECORDS',0),('FAIL_TPU_NESTED_RECORDS',2),
    ('PASS_TPU_NESTED_STATE_RECORDS',0),('FAIL_TPU_NESTED_STATE_RECORDS',2),
    ('FAIL_TPU_STATE_RECORDS',2),('BLOCKED_CLEANUP',2),('PASS_UNKNOWN_STATUS',2),('UNKNOWN',2),
])
def test_actual_cli_exact_exit_mapping(tmp_path,monkeypatch,capsys,status,expected):
    assert run_cli(OWNER,status,tmp_path,monkeypatch,capsys)==expected
