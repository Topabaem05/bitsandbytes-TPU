"""CPU differential controls and independent synthetic records, no TPU claims."""
import copy
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import socket
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]
SCIENCE=HERE.parents[1]
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
N=load(SCIENCE/'probe_nested.py','nested_tested')

args=SimpleNamespace(backend_probe=SCIENCE/'probe_backend.py',route_probe=SCIENCE/'probe_routes.py',
    precision_probe=SCIENCE/'probe_precision.py',transfer_admission=SCIENCE/'transfer_admission.py',
    patch_manifest=ROOT/'patches/params4bit-xla-v1.json',admission_sha256='a'*64)
B,R,P,A=N.helpers(args); DATA,SOURCE,SCHEMAS=N.spec(B); PROFILE,_=B.load_spec()

def hlo(case):
    if case['kind']=='module' and case['dtype']=='float32' and len(case['x_shape'])==2:
        m,k=case['x_shape']; n=case['weight_shape'][0]
        return ('HloModule SYNTHETIC_ONLY\nENTRY main {\n'
            f'x = f32[{m},{k}] parameter(0)\nwt = f32[{k},{n}] parameter(1)\ndy = f32[{m},{n}] parameter(2)\nw = f32[{n},{k}] parameter(3)\n'
            f'y = f32[{m},{n}] dot(x, wt), lhs_contracting_dims={{1}}, rhs_contracting_dims={{0}}, operand_precision={{highest,highest}}\n'
            f'dx = f32[{m},{k}] dot(dy, w), lhs_contracting_dims={{1}}, rhs_contracting_dims={{0}}, operand_precision={{highest,highest}}\n'
            f'ROOT result = (f32[{m},{n}], f32[{m},{k}]) tuple(y, dx)\n}}\n')
    return 'HloModule SYNTHETIC_ONLY\nENTRY main {\nROOT zero = f32[] constant(0)\n}\n'

def reseal(fixture,oracle=False):
    root=fixture.oracle if oracle else fixture.actual; name='oracle-seal.json' if oracle else 'receipt.json'
    rec=B.read(root/name); rec['artifacts']=B.inventory(root,name); B.write(root/name,rec)
    if oracle:
        fixture.oracle_sha256=B.sha(root/name)
        receipt=B.read(fixture.actual/'receipt.json'); receipt['oracle_sha256']=fixture.oracle_sha256; B.write(fixture.actual/'receipt.json',receipt)

def make_fixture(base,reference,admission='a'*64):
    base=Path(base); base.mkdir(parents=True,exist_ok=True); oracle=base/'oracle'; actual=base/'actual'; oracle.mkdir(); actual.mkdir()
    fixture=SimpleNamespace(**vars(args),oracle=oracle,actual=actual); fixture.admission_sha256=admission
    gold=dict(N.bindings(B,fixture),status='COMPLETE',pid=1001,pgid=1001,process_token='1'*32,
        source_pre=admission,source_post=admission,public_api=N.PUBLIC.copy(),schemas=SCHEMAS,
        runtime={'python':'3.12','torch':'2.9.0+cpu','platform':'linux','machine':'x86_64'},
        oracle_method='EXPLICIT_PINNED_DEFAULT_BODIES_NO_NATIVE_DISPATCH')
    rec=dict(gold,pid=1002,pgid=1002,process_token='2'*32,runtime=PROFILE['runtime'],
        device={'type':'xla','hardware':'TPU','pjrt':'TPU'},dispatch={key:True for key in SCHEMAS},
        precision_environment={key:None for key in P.ENV_KEYS},
        precision={'requested':'highest','readback':'highest','set_calls':1,'before_graph':True})
    for case in DATA['cases']:
        raw=copy.deepcopy(reference[case['id']]); raw.update(pid=gold['pid'],process_token=gold['process_token'])
        B.write(oracle/'raw'/(case['id']+'.json'),raw)
        raw.update(pid=rec['pid'],process_token=rec['process_token'],placements={key:'xla:0' for key in raw['outputs']},
            counters={},execution_metrics={'ExecuteTime':[1,1.0,[[1.0,1.0]]]})
        text=hlo(case); raw['hlo_precision']=P.hlo_precision(text) if case['kind']=='module' and case['dtype']=='float32' and len(case['x_shape'])==2 else None
        prefix=actual/'raw'/case['id']; B.write(prefix.with_suffix('.json'),raw)
        prefix.with_suffix('.hlo.txt').write_text(text); prefix.with_suffix('.metrics.txt').write_text('SYNTHETIC_ONLY\n')
    gold['artifacts']=B.inventory(oracle,'oracle-seal.json'); B.write(oracle/'oracle-seal.json',gold)
    fixture.oracle_sha256=B.sha(oracle/'oracle-seal.json'); rec['oracle_sha256']=fixture.oracle_sha256
    rec['artifacts']=B.inventory(actual,'receipt.json'); B.write(actual/'receipt.json',rec)
    return fixture



def make_saved_fixture(base,admission='a'*64):
    seed=HERE/'fixtures/nested_cpu_seed.json.gz'
    assert hashlib.sha256(seed.read_bytes()).hexdigest()=="9b7096cbab45da88e046fddd43bb070bf0e42bbbb35120c0c84fd6aa82bcce6b"
    raw=gzip.decompress(seed.read_bytes())
    assert hashlib.sha256(raw).hexdigest()=='7ae0b334645b4d8595c5445dd6c413a1dde89ccb3b113b3cd6861276a570ff86'
    reference=json.loads(raw)
    assert set(reference)=={case['id'] for case in DATA['cases']}
    return make_fixture(base,reference,admission=admission)
