"""Isolated synthetic records and public-call stubs. No CPU/TPU science runs."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[3]
PROBE=Path(__file__).resolve().parents[1]/'probe_routes.py'
spec=importlib.util.spec_from_file_location('route_probe',PROBE);P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
BACKEND=ROOT/'experiments/2026-10-04-bitsandbytes-tpu/probe_backend.py'
B=P.backend(BACKEND)
spec=importlib.util.spec_from_file_location('backend_fixtures',ROOT/'tests/test_backend_probe.py');F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)
F.ProbeControls.setUpClass()


def make_fixture(root,admission='a'*64):
    """Reusable fake-owner fixture. Fields assert claims only; no device proof."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    f=F.ProbeControls(methodName='test_complete_fixture_is_valid')
    f.setUp()
    try:
        oracle=root/'oracle';actual=root/'actual';shutil.copytree(f.cpu,oracle);actual.mkdir()
        oracle_rec=B.read(oracle/'oracle-seal.json');oracle_rec['source_admission_sha256']=admission;B.write(oracle/'oracle-seal.json',oracle_rec)
        oracle_sha=F.reseal(oracle,'oracle-seal.json')
        nonlinear,cases=P.selection(B);references={}
        for c in cases:
            references[c['id']]=P.gradient_math(c,B.read(oracle/'raw'/(c['id']+'.json')))
        B.write(actual/'references.json',references)
        for collection,route,case in [('nonlinear',None,c) for c in nonlinear]+[('routes',r,c) for c in cases for r in P.ROUTES]:
            name='raw/'+collection+'-'+(route+'-' if route else '')+case['id'];path=actual/(name+'.json');path.parent.mkdir(exist_ok=True)
            raw=copy.deepcopy(B.read(f.tpu/'raw'/(case['id']+'.json')))
            raw.update(quantization_source=P.PROTOCOL['quantization_source'][route] if route else 'TPU_FUNCTIONAL_QUANTIZE_OR_DIRECT_DECODE',collection=collection,route=route,pid=1234,process_token='b'*32,status='PASS',stage='record_validation',expected_error_observed=False)
            if route=='cpu_to_xla':
                raw.update(status='ERROR',stage='construct',error_type='RuntimeError',error_message=P.KNOWN_ERROR,traceback=name+'.error.log',expected_error_observed=True)
                (actual/raw['traceback']).write_text('SYNTHETIC expected incompatible TensorImpl error\n')
            else:
                (actual/(name+'.hlo.txt')).write_text('SYNTHETIC HLO\n');(actual/(name+'.metrics.txt')).write_text('SYNTHETIC metrics\n')
                if route:
                    raw['outputs'].update(references[case['id']]);raw['placements'].update({k:'xla:0' for k in references[case['id']]})
                    blob=json.dumps({'quant_type':'nf4','blocksize':64,'dtype':case['dtype'],'shape':case['weight_shape']}).encode()
                    raw.update(original_classes=True,module_training=True,module_state_alias=True,module_state_absent=False,weight_requires_grad=False,weight_grad_is_none=True,bnb_quantized=True,
                        packed_before=raw['outputs']['packed'],packed_after=raw['outputs']['packed'],roundtrip_scope='SAME_PID_DIAGNOSTIC_ONLY',roundtrip_outputs=copy.deepcopy(raw['outputs']),
                        state_dict={'weight':raw['outputs']['packed'],'weight.absmax':raw['outputs']['absmax'],'weight.quant_map':raw['outputs']['code'],
                            'bias':{'shape':[case['weight_shape'][0]],'dtype':case['dtype'],'values':case['bias']},
                            'weight.quant_state.bitsandbytes__nf4':{'shape':[len(blob)],'dtype':'uint8','values':list(blob)}})
            if route and raw['status']=='PASS': raw['restored_state_dict']=copy.deepcopy(raw['state_dict'])
            B.write(path,raw)
        profile,_=B.load_spec()
        rec=dict(f.receipt,kind=P.PROTOCOL['kind'],protocol_sha256=P.PROTOCOL_SHA,probe_sha256=B.sha(PROBE),backend_probe_sha256=P.BACKEND_SHA,pid=1234,process_token='b'*32,
                 routes=P.ROUTES,route_case_ids=P.CASE_IDS,nonlinear_case_ids=[c['id'] for c in nonlinear],oracle_sha256=oracle_sha,
                 source_admission_sha256=admission,source_pre=admission,source_post=admission)
        rec.pop('case_ids',None);B.write(actual/'receipt.json',rec);F.reseal(actual,'receipt.json')
        return SimpleNamespace(backend_probe=str(BACKEND),oracle=oracle,oracle_sha256=oracle_sha,actual=actual,admission_sha256=admission)
    finally:f.doCleanups()


def change(args,relative,mutate):
    path=args.actual/relative;raw=B.read(path);mutate(raw);B.write(path,raw);F.reseal(args.actual,'receipt.json')


def test_complete_records_keep_expected_errors(tmp_path):
    result=P.verify(B,make_fixture(tmp_path))
    assert result['record_validation']=='PASS' and result['api42_status']==result['m3_status']=='NOT_QUALIFIED'
    assert result['numerical_status']=='INCOMPLETE' and len(result['rows'])==40
    assert sum(r['expected_error_observed'] for r in result['rows'])==2


@pytest.mark.parametrize('fault,expected',[
 ('source','SOURCE_BINDING'),('pid','ROW_ORIGIN'),('partial','DIAGNOSTIC_TERMINAL'),('missing','DIAGNOSTIC_MATRIX'),
 ('fallback','CPU_FALLBACK'),('outputs','OUTPUT_SET'),('expected-error','EXPECTED_ERROR_CLASSIFICATION'),('bias','STATE_VALUE_ROUNDTRIP'),
 ('metadata','STATE_VALUE_ROUNDTRIP'),('runtime','RUNTIME_PAIR'),('oracle','ORACLE_SEAL_HASH')])
def test_invalid_records(tmp_path,fault,expected):
    a=make_fixture(tmp_path)
    first='raw/nonlinear-quant-float32-codes-n1.json'
    nonlinear,_=P.selection(B);first='raw/nonlinear-'+nonlinear[0]['id']+'.json'
    route='raw/routes-target_constructor-'+P.CASE_IDS[0]+'.json'
    if fault=='source':change(a,'receipt.json',lambda r:r.update(source_post='c'*64))
    elif fault=='pid':change(a,first,lambda r:r.update(pid=9999))
    elif fault=='partial':change(a,'receipt.json',lambda r:r.update(status='PARTIAL'))
    elif fault=='missing':change(a,'receipt.json',lambda r:r['nonlinear_case_ids'].pop())
    elif fault=='fallback':change(a,first,lambda r:r['counters'].update({'aten::copy':1}))
    elif fault=='outputs':change(a,first,lambda r:r['outputs'].pop('packed'))
    elif fault=='expected-error':change(a,'raw/routes-cpu_to_xla-'+P.CASE_IDS[0]+'.json',lambda r:r.update(error_message='unrelated failure'))
    elif fault=='bias':change(a,route,lambda r:r['state_dict']['bias']['values'].__setitem__(0,999))
    elif fault=='metadata':change(a,route,lambda r:r['state_dict']['weight.quant_state.bitsandbytes__nf4'].update(dtype='float32'))
    elif fault=='roundtrip':change(a,route,lambda r:r['roundtrip_outputs']['y']['values'].__setitem__(0,999))
    elif fault=='runtime':change(a,'receipt.json',lambda r:r['runtime'].update(torch='2.14.1'))
    elif fault=='oracle':a.oracle_sha256='0'*64
    with pytest.raises((ValueError,KeyError),match=expected):P.verify(B,a)


def test_numeric_failure_remains_fail(tmp_path):
    a=make_fixture(tmp_path);cases,_=P.selection(B)
    change(a,'raw/nonlinear-'+cases[0]['id']+'.json',lambda r:r['outputs']['packed']['values'].__setitem__(0,1))
    r=P.verify(B,a)
    assert r['record_validation']=='PASS' and r['numerical_status']=='FAIL'
    assert r['api42_status']=='NOT_QUALIFIED'


def test_wrong_probe_rejected_before_import(tmp_path):
    p=tmp_path/'wrong.py';p.write_text('raise AssertionError("must not import")')
    with pytest.raises(ValueError,match='BACKEND_SOURCE_IDENTITY'):P.backend(p)


def test_target_restore_has_no_cpu_bias_data_assignment():
    calls=[]
    class Module:
        def __init__(self,*a,**kw):calls.append(('constructor',kw['device']));self.quant_state=None
        def to(self,**kw):calls.append(('to',kw));return self
        def train(self):return self
        def load_state_dict(self,state,strict):calls.append(('load',set(state),strict))
        @property
        def bias(self):raise AssertionError('Do not access or replace CPU bias.data')
    class Params:
        @staticmethod
        def from_prequantized(data,stats,**kw):
            calls.append(('from_prequantized',kw['device'],dict(stats)));module=kw['module'];module.quant_state=object();return SimpleNamespace(quant_state=module.quant_state)
    torch=SimpleNamespace(float32='float32',uint8='uint8')
    bnb=SimpleNamespace(nn=SimpleNamespace(Linear4bit=Module,Params4bit=Params))
    case={'dtype':'float32','weight_shape':[19,17],'bias':[0.0]*19}
    module=P.restore_target(case,{'weight':'CPU_PACKED','bias':'CPU_BIAS','weight.absmax':'CPU_SCALE'},bnb,torch,'xla:0')
    assert calls==[('constructor','xla:0'),('to',{'dtype':'float32'}),('from_prequantized','xla:0',{'absmax':'CPU_SCALE'}),('load',{'weight','bias'},True)]
    assert module.weight.quant_state is module.quant_state


def test_wrong_gradient_reference_and_agreeing_outputs_rejected(tmp_path):
    a=make_fixture(tmp_path);refs=B.read(a.actual/'references.json');refs[P.CASE_IDS[0]]['dx']['values'][0]=123
    B.write(a.actual/'references.json',refs)
    for route in ('target_constructor','from_prequantized'):
        path='raw/routes-'+route+'-'+P.CASE_IDS[0]+'.json'
        change(a,path,lambda r:(r['outputs']['dx']['values'].__setitem__(0,123),r['roundtrip_outputs']['dx']['values'].__setitem__(0,123)))
    with pytest.raises(ValueError,match='GRADIENT_REFERENCE_WITNESS'):P.verify(B,a)


@pytest.mark.parametrize('delta,expected',[(1e-6,'PASS'),(10.0,'FAIL')])
def test_float_roundtrip_uses_fixed_gate(tmp_path,delta,expected):
    a=make_fixture(tmp_path);path='raw/routes-target_constructor-'+P.CASE_IDS[0]+'.json'
    change(a,path,lambda r:r['roundtrip_outputs']['y']['values'].__setitem__(0,delta))
    r=P.verify(B,a);row=next(x for x in r['rows'] if x['route']=='target_constructor' and x['case_id']==P.CASE_IDS[0])
    assert r['record_validation']=='PASS' and row['roundtrip_gates']['y']['status']==expected
    assert r['numerical_status']==('FAIL' if expected=='FAIL' else 'INCOMPLETE')


def test_packed_roundtrip_must_be_exact(tmp_path):
    a=make_fixture(tmp_path);path='raw/routes-target_constructor-'+P.CASE_IDS[0]+'.json'
    change(a,path,lambda r:r['roundtrip_outputs']['packed']['values'].__setitem__(0,1))
    with pytest.raises(ValueError,match='STATE_PACKED_ROUNDTRIP'):P.verify(B,a)


def test_checkpoint_route_source_label_mandatory(tmp_path):
    a=make_fixture(tmp_path);path='raw/routes-from_prequantized-'+P.CASE_IDS[0]+'.json'
    change(a,path,lambda r:r.update(quantization_source='TPU_PUBLIC_MODULE_QUANTIZE'))
    with pytest.raises(ValueError,match='QUANTIZATION_SOURCE'):P.verify(B,a)


@pytest.mark.parametrize('flag',[True,False])
def test_later_same_error_does_not_claim_cpu_to_failed(tmp_path,flag):
    a=make_fixture(tmp_path);path='raw/routes-cpu_to_xla-'+P.CASE_IDS[0]+'.json'
    change(a,path,lambda r:r.update(stage='forward_gradient_state',expected_error_observed=flag))
    if flag:
        with pytest.raises(ValueError,match='EXPECTED_ERROR_CLASSIFICATION'):P.verify(B,a)
    else:
        result=P.verify(B,a)
        row=next(r for r in result['rows'] if r['case_id']==P.CASE_IDS[0] and r['route']=='cpu_to_xla')
        assert row['status']=='ERROR' and row['expected_error_observed'] is False
