"""Synthetic scalar values with genuine recovered HLO. No XLA client/provider."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'packages/bitsandbytes-tpu/pyproject.toml').is_file())
SCIENCE=ROOT/'experiments/2026-10-04-bitsandbytes-tpu'
SOURCE=Path(os.environ.get('BNB_PRIMITIVE_PARAMETER_SOURCE',SCIENCE/'probe_primitives.py'))
CONTRACT=Path(os.environ.get('BNB_PRIMITIVE_PARAMETER_CONTRACT',SCIENCE/'cloud/primitive_contract.py'))
FIXTURE=Path(__file__).resolve().parent/'fixtures/actual_float_arithmetic.hlo.txt'

def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

V=load(SOURCE,'parameter_candidate');B=load(SCIENCE/'probe_backend.py','parameter_B');Ref,H=V.pure();PC=load(CONTRACT,'parameter_PC')
DATA=json.loads((SOURCE.parent/'primitive-inputs.json').read_text());EXPECTED=Ref.expected(DATA)['float-arithmetic']


def row():
    materialized=Ref.expected(DATA)['device-int-float']['float'];source=V.source_inputs(Ref,DATA,'float-arithmetic',materialized)
    raw={'fixture_scope':'SYNTHETIC_SCALAR_VALUES_NOT_ACTUAL_RECONSTRUCTION','status':'ERROR','output_order':list(EXPECTED),'source_inputs':source,'input_parameter_ids':{'float_input':0},'parameter_values':{'0':copy.deepcopy(materialized),'1':V.auxiliary_sources(Ref,'float-arithmetic')['clamp_max'],'2':V.auxiliary_sources(Ref,'float-arithmetic')['clamp_min']}}
    return raw,FIXTURE.read_text()


def test_original_strict_inventory_rejects_source_proven_implicit_bounds():
    original=os.environ.get('BNB_PRIMITIVE_PARAMETER_ORIGINAL_SOURCE')
    if not original:pytest.skip('Requires explicit preserved original science source')
    old=load(Path(original),'parameter_original');raw,hlo=row()
    with pytest.raises(ValueError,match='PARAMETER_VALUE_MATRIX'):old.row_graph(B,H,'float-arithmetic',raw,hlo,EXPECTED)


def test_genuine_hlo_roles_and_synthetic_values_are_exact():
    raw,hlo=row();w=V.row_graph(B,H,'float-arithmetic',raw,hlo,EXPECTED)
    roles=w['clamp']['auxiliary_parameters'];assert roles['clamp_min']['parameter']==2 and roles['clamp_max']['parameter']==1
    assert w['clamp']['parameters']==[0,1,2] and all(v['parameters']==[0] for k,v in w.items() if k!='clamp')
    assert all(V.bit_gate(raw['parameter_values'][str(roles[k]['parameter'])],v)['status']=='PASS' for k,v in V.auxiliary_sources(Ref,'float-arithmetic').items())
    # The actual scalar values were never recovered. This is a hypothetical correct record, not a replay of them.
    assert raw['fixture_scope'].startswith('SYNTHETIC_')


@pytest.mark.parametrize('fault',['missing','extra-mapping','unused-entry','type','shape','nonfinite','wrong-alias','wrong-middle','other-output-extra'])
def test_extra_unknown_unused_or_malformed_parameters_fail_closed(fault):
    raw,hlo=row()
    if fault=='missing':raw['parameter_values'].pop('1')
    elif fault=='extra-mapping':raw['parameter_values']['3']=Ref.array([1.],[],'float32')
    elif fault=='unused-entry':hlo=hlo.replace('  %p0.3 = f32[20]{0} parameter(0)','  %unused = f32[] parameter(3)\n  %p0.3 = f32[20]{0} parameter(0)')
    elif fault=='type':raw['parameter_values']['1']=Ref.array([1],[],'int32')
    elif fault=='shape':raw['parameter_values']['1']=Ref.array([Ref.from_bits(0x7f7fffff)],[1],'float32')
    elif fault=='nonfinite':raw['parameter_values']['1']['values']=[float('inf')]
    elif fault=='wrong-alias':hlo=hlo.replace('f32[20]{0} broadcast(f32[] %p2.31)','f32[20]{0} abs(f32[] %p2.31)')
    elif fault=='wrong-middle':hlo=hlo.replace('clamp(f32[20]{0} %broadcast.32, f32[20]{0} %p0.3,','clamp(f32[20]{0} %broadcast.32, f32[20]{0} %broadcast.33,')
    else:hlo=hlo.replace('f32[20]{0} abs(f32[20]{0} %p0.3)','f32[20]{0} add(f32[20]{0} %p0.3, f32[20]{0} %broadcast.33)',1)
    with pytest.raises(ValueError):V.row_graph(B,H,'float-arithmetic',raw,hlo,EXPECTED)


def test_unrelated_row_still_rejects_any_auxiliary_parameter():
    source=V.source_inputs(Ref,DATA,'host-float-bits');raw={'output_order':['bits'],'source_inputs':source,'input_parameter_ids':{'float_input':0},'parameter_values':{'0':source['float_input'],'1':Ref.array([1.],[],'float32')}}
    with pytest.raises(ValueError,match='PARAMETER_VALUE_MATRIX'):V.row_graph(B,H,'host-float-bits',raw,'unused',Ref.expected(DATA)['host-float-bits'])


@pytest.fixture(scope='module')
def full(tmp_path_factory):
    F=load(Path(os.environ.get('BNB_PRIMITIVE_PARAMETER_FIXTURE',SCIENCE/'cloud/tests/primitive_saved_fixture.py')),'parameter_synthetic_fixture')
    f=F.make_fixture(tmp_path_factory.mktemp('parameter-synthetic'))
    # Update only isolated synthetic receipts to this explicit proposed generation.
    for path in (f.oracle/'oracle-seal.json',f.actual/'receipt.json',f.actual/'native-view/receipt.json',f.actual/'builder/receipt.json'):
        record=F.B.read(path);record.update(V.binding(F.B,F.N,f));F.B.write(path,record)
    path=f.actual/'builder/raw/float-arithmetic.json';raw=F.B.read(path);hypothetical,hlo=row();raw.update(parameter_values=hypothetical['parameter_values'],fixture_scope=hypothetical['fixture_scope']);raw['hlo_witnesses']=V.row_graph(F.B,H,'float-arithmetic',raw,hlo,EXPECTED);F.B.write(path,raw);path.with_suffix('.hlo.txt').write_text(hlo)
    F.reseal(f,oracle=True)
    packet=f.actual.parent/'synthetic-report-science';packet.mkdir()
    for name in ('probe_primitives.py','primitive_kernels.py','primitive-inputs.json','primitive_reference.py','hlo_bitcast.py'):shutil.copyfile(SOURCE.parent/name,packet/name)
    for name in ('probe_backend.py','probe_routes.py','probe_precision.py','transfer_admission.py','probe_nested.py','probe_nested_state.py','nested-inputs.json','nested-source.json','nested-schemas.json','probe-profile.json','probe-inputs.json'):shutil.copyfile(SCIENCE/name,packet/name)
    (packet/'patches').mkdir();shutil.copyfile(ROOT/'patches/params4bit-xla-v1.json',packet/'patches/params4bit-xla-v1.json')
    F.B.write(packet/'manifest.json',{'fixture_scope':'SYNTHETIC_REPORT_VERIFIER_SOURCE_ONLY_NOT_A_DISPATCH_PACKET','source_admission_sha256':f.admission_sha256});f.synthetic_packet=packet
    return F,f


@pytest.mark.parametrize('fault',[None,'wrong-min','wrong-max','swapped-bound-roles','error-row','missing-evidence','forged-report','forged-expected'])
def test_real_full_verifier_auxiliary_values_never_hide_failure(full,fault):
    F,f=full;path=f.actual/'builder/raw/float-arithmetic.json';original=path.read_bytes();hpath=path.with_suffix('.hlo.txt');hbefore=hpath.read_bytes()
    try:
        raw=F.B.read(path)
        if fault=='wrong-min':raw['parameter_values']['2']=Ref.array([0.],[],'float32')
        elif fault=='wrong-max':raw['parameter_values']['1']=Ref.array([1.],[],'float32')
        elif fault=='swapped-bound-roles':
            text=hpath.read_text().replace('clamp(f32[20]{0} %broadcast.32, f32[20]{0} %p0.3, f32[20]{0} %broadcast.33)','clamp(f32[20]{0} %broadcast.33, f32[20]{0} %p0.3, f32[20]{0} %broadcast.32)');hpath.write_text(text);raw['hlo_witnesses']=V.row_graph(F.B,H,'float-arithmetic',raw,text,EXPECTED)
        elif fault=='error-row':raw.update(status='ERROR',error_type='ValueError',error_message='SYNTHETIC_VALIDATION_FAILURE',traceback='SYNTHETIC_ONLY')
        elif fault=='missing-evidence':raw['parameter_values'].pop('1')
        F.B.write(path,raw);F.reseal(f)
        if fault in ('error-row','missing-evidence'):
            with pytest.raises(ValueError):V.verify(F.B,F.R,F.P,F.A,F.N,F.S,f)
        else:
            report=V.verify(F.B,F.R,F.P,F.A,F.N,F.S,f)
            assert report['record_validation']=='PASS' and report['row_count']==32 and report['m4_status']=='NOT_QUALIFIED'
            assert report['numerical_status']==('FAIL' if fault in ('wrong-min','wrong-max','swapped-bound-roles') else 'PASS')
            assert PC.validate_report(report,f.synthetic_packet)
            if fault=='forged-expected':
                target=next(r for r in report['rows'] if r['case_id']=='float-arithmetic');gate=target['gates']['auxiliary_clamp_max'];gate.update(actual_bytes_sha256='f'*64,expected_bytes_sha256='f'*64,status='PASS')
                with pytest.raises(ValueError,match='PRIMITIVE_AUXILIARY_SOURCE_GATE'):PC.validate_report(report,f.synthetic_packet)
            if fault=='forged-report':
                target=next(r for r in report['rows'] if r['case_id']=='float-arithmetic');target['gates'].pop('auxiliary_clamp_max')
                with pytest.raises(ValueError,match='PRIMITIVE_GATE_MATRIX'):PC.validate_report(report,f.synthetic_packet)
    finally:path.write_bytes(original);hpath.write_bytes(hbefore);F.reseal(f)


@pytest.mark.parametrize('nonfinite',[False,True])
def test_validation_failure_retains_full_partial_row_without_passing(tmp_path,nonfinite):
    raw,hlo=row();raw.update(case_id='float-arithmetic',pid=1,process_token='a'*32,input_manifest_sha256=V.INPUT_SHA,outputs=copy.deepcopy(EXPECTED))
    if nonfinite:raw['parameter_values']['1']['values'][0]=float('nan')
    prior=tmp_path/'host-float-bits.json';prior.write_text('PRIOR_COMPLETE_RECORD_UNCHANGED');prefix=tmp_path/'float-arithmetic'
    try:raise ValueError('SYNTHETIC_POST_EXTRACTION_VALIDATION_FAILURE')
    except ValueError as error:V.retain_failed_row(B,prefix,raw,error)
    rec=B.read(prefix.with_suffix('.json'));assert rec['status']=='ERROR' and rec['error_type']=='ValueError' and prior.read_text()=='PRIOR_COMPLETE_RECORD_UNCHANGED'
    if nonfinite:
        retained=prefix.with_suffix('.partial.txt');assert 'nan' in retained.read_text() and rec['partial_text_sha256']==B.sha(retained)
    else:assert rec['parameter_values']==raw['parameter_values'] and rec['outputs']==EXPECTED


def test_prevalidation_snapshot_and_failure_hook_are_on_real_path():
    tree=ast.parse(SOURCE.read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_phase')
    graph=next(n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='row_graph')
    write_lines=[n.lineno for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='write' and any(isinstance(a,ast.Name) and a.id=='raw' for a in n.args)]
    assert graph.lineno-1 in write_lines
    outer=next(n for n in run.body if isinstance(n,ast.Try));assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='retain_failed_row' for n in ast.walk(outer.handlers[0]))


def test_early_source_failure_is_not_masked_by_partial_retention(tmp_path,full):
    import time
    from unittest.mock import patch
    F,f=full;args=SimpleNamespace(**vars(f));args.output=tmp_path/'early-failure';args.phase='builder';args.process_token='b'*32;args.parent_process_token='a'*32;args.parent_pid=100;args.deadline_epoch=time.time()+60
    with patch.object(V,'verify_cpu_oracle',return_value={'fixture_scope':'SYNTHETIC_PRE_DEVICE_BOUNDARY'}), patch.object(V.os,'getpid',return_value=101), patch.object(V.os,'getppid',return_value=100), patch.object(V.os,'getpgid',return_value=100), patch.object(F.N,'admit',side_effect=RuntimeError('SYNTHETIC_SOURCE_ADMISSION_FAILURE')):
        with pytest.raises(RuntimeError,match='SYNTHETIC_SOURCE_ADMISSION_FAILURE'):V.run_phase(F.B,F.R,F.P,F.A,F.N,F.S,args)
    assert F.B.read(args.output/'receipt.json')['status']=='FAILED' and 'SYNTHETIC_SOURCE_ADMISSION_FAILURE' in (args.output/'error.log').read_text()
    assert not (args.output/'raw').exists()
