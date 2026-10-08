"""Post-structural-validation faults with synthetic device execution and real checkpoints."""
import copy
import importlib.util
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('post_gate_fixtures', HERE / 'state_device_fixture.py')
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
D = F.D


def test_post_validation_gate_exception_retained_all8(tmp_path, monkeypatch):
    fixture = F.make_device_fixture(tmp_path / 'fixture')
    seeds = {case['id']: D.read(fixture.actual / 'save/raw' / (case['id'] + '.json')) for case in D.CASES}
    checkpoints = {case['id']: (fixture.actual / 'save/raw' / (case['id'] + '.state.pt')).read_bytes() for case in D.CASES}
    current = {}
    attempted, structurally_validated = [], []
    class Tensor:
        def __init__(self, array): self.array = copy.deepcopy(array); self.device = 'xla:0'
        def detach(self): return self
        def cpu(self): return self
        def clone(self): return self
    class QuantState: pass
    class Params4bit: pass
    class Linear4bit:
        def state_dict(self): return {key: Tensor(value) for key, value in current['seed']['state_dict'].items()}
    def route(route, case, saved, bnb, torch, device):
        attempted.append(case['id']); current.update(case=case, seed=seeds[case['id']])
        module = Linear4bit(); module.training = True
        state = QuantState()
        for key, value in seeds[case['id']]['metadata'].items(): setattr(state, key, value)
        module.quant_state = state; module.weight = Params4bit(); module.weight.quant_state = state
        module.weight.data = Tensor(seeds[case['id']]['outputs']['packed'])
        module.weight.requires_grad = False; module.weight.grad = None; module.weight.module = module
        return module
    def compute(case, module, torch, bnb, device):
        return {name: Tensor(value) for name, value in seeds[case['id']]['outputs'].items()}, module.weight.data
    original_validate = D.S.validate_raw
    def validate(raw, case, tpu):
        result = original_validate(raw, case, tpu)
        structurally_validated.append(case['id'])
        return result
    original_gates = D.device_gates
    def gates(case, raw, original, profile):
        assert case['id'] in structurally_validated
        if case['id'] == D.CASES[0]['id']: raise RuntimeError('INJECTED_AFTER_STRUCTURAL_VALIDATION')
        return original_gates(case, raw, original, profile)
    monkeypatch.setattr(D.R, 'make_route', route)
    monkeypatch.setattr(D.S, 'compute', compute)
    monkeypatch.setattr(D.S, 'validate_raw', validate)
    original_tensor_record = D.B.tensor_record
    monkeypatch.setattr(D.B, 'tensor_record', lambda tensor: copy.deepcopy(tensor.array))
    monkeypatch.setattr(D, 'device_gates', gates)
    torch = SimpleNamespace(save=lambda state, path: Path(path).write_bytes(checkpoints[current['case']['id']]))
    bnb = SimpleNamespace(nn=SimpleNamespace(Linear4bit=Linear4bit, Params4bit=Params4bit),
                          functional=SimpleNamespace(QuantState=QuantState))
    metrics = SimpleNamespace(clear_all=lambda: None, counter_names=lambda: ['xla::dot'], counter_value=lambda key: 1,
        metric_data=lambda key: (1, 0.01, ((0, 0.01),)) if key == 'ExecuteTime' else None, metrics_report=lambda: 'SYNTHETIC METRICS\n')
    xla = (SimpleNamespace(_XLAC=SimpleNamespace(_get_xla_tensors_hlo=lambda tensors: F.fixture_hlo(current['case']))),
           SimpleNamespace(mark_step=lambda wait: None, wait_device_ops=lambda: None), metrics)
    receipt = D.read(fixture.actual / 'save/receipt.json'); receipt['status'] = 'PARTIAL'
    output = tmp_path / 'injected-save'
    args = SimpleNamespace(output=output, phase='save', oracle=fixture.oracle, deadline_epoch=time.time()+60)
    assert D.case_loop(args, bnb, torch, 'xla:0', receipt, xla) == 2
    assert attempted == structurally_validated == [case['id'] for case in D.CASES]
    rows = [D.read(output / 'raw' / (case['id']+'.json')) for case in D.CASES]
    assert rows[0]['status'] == 'ERROR' and rows[0]['error_message'] == 'INJECTED_AFTER_STRUCTURAL_VALIDATION'
    assert (output / 'raw' / (D.CASES[0]['id']+'.error.log')).is_file()
    assert all(row['status']=='PASS' and set(row['numerical_gates'])==D.expected_gate_names(case)
               for row,case in zip(rows[1:],D.CASES[1:]))
    assert D.read(output/'receipt.json')['numerical_status'] == 'FAIL'
    # Recover the produced phase without rerunning case science. The true first
    # case error must remain a valid failure record and cannot become a pass.
    monkeypatch.setattr(D, 'device_gates', original_gates)
    monkeypatch.setattr(D.B, 'tensor_record', original_tensor_record)
    shutil.rmtree(fixture.actual/'save'); shutil.copytree(output, fixture.actual/'save')
    parent=D.read(fixture.actual/'parent.json'); parent['children'][0]['exit_code']=2; D.write(fixture.actual/'parent.json',parent)
    F.reseal_fixture(fixture)
    result=F.verify_fixture(fixture)
    assert result['record_validation']=='PASS' and result['numerical_status']=='FAIL'
    assert result['rows'][0]['status']=='ERROR' and all(row['status']=='PASS' for row in result['rows'][1:])


@pytest.mark.parametrize('fault', ['error-pass','missing','extra','bad-status','false-pass','bad-metric'])
def test_recovered_pass_requires_complete_true_device_gates(tmp_path, fault):
    fixture=F.make_device_fixture(tmp_path)
    path=fixture.actual/'restore/raw'/(D.CASES[0]['id']+'.json');raw=D.read(path)
    if fault=='error-pass':raw.update(error_type='RuntimeError',error_message='INJECTED_AFTER_VALIDATION',traceback='SYNTHETIC ERROR')
    elif fault=='missing':raw['numerical_gates'].pop('y')
    elif fault=='extra':raw['numerical_gates']['invented']=copy.deepcopy(raw['numerical_gates']['y'])
    elif fault=='bad-status':raw['numerical_gates']['y']['status']='NOT_RUN'
    elif fault=='false-pass':raw['numerical_gates']['y']['max_abs']=1
    elif fault=='bad-metric':raw['numerical_gates']['y']['rmse']='NaN'
    D.write(path,raw);F.reseal_fixture(fixture,refresh_gates=False)
    with pytest.raises(ValueError,match='PASS_WITH_ERROR_METADATA|DEVICE_GATE_'):
        F.verify_fixture(fixture)
