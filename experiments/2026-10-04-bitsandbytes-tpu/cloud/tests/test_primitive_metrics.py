"""CPU-only pinned API shape controls. No XLA client, tensor execution or provider."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'packages/bitsandbytes-tpu/pyproject.toml').is_file())
SCIENCE=ROOT/'experiments/2026-10-04-bitsandbytes-tpu'
CANDIDATE=Path(os.environ.get('BNB_PRIMITIVE_METRIC_SOURCE',SCIENCE/'probe_primitives.py')).resolve()
ORIGINAL=Path(os.environ.get('BNB_PRIMITIVE_ORIGINAL_SOURCE',SCIENCE/'probe_primitives.py')).resolve()

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

V=load(CANDIDATE,'primitive_metrics_candidate');OLD=load(ORIGINAL,'primitive_metrics_original');B=load(SCIENCE/'probe_backend.py','primitive_metrics_B');P=load(SCIENCE/'probe_precision.py','primitive_metrics_P');PROFILE,_=B.load_spec()

class Metrics:
    """Exact pinned C++ outer/sample tuple shapes, not a list-shaped fixture."""
    def __init__(self,value):self.value=value;self.calls=[]
    def counter_names(self):return list(self.value['counters'])
    def counter_value(self,key):return self.value['counters'][key]
    def metric_data(self,key):self.calls.append(key);return self.value['execution_metrics'].get(key)


def sample():return {'counters':{'ExecuteComputation':1},'execution_metrics':{'ExecuteTime':(1,0.0006279,((1.0,0.0006279),))}}

def test_original_shallow_conversion_reproduces_actual_failure():
    if CANDIDATE==ORIGINAL:pytest.skip('Old-source incorrect control requires explicit original file argument')
    evidence=OLD.metrics_record(Metrics(sample()),PROFILE)
    assert isinstance(evidence['execution_metrics']['ExecuteTime'],list)
    assert isinstance(evidence['execution_metrics']['ExecuteTime'][2],tuple)
    with pytest.raises(ValueError,match='EXECUTION_METRIC_SAMPLES'):P.validate_execution(B,evidence)


def test_tuple_snapshot_canonicalizes_without_drop_or_mutation():
    value=sample();before=copy.deepcopy(value);api=Metrics(value);evidence=V.metrics_record(api,PROFILE)
    assert evidence=={'counters':{'ExecuteComputation':1},'execution_metrics':{'ExecuteTime':[1,0.0006279,[[1.0,0.0006279]]]}}
    assert value==before and len(api.calls)==len(set(api.calls))==len(PROFILE['execution_metrics'])
    P.validate_execution(B,evidence);P.validate_execution(B,json.loads(json.dumps(evidence,allow_nan=False)))


@pytest.mark.parametrize('phase',['native-view','builder'])
def test_complete_supported_collection_preserves_both_metrics(phase):
    value=sample();value['execution_metrics']['ExecuteReplicatedTime']=(2,0.25,((1.,0.1),(2.,0.15)))
    evidence=V.metrics_record(Metrics(value),PROFILE);P.validate_execution(B,evidence)
    assert evidence['execution_metrics']['ExecuteReplicatedTime']==[2,0.25,[[1.,0.1],[2.,0.15]]]


def test_native_unsupported_empty_execution_and_positive_fallback_retained():
    value={'counters':{'aten::view_copy.dtype':2},'execution_metrics':{}}
    evidence=V.metrics_record(Metrics(value),PROFILE);assert evidence==value
    V.validate_counter_observation(B,evidence)
    with pytest.raises(ValueError,match='CPU_FALLBACK'):P.validate_execution(B,evidence)


@pytest.mark.parametrize('fault,message',[
 ('nan','Out of range float'),('infinity','Out of range float'),('negative_sample','EXECUTION_METRIC_SAMPLES'),
 ('bad_sample','EXECUTION_METRIC_SAMPLES'),('boolean_sample','EXECUTION_METRIC_SAMPLES'),('missing_samples','EXECUTION_METRIC_SAMPLES'),
 ('negative_accumulator','EXECUTION_METRICS'),('bad_total','EXECUTION_METRICS'),('negative_counter','COUNTERS_UNKNOWN'),
 ('fallback','CPU_FALLBACK'),('no_execution','EXECUTION_NOT_OBSERVED')])
def test_invalid_metrics_are_not_normalized_into_valid(fault,message):
    value=sample();count,acc,samples=value['execution_metrics']['ExecuteTime']
    if fault=='nan':samples=((1.,float('nan')),)
    elif fault=='infinity':acc=float('inf')
    elif fault=='negative_sample':samples=((1.,-1.),)
    elif fault=='bad_sample':samples=((1.,0.1,2.),)
    elif fault=='boolean_sample':samples=((1.,True),)
    elif fault=='missing_samples':samples=()
    elif fault=='negative_accumulator':acc=-1.
    elif fault=='bad_total':count=True
    elif fault=='negative_counter':value['counters']['ExecuteComputation']=-1
    elif fault=='fallback':value['counters']['aten::mm']=1
    elif fault=='no_execution':count=0;samples=()
    value['execution_metrics']['ExecuteTime']=(count,acc,samples)
    with pytest.raises(ValueError,match=message):P.validate_execution(B,V.metrics_record(Metrics(value),PROFILE))


def test_shared_collector_covers_exception_host_and_synced_phase_paths():
    module=ast.parse(CANDIDATE.read_text());phase=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='run_phase')
    calls=[n for n in ast.walk(phase) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='metrics_record']
    assert len(calls)==3
    # All device collections go through the one canonical snapshot function.
    direct=[n for n in ast.walk(phase) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='metric_data']
    assert not direct
