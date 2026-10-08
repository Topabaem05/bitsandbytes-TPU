"""Small in-memory defensive traversal fixtures; corrected source only, no device or compiler."""
import copy,importlib.util,os
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'packages/bitsandbytes-tpu/pyproject.toml').is_file())
SCIENCE=ROOT/'experiments/2026-10-04-bitsandbytes-tpu'
SOURCE=Path(os.environ.get('BNB_PRIMITIVE_PARAMETER_SOURCE',SCIENCE/'probe_primitives.py'))
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
V=load(SOURCE,'defensive_alias_candidate');B=load(SCIENCE/'probe_backend.py','defensive_alias_B');Ref,H=V.pure()
FIXTURE=Path(__file__).resolve().parent/'fixtures/actual_float_arithmetic.hlo.txt'
class BoundedNodes(dict):
    """A finite lookup budget makes these unit fixtures bounded without a hanging process."""
    def __init__(self,data):super().__init__(data);self.lookups=0
    def __getitem__(self,name):
        self.lookups+=1
        assert self.lookups<=24,'Unit traversal exceeded finite lookup budget'
        return super().__getitem__(name)
def node(op,shape,deps):return {'opcode':op,'dtype':'f32','shape':shape,'deps':deps,'line':'IN_MEMORY_DEFENSIVE_FIXTURE','body':'dimensions={}'}
def check(mode):
    comps,entry=copy.deepcopy(H.parse(FIXTURE.read_text()));nodes=comps[entry]['nodes'];start='clamp.34'
    if mode=='scalar-copy':nodes['scalar_copy']=node('copy',[],['p2.31']);nodes['broadcast.32']['deps']=['scalar_copy']
    elif mode=='scalar-reshape':
        nodes['unit_vector']=node('reshape',[1],['p2.31']);nodes['scalar_back']=node('reshape',[],['unit_vector']);nodes['broadcast.32']['deps']=['scalar_back']
    elif mode in ('output-copy','output-reshape'):
        nodes['output_alias']=node(mode.split('-')[1],[20],['clamp.34']);start='output_alias'
    elif mode=='scalar-copy-shape':
        nodes['unit_vector']=node('copy',[1],['p2.31']);nodes['scalar_back']=node('reshape',[],['unit_vector']);nodes['broadcast.32']['deps']=['scalar_back']
    elif mode=='output-copy-shape':nodes['output_alias']=node('copy',[20],['p2.31']);start='output_alias'
    elif mode.startswith('output-repeat'):
        start='alias_a';nodes['alias_a']=node('copy',[20],['alias_a'if mode.endswith('self')else'alias_b']);nodes['alias_b']=node('reshape',[20],['alias_a'])
    elif mode.startswith('scalar-repeat'):
        nodes['broadcast.32']['deps']=['alias_a'];nodes['alias_a']=node('copy',[],['alias_a'if mode.endswith('self')else'alias_b']);nodes['alias_b']=node('reshape',[],['alias_a'])
    elif mode.startswith('middle-repeat'):
        nodes['clamp.34']['deps'][1]='alias_a';nodes['alias_a']=node('copy',[20],['alias_a'if mode.endswith('self')else'alias_b']);nodes['alias_b']=node('reshape',[20],['alias_a'])
    bounded=BoundedNodes(nodes);comps[entry]['nodes']=bounded
    helper=SimpleNamespace(parse=lambda _: (comps,entry),witnesses=lambda *_: {'clamp':{'call_input':start}},input_lineage=H.input_lineage)
    roles=V.clamp_parameter_roles(B,helper,'IN_MEMORY_NO_COMPILER',[],{},0)
    assert roles['clamp_min']['parameter']==2 and roles['clamp_max']['parameter']==1
    assert bounded.lookups<=24
    return roles
@pytest.mark.parametrize('mode',['broadcast','scalar-copy','scalar-reshape','output-copy','output-reshape'])
def test_valid_typed_paths_remain_accepted(mode):check(mode)
@pytest.mark.parametrize('mode,label',[
    ('scalar-copy-shape','CLAMP_SCALAR_COPY'),('output-copy-shape','CLAMP_OUTPUT_COPY'),
    ('output-repeat-self','CLAMP_OUTPUT_ALIAS_CYCLE'),('output-repeat-two','CLAMP_OUTPUT_ALIAS_CYCLE'),
    ('scalar-repeat-self','CLAMP_SCALAR_ALIAS_CYCLE'),('scalar-repeat-two','CLAMP_SCALAR_ALIAS_CYCLE'),
    ('middle-repeat-self','HLO_INPUT_LINEAGE'),('middle-repeat-two','HLO_INPUT_LINEAGE')])
def test_shape_changes_and_repeated_ids_reject_with_finite_lookup_budget(mode,label):
    with pytest.raises(ValueError,match=label):check(mode)
