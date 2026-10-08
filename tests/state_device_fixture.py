"""Synthetic device records with real CPU checkpoints. No TPU execution proof."""
import copy
import importlib.util
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r3_fixture_probe',HERE.parent/'experiments/2026-10-04-bitsandbytes-tpu/probe_state_roundtrip.py')
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
SEED=HERE/'fixtures/state_roundtrip_cpu.json'
SEED_SHA='655077e7a78008ab55fb5421b9ff636f865f3d31a6c87ee020a7c838288cdf42'


def reseal_fixture(fixture, refresh_gates=True):
    """Rebuild isolated byte inventories after an intentional control mutation."""
    root=Path(fixture.actual if hasattr(fixture,'actual') else fixture)
    if refresh_gates and hasattr(fixture,'oracle'):
        profile,_=D.B.load_spec()
        for phase in ('save','restore'):
            for case in D.CASES:
                path=root/phase/'raw'/(case['id']+'.json');raw=D.read(path)
                if raw['status']=='PASS':
                    raw['numerical_gates']=D.device_gates(case,raw,D.read(Path(fixture.oracle)/'raw'/(case['id']+'.json')),profile)
                    D.write(path,raw)
    receipt=D.read(root/'save/receipt.json');receipt['artifacts']=D.B.inventory(root/'save','receipt.json');D.write(root/'save/receipt.json',receipt)
    saved_sha=D.sha(root/'save/receipt.json')
    for case in D.CASES:
        path=root/'restore/raw'/(case['id']+'.json');raw=D.read(path)
        if raw['status']=='PASS':raw['saved_receipt_sha256']=saved_sha;D.write(path,raw)
    receipt=D.read(root/'restore/receipt.json');receipt['saved_seal_sha256']=saved_sha;receipt['artifacts']=D.B.inventory(root/'restore','receipt.json');D.write(root/'restore/receipt.json',receipt)
    parent=D.read(root/'parent.json')
    for launch in parent['children']:launch['receipt_sha256']=D.sha(root/launch['receipt'])
    parent['artifacts']=D.B.inventory(root,'parent.json');D.write(root/'parent.json',parent)
    return D.sha(root/'parent.json')


def fixture_hlo(case):
    dtype='f32' if case['dtype']=='float32' else 'bf16'
    m=math.prod(case['x_shape'][:-1]);n,k=case['weight_shape']
    text=f'''HloModule SYNTHETIC_ONLY
ENTRY main {{
 x = {dtype}[{m},{k}] parameter(0)
 wt = {dtype}[{k},{n}] parameter(1)
 y = {dtype}[{m},{n}] dot(x, wt), lhs_contracting_dims={{1}}, rhs_contracting_dims={{0}}, operand_precision={{highest,highest}}
'''
    if dtype=='f32':
        text+=f''' dy = f32[{m},{n}] parameter(2)
 w = f32[{n},{k}] parameter(3)
 dx = f32[{m},{k}] dot(dy, w), lhs_contracting_dims={{1}}, rhs_contracting_dims={{0}}, operand_precision={{highest,highest}}
 ROOT out = (f32[{m},{n}],f32[{m},{k}]) tuple(y, dx)
'''
    else:text+=f' ROOT out = {dtype}[{m},{n}] copy(y)\n'
    return text+'}\n'


def materialize_seed(root):
    import torch
    require_seed = D.read(SEED)
    assert D.sha(SEED) == SEED_SHA
    assert require_seed['kind'] == 'SYNTHETIC_STATE_TEST_SEED'
    assert require_seed['profile_sha256'] == D.B.PROFILE_SHA and require_seed['inputs_sha256'] == D.B.INPUTS_SHA
    assert set(require_seed['phases']) == {'save', 'restore'}
    for phase in ('save', 'restore'):
        phase_seed = require_seed['phases'][phase]
        assert set(phase_seed['rows']) == {case['id'] for case in D.CASES}
        D.write(root/phase/'receipt.json', phase_seed['receipt'])
        for case in D.CASES:
            name = case['id']; raw = copy.deepcopy(phase_seed['rows'][name])
            target = root/phase/'raw'/(name+'.state.pt')
            target.parent.mkdir(parents=True, exist_ok=True)
            if phase == 'save':
                state = {key:torch.tensor(value['values'],dtype=getattr(torch,value['dtype'])).reshape(value['shape'])
                         for key,value in raw['state_dict'].items()}
                torch.save(state, target)
                _, records = D.checkpoint_records(torch,target,case)
                assert records == raw['state_dict']
            else:
                shutil.copyfile(root/'save/raw'/(name+'.state.pt'),target)
                raw['loaded_checkpoint_sha256'] = D.sha(target)
            D.write(root/phase/'raw'/(name+'.json'), raw)
    D.write(root/'parent.json', require_seed['parent'])


def make_device_fixture(base, admission='a'*64):
    """Return an actual-shaped, synthetic-only record accepted by the real verifier."""
    base=Path(base);base.mkdir(parents=True,exist_ok=True)
    test_path=HERE/'test_transfer_probe.py'
    spec=importlib.util.spec_from_file_location('r2_fixture_controls',test_path)
    F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)
    r2=F.make_fixture(base/'r2-fixture',admission=admission)
    root=base/'actual';materialize_seed(root)
    profile,_=D.B.load_spec()
    # Keep all 42 CPU records and replace the eight linear witnesses with real CPU values.
    for case in D.CASES:
        raw=D.read(root/'save/raw'/(case['id']+'.json'))
        original={key:copy.deepcopy(raw[key]) for key in ('case_id','input_sha256','metadata','counters')}
        original['outputs']={key:raw['outputs'][key] for key in D.B.expected_outputs(case)}
        original['placements']={key:'cpu' for key in original['outputs']}
        D.write(r2.oracle/'raw'/(case['id']+'.json'),original)
    oracle_seal=D.read(r2.oracle/'oracle-seal.json');oracle_seal['artifacts']=D.B.inventory(r2.oracle,'oracle-seal.json');D.write(r2.oracle/'oracle-seal.json',oracle_seal)
    oracle_sha=D.sha(r2.oracle/'oracle-seal.json')
    parent=D.read(root/'parent.json');parent.update(probe_sha256=D.sha(D.__file__),helper_pins=D.PINS,protocol_sha256=D.PROTOCOL_SHA,pid=1000,pgid=1000,source_admission_sha256=r2.admission_sha256,oracle=str(r2.oracle),oracle_sha256=oracle_sha,precision_environment={key:None for key in D.P.ENV_KEYS})
    for index,phase in enumerate(('save','restore')):
        launch=parent['children'][index];launch.update(pid=2000+index,launched_pid=2000+index,pgid=1000,parent_pid=1000)
        receipt=D.read(root/phase/'receipt.json')
        receipt.update(probe_sha256=D.sha(D.__file__),helper_pins=D.PINS,protocol_sha256=D.PROTOCOL_SHA,pid=2000+index,pgid=1000,parent_pid=1000,scope='TPU_CANDIDATE',runtime=profile['runtime'],runtime_lock_sha256=profile['runtime_lock_sha256'],source_pre=r2.admission_sha256,source_post=r2.admission_sha256,oracle_sha256=oracle_sha,patch_manifest_sha256=D.A.PATCH_MANIFEST_SHA,device={'type':'xla','hardware':'TPU','pjrt':'TPU'},dispatch={op:True for op in D.B.DISPATCH_OPS},precision={'requested':'highest','readback':'highest','set_calls':1,'before_graph':True},precision_environment=parent['precision_environment'])
        if phase=='restore':receipt['saved_pid']=2000
        D.write(root/phase/'receipt.json',receipt)
        for case in D.CASES:
            prefix=root/phase/'raw'/case['id'];raw=D.read(prefix.with_suffix('.json'))
            hlo=fixture_hlo(case);prefix.with_suffix('.hlo.txt').write_text(hlo);prefix.with_suffix('.metrics.txt').write_text('SYNTHETIC METRICS ONLY\n')
            raw.update(pid=2000+index,placements={key:'xla:0' for key in raw['outputs']},counters={'xla::dot':1},execution_metrics={'ExecuteTime':[1,0.01,[[0,0.01]]]},hlo_precision=D.P.hlo_precision(hlo))
            D.write(prefix.with_suffix('.json'),raw)
    D.write(root/'parent.json',parent)
    fixture=SimpleNamespace(actual=root,oracle=r2.oracle,oracle_sha256=oracle_sha,admission_sha256=r2.admission_sha256,synthetic=True)
    fixture.parent_sha256=reseal_fixture(fixture)
    return fixture


def verify_fixture(fixture):
    return D.verify_pair(fixture.actual,False,fixture.oracle,fixture.oracle_sha256,fixture.admission_sha256,D.sha(fixture.actual/'parent.json'))


def reseal_oracle(fixture):
    seal=D.read(fixture.oracle/'oracle-seal.json');seal['artifacts']=D.B.inventory(fixture.oracle,'oracle-seal.json');D.write(fixture.oracle/'oracle-seal.json',seal)
    fixture.oracle_sha256=D.sha(fixture.oracle/'oracle-seal.json')
    parent=D.read(fixture.actual/'parent.json');parent['oracle_sha256']=fixture.oracle_sha256;D.write(fixture.actual/'parent.json',parent)
    for phase in ('save','restore'):
        receipt=D.read(fixture.actual/phase/'receipt.json');receipt['oracle_sha256']=fixture.oracle_sha256;D.write(fixture.actual/phase/'receipt.json',receipt)
    return reseal_fixture(fixture)
