"""Read-only old evidence and isolated packet maps. No provider or tensor runtime."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'packages/bitsandbytes-tpu/pyproject.toml').is_file())
SCIENCE=ROOT/'experiments/2026-10-04-bitsandbytes-tpu'
CLOUD=Path(os.environ.get('BNB_PRIMITIVE_PARAMETER_CONTRACT',SCIENCE/'cloud/primitive_contract.py'))
PROBE=Path(os.environ.get('BNB_PRIMITIVE_PARAMETER_SOURCE',SCIENCE/'probe_primitives.py'))

def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

PC=load(CLOUD,'metric_admission_contract')
V=load(PROBE,'metric_admission_probe')


def packet():
    path=os.environ.get('BNB_PRIMITIVE_PARAMETER_PRIOR_PACKET_MANIFEST')
    if not path:pytest.skip('Requires explicit original genuine packet manifest')
    return read(path)


def current_manifest():
    value=copy.deepcopy(packet())
    for field,name in PC.BINDINGS.items():
        pin={'probe_primitives.py':PC.PROBE_SHA,'primitive_kernels.py':PC.KERNEL_SHA,'primitive-inputs.json':PC.INPUT_SHA,
             'primitive_reference.py':PC.REFERENCE_SHA,'hlo_bitcast.py':PC.HLO_SHA,
             'probe_nested_state.py':PC.STATE_SHA,'primitive-manifest.json':PC.MANIFEST_SHA}[name]
        value[field]=pin;value['files'][name]['sha256']=pin
        if name=='probe_primitives.py':value['files'][name]['bytes']=PROBE.stat().st_size
        if name=='primitive-manifest.json':value['files'][name]['bytes']=(PROBE.parent/name).stat().st_size
    return value


def test_current_complete_source_map_admitted():
    PC.verify_manifest(current_manifest())
    protocol=read(PROBE.parent/'primitive-manifest.json')
    assert protocol['sources']==PC.SOURCES and sha(PROBE)==PC.PROBE_SHA
    assert sha(PROBE.parent/'primitive-manifest.json')==PC.MANIFEST_SHA
    assert {name:{'sha256':sha(PROBE.parent/name),'bytes':(PROBE.parent/name).stat().st_size} for name in PC.SOURCES}==PC.SOURCES


@pytest.mark.parametrize('fault',['original','wrong-probe','wrong-file-map','wrong-manifest','wrong-variant'])
def test_wrong_or_old_generation_rejected(fault):
    value=current_manifest()
    if fault=='original':value=packet()
    elif fault=='wrong-probe':value['primitive_probe_sha256']='f'*64
    elif fault=='wrong-file-map':value['files']['probe_primitives.py']['sha256']='f'*64
    elif fault=='wrong-manifest':value['primitive_manifest_sha256']='f'*64
    else:value['primitive_variant']='arithmetic-v2'
    with pytest.raises(ValueError,match='PRIMITIVE_'):PC.verify_manifest(value)


def test_original_actual_cpu_oracle_rejected_without_mutation():
    root=os.environ.get('BNB_PRIMITIVE_PARAMETER_PRIOR_CPU_ORACLE')
    if not root:pytest.skip('Requires explicit original recovered CPU oracle')
    root=Path(root);before={p.name:sha(p) for p in root.iterdir() if p.is_file()};seal=read(root/'oracle-seal.json')
    args=SimpleNamespace(backend_probe=SCIENCE/'probe_backend.py',route_probe=SCIENCE/'probe_routes.py',
        precision_probe=SCIENCE/'probe_precision.py',transfer_admission=SCIENCE/'transfer_admission.py',
        nested_probe=SCIENCE/'probe_nested.py',nested_state_probe=SCIENCE/'probe_nested_state.py',
        patch_manifest=ROOT/'patches/params4bit-xla-v1.json',admission_sha256=seal['source_admission_sha256'],
        oracle=root,oracle_sha256=sha(root/'oracle-seal.json'))
    B,R,P,A,N,S=V.helpers(args)
    with pytest.raises(ValueError,match='BINDING_probe_sha256'):V.verify_cpu_oracle(B,R,P,A,N,S,args)
    assert {p.name:sha(p) for p in root.iterdir() if p.is_file()}==before
