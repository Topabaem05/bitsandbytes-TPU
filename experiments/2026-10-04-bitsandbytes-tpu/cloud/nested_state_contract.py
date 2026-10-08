"""Exact state8 supplement and accepted nested79 dependency. No milestone acceptance."""
import hashlib
import importlib.util
import json
import re
from pathlib import Path
_spec = importlib.util.spec_from_file_location('_bnb_state_admitted_nested_contract', Path(__file__).resolve().parent / 'nested_contract.py')
_contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_contract)
for _name in ('NESTED_BINDINGS', 'NESTED_PLUGIN_MANIFEST_SHA', 'NESTED_PROBE_SHA', 'NESTED_INPUT_SHA',
              'NESTED_SOURCE_SHA', 'NESTED_SCHEMA_SHA', 'verify_nested_manifest', 'verify_nested_payload', 'nested_cli'):
    globals()[_name] = getattr(_contract, _name)

NESTED_STATE_PROBE_SHA='1233803f173b9e0d6e864e1fa3081f66408b357ffe17d925b0a69459514fb1f1'
NESTED_STATE_SCOPE='FRESH_SAVE_AND_RESTORE_PROCESSES_ALL8_NESTED_LINEAR'
NESTED_STATE_BINDINGS={'nested_state_probe_sha256':'probe_nested_state.py','nested79_dependency_sha256':'nested79-dependency.json'}
RUNTIME_SHA='323371ff61c5fbcc4f79fd6a358cf2ba17cb72b907382a5ceaac07f91dc66ed6'
PROFILE_SHA='ad53f6416bdabf6440f08b313963c947a468cd39afd413bfe41bb381c4a7a166'
PATCH_SHA='e745fbf21aac10ed9118167a131d6505bf6dbe03e1fab0a669c663b26c5a5732'


def require(ok,name):
    if not ok:raise ValueError(name)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_dependency(record):
    fixed={'format':'bnb-tpu.nested79-dependency.v1','status':'ACCEPTED_NESTED79_DEVICE_RECORDS',
        'record_validation':'PASS','numerical_status':'PASS','cases':79,
        'plugin_source_manifest_sha256':NESTED_PLUGIN_MANIFEST_SHA,'nested_probe_sha256':NESTED_PROBE_SHA,
        'nested_inputs_sha256':NESTED_INPUT_SHA,'nested_source_sha256':NESTED_SOURCE_SHA,
        'nested_schemas_sha256':NESTED_SCHEMA_SHA,'patch_manifest_sha256':PATCH_SHA,
        'runtime_lock_sha256':RUNTIME_SHA,'profile_sha256':PROFILE_SHA,
        'source_admission_sha256':'6647dbe8d4928a201020e7116105e836585bc1d4ad1adf3cc730976d9885b139'}
    provenance={'packet_sha256','actual_receipt_sha256','qualified_oracle_sha256','accepted_result_sha256'}
    require(type(record) is dict and set(record)==set(fixed)|provenance,'NESTED79_DEPENDENCY_FIELDS')
    require(all(record.get(k)==v for k,v in fixed.items()) and type(record['cases']) is int,'NESTED79_ACCEPTED_DEPENDENCY')
    require(all(isinstance(record[k],str) and re.fullmatch('[0-9a-f]{64}',record[k]) for k in provenance),'NESTED79_DEPENDENCY_PROVENANCE')
    return record

def verify_state_manifest(manifest):
    verify_nested_manifest(manifest)
    for field,name in NESTED_STATE_BINDINGS.items():
        value=manifest.get(field)
        require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) and value==manifest['files'].get(name,{}).get('sha256'),'NESTED_STATE_BINDING:'+field)
    require(manifest['nested_state_probe_sha256']==NESTED_STATE_PROBE_SHA and manifest.get('nested_state_scope')==NESTED_STATE_SCOPE and
            manifest.get('nested_state_variant')=='nested-state-v1' and manifest.get('source_admission_sha256')=='6647dbe8d4928a201020e7116105e836585bc1d4ad1adf3cc730976d9885b139','NESTED_STATE_REVIEWED_SCOPE')

def verify_state_payload(payload,manifest):
    verify_state_manifest(manifest);verify_nested_payload(payload,manifest)
    verify_dependency(json.loads((payload/'nested79-dependency.json').read_text()))

def state_cli(payload,nested_oracle,nested_oracle_sha):
    return [*nested_cli(payload),'--nested-probe',payload/'probe_nested.py',
        '--nested-oracle',nested_oracle,'--nested-oracle-sha256',nested_oracle_sha]
