"""State8 dependency and ordering contract. It cannot create acceptance."""
from pathlib import Path
import exact_admission as A

def fixed_cases(inputs):
    A.require(inputs == A.read(Path(__file__).with_name('nested-inputs.json')), 'ORIGINAL79_STATE8_INPUTS')
    cases = [case for case in inputs['cases'] if case['kind'] == 'module']
    A.require(len(cases) == 8 and {case['dtype'] for case in cases} == {'float32', 'bfloat16'}
              and {len(case['x_shape']) for case in cases} == {2, 3}
              and {case['bias'] is None for case in cases} == {True, False}
              and all(case['weight_shape'] == [5, 67] for case in cases), 'UNCHANGED_STATE8_MATRIX')
    return cases

def stage_order(*, accepted79_path, accepted79_sha256, science_manifest_sha256,
                fresh_cpu_gate, fresh_nested_oracle_sha256, inputs):
    """Old79 FAIL, another generation, missing state, or stale fresh-CPU seal reject."""
    record = A.read(accepted79_path)
    A.state_dependency(record, accepted79_path, accepted79_sha256, science_manifest_sha256)
    A.require(fresh_cpu_gate.get('kind') == 'ROOT_NESTED_EXACT_CPU_ORACLE_GATE'
              and fresh_cpu_gate.get('status') == 'QUALIFIED_LINUX_CPU_ORACLE_VERIFIED'
              and fresh_cpu_gate.get('source_variant') == A.VARIANT
              and fresh_cpu_gate.get('source_admission_sha256') == A.INTEGRATION_ADMISSION_SHA
              and fresh_cpu_gate.get('science_manifest_sha256') == science_manifest_sha256
              and fresh_cpu_gate.get('oracle_sha256') == fresh_nested_oracle_sha256
              and fresh_cpu_gate.get('case_count') == 79 and fresh_cpu_gate.get('array_count') == 540,
              'FRESH_STATE_ALLOCATION_CPU_ORACLE_REQUIRED')
    cases = fixed_cases(inputs)
    return {'kind': 'NESTED_EXACT_STATE8_PREPARATION_CONTRACT', 'status': 'DEPENDENCIES_VALIDATED_STATE_NOT_EXECUTED',
            'accepted79_result_sha256': accepted79_sha256, 'fresh_nested_oracle_sha256': fresh_nested_oracle_sha256,
            'source_variant': A.VARIANT, 'case_ids': [c['id'] for c in cases],
            'restore_routes': ['target_constructor', 'from_prequantized', 'strict_assign_false'],
            'checkpoint': 'ALL_ORIGINAL_STATE_DICT_KEYS_AND_CPU_TENSOR_BYTES',
            'identity': 'ORIGINAL_CLASSES_FRESH_PARAMS_AND_BIAS_IDENTITY',
            'process': 'FRESH_SAVE_REAPED_THEN_RESTORE_SHARED_OWNED_PGID_DISTINCT_PIDS_TOKENS',
            'state_execution_bridge': 'PROBE_STATE8_EXACT_REQUIRES_SEPARATE_ROOT_REVIEW_AND_DEVICE_RECORDS',
            'm4_status': 'NOT_QUALIFIED'}
