"""Host CPU recovery gate. No account call, install, device import or source mutation."""
import hashlib
import json
from pathlib import Path
import re
import exact_admission as A
import generation_gate as S
import record_validation as V

def require(value, message):
    if not value:
        raise ValueError(message)

def audit(B, P, N, *, oracle, oracle_sha256, binding, ownership, step_result,
          expected_argv, installed_source, upstream_wheel, plugin_wheel, admission_path):
    gold, _, numerical = V.verify_cpu(B, P, N, oracle, oracle_sha256, binding)
    V.validate_cpu_launch(gold, ownership, step_result, expected_argv)
    admission = A.admission(admission_path, A.INTEGRATION_ADMISSION_SHA)
    require(installed_source.get('kind') == 'NESTED_EXACT_INSTALLED_SOURCE_READBACK'
            and installed_source.get('status') == 'SOURCE_BYTES_PASS'
            and installed_source.get('source_admission_sha256') == A.INTEGRATION_ADMISSION_SHA
            and installed_source.get('upstream_python_files') == admission['upstream_python_files']
            and installed_source.get('plugin_python_files') == admission['plugin_python_files']
            and installed_source.get('py_typed_sha256') == admission['upstream_package_files']['py.typed'],
            'CPU_GATE_INSTALLED_COMPLETE_SOURCE')
    S.verify(Path(admission_path), A.INTEGRATION_ADMISSION_SHA,
             upstream_wheel=Path(upstream_wheel), plugin_wheel=Path(plugin_wheel))
    # Qualification belongs to the recovered owned Linux command. The host may be macOS.
    body = {'kind': 'ROOT_NESTED_EXACT_CPU_ORACLE_GATE', 'status': 'QUALIFIED_LINUX_CPU_ORACLE_VERIFIED',
            'oracle_sha256': oracle_sha256, 'source_admission_sha256': A.INTEGRATION_ADMISSION_SHA,
            'science_manifest_sha256': binding['science_manifest_sha256'], 'case_count': 79, 'array_count': 540,
            'owned_cpu_pid': gold['pid'], 'owned_cpu_pgid': gold['pgid'], 'owned_cpu_token': gold['process_token'],
            'upstream_wheel_sha256': A.sha(upstream_wheel), 'plugin_wheel_sha256': A.sha(plugin_wheel),
            'cpu_step_result_sha256': hashlib.sha256(json.dumps(step_result, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
            'observation_report_sha256': hashlib.sha256(json.dumps(numerical, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
            'source_variant': A.VARIANT, 'm4_status': 'NOT_QUALIFIED'}
    return body, numerical
