"""Fresh-process native repair controls. Ordinary cloud tests never import JAX."""
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

HERE = Path(__file__).resolve().parent


def test_native_harness_regressions(tmp_path):
    report = tmp_path / 'harness.xml'
    command = [sys.executable, '-B', '-m', 'pytest', str(HERE / 'native_harness_controls.py'),
        '-q', '--basetemp=' + str(tmp_path / 'controls'), '--junitxml=' + str(report)]
    result = subprocess.run(command, capture_output=True, timeout=60,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'))
    (tmp_path / 'harness.stdout').write_bytes(result.stdout)
    (tmp_path / 'harness.stderr').write_bytes(result.stderr)
    assert result.returncode == 0, result.stderr.decode(errors='replace') + result.stdout.decode(errors='replace')
    suite = ET.parse(report).getroot().find('testsuite')
    assert int(suite.attrib['tests']) == 26 and int(suite.attrib['failures']) == int(suite.attrib['errors']) == 0


def test_native_pallas_regressions_optional(tmp_path):
    python = os.environ.get('BNB_NATIVE_JAX071_PYTHON')
    if not python:
        pytest.skip('Set BNB_NATIVE_JAX071_PYTHON to a reviewed JAX/jaxlib0.7.1 CPU interpreter.')
    native = Path(os.environ.get('BNB_NATIVE_REPAIR_SOURCE', HERE.parents[1] / 'native')).resolve()
    output = tmp_path / 'pallas'
    command = [python, '-B', str(HERE / 'native_pallas_regression.py'),
        '--native-source', str(native), '--output', str(output)]
    result = subprocess.run(command, capture_output=True, timeout=90,
        env=dict(os.environ, JAX_PLATFORMS='cpu', PYTHONDONTWRITEBYTECODE='1'))
    (tmp_path / 'pallas.stdout').write_bytes(result.stdout)
    (tmp_path / 'pallas.stderr').write_bytes(result.stderr)
    assert result.returncode == 0, result.stderr.decode(errors='replace') + result.stdout.decode(errors='replace')
    record = json.loads((output / 'results.json').read_text())
    assert record['status'] == 'PASS_OFFLINE_REGRESSION' and len(record['controls']) == 7
    assert record['actual_tpu'] == 'NOT_RUN' and record['provider_calls'] == 0
