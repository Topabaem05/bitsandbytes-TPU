"""Run portable synthetic scientific controls under the unchanged local ownership guard."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'ownership'))
from cleanup_lifecycle import Ownership


def test_portable_primitive_saved_record_scientific_controls(tmp_path):
    directory = tmp_path / 'owner'; directory.mkdir()
    guard = Ownership(directory)
    stdout = (tmp_path / 'science.stdout').open('wb')
    stderr = (tmp_path / 'science.stderr').open('wb')
    try:
        with guard.guard(120):
            child = guard.launch([sys.executable, '-B', str(HERE / 'tests/primitive_saved_fixture.py')],
                                 stdout=stdout, stderr=stderr, record=directory / 'ownership.json')
            child.wait()
    finally:
        stdout.close(); stderr.close()
    assert child.returncode == 0, (tmp_path / 'science.stderr').read_text()
    assert not guard.summary()['errors']
    assert 'Ran 25 tests' in (tmp_path / 'science.stderr').read_text()
