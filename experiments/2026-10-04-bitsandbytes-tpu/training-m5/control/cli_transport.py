"""Frozen official CLI ownership/identity helpers from baseline bf7c843 owner.py."""
import email,os,sys
from pathlib import Path
import contract as C
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/"ownership"))
from cleanup_lifecycle import Ownership
from lifecycle import durable_json
sha=C.sha

def verify_cli_identity(python, identity):
    """Read the admitted installed CLI files. Do not import auth or execute a command."""
    if identity.get('distribution') != 'google-colab-cli' or identity.get('version') != '0.7.4':
        raise ValueError('CLI_PROFILE')
    site = python.absolute().parent.parent / 'lib/python3.12/site-packages'
    metadata_path = site / 'google_colab_cli-0.7.4.dist-info/METADATA'
    record_path = site / 'google_colab_cli-0.7.4.dist-info/RECORD'
    cfg = python.absolute().parent.parent / 'pyvenv.cfg'
    if sha(python.resolve()) != identity.get('python_sha256') or sha(cfg) != identity.get('venv_config_sha256') or sha(metadata_path) != identity.get('metadata_sha256') or sha(record_path) != identity.get('record_sha256'):
        raise ValueError('CLI_INTERPRETER_OR_DISTRIBUTION_SHA')
    meta = email.message_from_bytes(metadata_path.read_bytes())
    if meta['Name'] != 'google-colab-cli' or meta['Version'] != '0.7.4':
        raise ValueError('CLI_INSTALLED_VERSION')
    files = identity.get('files', {})
    actual = {p.relative_to(site).as_posix() for p in (site / 'colab_cli').rglob('*.py')}
    if not files or actual != set(files):
        raise ValueError('CLI_SOURCE_INVENTORY')
    for name, digest in files.items():
        p = site / name
        if not name.startswith('colab_cli/') or '..' in Path(name).parts or p.is_symlink() or sha(p) != digest:
            raise ValueError('CLI_SOURCE_SHA')

class OfficialCLI:
    def __init__(self, python, out):
        self.prefix = [str(python), '-B', str(HERE / 'official_cli_no_oauth.py')]
        self.out = out

    def __call__(self, label, argv, timeout):
        directory = self.out / 'cli' / label
        directory.mkdir(parents=True)
        owner = Ownership(directory)
        record = {'status': 'BLOCKED', 'argv': self.prefix + argv, 'timeout_seconds': timeout}
        stream = None
        try:
            stream = (directory / 'stdout-stderr.raw').open('wb', buffering=0)
            with owner.guard(timeout):
                p = owner.launch(record['argv'], record=directory / 'ownership.json', cwd=HERE,
                                 env={**{k: v for k, v in os.environ.items() if k not in {'PYTHONPATH', 'PYTHONHOME'}}, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1'}, stdout=stream, stderr=stream)
                p.wait()
                record.update(status='PASS' if p.returncode == 0 else 'CLI_FAILED', exit_code=p.returncode)
        except BaseException as error:
            record['error'] = {'type': type(error).__name__, 'message': str(error)}
        finally:
            record['original_outcome'] = {'status': record['status'], 'exit_code': record.get('exit_code'), 'error': record.get('error')}
            if stream is not None:
                try:
                    os.fsync(stream.fileno())
                except Exception as error:
                    owner.note_error('fsync_cli_log', error)
                finally:
                    try:
                        stream.close()
                    except Exception as error:
                        owner.note_error('close_cli_log', error)
            record['cleanup'] = owner.summary()
            if record['cleanup']['errors']:
                record['status'] = 'BLOCKED_CLEANUP'
            durable_json(directory / 'result.json', record)
        text = (directory / 'stdout-stderr.raw').read_text(errors='replace') if (directory / 'stdout-stderr.raw').exists() else ''
        return record, text
