"""Standard-library controls. Do not import account or provider libraries."""
import argparse
import builtins
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Response:
    def __init__(self, status=400, body=b'{}'):
        self.status_code = status
        self._content = body

    def __bool__(self):
        return self.status_code < 400


class HTTPError(Exception):
    def __init__(self, response=None):
        super().__init__('SYNTHETIC_SECRET')
        self.response = response


class Controls(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='m8-retention-control-')
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_success_has_no_error_record(self):
        self.assertFalse(H.record_response_error(Response(200), self.root))
        self.assertFalse((self.root/'http-error.sanitized.json').exists())

    def test_400_retains_bounded_classification(self):
        response = Response(body=b'{"code":400,"message":"Source size exceeds limit"}')
        self.assertFalse(response)
        self.assertTrue(H.record_response_error(response, self.root))
        result = json.loads((self.root/'http-error.sanitized.json').read_text())
        self.assertTrue(result['response_present'])
        self.assertEqual(result['http_status'], 400)
        self.assertEqual(result['constraint_classes'], ['SOURCE_SIZE'])
        self.assertEqual(result['remote_effect'], 'UNKNOWN')
        self.assertFalse(result['automatic_retry'])
        self.assertEqual(result['acceptance'], 'NOT_QUALIFIED')
        self.assertEqual((self.root/'http-error.sanitized.json').stat().st_mode & 0o777, 0o600)

    def test_buffer_cap_does_not_parse_oversize(self):
        result = H.describe(HTTPError(Response(body=b'x'*(H.MAX_BODY+1))))
        self.assertEqual(result['prefix_bytes'], H.MAX_BODY)
        self.assertEqual(result['response_body'], 'OVERSIZE_NOT_PARSED')
        self.assertEqual(result['constraint_classes'], [])

    def test_no_stream_read(self):
        response = Response(body=False)
        response.raw = SimpleNamespace(read=lambda *_: self.fail('stream read'))
        self.assertEqual(H.describe(HTTPError(response))['response_body'], 'NOT_BUFFERED')

    def test_unknown_and_secret_text_dropped(self):
        result = H.describe(HTTPError(Response(body=b'{"message":"SYNTHETIC_SECRET https://signed.invalid/?token=secret"}')))
        self.assertEqual(result['constraint_classes'], [])
        for value in ['SYNTHETIC_SECRET', 'signed.invalid', 'token=', 'message']:
            self.assertNotIn(value, json.dumps(result))

    def test_writer_and_stderr_failures_preserve_response(self):
        response = Response()
        def fail(*_): raise OSError('synthetic write failure')
        with patch.object(H.sys, 'stderr', SimpleNamespace(write=fail)):
            self.assertFalse(H.record_response_error(response, self.root, writer=fail))
        self.assertEqual(response.status_code, 400)

    def test_malformed_observer_preserves_response(self):
        response = Response()
        with patch.object(H, 'describe', side_effect=ValueError('synthetic observer failure')):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertFalse(H.record_response_error(response, self.root))
        self.assertEqual(response.status_code, 400)

    def test_original_exception_identity(self):
        original = HTTPError(Response())
        def fail(*_): raise OSError('synthetic write failure')
        try:
            with H.retained_errors(self.root, writer=fail):
                raise original
        except HTTPError as caught:
            self.assertIs(caught, original)
        else:
            self.fail('primary error disappeared')

    def test_note_failure_preserves_exception(self):
        class Primary(HTTPError):
            def add_note(self, _): raise OSError('synthetic note failure')
        original = Primary(Response())
        def fail(*_): raise OSError('synthetic write failure')
        try:
            with H.retained_errors(self.root, writer=fail): raise original
        except Primary as caught: self.assertIs(caught, original)
        else: self.fail('primary error disappeared')

    def worker(self):
        worker = load(SOURCE/'sdk_worker.py', 'portable_retention_worker')
        worker.HERE = self.root
        return worker

    def test_exact_helper_admission(self):
        shutil.copyfile(SOURCE/'http_error_retention.py', self.root/'http_error_retention.py')
        self.worker().admit_retention()

    def rejected_before_source_or_auth(self):
        worker = self.worker()
        worker.assert_sources = lambda: self.fail('source/auth path reached')
        with self.assertRaisesRegex(ValueError, 'ERROR_RETENTION_SOURCE_CHANGED'):
            worker.run({'operation':'submit', 'deadline_epoch':time.time()+60}, self.root)

    def test_changed_helper_rejected_before_auth(self):
        (self.root/'http_error_retention.py').write_bytes((SOURCE/'http_error_retention.py').read_bytes()+b'\n')
        self.rejected_before_source_or_auth()

    def test_missing_helper_rejected_before_auth(self):
        self.rejected_before_source_or_auth()

    def test_symlink_helper_rejected_before_auth(self):
        (self.root/'http_error_retention.py').symlink_to(SOURCE/'http_error_retention.py')
        self.rejected_before_source_or_auth()

    def test_one_post_and_transport_restoration(self):
        worker = load(SOURCE/'sdk_worker.py', 'portable_retention_transport')
        calls = []
        response = Response()
        class Session:
            def get_adapter(self, _): return SimpleNamespace(max_retries=SimpleNamespace(total=0))
            def send(self, request, **kwargs):
                calls.append((request, kwargs))
                return response
        fake = SimpleNamespace(sessions=SimpleNamespace(Session=Session))
        original = Session.send
        request = SimpleNamespace(url=worker.SAVE_URL, method='POST', body=b'{}')
        # Temporary sys.path permits only the reviewed sibling helper import.
        sys.path.insert(0, str(SOURCE))
        try:
            with worker.bounded_transport(time.time()+60, fake, error_output=self.root):
                self.assertIs(Session().send(request, verify=True), response)
                with self.assertRaisesRegex(ValueError, 'SAVE_REPLAY_OR_METHOD'):
                    Session().send(request, verify=True)
        finally:
            sys.path.pop(0)
        self.assertIs(Session.send, original)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], request)
        self.assertTrue(calls[0][1]['verify'])
        self.assertFalse(calls[0][1]['allow_redirects'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host-source', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    SOURCE = args.host_source.resolve()
    attempts = {'network':0, 'credential':0}
    def forbidden_network(*_, **__):
        attempts['network'] += 1
        raise AssertionError('NETWORK_FORBIDDEN')
    original_open = builtins.open
    def guarded_open(file, *a, **k):
        if isinstance(file, (str, bytes, Path)) and any(x in str(file) for x in ('/.kaggle/', 'access_token', 'credentials.json')):
            attempts['credential'] += 1
            raise AssertionError('CREDENTIAL_READ_FORBIDDEN')
        return original_open(file, *a, **k)
    with patch.object(socket.socket, 'connect', forbidden_network), patch.object(socket, 'create_connection', forbidden_network), patch.object(builtins, 'open', guarded_open):
        H = load(SOURCE/'http_error_retention.py', 'portable_retention_helper')
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    report = {'status':'PASS' if result.wasSuccessful() else 'FAIL', 'tests':result.testsRun,
              'failures':len(result.failures), 'errors':len(result.errors),
              'network_attempts':attempts['network'], 'credential_read_attempts':attempts['credential'],
              'provider_calls':0, 'm8_status':'NOT_QUALIFIED'}
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'controls.json').write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() and not any(attempts.values()) else 1)
