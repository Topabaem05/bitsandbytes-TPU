"""Local controls. No provider, credentials, packet change, or scientific execution."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import owner
OBSERVATIONS = []

NAMES = ['cloud/owner.py', 'cloud/ownership/lifecycle.py',
         'cloud/exact-nested/allocation_transport.py',
         'cloud/exact-nested/ownership/lifecycle.py',
         'cloud/exact-nested/ownership/cleanup_lifecycle.py',
         'cloud/deep/shared/one.py', 'cloud/deep/shared/two.py',
         'cloud/deep/other/three.py', 'source-admission.json']


def manifest(names=NAMES):
    return {'files': {n: {'sha256': '0'*64, 'bytes': 0} for n in names},
            'plugin_source_manifest_sha256': 'plugin', 'runtime_lock_sha256': 'runtime',
            'budget': {'total': 3600, 'install': 1200, 'science': 1800, 'retrieval': 600, 'cleanup': 60}}


def execute(code):
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        exec(compile(code, '<real-generated-readiness>', 'exec'), {})
    return capture.getvalue()


class ParentControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='bnb-owner-parent-controls-')
        self.root = Path(self.temp.name).resolve()
        self.base = self.root/'remote'

    def tearDown(self):
        self.temp.cleanup()
        self.assertFalse(self.root.exists())

    def code(self, base=None):
        base = str(base or self.base)
        return owner.control_readiness(base, owner.control_uploads(manifest(), base), 'n'*64)

    def test_old_generated_readiness_reproduces_missing_parent(self):
        # Literal fixture from the original owner, before the parent-directory correction.
        # Source SHA256: dcfa490f94de91bb1dc6734f2b89c564d37dc27886ff122e64c7ed64ba99736b.
        old = ('from pathlib import Path\nimport json,platform,sys,os\n'
               f'b=Path({str(self.base)!r})\n'
               'if b.exists() or b.is_symlink(): raise ValueError("FRESH_REMOTE_ROOT_REQUIRED")\n'
               'b.mkdir(mode=0o700); (b/"control/ownership").mkdir(parents=True)\n'
               'print(json.dumps({"python":platform.python_version(),"libc":platform.libc_ver(),"executable":sys.executable,"uid":os.getuid(),"gid":os.getgid(),"system":platform.system(),"machine":platform.machine(),"parents":["control","control/ownership"]}))\n')
        execute(old)
        with self.assertRaises(FileNotFoundError):
            (self.base/'control/exact-nested/allocation_transport.py').write_bytes(b'x')
        self.assertTrue((self.base/'control/ownership').is_dir())

    def test_multiple_depth_shared_parent_complete_map(self):
        code, proof = self.code()
        uploads = owner.control_uploads(manifest(), str(self.base))
        self.assertEqual([n for n, _ in uploads], sorted(n for n in NAMES if n.startswith('cloud/')))
        self.assertEqual(proof['parents'], sorted(set(proof['parents']), key=lambda n: (len(Path(n).parts), n)))
        self.assertEqual(owner.verify_control_readiness(execute(code), proof), proof)
        for n, dest in uploads:
            self.assertTrue(Path(dest).parent.is_dir())
            Path(dest).write_text(n)
        self.assertEqual(len(uploads), 8)

    def test_all_admitted_cloud_sources_have_parents(self):
        canonical = HERE
        files = {'cloud/'+p.relative_to(canonical).as_posix():
                 {'sha256': owner.sha(p), 'bytes': p.stat().st_size}
                 for p in canonical.rglob('*.py')
                 if 'tests' not in p.relative_to(canonical).parts and p.name != 'build_packet.py'}
        before = json.dumps(files, sort_keys=True)
        uploads = owner.control_uploads({'files': files}, str(self.base))
        code, proof = owner.control_readiness(str(self.base), uploads, 'c'*64)
        owner.verify_control_readiness(execute(code), proof)
        self.assertEqual({n for n, _ in uploads}, set(files))
        self.assertEqual(before, json.dumps(files, sort_keys=True))
        self.assertIn('cloud/exact-nested/allocation_transport.py', files)
        for _, destination in uploads: self.assertTrue(Path(destination).parent.is_dir())

    def test_existing_ownership_directory_supported(self):
        code, proof = self.code()
        original = Path.mkdir
        def mkdir(p, *args, **kwargs):
            result = original(p, *args, **kwargs)
            if p == self.base:
                original(p/'control'); original(p/'control/ownership')
            return result
        with patch.object(Path, 'mkdir', mkdir):
            self.assertEqual(owner.verify_control_readiness(execute(code), proof), proof)

    def test_existing_fresh_root_rejected(self):
        self.base.mkdir()
        with self.assertRaisesRegex(ValueError, 'FRESH_REMOTE_ROOT_REQUIRED'):
            execute(self.code()[0])

    def test_fresh_root_symlink_rejected(self):
        self.base.symlink_to(self.root/'absent')
        with self.assertRaisesRegex(ValueError, 'FRESH_REMOTE_ROOT_REQUIRED'):
            execute(self.code()[0])

    def test_file_and_symlink_existing_ancestor_rejected(self):
        for mode in ('file', 'symlink'):
            with self.subTest(mode=mode):
                base = self.root/('remote-'+mode)
                code, _ = self.code(base)
                original = Path.mkdir
                def mkdir(p, *args, **kwargs):
                    result = original(p, *args, **kwargs)
                    if p == base:
                        original(p/'control')
                        if mode == 'file': (p/'control/exact-nested').write_text('bad')
                        else: (p/'control/exact-nested').symlink_to(self.root, target_is_directory=True)
                    return result
                with patch.object(Path, 'mkdir', mkdir):
                    with self.assertRaisesRegex(ValueError, 'CONTROL_PARENT_NOT_DIRECTORY'):
                        execute(code)

    def test_symlink_ancestor_above_fresh_base_rejected(self):
        link = self.root/'alias'; link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'CONTROL_PARENT_NOT_DIRECTORY'):
            execute(self.code(link/'remote')[0])

    def test_noncanonical_manifest_paths_rejected(self):
        for name in ('/cloud/a.py', '../cloud/a.py', 'cloud/../a.py', 'cloud//a.py',
                     'cloud/./a.py', 'cloud/a\\b.py', '', 'cloud/a.py/'):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, 'CONTROL_SOURCE_PATH'):
                    owner.control_uploads(manifest([name]), str(self.base))

    def test_noncanonical_remote_base_rejected(self):
        for base in ('relative', '/x/../y', '/x//y', '/x/./y', '/x\\y', '/x/', ''):
            with self.subTest(base=base):
                with self.assertRaisesRegex(ValueError, 'CONTROL_REMOTE_BASE_PATH'):
                    owner.control_uploads(manifest(), base)

    def test_cli_zero_incorrect_or_absent_completion_rejected(self):
        _, proof = self.code()
        for output in ('', 'Traceback\nValueError: failure', '{}',
                       json.dumps({**proof, 'nonce': 'wrong'}),
                       json.dumps({**proof, 'parents': ['control']})):
            with self.subTest(output=output):
                with self.assertRaisesRegex(RuntimeError, 'CONTROL_PARENTS_NOT_PROVED'):
                    owner.verify_control_readiness(output, proof)

    def test_runtime_info_then_one_final_completion_accepted(self):
        _, proof = self.code()
        text = json.dumps({'python': '3.12.12', 'parents': proof['parents']})+'\n\n'+json.dumps(proof)+'\n \n'
        self.assertEqual(owner.verify_control_readiness(text, proof), proof)

    def test_duplicate_completion_records_rejected(self):
        _, proof = self.code()
        for earlier in (proof, {**proof, 'nonce': 'incorrect'}):
            with self.subTest(earlier=earlier):
                with self.assertRaisesRegex(RuntimeError, 'CONTROL_PARENTS_NOT_PROVED'):
                    owner.verify_control_readiness(json.dumps(earlier)+'\n'+json.dumps(proof), proof)

    def test_correct_completion_then_error_or_output_rejected(self):
        _, proof = self.code()
        for trailing in ('Traceback\nRuntimeError: failure', 'upload starting', '{}', json.dumps({'status': 'ERROR'})):
            with self.subTest(trailing=trailing):
                with self.assertRaisesRegex(RuntimeError, 'CONTROL_PARENTS_NOT_PROVED'):
                    owner.verify_control_readiness(json.dumps(proof)+'\n'+trailing, proof)

    def test_duplicate_json_fields_rejected(self):
        _, proof = self.code()
        text = json.dumps(proof)
        for output in (text.replace('"nonce":', '"nonce":"incorrect","nonce":', 1),
                       '{"runtime":{"python":"3.12","python":"3.12"}}\n'+text):
            with self.subTest(output=output):
                with self.assertRaisesRegex(RuntimeError, 'CONTROL_PARENTS_NOT_PROVED'):
                    owner.verify_control_readiness(output, proof)

    def drive_case(self, fault):
        packet = self.root/'packet'; packet.mkdir()
        out = self.root/'output'
        ident = self.root/'identity.json'; ident.write_text('{}')
        m = manifest()
        gate = {'status': 'ACTUAL_DISPATCH_AUTHORIZED', 'packet_sha256': 'fixture',
                'driver_sha256': owner.sha(HERE/'owner.py'), 'output': str(out),
                'plugin_source_manifest_sha256': 'plugin', 'runtime_lock_sha256': 'runtime',
                'budget': m['budget'], 'one_allocation_only': True, 'provider_or_solver': 'FORBIDDEN',
                'cli_identity_sha256': owner.sha(ident)}
        acceptance = self.root/'acceptance.json'; acceptance.write_text(json.dumps(gate))
        calls = []; clock = [1000.0]; session = []
        handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        failed_child = None
        def api(label, argv, timeout):
            nonlocal failed_child
            calls.append((label, argv, timeout))
            if label == '03-one-allocation': session.append(argv[3])
            if label == '04-sessions-after':
                if fault == 'expired-before-readiness': clock[0] += 3000
                return {'status': 'PASS'}, '['+session[0]+'] | Hardware: V6E1'
            if label == '05-readiness':
                code = Path(argv[-1]).read_text().replace(repr('/content/bnb-tpu-first'), repr(str(self.base)))
                text = execute(code)
                if fault == 'missing-marker': text = 'Kernel traceback with CLI exit 0'
                if fault == 'trailing-error': text += 'Traceback\nRuntimeError: readiness failed\n'
                if fault == 'expired-after-readiness': clock[0] += 3000
                return {'status': 'PASS'}, text
            if label == '06-cloud-exact-nested-allocation_transport.py' and fault == 'upload-child-failure':
                cli = owner.OfficialCLI(Path(sys.executable), out)
                cli.prefix = [sys.executable, '-B', '-c', 'raise SystemExit(17)']
                record, text = cli(label, argv, timeout)
                failed_child = record
                return record, text
            if argv[0] == 'upload':
                if label == '07-payload': return {'status': 'CLI_FAILED'}, 'FIXTURE_STOP_BEFORE_SCIENCE'
                dest = self.base/argv[-1].removeprefix('content/bnb-tpu-first/')
                self.assertTrue(dest.parent.is_dir()); dest.write_text(argv[-2])
            if argv[0] == 'download': return {'status': 'CLI_FAILED'}, 'NO_SCIENTIFIC_RECORDS_IN_FIXTURE'
            return {'status': 'PASS'}, 'No active sessions found on server.' if argv == ['sessions'] else 'Active assignments: 0\nUsage rate: 0.00/hr'
        with patch.object(owner, 'preflight', return_value=m), patch.object(owner, 'verify_cli_identity'), patch.object(owner.time, 'time', side_effect=lambda: clock[0]):
            result = owner.drive(packet, out, 'fixture', acceptance, owner.sha(acceptance), Path('fixture-python'), ident, api=api, simulated=True)
        labels = [label for label, _, _ in calls]
        self.assertEqual(labels[-4:], ['90-before-stop','91-stop-exact','92-after-stop','93-usage-after'])
        self.assertEqual(sum(t for l, _, t in calls if l.startswith(('90-', '91-', '92-', '93-'))), 60)
        self.assertEqual(next(a for l, a, _ in calls if l == '91-stop-exact'), ['stop', '-s', session[0]])
        self.assertEqual(result['allocation_epoch'], 1000.0)
        self.assertEqual(result['allocation_attempts'], 1)
        self.assertTrue(result['server_empty_observed'] and result['usage_zero_observed'])
        self.assertTrue(all(signal.getsignal(s) == handler for s, handler in handlers.items()))
        self.assertFalse(any(label in labels for label in ('08-unpack', '09-install', '10-cpu', '13-tpu')))
        OBSERVATIONS.append({'fault': fault, 'status': result['status'],
                             'original_error': result['original_error'],
                             'allocation_epoch': result['allocation_epoch'],
                             'allocation_attempts': result['allocation_attempts'],
                             'command_labels': labels,
                             'exact_stop': True, 'server_empty_observed': result['server_empty_observed'],
                             'usage_zero_observed': result['usage_zero_observed'],
                             'real_failed_child_cleanup': failed_child['cleanup'] if failed_child else None})
        return result, labels, failed_child

    def test_full_drive_complete_upload_inventory(self):
        result, labels, _ = self.drive_case('complete')
        self.assertEqual([l for l in labels if l.startswith('06-')], ['06-'+n.replace('/', '-') for n in sorted(NAMES) if n.startswith('cloud/')])
        self.assertEqual(result['original_error']['message'], 'CLI_STEP_BLOCKED:07-payload')
        self.assertEqual(result['status'], 'BLOCKED')

    def test_full_drive_cli_zero_missing_marker_no_upload(self):
        result, labels, _ = self.drive_case('missing-marker')
        self.assertEqual(result['original_error']['message'], 'CONTROL_PARENTS_NOT_PROVED')
        self.assertFalse(any(l.startswith(('06-', '07-')) for l in labels))

    def test_full_drive_cli_zero_trailing_error_no_upload(self):
        result, labels, _ = self.drive_case('trailing-error')
        self.assertEqual(result['original_error']['message'], 'CONTROL_PARENTS_NOT_PROVED')
        self.assertFalse(any(l.startswith(('06-', '07-')) for l in labels))

    def test_full_drive_real_upload_child_failure_closed(self):
        result, labels, child = self.drive_case('upload-child-failure')
        self.assertEqual(result['original_error']['message'], 'CLI_STEP_BLOCKED:06-cloud-exact-nested-allocation_transport.py')
        self.assertEqual(child['exit_code'], 17)
        self.assertEqual(child['cleanup']['errors'], [])
        self.assertEqual(len(child['cleanup']['groups']), 1)
        self.assertEqual(child['cleanup']['groups'][0]['status'], 'CLEANUP_VERIFIED')
        self.assertNotIn('07-payload', labels)

    def test_original_deadline_no_readiness_or_upload(self):
        result, labels, _ = self.drive_case('expired-before-readiness')
        self.assertEqual(result['original_error']['message'], 'TOTAL_LIFECYCLE_RESERVE')
        self.assertNotIn('05-readiness', labels)
        self.assertFalse(any(l.startswith('06-') for l in labels))

    def test_original_deadline_no_upload_after_readiness(self):
        result, labels, _ = self.drive_case('expired-after-readiness')
        self.assertEqual(result['original_error']['message'], 'TOTAL_LIFECYCLE_RESERVE')
        self.assertIn('control_parents_readback', result)
        self.assertFalse(any(l.startswith('06-') for l in labels))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--report', type=Path)
    options, unittest_args = parser.parse_known_args()
    if options.report is not None:
        options.report = options.report.resolve()
        if options.report.is_relative_to(HERE):
            parser.error('REPORT_PATH_MUST_BE_OUTSIDE_CLOUD_SOURCE')
    program = unittest.main(argv=[sys.argv[0], *unittest_args], verbosity=2, exit=False)
    result = program.result
    if options.report is not None:
        report = {'scope': 'LOCAL_NO_PROVIDER_NO_SCIENCE', 'tests_run': result.testsRun,
                  'failures': len(result.failures), 'errors': len(result.errors),
                  'candidate_owner_sha256': owner.sha(HERE/'owner.py'),
                  'observations': OBSERVATIONS, 'fixtures_deleted_by_teardown': True}
        options.report.write_text(json.dumps(report, indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
