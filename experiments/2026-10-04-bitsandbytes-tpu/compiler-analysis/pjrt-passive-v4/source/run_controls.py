"""Build and exercise the actual table interposer. Delete each fixture promptly."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import urllib.request

import sys
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from verify import envelope, identity, seal, verify

ROOT = Path(__file__).resolve().parent
HEADER_URL = 'https://raw.githubusercontent.com/openxla/xla/92f7b5952dd585c5be17c9a5caad27407005b513/xla/pjrt/c/pjrt_c_api.h'
HEADER_SHA = 'a946f54b0df19ecae28015b86e4c10fad23ba0ce6fc3f14941157db79c437cf6'
PASS_CASES = {'cold', 'warm_reuse', 'destroy_reuse', 'multiple_executables',
              'delete_after', 'inflight', 'entry_boundary', 'byte_boundary', 'capacity_larger',
              'unrelated_query_inflight', 'query_after_execute', 'error_after_execute',
              'destroy_executable_before', 'destroy_executable_reuse'}
UNKNOWN_CASES = {'wrong_loaded', 'compile_error', 'execute_error', 'short_api',
                 'short_args', 'deserialize', 'changed_length', 'changed_format',
                 'oversize_query', 'unknown_fill', 'duplicate_loaded',
                 'duplicate_executable', 'delete_before', 'entry_quota',
                 'byte_quota', 'expired', 'deadline_inflight', 'overlap', 'write_failure', 'existing_output', 'stale_optimized', 'changed_pointer', 'unsupported_abi', 'short_program', 'wrong_executable', 'overlap_destroy', 'optimized_query_error_inflight',
                 'optimized_query_success_inflight', 'execute_then_query',
                 'execute_then_fill', 'stale_short_program'}


def run(argv, **kwargs):
    result = subprocess.run([str(x) for x in argv], text=True, capture_output=True,
                            timeout=45, **kwargs)
    if result.returncode:
        raise AssertionError({'command': [str(x) for x in argv],
                              'returncode': result.returncode,
                              'stdout': result.stdout, 'stderr': result.stderr})
    return result


def rewrite(root, edit):
    """Mutate a private fixture and establish a new, full host seal."""
    records = [json.loads(line) for line in (root / 'journal.jsonl').read_text().splitlines()]
    edit(records)
    (root / 'journal.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    (root / 'closed.json').unlink()
    seal(root, 0)


def mutations(root):
    results = []
    edits = {
        'wrong_executable_parent': lambda r: next(x for x in r if x.get('op') == 'get_executable').update(parent='fixture-L-missing'),
        'missing_submit_bytes': lambda r: next(x for x in r if x.get('op') == 'compile').update(submitted=None),
        'failed_execute': lambda r: next(x for x in r if x.get('op') == 'execute').update(error_returned=True),
        'wrong_query_link': lambda r: next(x for x in r if x.get('mode') == 'fill').update(query_seq=999),
        'false_query_size': lambda r: next(x for x in r if x.get('mode') == 'fill').update(query_size=8),
        'missing_event': lambda r: r.pop(2),
        'foreign_nonce': lambda r: r[0].update(nonce='foreign'),
    }
    for name, edit in edits.items():
        with tempfile.TemporaryDirectory(prefix='wrong-', dir=root.parent) as td:
            wrong = Path(td) / 'capture'
            shutil.copytree(root, wrong)
            rejected = False
            try:
                rewrite(wrong, edit)
                rejected = verify(wrong)['status'] != 'PASS'
            except (AssertionError, KeyError):
                rejected = True
            assert rejected, name
            results.append({'name': name, 'result': 'PASS', 'expected': 'REJECT_OR_UNKNOWN'})
    for name, action in {
        'payload_tamper': lambda p: next(p.glob('*optimized.bin')).write_bytes(b'tampered'),
        'unlisted_file': lambda p: (p / 'foreign.bin').write_bytes(b'x'),
        'unclosed': lambda p: (p / 'closed.json').unlink(),
    }.items():
        with tempfile.TemporaryDirectory(prefix='wrong-', dir=root.parent) as td:
            wrong = Path(td) / 'capture'
            shutil.copytree(root, wrong)
            action(wrong)
            rejected = False
            try:
                verify(wrong)
            except (AssertionError, FileNotFoundError):
                rejected = True
            assert rejected, name
            results.append({'name': name, 'result': 'PASS', 'expected': 'REJECT'})
    return results


def parent_controls(root):
    """Reseal private ordered lineages with two known compiled parents."""
    import copy
    results = []
    for wrong in (False, True):
        name = 'existing_parent_mismatch' if wrong else 'existing_parent_matches_input'
        with tempfile.TemporaryDirectory(prefix=name + '-', dir=root.parent) as td:
            fixture = Path(td) / 'capture'
            shutil.copytree(root, fixture)
            def edit(records):
                extra = copy.deepcopy(records[1])
                extra.update(pre_seq=3, seq=4, call=2, created='fixture-L-4')
                for c in records[2:]:
                    c['pre_seq'] += 2
                    c['seq'] += 2
                    c['call'] += 1
                    if c.get('op') == 'get_executable':
                        c['created'] = 'fixture-E-6'
                        c['parent'] = 'fixture-L-4'
                        if not wrong:
                            c['handle'] = 'fixture-L-4'
                    if c['handle'].startswith('fixture-E-'):
                        c['handle'] = 'fixture-E-6'
                    if c.get('query_seq'):
                        c['query_seq'] += 2
                    if 'optimized_epoch' in c:
                        c['optimized_epoch'] += 2
                    if c['op'] == 'execute':
                        c['handle'] = 'fixture-L-4'
                records.insert(2, extra)
            records = [json.loads(line) for line in (fixture / 'journal.jsonl').read_text().splitlines()]
            edit(records)
            (fixture / 'journal.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in records))
            footer = json.loads((fixture / 'footer.json').read_text())
            footer['last_seq'] = 12
            (fixture / 'footer.json').write_text(json.dumps(footer))
            (fixture / 'closed.json').unlink()
            seal(fixture, 0)
            if wrong:
                try:
                    verify(fixture)
                except AssertionError as error:
                    assert 'parent differs from input' in str(error)
                    result = 'REJECTED'
                else:
                    raise AssertionError('Existing wrong parent accepted')
            else:
                assert verify(fixture)['status'] == 'PASS'
                result = 'PASS'
            results.append({'name': name, 'result': result, 'test_status': 'PASS'})
    return results


def controls(build, harness, provider, shim, cases):
    results, mutation_results = [], []
    for name in cases:
        with tempfile.TemporaryDirectory(prefix=name + '-', dir=build) as td:
            capture = Path(td) / 'capture'
            if name == 'existing_output':
                capture.mkdir()
                (capture / 'sentinel').write_bytes(b'preserve-existing-output')
            execution = run([harness, name, provider, shim, capture])
            if name == 'self_recursion':
                assert not capture.exists()
                results.append({'name': name, 'result': 'PASS', 'stdout': execution.stdout.strip()})
                continue
            if name == 'existing_output':
                assert list(p.name for p in capture.iterdir()) == ['sentinel']
                assert (capture / 'sentinel').read_bytes() == b'preserve-existing-output'
                assert 'finalize=-1' in execution.stdout
                results.append({'name': name, 'result': 'PASS', 'expected_claim_status': 'UNKNOWN', 'stdout': execution.stdout.strip()})
                continue
            finalizer_return = int(re.search(r'finalize=(-?\d+)', execution.stdout).group(1))
            seal(capture, finalizer_return)
            header = json.loads((capture / "journal.jsonl").read_text().splitlines()[0])
            assert header["provider_sha256"] == hashlib.sha256(provider.read_bytes()).hexdigest()
            assert header["clock"] == "CLOCK_MONOTONIC"
            report = verify(capture)
            expected = 'PASS' if name in PASS_CASES else 'UNKNOWN'
            assert report['status'] == expected, (name, report)
            if name == 'warm_reuse':
                assert len(report['claims']) == 2
            if name == 'multiple_executables':
                assert len(report['claims'][0]['executables']) == 2
            if name == 'destroy_reuse':
                assert report['claims'][0]['loaded'] != 'fixture-L-2'
            if name == 'cold':
                mutation_results = mutations(capture) + parent_controls(capture)
                fields = report['extracted_fields'][0]['fields']
                assert [(f['field'], f['offset'], f['bytes']) for f in fields] == [(1, 2, 3), (2, 7, 2)]
                assert fields[0]['sha256'] == hashlib.sha256(b'abc').hexdigest()
                assert next(capture.glob('*submitted.bin')).read_bytes() == b'module\0whole'
            results.append({'name': name, 'result': 'PASS', 'expected_claim_status': expected,
                            'stdout': execution.stdout.strip(), 'report': report,
                            'closed_inventory': identity(capture / 'closed.json'),
                            'retention': 'Fixture deleted after validation'})
    return results, mutation_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--header', type=Path)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    compiler = shutil.which('clang++') or shutil.which('c++')
    assert compiler
    with tempfile.TemporaryDirectory(prefix='m6-pjrt-controls-') as td:
        build = Path(td)
        body = args.header.read_bytes() if args.header else urllib.request.urlopen(HEADER_URL, timeout=30).read()
        assert len(body) == 102174 and hashlib.sha256(body).hexdigest() == HEADER_SHA
        (build / 'pjrt_c_api.h').write_bytes(body)
        extension = '.dylib' if platform.system() == 'Darwin' else '.so'
        shared = '-dynamiclib' if platform.system() == 'Darwin' else '-shared'
        common = [compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-pthread', '-I', build]
        extra = [] if platform.system() == 'Darwin' else ['-ldl']
        shim, provider, harness = build / ('recorder' + extension), build / ('fake' + extension), build / 'harness'
        run(common + [shared, '-fPIC', ROOT / 'recorder.cpp', '-o', shim] + extra)
        run(common + [shared, '-fPIC', ROOT / 'fake_provider.cpp', '-o', provider])
        run(common + [ROOT / 'harness.cpp', '-o', harness] + extra)
        results, mutation_results = controls(build, harness, provider, shim,
                                             sorted(PASS_CASES | UNKNOWN_CASES) + ['self_recursion'])
        malformed = [b'\x0a\x04abc', b'\x00', b'\x0a\x00\x0a\x00', b'\x0f', b'\x0a\x80']
        for body in malformed:
            try:
                envelope(body)
            except AssertionError:
                continue
            raise AssertionError('Malformed envelope accepted')
        # ASan and UBSan detect memory and undefined-behavior failures here.
        # Their successful results do not establish absence of uninitialized reads.
        sanitizer_common = common + ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
        san_shim, san_provider, san_harness = build / ('san-recorder' + extension), build / ('san-fake' + extension), build / 'san-harness'
        run(sanitizer_common + [shared, '-fPIC', ROOT / 'recorder.cpp', '-o', san_shim] + extra)
        run(sanitizer_common + [shared, '-fPIC', ROOT / 'fake_provider.cpp', '-o', san_provider])
        run(sanitizer_common + [ROOT / 'harness.cpp', '-o', san_harness] + extra)
        sanitizer_results, _ = controls(build, san_harness, san_provider, san_shim,
                                        ['cold', 'short_args', 'short_api', 'destroy_reuse', 'overlap', 'deadline_inflight', 'optimized_query_error_inflight',
                                         'execute_then_fill', 'unrelated_query_inflight'])
        host_build = run([os.sys.executable, ROOT / 'build_recorder.py', '--header', build / 'pjrt_c_api.h', '--output', build / ('host-recorder' + extension)])
        host_code = '''import ctypes,json,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from recorder_host import PassiveRecorder
p=ctypes.CDLL(sys.argv[3]);p.FakeSetup.argtypes=[ctypes.c_int,ctypes.c_size_t];p.FakeSetup(0,0)
r=PassiveRecorder(sys.argv[2],sys.argv[3],sys.argv[4],'host-fixture',time.clock_gettime_ns(time.CLOCK_MONOTONIC)+30_000_000_000,32*1024*1024)
r.library.GetPjrtApi.restype=ctypes.c_void_p
assert r.library.GetPjrtApi()
r.set_phase('host-fixture-phase')
result=r.finalize();assert result['closed'] and result['status']=='UNKNOWN' and result['calls']==0
print(json.dumps(result))
'''
        host_smoke = run([os.sys.executable, '-c', host_code, ROOT, build / ('host-recorder' + extension), provider, build / 'host-capture'])
        shutil.rmtree(build / 'host-capture')
        msan_probe = subprocess.run(common + ['-fsanitize=memory', ROOT / 'harness.cpp', '-o', build / 'msan-harness'] + extra,
                                    text=True, capture_output=True, timeout=30)
        msan = {'status': 'UNAVAILABLE' if msan_probe.returncode else 'AVAILABLE_NOT_RUN',
                'reason': msan_probe.stderr[-1000:] if msan_probe.returncode else 'Instrumented dependencies need separate qualification'}
        result = {'schema': 'M6_DIRECT_PJRT_TRACE_V4_LOCAL_CONTROLS', 'status': 'PASS',
                  'platform': platform.platform(), 'compiler': run([compiler, '--version']).stdout.splitlines()[0],
                  'header_sha256': HEADER_SHA, 'cases': results,
                  'independent_mutations': mutation_results,
                  'malformed_envelopes_rejected': len(malformed),
                  'asan_ubsan': sanitizer_results, 'host_helper': {'build': json.loads(host_build.stdout), 'smoke': json.loads(host_smoke.stdout)}, 'uninitialized_read_sanitizer': msan,
                  'claim': 'LOCAL_FAKE_PROVIDER_PREPARATION_ONLY', 'TPU': 'NOT_RUN',
                  'bulk_cleanup': 'All temporary build files and fixture bodies deleted before result publication'}
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': 'PASS', 'cases': len(results), 'mutations': len(mutation_results),
                      'sanitizer_cases': len(sanitizer_results), 'result': str(args.result)}))


if __name__ == '__main__':
    main()
