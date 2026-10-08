"""Independent synthetic record controls; these fixtures do not show TPU execution."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT/'experiments/2026-10-04-bitsandbytes-tpu/probe_precision.py'
ROUTE = ENTRY.with_name('probe_routes.py')
BACKEND = ENTRY.with_name('probe_backend.py')
spec = importlib.util.spec_from_file_location('precision_tested', ENTRY)
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
R = P.route_probe(ROUTE); B = R.backend(BACKEND)


def fixture_hlo(mode, dtype='f32'):
    precision = '' if mode == 'default' else ', operand_precision={'+mode+','+mode+'}'
    return ('HloModule SYNTHETIC_ONLY\nENTRY main {\n'
            f'  x = {dtype}[3,17] parameter(0)\n'
            f'  wt = {dtype}[17,19] parameter(1)\n'
            f'  dy = {dtype}[3,19] parameter(2)\n'
            f'  w = {dtype}[19,17] parameter(3)\n'
            f'  y = {dtype}[3,19] dot(x, wt), lhs_contracting_dims={{1}}, rhs_contracting_dims={{0}}{precision}\n'
            f'  dx = {dtype}[3,17] dot(dy, w), lhs_contracting_dims={{1}}, rhs_contracting_dims={{0}}{precision}\n'
            f'  ROOT result = ({dtype}[3,19], {dtype}[3,17]) tuple(y, dx)\n}}\n')


def reseal_fixture(fixture):
    for mode in P.MODES:
        root = fixture.actual/'modes'/mode
        rec = B.read(root/'receipt.json'); rec['artifacts'] = B.inventory(root, 'receipt.json'); B.write(root/'receipt.json', rec)
    parent = B.read(fixture.actual/'receipt.json')
    for child in parent['child_processes']:
        child['receipt_sha256'] = B.sha(fixture.actual/child['receipt'])
    parent['artifacts'] = B.inventory(fixture.actual, 'receipt.json'); B.write(fixture.actual/'receipt.json', parent)


def make_fixture(base, admission='a'*64):
    """Return sealed independent records for the real verifier and owner transport tests."""
    base = Path(base); base.mkdir(parents=True, exist_ok=True)
    oracle = base/'oracle'; actual = base/'actual'; oracle.mkdir(); actual.mkdir()
    profile, inputs = B.load_spec(); cases = P.selection(B, R)
    for case in inputs['cases']:
        outputs = {key: {'shape': shape, 'dtype': dtype, 'values': [0 if dtype == 'uint8' else 0.0]*math.prod(shape)}
                   for key, (shape, dtype) in B.expected_outputs(case).items()}
        if case['op'] == 'linear' and case['bias'] is not None:
            outputs['y']['values'] = case['bias']*math.prod(case['x_shape'][:-1])
        raw = {'case_id': case['id'], 'input_sha256': B.case_sha(case), 'outputs': outputs,
               'metadata': {'shape': case['weight_shape'], 'dtype': case['dtype'], 'blocksize': 64,
                            'quant_type': 'nf4', 'nested': False, 'packing_format_for_cpu': False},
               'placements': {key: 'cpu' for key in outputs}, 'counters': {}, 'execution_metrics': {}}
        B.write(oracle/'raw'/(case['id']+'.json'), raw)
    seal = {'kind': 'CPU_ORACLE', 'status': 'COMPLETE', 'oracle_method': B.ORACLE_METHOD,
            'runtime_lock_sha256': profile['runtime_lock_sha256'], 'profile_sha256': B.PROFILE_SHA,
            'inputs_sha256': B.INPUTS_SHA, 'source_admission_sha256': admission,
            'runtime': {'python': '3.12', 'torch': '2.9.0+cpu'}, 'case_ids': profile['case_ids'],
            'artifacts': B.inventory(oracle, 'oracle-seal.json')}
    B.write(oracle/'oracle-seal.json', seal); oracle_sha = B.sha(oracle/'oracle-seal.json')
    args = SimpleNamespace(oracle=oracle, oracle_sha256=oracle_sha, admission_sha256=admission,
                           backend_probe=BACKEND, route_probe=ROUTE, actual=actual)
    parent = P.identities(B, args, profile)
    parent.update(status='COMPLETE', pid=1000, pgid=1000, process_token='0'*32,
                  source_pre=admission, source_post=admission, runtime=profile['runtime'],
                  modes=P.MODES, child_processes=[], precision_environment={key: None for key in P.ENV_KEYS})
    for index, mode in enumerate(P.MODES, 1):
        mode_root = actual/'modes'/mode; mode_root.mkdir(parents=True)
        rec = dict(parent); rec.pop('modes'); rec.pop('child_processes')
        rec.update(mode=mode, pid=1000+index, process_token=str(index)*32,
                   parent_pid=parent['pid'], parent_process_token=parent['process_token'],
                   precision={'requested': mode, 'readback': mode, 'set_calls': 1, 'before_graph': True},
                   device={'type': 'xla', 'hardware': 'TPU', 'pjrt': 'TPU'}, bf16_gradients='NOT_RUN',
                   public_api={'Linear4bit': 'bitsandbytes.nn.modules.Linear4bit', 'Params4bit': 'bitsandbytes.nn.modules.Params4bit', 'original_class_identity': True},
                   dispatch={op: True for op in B.DISPATCH_OPS})
        refs = {case['id']: R.gradient_math(case, B.read(oracle/'raw'/(case['id']+'.json'))) for case in cases}
        B.write(mode_root/'references.json', refs)
        for route, case in P.row_spec(cases):
            original = B.read(oracle/'raw'/(case['id']+'.json'))
            outputs = copy.deepcopy(original['outputs']) if route != 'plain' else {'y': copy.deepcopy(original['outputs']['y'])}
            outputs.update(copy.deepcopy(refs[case['id']]))
            hlo = fixture_hlo(mode if case['dtype'] == 'float32' else 'default', 'f32' if case['dtype'] == 'float32' else 'bf16')
            raw = {'mode': mode, 'route': route, 'case_id': case['id'], 'input_sha256': B.case_sha(case),
                   'pid': rec['pid'], 'process_token': rec['process_token'], 'parent_pid': parent['pid'],
                   'parent_process_token': parent['process_token'], 'status': 'PASS', 'precision': rec['precision'],
                   'outputs': outputs, 'placements': {key: 'xla:0' for key in outputs},
                   'counters': {'xla::dot': 1}, 'execution_metrics': {'ExecuteTime': [1, 0.01, [[0, 0.01]]]},
                   'gradients': 'EXECUTED' if case['dtype'] == 'float32' else 'NOT_RUN',
                   'hlo_scope': P.PROTOCOL['hlo_scope'], 'hlo_precision': P.hlo_precision(hlo)}
            if route == 'plain':
                raw.update(operands=P.plain_operands(B, case, original), operand_placements={key: 'xla:0' for key in ('x', 'weight', 'bias', 'dy')})
            else:
                raw.update(metadata=original['metadata'], original_classes=True, module_training=True,
                           module_state_alias=True, module_state_absent=False, weight_requires_grad=False,
                           weight_grad_is_none=True, bnb_quantized=True, packed_before=outputs['packed'], packed_after=outputs['packed'],
                           quantization_source=R.PROTOCOL['quantization_source'][route])
            prefix = mode_root/'raw'/(route+'-'+case['id']); B.write(prefix.with_suffix('.json'), raw)
            prefix.with_suffix('.hlo.txt').write_text(hlo); prefix.with_suffix('.metrics.txt').write_text('SYNTHETIC METRICS ONLY\n')
        rec['artifacts'] = B.inventory(mode_root, 'receipt.json'); B.write(mode_root/'receipt.json', rec)
        stdout, stderr = f'processes/{mode}.stdout.log', f'processes/{mode}.stderr.log'
        (actual/stdout).parent.mkdir(exist_ok=True); (actual/stdout).write_text('SYNTHETIC ONLY\n'); (actual/stderr).write_text('')
        parent['child_processes'].append({'mode': mode, 'pid': rec['pid'], 'launched_pid': rec['pid'],
            'process_token': rec['process_token'], 'parent_pid': parent['pid'], 'parent_process_token': parent['process_token'],
            'pgid': rec['pgid'], 'exit_code': 0, 'reaped': True, 'receipt': f'modes/{mode}/receipt.json',
            'receipt_sha256': B.sha(mode_root/'receipt.json'), 'stdout': stdout, 'stderr': stderr})
    parent['artifacts'] = B.inventory(actual, 'receipt.json'); B.write(actual/'receipt.json', parent)
    return SimpleNamespace(oracle=oracle, actual=actual, oracle_sha256=oracle_sha, admission_sha256=admission, args=args, B=B, R=R, P=P)


class PrecisionControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.f = make_fixture(Path(self.tmp.name))

    def verify(self): return P.verify(B, R, self.f.args)

    def mutate(self, path, function):
        file = self.f.actual/path; value = B.read(file); function(value); B.write(file, value); reseal_fixture(self.f)

    def raw_path(self, route='plain', mode='high', dtype='float32'):
        return f'modes/{mode}/raw/{route}-linear-{dtype}-rank2-bias1.json'

    def test_valid_fixture(self):
        result = self.verify(); self.assertEqual(result['record_validation'], 'PASS'); self.assertEqual(result['numerical_status'], 'PASS')
        self.assertEqual(len(result['modes']), 3); self.assertEqual(sum(len(mode['rows']) for mode in result['modes']), 15)
        self.assertEqual(result['api42_status'], 'NOT_QUALIFIED'); self.assertEqual(result['m3_status'], 'NOT_QUALIFIED')

    def test_numerical_failure_is_valid_diagnostic(self):
        self.mutate(self.raw_path(), lambda raw: raw['outputs']['y']['values'].__setitem__(0, 10.0))
        result = self.verify(); self.assertEqual(result['record_validation'], 'PASS'); self.assertEqual(result['numerical_status'], 'FAIL')

    def test_missing_mode(self):
        self.mutate('receipt.json', lambda rec: rec['child_processes'].pop())
        with self.assertRaisesRegex(ValueError, 'MODE_MATRIX'): self.verify()

    def test_reused_pid(self):
        self.mutate('receipt.json', lambda rec: rec['child_processes'][1].update(pid=rec['child_processes'][0]['pid']))
        with self.assertRaisesRegex(ValueError, 'FRESH_PID'): self.verify()

    def test_reused_token(self):
        self.mutate('receipt.json', lambda rec: rec['child_processes'][1].update(process_token=rec['process_token']))
        with self.assertRaisesRegex(ValueError, 'FRESH_PROCESS_TOKEN'): self.verify()

    def test_wrong_precision_readback(self):
        self.mutate('modes/high/receipt.json', lambda rec: rec['precision'].update(readback='default'))
        with self.assertRaisesRegex(ValueError, 'PRECISION_READBACK'): self.verify()

    def test_extra_setter_call(self):
        self.mutate('modes/high/receipt.json', lambda rec: rec['precision'].update(set_calls=2))
        with self.assertRaisesRegex(ValueError, 'PRECISION_READBACK'): self.verify()

    def test_wrong_parent_identity(self):
        self.mutate('modes/high/receipt.json', lambda rec: rec.update(parent_pid=9999))
        with self.assertRaisesRegex(ValueError, 'CHILD_PARENT'): self.verify()

    def test_wrong_launched_pid(self):
        self.mutate('receipt.json', lambda rec: rec['child_processes'][1].update(launched_pid=9999))
        with self.assertRaisesRegex(ValueError, 'CHILD_LAUNCH_IDENTITY'): self.verify()

    def test_nonzero_exit(self):
        self.mutate('receipt.json', lambda rec: rec['child_processes'][1].update(exit_code=1))
        with self.assertRaisesRegex(ValueError, 'CHILD_EXIT'): self.verify()

    def test_missing_hlo(self):
        (self.f.actual/self.raw_path()).with_suffix('.hlo.txt').unlink(); reseal_fixture(self.f)
        with self.assertRaises(FileNotFoundError): self.verify()

    def test_wrong_hlo_precision(self):
        path = (self.f.actual/self.raw_path()).with_suffix('.hlo.txt'); hlo = fixture_hlo('default'); path.write_text(hlo)
        self.mutate(self.raw_path(), lambda raw: raw.update(hlo_precision=P.hlo_precision(hlo)))
        with self.assertRaisesRegex(ValueError, 'HLO_PRECISION'): self.verify()

    def test_missing_gradient_dot(self):
        path = (self.f.actual/self.raw_path()).with_suffix('.hlo.txt'); hlo = '\n'.join(line for line in fixture_hlo('high').replace('tuple(y, dx)', 'tuple(y)').splitlines() if 'dx =' not in line); path.write_text(hlo)
        self.mutate(self.raw_path(), lambda raw: raw.update(hlo_precision=P.hlo_precision(hlo)))
        with self.assertRaisesRegex(ValueError, 'HLO_FP32_FORWARD_GRADIENT'): self.verify()

    def test_positive_fallback(self):
        self.mutate(self.raw_path(), lambda raw: raw['counters'].update({'aten::mm': 1}))
        with self.assertRaisesRegex(ValueError, 'CPU_FALLBACK'): self.verify()

    def test_zero_fallback_counter_is_allowed(self):
        self.mutate(self.raw_path(), lambda raw: raw['counters'].update({'aten::mm': 0}))
        self.assertEqual(self.verify()['record_validation'], 'PASS')

    def test_missing_fallback_record(self):
        self.mutate(self.raw_path(), lambda raw: raw.pop('counters'))
        with self.assertRaisesRegex(ValueError, 'COUNTERS_UNKNOWN'): self.verify()

    def test_missing_execution_metric(self):
        self.mutate(self.raw_path(), lambda raw: raw.update(execution_metrics={}))
        with self.assertRaisesRegex(ValueError, 'EXECUTION_NOT_OBSERVED'): self.verify()

    def test_changed_child_source(self):
        self.mutate('modes/high/receipt.json', lambda rec: rec.update(source_post='b'*64))
        with self.assertRaisesRegex(ValueError, 'SOURCE_BINDING'): self.verify()

    def test_changed_data_hash(self):
        self.mutate(self.raw_path(), lambda raw: raw.update(input_sha256='b'*64))
        with self.assertRaisesRegex(ValueError, 'ROW_BINDING'): self.verify()

    def test_changed_plain_operands(self):
        self.mutate(self.raw_path(), lambda raw: raw['operands']['x']['values'].__setitem__(0, 99.0))
        with self.assertRaisesRegex(ValueError, 'PLAIN_OPERAND_BINDING'): self.verify()

    def test_bad_scalar_reference(self):
        self.mutate('modes/high/references.json', lambda refs: refs[P.CASE_IDS[0]]['dx']['values'].__setitem__(0, 99.0))
        with self.assertRaisesRegex(ValueError, 'SCALAR_GRADIENT_REFERENCE'): self.verify()

    def test_changed_frozen_weight(self):
        self.mutate(self.raw_path(route='target_constructor'), lambda raw: raw.update(packed_after=dict(raw['packed_after'], values=[1]*len(raw['packed_after']['values']))))
        with self.assertRaisesRegex(ValueError, 'PACKED_CHANGED'): self.verify()

    def test_bf16_gradient_claim(self):
        self.mutate(self.raw_path(route='target_constructor', dtype='bfloat16'), lambda raw: raw.update(gradients='EXECUTED'))
        with self.assertRaisesRegex(ValueError, 'GRADIENT_SCOPE'): self.verify()

    def test_precision_environment_override(self):
        self.mutate('receipt.json', lambda rec: rec['precision_environment'].update(XLA_USE_BF16='1'))
        with self.assertRaisesRegex(ValueError, 'PRECISION_ENVIRONMENT_OVERRIDE'): self.verify()

    def test_hlo_default_and_explicit_shapes(self):
        self.assertTrue(all(dot['implicit_default'] for dot in P.hlo_precision(fixture_hlo('default'))))
        self.assertEqual(P.hlo_precision(fixture_hlo('highest'))[1]['result_shape'], [3, 17])

    def test_changed_route_helper_rejected(self):
        path = Path(self.tmp.name)/'changed.py'; path.write_text(ROUTE.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'ROUTE_SOURCE_IDENTITY'): P.route_probe(path)

    def test_cli_verifier_uses_real_fixture(self):
        result = subprocess.run([sys.executable, '-B', str(ENTRY), 'verify', '--backend-probe', str(BACKEND),
            '--route-probe', str(ROUTE), '--oracle', str(self.f.oracle), '--oracle-sha256', self.f.oracle_sha256,
            '--admission-sha256', self.f.admission_sha256, '--actual', str(self.f.actual)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr+result.stdout)
        self.assertEqual(json.loads(result.stdout)['record_validation'], 'PASS')

    def test_typed_hlo_operands_and_layouts(self):
        hlo = ('HloModule SYNTHETIC_ONLY\nENTRY %main {\n'
               '%reshape.102 = f32[3,17]{1,0} parameter(0)\n'
               '%transpose.100 = f32[17,19]{0,1} parameter(1)\n'
               'ROOT %dot.103 = f32[3,19]{1,0} dot(f32[3,17]{1,0} %reshape.102, '
               'f32[17,19]{0,1} %transpose.100), lhs_contracting_dims={1}, rhs_contracting_dims={0}\n}\n')
        parsed = P.hlo_precision(hlo)
        self.assertEqual(parsed[0]['operand_dtypes'], ['f32', 'f32'])
        self.assertEqual(parsed[0]['result_shape'], [3, 19])

    def test_tuple_operand_and_root_alias(self):
        hlo = fixture_hlo('default').replace('ROOT result = (f32[3,19], f32[3,17]) tuple(y, dx)',
            'pair = (f32[3,19], f32[3,17]) tuple(y, dx)\n'
            'alias = f32[3,19] get-tuple-element((f32[3,19], f32[3,17]) pair), index=0\n'
            'ROOT result = f32[3,19] copy(f32[3,19] alias)')
        parsed = P.hlo_precision(hlo)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]['result_shape'], [3, 19])
        self.assertTrue(all(dot['entry_root'] == 'result' for dot in parsed))

    def test_wrong_contracting_dimensions_rejected(self):
        hlo = fixture_hlo('high').replace('lhs_contracting_dims={1}', 'lhs_contracting_dims={0}')
        (self.f.actual/self.raw_path()).with_suffix('.hlo.txt').write_text(hlo)
        self.mutate(self.raw_path(), lambda raw: raw.update(hlo_precision=P.hlo_precision(hlo)))
        with self.assertRaisesRegex(ValueError, 'HLO_DOT_CONTRACTION'): self.verify()

    def test_unused_computation_dots_rejected(self):
        unused = fixture_hlo('high').replace('ENTRY main', 'unused')
        entry = ('ENTRY actual {\n z = f32[] constant(0)\n'
                 ' y = f32[3,19] broadcast(z), dimensions={}\n'
                 ' dx = f32[3,17] broadcast(z), dimensions={}\n'
                 ' ROOT out = (f32[3,19], f32[3,17]) tuple(y, dx)\n}\n')
        hlo = unused+entry
        (self.f.actual/self.raw_path()).with_suffix('.hlo.txt').write_text(hlo)
        self.mutate(self.raw_path(), lambda raw: raw.update(hlo_precision=P.hlo_precision(hlo)))
        with self.assertRaisesRegex(ValueError, 'HLO_RECORD'): self.verify()

    def test_disconnected_entry_dots_rejected(self):
        hlo = fixture_hlo('high').replace('ROOT result = (f32[3,19], f32[3,17]) tuple(y, dx)', 'zero = f32[] constant(0)\n ROOT result = f32[] copy(zero)')
        (self.f.actual/self.raw_path()).with_suffix('.hlo.txt').write_text(hlo)
        self.mutate(self.raw_path(), lambda raw: raw.update(hlo_precision=P.hlo_precision(hlo)))
        with self.assertRaisesRegex(ValueError, 'HLO_RECORD'): self.verify()

    def test_invalid_metric_samples_rejected(self):
        self.mutate(self.raw_path(), lambda raw: raw['execution_metrics']['ExecuteTime'].__setitem__(2, 'not metric samples'))
        with self.assertRaisesRegex(ValueError, 'EXECUTION_METRIC_SAMPLES'): self.verify()

    def test_nonfinite_metric_samples_rejected(self):
        raw = B.read(self.f.actual/self.raw_path())
        raw['execution_metrics']['ExecuteTime'][2] = [[0.0, float('inf')]]
        with self.assertRaisesRegex(ValueError, 'EXECUTION_METRIC_SAMPLES'): P.validate_execution(B, raw)

    def test_actual_retained_hlo_root_dependencies(self):
        # Copy the retained text shape into this positive fixture; no device computation executes.
        hlo = ('HloModule RETAINED_FORMAT_FIXTURE\n'
               '%Add (x: f32[], y: f32[]) -> f32[] {\n'
               ' %x = f32[] parameter(0)\n %y = f32[] parameter(1)\n ROOT %add = f32[] add(f32[] %x, f32[] %y)\n}\n'
               'ENTRY %main (p0: f32[3,17], p1: f32[17,19], p2: f32[3,19], p3: f32[19,17]) -> (f32[3,19], f32[3,17]) {\n'
               ' %p0 = f32[3,17]{1,0} parameter(0)\n %p1 = f32[17,19]{0,1} parameter(1)\n'
               ' %p2 = f32[3,19]{1,0} parameter(2)\n %p3 = f32[19,17]{1,0} parameter(3)\n'
               ' %dot.y = f32[3,19]{1,0} dot(f32[3,17]{1,0} %p0, f32[17,19]{0,1} %p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}\n'
               ' %dot.dx = f32[3,17]{1,0} dot(f32[3,19]{1,0} %p2, f32[19,17]{1,0} %p3), lhs_contracting_dims={1}, rhs_contracting_dims={0}\n'
               ' %alias = f32[3,19]{1,0} copy(f32[3,19]{1,0} %dot.y)\n'
               ' ROOT %tuple = (f32[3,19]{1,0}, f32[3,17]{1,0}) tuple(f32[3,19]{1,0} %alias, f32[3,17]{1,0} %dot.dx)\n}\n')
        parsed = P.hlo_precision(hlo)
        self.assertEqual(len(parsed), 2)
        self.assertTrue(all(dot['root_reachable'] and dot['entry_root'] == 'tuple' for dot in parsed))
        for dot in parsed: P.validate_dot_matrix(B, dot)
        raw = B.read(self.f.actual/self.raw_path())
        raw['execution_metrics']['ExecuteTime'] = [4, 6140487.0, [[1791098289.9015226, 1315859.0]]]
        P.validate_execution(B, raw)

    def test_clean_foreground_child_exit(self):
        process = subprocess.Popen([sys.executable, '-c', 'pass'])
        self.assertEqual(P.reap(process, 5), 0)
        self.assertEqual(process.poll(), 0)

    def test_bad_oracle_and_agreeing_outputs_rejected(self):
        path = self.f.oracle/'raw'/(P.CASE_IDS[0]+'.json')
        original = B.read(path); original['outputs']['y']['values'][0] += 10.0; B.write(path, original)
        seal = B.read(self.f.oracle/'oracle-seal.json'); seal['artifacts'] = B.inventory(self.f.oracle, 'oracle-seal.json')
        B.write(self.f.oracle/'oracle-seal.json', seal); self.f.oracle_sha256 = B.sha(self.f.oracle/'oracle-seal.json')
        self.f.args.oracle_sha256 = self.f.oracle_sha256
        parent = B.read(self.f.actual/'receipt.json'); parent['oracle_sha256'] = self.f.oracle_sha256; B.write(self.f.actual/'receipt.json', parent)
        for mode in P.MODES:
            child_path = self.f.actual/'modes'/mode/'receipt.json'
            child = B.read(child_path); child['oracle_sha256'] = self.f.oracle_sha256; B.write(child_path, child)
            for route in ['plain']+P.ROUTES:
                row_path = self.f.actual/self.raw_path(route=route, mode=mode)
                row = B.read(row_path); row['outputs']['y']['values'][0] = original['outputs']['y']['values'][0]; B.write(row_path, row)
        reseal_fixture(self.f)
        with self.assertRaisesRegex(ValueError, 'SCALAR_FORWARD_REFERENCE'): self.verify()

    def test_deadline_reaps_foreground_child(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
        with self.assertRaisesRegex(ValueError, 'CHILD_DEADLINE'): P.reap(process, 0.01)
        self.assertIsNotNone(process.poll())


if __name__ == '__main__': unittest.main()
