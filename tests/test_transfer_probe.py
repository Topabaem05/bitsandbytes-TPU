"""Independent synthetic controls. These tests do not show device execution."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT/'experiments/2026-10-04-bitsandbytes-tpu/probe_transfer.py'
BACKEND = ENTRY.with_name('probe_backend.py'); ROUTE = ENTRY.with_name('probe_routes.py')
PRECISION = ENTRY.with_name('probe_precision.py'); MANIFEST = ROOT/'patches/params4bit-xla-v1.json'
spec = importlib.util.spec_from_file_location('transfer_tested', ENTRY)
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
BASE_ARGS = SimpleNamespace(backend_probe=BACKEND, route_probe=ROUTE, precision_probe=PRECISION, patch_manifest=MANIFEST)
B, R, P, A = T.helpers(BASE_ARGS)


def fixture_hlo(dtype='f32', gradient=False):
    text = ('HloModule SYNTHETIC_ONLY\nENTRY main {\n'
            f'  x = {dtype}[3,17] parameter(0)\n'
            f'  wt = {dtype}[17,19] parameter(1)\n'
            f'  y = {dtype}[3,19] dot(x, wt), lhs_contracting_dims={{1}}, rhs_contracting_dims={{0}}, operand_precision={{highest,highest}}\n')
    if gradient:
        text += (f'  dy = {dtype}[3,19] parameter(2)\n  w = {dtype}[19,17] parameter(3)\n'
                 f'  dx = {dtype}[3,17] dot(dy, w), lhs_contracting_dims={{1}}, rhs_contracting_dims={{0}}, operand_precision={{highest,highest}}\n'
                 f'  ROOT result = ({dtype}[3,19], {dtype}[3,17]) tuple(y, dx)\n')
    else: text += f'  ROOT result = {dtype}[3,19] copy(y)\n'
    return text + '}\n'


def reseal_fixture(fixture):
    rec = B.read(fixture.actual/'receipt.json')
    rec['artifacts'] = B.inventory(fixture.actual, 'receipt.json'); B.write(fixture.actual/'receipt.json', rec)


def make_fixture(base, admission='a'*64):
    """Return a complete sealed record for real verifier/owner-chain controls."""
    base = Path(base); base.mkdir(parents=True, exist_ok=True)
    oracle = base/'oracle'; actual = base/'actual'; oracle.mkdir(); actual.mkdir()
    profile, inputs = B.load_spec(); cases = P.selection(B, R)
    for case in inputs['cases']:
        outputs = {name: {'shape': shape, 'dtype': dtype, 'values': [0 if dtype == 'uint8' else 0.0]*math.prod(shape)}
                   for name, (shape, dtype) in B.expected_outputs(case).items()}
        if case['op'] == 'linear' and case['bias'] is not None: outputs['y']['values'] = case['bias']*math.prod(case['x_shape'][:-1])
        raw = {'case_id': case['id'], 'input_sha256': B.case_sha(case), 'outputs': outputs,
               'metadata': {'shape': case['weight_shape'], 'dtype': case['dtype'], 'blocksize': 64,
                            'quant_type': 'nf4', 'nested': False, 'packing_format_for_cpu': False},
               'placements': {name: 'cpu' for name in outputs}, 'counters': {}}
        B.write(oracle/'raw'/(case['id']+'.json'), raw)
    seal = {'kind': 'CPU_ORACLE', 'status': 'COMPLETE', 'oracle_method': B.ORACLE_METHOD,
            'profile_sha256': B.PROFILE_SHA, 'inputs_sha256': B.INPUTS_SHA,
            'runtime_lock_sha256': profile['runtime_lock_sha256'], 'source_admission_sha256': admission,
            'runtime': {'python': '3.12', 'torch': '2.9.0+cpu'}, 'case_ids': profile['case_ids'],
            'patch_manifest_sha256': A.PATCH_MANIFEST_SHA, 'transfer_admission_sha256': B.sha(ENTRY.with_name('transfer_admission.py')),
            'transfer_probe_sha256': B.sha(ENTRY), 'precision_probe_sha256': T.PRECISION_SHA, 'route_probe_sha256': P.ROUTE_SHA,
            'backend_probe_sha256': R.BACKEND_SHA, 'source_pre': admission, 'source_post': admission,
            'public_api': {'Linear4bit': 'bitsandbytes.nn.modules.Linear4bit', 'Params4bit': 'bitsandbytes.nn.modules.Params4bit', 'original_class_identity': True, 'original_method_identity': True},
            'artifacts': B.inventory(oracle, 'oracle-seal.json')}
    B.write(oracle/'oracle-seal.json', seal); oracle_sha = B.sha(oracle/'oracle-seal.json')
    args = SimpleNamespace(**vars(BASE_ARGS), oracle=oracle, oracle_sha256=oracle_sha,
                           admission_sha256=admission, actual=actual)
    rec = T.identity(B, R, P, A, args)
    rec.update(status='COMPLETE', pid=1000, pgid=1000, process_token='0'*32,
               source_pre=admission, source_post=admission, oracle_sha256=oracle_sha,
               runtime=profile['runtime'], device={'type': 'xla', 'hardware': 'TPU', 'pjrt': 'TPU'},
               dispatch={op: True for op in B.DISPATCH_OPS},
               precision={'requested': 'highest', 'readback': 'highest', 'set_calls': 1, 'before_graph': True},
               precision_environment={key: None for key in P.ENV_KEYS},
               public_api={'Linear4bit': 'bitsandbytes.nn.modules.Linear4bit', 'Params4bit': 'bitsandbytes.nn.modules.Params4bit', 'original_class_identity': True, 'original_method_identity': True})
    refs = {case['id']: R.gradient_math(case, B.read(oracle/'raw'/(case['id']+'.json'))) for case in cases}
    B.write(actual/'references.json', refs)
    rows = [('matrix', 'api42', case) for case in inputs['cases']] + [('transfer', route, case) for case in cases for route in T.ROUTES]
    for group, route, case in rows:
        original = B.read(oracle/'raw'/(case['id']+'.json')); outputs = copy.deepcopy(original['outputs'])
        if route == 'params_to_xla': outputs.pop('y')
        if route == 'module_to_xla': outputs.update(copy.deepcopy(refs[case['id']]))
        observed_hlo = route != 'params_to_xla' and case['op'] == 'linear' and case['dtype'] == 'float32' and len(case['x_shape']) == 2
        hlo = fixture_hlo(gradient=route == 'module_to_xla') if observed_hlo else 'HloModule SYNTHETIC_ONLY\nENTRY main {\n ROOT out = f32[1] constant({0})\n}\n'
        raw = T.row_identity(B, rec, case, route)
        raw.update(metadata=original['metadata'], outputs=outputs, placements={name: 'xla:0' for name in outputs},
                   counters={'xla::device': 1}, execution_metrics={'ExecuteTime': [1, 0.01, [[0, 0.01]]]},
                   hlo_precision=P.hlo_precision(hlo) if observed_hlo else None)
        if group == 'transfer':
            raw.update(original_classes=True, parameter_identity=True, custom_attribute_retained=True,
                       quant_state_class='bitsandbytes.functional.QuantState', parameter_module_alias=True,
                       parameter_attrs={'blocksize': 64, 'compress_statistics': False, 'quant_type': 'nf4', 'quant_storage': 'uint8', 'bnb_quantized': True},
                       weight_requires_grad=False, weight_grad_is_none=True,
                       quant_state_placements={'absmax': 'xla:0', 'code': 'xla:0'}, packed_before=outputs['packed'], packed_after=outputs['packed'],
                       gradients='EXECUTED' if route == 'module_to_xla' and case['dtype'] == 'float32' else 'NOT_RUN')
            if route == 'module_to_xla':
                raw.update(module_identity=True, module_state_alias=True, module_training=True,
                           bias={'shape': [case['weight_shape'][0]], 'dtype': case['dtype'], 'values': case['bias']},
                           bias_placement='xla:0', bias_class='torch.nn.parameter.Parameter')
        prefix = actual/group/(route+'-'+case['id']); B.write(prefix.with_suffix('.json'), raw)
        prefix.with_suffix('.hlo.txt').write_text(hlo); prefix.with_suffix('.metrics.txt').write_text('SYNTHETIC METRICS ONLY\n')
    rec['artifacts'] = B.inventory(actual, 'receipt.json'); B.write(actual/'receipt.json', rec)
    return SimpleNamespace(oracle=oracle, actual=actual, oracle_sha256=oracle_sha,
                           admission_sha256=admission, args=args, B=B, R=R, P=P, A=A, T=T)


class TransferControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.f = make_fixture(Path(self.tmp.name))

    def verify(self): return T.verify(B, R, P, A, self.f.args)

    def mutate(self, path, change):
        file = self.f.actual/path; raw = B.read(file); change(raw); B.write(file, raw); reseal_fixture(self.f)

    def transfer(self, route='module_to_xla', dtype='float32'):
        return f'transfer/{route}-linear-{dtype}-rank2-bias1.json'

    def test_full_correct_record(self):
        result = self.verify(); self.assertEqual(result['record_validation'], 'PASS'); self.assertEqual(result['api42_status'], 'PASS')
        self.assertEqual(len(result['matrix']), 42); self.assertEqual(len(result['transfers']), 4)
        self.assertEqual(result['m3_status'], 'NOT_QUALIFIED')

    def test_numerical_failure_stays_failure(self):
        self.mutate(self.transfer(), lambda raw: raw['outputs']['y']['values'].__setitem__(0, 10.0))
        result = self.verify(); self.assertEqual(result['record_validation'], 'PASS'); self.assertEqual(result['api42_status'], 'FAIL')
        self.assertEqual(result['numerical_status'], 'FAIL')

    def test_matrix_numerical_failure_stays_failure(self):
        self.mutate('matrix/api42-linear-float32-rank2-bias1.json', lambda raw: raw['outputs']['packed']['values'].__setitem__(0, 1))
        result = self.verify(); self.assertEqual(result['api42_status'], 'FAIL'); self.assertEqual(result['transfer_status'], 'PASS')

    def test_transfer_error_retained(self):
        self.mutate(self.transfer(), lambda raw: raw.update(status='ERROR', error_type='RuntimeError', error_message='fixture error', traceback='SYNTHETIC TRACEBACK'))
        result = self.verify(); self.assertEqual(result['api42_status'], 'FAIL'); self.assertEqual(result['transfers'][1]['status'], 'ERROR')

    def test_missing_matrix_case(self):
        (self.f.actual/'matrix/api42-linear-float32-rank2-bias1.json').unlink(); reseal_fixture(self.f)
        with self.assertRaises(FileNotFoundError): self.verify()

    def test_missing_transfer(self):
        (self.f.actual/self.transfer()).unlink(); reseal_fixture(self.f)
        with self.assertRaises(FileNotFoundError): self.verify()

    def test_missing_hlo(self):
        (self.f.actual/self.transfer()).with_suffix('.hlo.txt').unlink(); reseal_fixture(self.f)
        with self.assertRaises(FileNotFoundError): self.verify()

    def test_data_change(self):
        self.mutate(self.transfer(), lambda raw: raw.update(input_sha256='b'*64))
        with self.assertRaisesRegex(ValueError, 'ROW_BINDING_input_sha256'): self.verify()

    def test_wrong_row_process(self):
        self.mutate(self.transfer(), lambda raw: raw.update(pid=1001))
        with self.assertRaisesRegex(ValueError, 'ROW_BINDING_pid'): self.verify()

    def test_wrong_state_attributes(self):
        self.mutate(self.transfer('params_to_xla'), lambda raw: raw['parameter_attrs'].update(compress_statistics=True))
        with self.assertRaisesRegex(ValueError, 'PARAMETER_ATTRS'): self.verify()

    def test_wrong_oracle_source(self):
        path = self.f.oracle/'oracle-seal.json'; seal = B.read(path); seal['source_post'] = 'b'*64; B.write(path, seal)
        self.f.args.oracle_sha256 = B.sha(path)
        self.mutate('receipt.json', lambda rec: rec.update(oracle_sha256=self.f.args.oracle_sha256))
        with self.assertRaisesRegex(ValueError, 'ORACLE_SOURCE_METHODS'): self.verify()

    def test_wrong_oracle_and_agreeing_forward(self):
        case = P.CASE_IDS[0]; path = self.f.oracle/'raw'/(case+'.json'); raw = B.read(path)
        raw['outputs']['y']['values'][0] = 10.0; B.write(path, raw)
        seal_path = self.f.oracle/'oracle-seal.json'; seal = B.read(seal_path)
        seal['artifacts'] = B.inventory(self.f.oracle, 'oracle-seal.json'); B.write(seal_path, seal)
        self.f.args.oracle_sha256 = B.sha(seal_path)
        self.mutate('receipt.json', lambda rec: rec.update(oracle_sha256=self.f.args.oracle_sha256))
        self.mutate(self.transfer(), lambda actual: actual['outputs']['y']['values'].__setitem__(0, 10.0))
        self.mutate('matrix/api42-'+case+'.json', lambda actual: actual['outputs']['y']['values'].__setitem__(0, 10.0))
        with self.assertRaisesRegex(ValueError, 'SCALAR_FORWARD_REFERENCE'): self.verify()

    def test_wrong_runtime(self):
        self.mutate('receipt.json', lambda rec: rec['runtime'].update(torch='2.14.1'))
        with self.assertRaisesRegex(ValueError, 'RUNTIME_PAIR'): self.verify()

    def test_class_replacement(self):
        self.mutate('receipt.json', lambda rec: rec['public_api'].update(original_class_identity=False))
        with self.assertRaisesRegex(ValueError, 'PUBLIC_API_REPLACEMENT'): self.verify()

    def test_method_replacement(self):
        self.mutate('receipt.json', lambda rec: rec['public_api'].update(original_method_identity=False))
        with self.assertRaisesRegex(ValueError, 'PUBLIC_API_REPLACEMENT'): self.verify()

    def test_wrong_patch_link(self):
        self.mutate('receipt.json', lambda rec: rec.update(patch_manifest_sha256='b'*64))
        with self.assertRaisesRegex(ValueError, 'TRANSFER_BINDING_patch_manifest'): self.verify()

    def test_wrong_source_post(self):
        self.mutate('receipt.json', lambda rec: rec.update(source_post='b'*64))
        with self.assertRaisesRegex(ValueError, 'SOURCE_BINDING'): self.verify()

    def test_wrong_precision(self):
        self.mutate('receipt.json', lambda rec: rec['precision'].update(readback='default'))
        with self.assertRaisesRegex(ValueError, 'PRECISION_READBACK'): self.verify()

    def test_fallback_counter(self):
        self.mutate(self.transfer(), lambda raw: raw['counters'].update({'aten::mm': 1}))
        with self.assertRaisesRegex(ValueError, 'CPU_FALLBACK'): self.verify()

    def test_zero_execution(self):
        self.mutate(self.transfer(), lambda raw: raw.update(execution_metrics={'ExecuteTime': [0, 0.0, []]}))
        with self.assertRaisesRegex(ValueError, 'EXECUTION_NOT_OBSERVED'): self.verify()

    def test_invalid_execution_samples(self):
        self.mutate(self.transfer(), lambda raw: raw['execution_metrics']['ExecuteTime'].__setitem__(2, 'bad samples'))
        with self.assertRaisesRegex(ValueError, 'EXECUTION_METRIC_SAMPLES'): self.verify()

    def test_wrong_parameter_identity(self):
        self.mutate(self.transfer('params_to_xla'), lambda raw: raw.update(parameter_identity=False))
        with self.assertRaisesRegex(ValueError, 'TRANSFER_IDENTITY'): self.verify()

    def test_lost_custom_attribute(self):
        self.mutate(self.transfer('params_to_xla'), lambda raw: raw.update(custom_attribute_retained=False))
        with self.assertRaisesRegex(ValueError, 'TRANSFER_IDENTITY'): self.verify()

    def test_wrong_quantstate_alias(self):
        self.mutate(self.transfer(), lambda raw: raw.update(module_state_alias=False))
        with self.assertRaisesRegex(ValueError, 'MODULE_ALIAS_IDENTITY'): self.verify()

    def test_wrong_bias(self):
        self.mutate(self.transfer(), lambda raw: raw['bias']['values'].__setitem__(0, 10.0))
        with self.assertRaisesRegex(ValueError, 'BIAS'): self.verify()

    def test_trainable_base(self):
        self.mutate(self.transfer(), lambda raw: raw.update(weight_requires_grad=True))
        with self.assertRaisesRegex(ValueError, 'FROZEN_WEIGHT'): self.verify()

    def test_missing_gradient(self):
        self.mutate(self.transfer(), lambda raw: (raw['outputs'].pop('dx'), raw['placements'].pop('dx')))
        with self.assertRaisesRegex(ValueError, 'OUTPUT_SET'): self.verify()

    def test_wrong_scalar_reference(self):
        self.mutate('references.json', lambda refs: refs[P.CASE_IDS[0]]['dx']['values'].__setitem__(0, 10.0))
        with self.assertRaisesRegex(ValueError, 'SCALAR_GRADIENT_REFERENCE'): self.verify()

    def test_hlo_precision_rejected(self):
        path = self.f.actual/self.transfer(); prefix = path.with_suffix('')
        hlo = prefix.with_suffix('.hlo.txt').read_text().replace('highest,highest', 'default,default')
        prefix.with_suffix('.hlo.txt').write_text(hlo)
        self.mutate(self.transfer(), lambda raw: raw.update(hlo_precision=P.hlo_precision(hlo)))
        with self.assertRaisesRegex(ValueError, 'HLO_PRECISION'): self.verify()

    def test_rank3_retained_without_rank2_shape_gate(self):
        result = self.verify(); self.assertEqual(result['api42_status'], 'PASS')
        profile, inputs = B.load_spec(); self.assertTrue(any(c['op'] == 'linear' and len(c['x_shape']) == 3 for c in inputs['cases']))

    def test_real_cli(self):
        args = [sys.executable, '-B', str(ENTRY), 'verify', '--backend-probe', str(BACKEND), '--route-probe', str(ROUTE), '--precision-probe', str(PRECISION), '--patch-manifest', str(MANIFEST), '--admission-sha256', self.f.admission_sha256, '--oracle', str(self.f.oracle), '--oracle-sha256', self.f.oracle_sha256, '--actual', str(self.f.actual)]
        result = subprocess.run(args, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr); self.assertEqual(json.loads(result.stdout)['api42_status'], 'PASS')
        self.mutate(self.transfer(), lambda raw: raw['outputs']['y']['values'].__setitem__(0, 10.0))
        result = subprocess.run(args, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 2); self.assertEqual(json.loads(result.stdout)['api42_status'], 'FAIL')


class OverlayAdmissionControls(unittest.TestCase):
    def setUp(self):
        self.manifest = A.load_manifest(B, MANIFEST); profile, _ = B.load_spec()
        self.admission = {'format': 'bnb-tpu.probe-source-admission.v1', 'runtime_lock_sha256': profile['runtime_lock_sha256'],
                          'bitsandbytes': {'commit': profile['source_commit'], 'patch_manifest_sha256': A.PATCH_MANIFEST_SHA, 'files': A.python_files(self.manifest['post_patch_package_files'])},
                          'bitsandbytes_tpu': {'files': {'__init__.py': 'a'*64}}}

    def test_exact_overlay(self):
        self.assertIs(A.validate_admission(B, self.admission, self.manifest), self.admission)
        self.assertEqual(len(self.admission['bitsandbytes']['files']), 47)
        self.assertNotIn('py.typed', self.admission['bitsandbytes']['files'])

    def test_base_source_rejected(self):
        self.admission['bitsandbytes']['files'] = A.python_files(self.manifest['base_package_files'])
        with self.assertRaisesRegex(ValueError, 'SOURCE_EXACT_OVERLAY'): A.validate_admission(B, self.admission, self.manifest)

    def test_wrong_overlay_file(self):
        self.admission['bitsandbytes']['files']['nn/modules.py'] = 'a'*64
        with self.assertRaisesRegex(ValueError, 'SOURCE_EXACT_OVERLAY'): A.validate_admission(B, self.admission, self.manifest)

    def test_extra_overlay_file(self):
        self.admission['bitsandbytes']['files']['extra.py'] = 'a'*64
        with self.assertRaisesRegex(ValueError, 'SOURCE_EXACT_OVERLAY'): A.validate_admission(B, self.admission, self.manifest)

    def test_missing_plugin_inventory(self):
        self.admission['bitsandbytes_tpu']['files'] = {}
        with self.assertRaisesRegex(ValueError, 'PLUGIN_FULL_MAP'): A.validate_admission(B, self.admission, self.manifest)

    def test_wrong_manifest_bytes(self):
        with tempfile.TemporaryDirectory() as base:
            path = Path(base)/'manifest.json'; path.write_text(json.dumps(self.manifest))
            with self.assertRaisesRegex(ValueError, 'PATCH_MANIFEST_IDENTITY'): A.load_manifest(B, path)

    def test_installed_full_tree_and_mutation(self):
        with tempfile.TemporaryDirectory() as base:
            roots = {}
            admission = copy.deepcopy(self.admission)
            for name in ('bitsandbytes', 'bitsandbytes_tpu'):
                root = Path(base)/name; root.mkdir(); file = root/'__init__.py'; file.write_text('# synthetic package\n')
                roots[name] = root; admission[name]['files'] = {'__init__.py': B.sha(file)}
            A.verify_installed(B, admission, roots)
            (roots['bitsandbytes']/'unexpected.py').write_text('# extra source\n')
            with self.assertRaisesRegex(ValueError, 'SOURCE_BYTES_OR_INVENTORY'): A.verify_installed(B, admission, roots)

    def test_actual_loaded_method_code_and_replacement(self):
        source = ('class Params4bit:\n def __new__(cls): pass\n def _quantize(self, device): pass\n def to(self): pass\n'
                  'class Linear4bit:\n def __init__(self): pass\n def forward(self, x): pass\n')
        with tempfile.TemporaryDirectory() as base:
            root = Path(base); file = root/'nn/modules.py'; file.parent.mkdir(); file.write_text(source)
            module = ModuleType('bitsandbytes.nn.modules'); module.__file__ = str(file)
            exec(compile(source, str(file), 'exec'), module.__dict__)
            bnb = SimpleNamespace(nn=SimpleNamespace(Linear4bit=module.Linear4bit, Params4bit=module.Params4bit))
            with patch.dict(sys.modules, {'bitsandbytes.nn.modules': module}):
                self.assertTrue(A.public_methods(B, bnb, {'bitsandbytes': root})['original_method_identity'])
                module.Params4bit._quantize = lambda self, device: None
                with self.assertRaisesRegex(ValueError, 'PUBLIC_METHOD_REPLACEMENT'): A.public_methods(B, bnb, {'bitsandbytes': root})

    def test_actual_loaded_class_replacement(self):
        with tempfile.TemporaryDirectory() as base:
            root = Path(base); file = root/'nn/modules.py'; file.parent.mkdir(); file.write_text('# synthetic\n')
            module = ModuleType('bitsandbytes.nn.modules'); module.__file__ = str(file)
            exec('class Linear4bit: pass\nclass Params4bit: pass\n', module.__dict__)
            bnb = SimpleNamespace(nn=SimpleNamespace(Linear4bit=type('Replacement', (), {}), Params4bit=module.Params4bit))
            with patch.dict(sys.modules, {'bitsandbytes.nn.modules': module}):
                with self.assertRaisesRegex(ValueError, 'PUBLIC_API_SUBSTITUTE'): A.public_methods(B, bnb, {'bitsandbytes': root})


if __name__ == '__main__': unittest.main()
