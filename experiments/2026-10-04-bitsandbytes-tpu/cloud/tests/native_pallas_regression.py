"""Optional JAX071 CPU interpretation and genuine TPU-target frontend replay; no TPU."""
import argparse
import ast
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jaxlib
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl  # Public initialization before internal registration.
from jax._src import sharding_impls
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
import jax._src.pallas.mosaic.pallas_call_registration

SOURCES = {'jax/_src/pallas/mosaic/lowering.py': '6e5d7b20367972ff2fc62604ee62c32dd33b2113f163bd6afde057086bed6ebb',
    'jax/_src/pallas/mosaic/pallas_call_registration.py': '5700557310159ac45b106ae56ceabf4d09a67c488e91fed41717766f88501c1b',
    'jax/_src/tpu_custom_call.py': 'ae78bf3e65571d8c15004e76863747e2b414e7f17073cee151c643f8c7932d37',
    'jax/_src/interpreters/mlir.py': 'f60cb1ccc0ea6e2b2864c24deb403b401a23971505bea50f53030de88bed9d67'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lower(kernel, dtype, output):
    output.mkdir()
    args = [jax.ShapeDtypeStruct((128, 512), dtype), jax.ShapeDtypeStruct((2, 128, 128), jnp.uint8),
        jax.ShapeDtypeStruct((2, 128, 4), jnp.float32)]
    closed = jax.make_jaxpr(lambda *xs: kernel.forward_grouped(*xs, interpret=False))(*args)
    lowered = mlir.lower_jaxpr_to_module('native_repair_regression', closed, num_const_args=0,
        in_avals=closed.in_avals, ordered_effects=[], platforms=('tpu',), backend=None,
        axis_context=sharding_impls.ReplicaAxisContext(sharding_impls.AxisEnv(1, (), ())),
        donated_args=(False,) * 3, lowering_parameters=mlir.LoweringParameters())
    lowered.module.operation.verify()
    (output / 'module.mlir').write_text(str(lowered.module))
    calls = []
    def visit(op):
        if op.name == 'stablehlo.custom_call' and ir.StringAttr(op.attributes['call_target_name']).value == 'tpu_custom_call':
            calls.append(op)
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations: visit(child.operation)
    visit(lowered.module.operation)
    assert len(calls) == 1
    with lowered.module.context:
        payload = ir.StringAttr(calls[0].attributes['backend_config']).value
    config = json.loads(payload)
    body = base64.b64decode(config['custom_call_config']['body'], validate=True)
    assert body and config['custom_call_config']['needs_layout_passes'] is True
    (output / 'native-config.json').write_text(payload)
    (output / 'mosaic-body.bin').write_bytes(body)


def run(native, output):
    native = Path(native).resolve()
    output = Path(output).resolve()
    output.mkdir(exist_ok=False)
    root = next(p for p in native.parents if (p / 'packages/bitsandbytes-tpu/pyproject.toml').is_file())
    assert jax.__version__ == jaxlib.__version__ == '0.7.1'
    for name, pin in SOURCES.items(): assert sha(Path(jax.__file__).parent.parent / name) == pin
    assert {device.platform for device in jax.devices()} == {'cpu'}
    contract = load(native.parent / 'cloud/native_contract.py', 'native_regression_contract')
    assert sha(native / 'manifest.json') == contract.MANIFEST_SHA
    source = json.loads((native / 'manifest.json').read_text())['sources']
    assert source == contract.SOURCES
    for name, record in source.items():
        path = native / name
        assert not path.is_symlink() and sha(path) == record['sha256'] and path.stat().st_size == record['bytes']
    assert sha(native / 'kernel.py') == source['kernel.py']['sha256']
    kernel = load(native / 'kernel.py', 'current_native_kernel')
    kernel.runtime_guard()
    backend = load(native / 'probe_backend.py', 'native_regression_backend')
    profile, _ = backend.load_spec()
    reference = root / 'packages/bitsandbytes-tpu/src/bitsandbytes_tpu/reference.py'
    protocol = json.loads((native / 'protocol.json').read_text())
    assert sha(reference) == protocol['plugin_python_files']['reference.py']
    node = next(n for n in ast.parse(reference.read_text()).body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'NF4_CODE' for t in n.targets))
    codebook = np.asarray(ast.literal_eval(node.value), dtype=np.float32)
    rows = []
    for dtype in (jnp.float32, jnp.bfloat16):
        lower(kernel, dtype, output / ('correct-' + str(dtype)))
        rows.append({'case': 'correct_frontend_' + str(dtype), 'status': 'PASS'})
    old = output / 'dynamic-slice-fixture.py'
    text = (native / 'kernel.py').read_text()
    before = 'byte_half=jnp.where(half==0,packed_group[:, :64],packed_group[:, 64:])'
    assert text.count(before) == 1
    old.write_text(text.replace(before, 'byte_half=lax.dynamic_slice(packed_group,(0,half*64),(BN,64))'))
    try:
        lower(load(old, 'old_dynamic_fixture'), jnp.float32, output / 'dynamic-slice')
    except NotImplementedError as error:
        assert 'Unimplemented primitive in Pallas TPU lowering for KernelType.TC: dynamic_slice' in str(error)
        (output / 'dynamic-slice/error.txt').write_text(str(error))
        rows.append({'case': 'dynamic_slice_rejected', 'status': 'PASS'})
    else: raise AssertionError('Unsupported dynamic slice accepted')
    # Distinct 64-value blocks make the two halves observably different.
    indices = np.arange(128 * 512)
    codes = ((indices * 7 + (indices // 64) * 5) % 16).reshape(128, 512).astype(np.uint8)
    packed = ((codes.reshape(-1)[::2] << 4) | codes.reshape(-1)[1::2]).reshape(-1, 1)
    scales = np.ones(128 * 512 // 64, dtype=np.float32)
    activation = (((np.arange(128 * 512) % 31) - 15).astype(np.float32) / 32).reshape(128, 512)
    grouped, gs = kernel.prepare_operands(jnp.asarray(packed), jnp.asarray(scales), (128, 512))
    def numeric(candidate, dtype):
        decoded = np.asarray(jnp.asarray(codebook[codes], dtype=dtype)).astype(np.float32)
        expected = np.asarray(jnp.asarray(activation @ decoded.T, dtype=dtype)).astype(np.float32)
        actual = np.asarray(candidate.forward_grouped(jnp.asarray(activation, dtype=dtype), grouped, gs,
            interpret=True).block_until_ready()).astype(np.float32)
        tolerance = profile['tolerances']['fp32_forward_gradient' if dtype == jnp.float32 else 'bf16_forward_unit_scale']
        return backend.numeric(actual.reshape(-1).tolist(), expected.reshape(-1).tolist(), tolerance)
    for dtype in (jnp.float32, jnp.bfloat16):
        gate = numeric(kernel, dtype)
        assert gate['status'] == 'PASS'
        rows.append({'case': 'correct_cpu_' + str(dtype), 'status': 'PASS', 'gate': gate})
    wrong = output / 'wrong-half-fixture.py'
    wrong.write_text(text.replace('half==0', 'jnp.bool_(True)'))
    bad = load(wrong, 'wrong_half_fixture')
    lower(bad, jnp.float32, output / 'wrong-half-frontend')
    rows.append({'case': 'wrong_half_frontend_can_serialize', 'status': 'PASS'})
    gate = numeric(bad, jnp.float32)
    assert gate['status'] == 'FAIL'
    rows.append({'case': 'wrong_half_numeric_rejected', 'status': 'PASS', 'gate': gate})
    result = {'status': 'PASS_OFFLINE_REGRESSION', 'controls': rows, 'kernel_sha256': sha(native / 'kernel.py'),
        'frontend_sources': SOURCES, 'profile_sha256': backend.PROFILE_SHA, 'jax': jax.__version__,
        'jaxlib': jaxlib.__version__, 'scope': 'CPU_INTERPRET_AND_TPU_TARGET_FRONTEND_CPU_CLIENT',
        'actual_tpu': 'NOT_RUN', 'libtpu_layout_backend': 'NOT_RUN', 'memory_performance': 'NOT_RUN', 'provider_calls': 0}
    (output / 'results.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-source', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    run(a.native_source, a.output)
