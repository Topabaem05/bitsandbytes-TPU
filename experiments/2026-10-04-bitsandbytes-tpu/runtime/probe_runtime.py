"""Run a small TPU runtime probe in a separate Linux CPython 3.12 environment."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

from resolve import CORE, SOURCES, validate_profile


def require_tpu(device_type, devices):
    if device_type != 'TPU' or not devices or any(not d.startswith('TPU:') for d in devices):
        raise ValueError('CPU_FALLBACK_OR_NON_TPU')


def origin(module):
    p = Path(module.__file__)
    return {'module': module.__name__, 'source_filename': p.name,
            'source_sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = {'status': 'BLOCKED', 'runtime_executed': False,
              'python': platform.python_version(), 'system': platform.system(),
              'machine': platform.machine(), 'libc': list(platform.libc_ver()),
              'requested_backend': os.environ.get('PJRT_DEVICE'),
              'versions': {}, 'metrics': None, 'sources': [], 'error': None}
    try:
        if platform.system() != 'Linux' or platform.machine() != 'x86_64' or sys.version_info[:2] != (3, 12):
            raise ValueError('LINUX_X86_64_CP312_REQUIRED')
        libc_name, libc_version = platform.libc_ver()
        if libc_name != 'glibc' or tuple(int(n) for n in libc_version.split('.')[:2]) < (2, 31):
            raise ValueError('GLIBC_2_31_REQUIRED')
        if os.environ.get('PJRT_DEVICE') != 'TPU':
            raise ValueError('EXPLICIT_PJRT_DEVICE_TPU_REQUIRED')
        if os.environ.get('TPU_LIBRARY_PATH') or os.environ.get('PTXLA_TPU_LIBRARY_PATH'):
            raise ValueError('PRESET_TPU_LIBRARY_PATH_FORBIDDEN')
        result['versions'] = {n: importlib.metadata.version(n) for n in CORE}
        validate_profile(result['versions'])
        import torch
        import torch_xla
        import torch_xla.runtime as xr
        import torch_xla.core.xla_model as xm
        import torch_xla.debug.metrics as met
        # Initialize the XLA computation client before importing JAX.
        from torch_xla._internal.jax_workarounds import maybe_get_jax
        jax = maybe_get_jax()
        if jax is None:
            raise ValueError('JAX_IMPORT_GUARD_FAILED')
        from torch_xla.experimental import custom_kernel
        import jaxlib
        devices = torch_xla._XLAC._xla_get_devices()
        require_tpu(xr.device_type(), devices)
        result['observed_backend'] = xr.device_type()
        result['devices'] = devices
        result['logical_devices'] = xm.get_xla_supported_devices()
        import libtpu
        library = Path(libtpu.get_library_path())
        if os.environ.get('PTXLA_TPU_LIBRARY_PATH') != str(library):
            raise ValueError('LIBTPU_ORIGIN_MISMATCH')
        result['libtpu_library'] = {'filename': library.name,
                                   'sha256': hashlib.sha256(library.read_bytes()).hexdigest()}
        result['sources'] = [origin(m) for m in [torch, torch_xla, xr, custom_kernel, jax, jaxlib]]
        if origin(custom_kernel)['source_sha256'] != SOURCES['torch_xla/experimental/custom_kernel.py']:
            raise ValueError('PALLAS_BRIDGE_SOURCE_MISMATCH')
        met.clear_all()
        device = torch_xla.device()
        x = torch.tensor([1.0, 2.0], device=device, requires_grad=True)
        loss = (x * x).sum()
        loss.backward()
        torch_xla.sync(wait=True)
        result['runtime_executed'] = True
        result['loss'] = loss.cpu().item()
        result['gradient'] = x.grad.cpu().tolist()
        result['metrics'] = met.metrics_report()
        if result['loss'] != 5.0 or result['gradient'] != [2.0, 4.0]:
            raise ValueError('RUNTIME_VALUE_OR_GRADIENT')
        if not callable(custom_kernel.make_kernel_from_pallas):
            raise ValueError('PALLAS_BRIDGE_MISSING')
        result['status'] = 'PASS_TPU_RUNTIME_PROBE_ONLY'
        result['pallas_kernel_executed'] = False
        result['nf4_executed'] = False
    except Exception as exc:
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': result['status'], 'error': result['error']}))
    return 0 if result['status'] == 'PASS_TPU_RUNTIME_PROBE_ONLY' else 1


if __name__ == '__main__':
    raise SystemExit(main())
