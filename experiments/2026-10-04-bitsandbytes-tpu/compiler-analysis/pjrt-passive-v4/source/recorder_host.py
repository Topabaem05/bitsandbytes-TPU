"""Explicit host setup. Import this module before importing Torch/XLA."""
import ctypes
import hashlib
import os
from pathlib import Path
import sys

import sys
sys.dont_write_bytecode = True

from verify import seal, verify


class PassiveRecorder:
    def __init__(self, shim, real_provider, output, nonce, original_deadline_ns,
                 recorder_budget_bytes, recorder_max_entries=256):
        # The caller supplies the original CLOCK_MONOTONIC deadline.
        # This helper does not create or renew an experiment deadline.
        assert not any(n == 'torch_xla' or n.startswith('torch_xla.') for n in sys.modules)
        self.shim = Path(shim).resolve(strict=True)
        provider = Path(real_provider).resolve(strict=True)
        self.output = Path(output).absolute()
        assert not self.output.exists()
        assert not os.path.samefile(self.shim, provider)
        recorder_id = hashlib.sha256(self.shim.read_bytes()).hexdigest()
        self.library = ctypes.CDLL(str(self.shim), mode=ctypes.RTLD_LOCAL)
        configure = self.library.M6TraceConfigure
        configure.argtypes = [ctypes.c_char_p] * 4 + [ctypes.c_uint64] + [ctypes.c_size_t] * 3
        configure.restype = ctypes.c_int
        self.library.M6TraceSetPhase.argtypes = [ctypes.c_char_p]
        self.library.M6TraceSetPhase.restype = ctypes.c_int
        self.library.M6TraceFinalize.argtypes = []
        self.library.M6TraceFinalize.restype = ctypes.c_int
        result = configure(os.fsencode(provider), os.fsencode(self.output), nonce.encode('ascii'),
                           recorder_id.encode('ascii'), original_deadline_ns,
                           recorder_max_entries, recorder_budget_bytes, 16 * 1024 * 1024)
        assert result == 0, 'Recorder configuration failed'
        os.environ['TPU_LIBRARY_PATH'] = str(self.shim)

    def set_phase(self, public_label):
        assert self.library.M6TraceSetPhase(public_label.encode('ascii')) == 0

    def finalize(self):
        # The caller must stop recorder capture only after its planned API calls.
        result = self.library.M6TraceFinalize()
        if result == 1:
            return {'status': 'INFLIGHT', 'closed': False}
        seal(self.output, result)
        report = verify(self.output)
        report['closed'] = True
        return report
