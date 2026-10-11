# Diagnostic raw input boundary

Date: 2026-10-11.

The diagnostic now synchronizes and unwraps native functional inputs before the backend call.
It retains the raw tensors in the input dictionary and preserves repeated references.
Each input must retain its dtype, shape, stride, and device.
The report records the original and normalized functional states.

The backend rejection remains unchanged.
The four bitcast cases still create their requested input states.
The arithmetic, isolated variants, fixed input bytes, runtime, limits, and original 600-second clock remain unchanged.
Five authored payload files changed; all other payload files remain byte-identical.

The root replay passed 24 CPU probes and rejected nine incorrect controls.
Five native functional controls passed, including all 24 wrapped-input results.
The adviser independently passed 14 correct and incorrect controls.
All 88 complete source archive members passed readback.
These local tests used Torch 2.14.1 and do not qualify the fixed Linux runtime or TPU.

The reviewer adopted 67 selected source files and deleted six temporary roots with 134 files.
Refer to the [source recipe](../../experiments/2026-10-04-bitsandbytes-tpu/cloud/word-diagnostic-v2/ADOPTION.md)
and [adoption record](../../experiments/2026-10-04-bitsandbytes-tpu/results/m4-word-diagnostic-v2-preparation/source-adoption.json).
M4 remains unqualified.
Actual Linux CPU and TPU execution are the next checks.
