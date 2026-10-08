# Nested quantization result on Colab

Date: 2026-10-08.

The 79-case nested experiment executed on Colab V6E1.
The reviewer independently repeated all 540 numerical comparisons.
There were 52 passing cases, 27 failing cases, and no execution errors.
M4 remains unqualified.

## Numerical result

The experiment used the unchanged inputs, reference implementation, and tolerances.
The CPU reference used Linux x86-64, Python 3.12, and PyTorch 2.9.0+cpu.
The TPU runtime used PyTorch/XLA 2.9.0 and libtpu 0.0.21.
Native precision was `highest` before graph construction.

| Output | Failed comparisons | Observed difference |
| --- | ---: | --- |
| `second_scales` | 27 | Subnormal values or small FP32 differences |
| `offset` | 9 | Subnormal values or small FP32 differences |
| `scale_codes` | 4 | Incorrect nested codes in zero-input cases |
| `codes` | 3 | Incorrect blockwise codes in subnormal-input cases |

All forward, input-gradient, bias-gradient, and decode comparisons passed their fixed tolerances.
Those passes do not compensate for the 43 failed exact comparisons.
The full result is `FAIL`.

Actual scale values frequently contained zero where the CPU reference retained a subnormal value.
Three cases also had normal FP32 offset differences.
The admitted upstream function calculates the mean and centered scales outside the seven plugin operators.
An operator-only correction cannot cover that complete path.
The next change must address this source boundary and retain the exact criteria.

The source inspection does not authorize dynamic replacement of public methods.
Any additional source patch needs separate admission and tests.
Keep the existing failed packet and records unchanged.
Do not execute the saved-state supplement or accept an M4 dependency from this failed result.

## Execution and closure

The CLI reused the identified browser runtime without another allocation request.
The source packet SHA-256 is `08189b55e8aef5507fab27017dcf824e21078cae9e068d3b4516108e231f1fbb`.
The complete result archive contains 409 members and 1,186,835 bytes.
Its SHA-256 is `a0a0d811ff4061b5b9f3b4b94188869da78a26676843fa0b43e4f2bc091d2f11`.

All 40 CLI process groups, scientific process groups, and the host process closed.
The server list was empty after exact-session termination.
Active runtime usage was zero.
The browser also showed a disconnected runtime before its temporary tab closed.
The original runtime lifecycle was 2,468.07 seconds, within the 3,600-second limit.
The host execution took 325.95 seconds.
The driver returned exit code 2 for the numerical failure.

The [result record](../../experiments/2026-10-04-bitsandbytes-tpu/results/nested-colab-failure.json) contains each comparison and the source bindings.
Accepted milestones remain **3 of 8**.
