# Arithmetic diagnostic preparation

This diagnostic separates FP32 reduction, division, and subnormal behavior on the TPU.
It supplies observations for the failed nested quantization result.
It does not change the original 79 cases, their inputs, or their acceptance criteria.
M4 remains unqualified.

Use the explicit `arithmetic-primitives` mode with the fixed `nested-v1` source and runtime.
The qualified Linux CPU phase produces 32 rows and 206 output byte comparisons.
It calculates 28 first-scale arrays from the original inputs.
The host recovers the complete CPU archive before device execution.
It verifies source, runtime, process identity, closure, exact input bytes, and the source-defined arithmetic reference.

The TPU phase uses two separate child processes in one owned process group.
The first observes native dtype views.
The second uses the fixed Torch/XLA bitcast builder and tests arithmetic on the same recorded inputs.
Each result includes input parameters, graph content, execution metrics, and exact byte comparisons.
These graphs do not establish an optimized instruction sequence.

| Limit | Value |
| --- | --- |
| Complete device phase | 1,500 seconds |
| Native view child | 180 seconds |
| Builder child | 900 seconds |
| Child cleanup reserve | 10 seconds |
| Complete allocation | Original 3,600-second deadline |

Each child deadline stays within the original remaining work interval.
A native view can give a valid `UNSUPPORTED` observation.
The builder must supply complete graph and execution records without CPU fallback.
Complete records can contain numerical differences.
`PASS_PRIMITIVE_RECORDS` confirms record validity; it does not qualify M4 or another milestone.

## Reviewer results

The reviewer repeated 72 focused cloud controls; all passed.
After file adoption, 18 test containers passed: one contained 25 scientific controls, and 17 tested CLI status handling.
The earlier independent review also examined six valid and incorrect HLO records.
All five scientific files match their reviewed hashes.
The fixed upstream Git source produced a 56-member packet with 1,082,391 bytes.
Every packet member passed inspection.
Live upload of this packet remains unverified.

The worker's complete local suite passed 477 of 478 tests.
A parent-SIGKILL test returned a macOS `PermissionError` during process group inspection.
The same test failed in the single focused repeat; its other three controls passed.
Both original failures remain `CLEANUP_UNVERIFIED`.
A later absence observation did not change those results.
The ownership implementation is unchanged, and actual execution still requires verified group closure.

The reviewer accepted this diagnostic preparation with the retained local platform limitation.
No Linux or TPU arithmetic result is claimed by these preparation tests.
The [preparation record](../../experiments/2026-10-04-bitsandbytes-tpu/results/primitive-preparation.json) supplies file hashes and exact test scope.

The arithmetic reference follows [PyTorch 2.9 SumKernel.cpp](https://github.com/pytorch/pytorch/blob/0fabc3ba44823f257e70ce397d989c8de5e362c1/aten/src/ATen/native/cpu/SumKernel.cpp).
The bitcast interface follows the [fixed Torch/XLA builder](https://github.com/pytorch/xla/blob/5fab7053df86c8d503b98d9e7202ca8b8d4978c7/torch_xla/core/xla_builder.py).
The exact sources, inputs, and byte criteria remain required for the actual experiment.
