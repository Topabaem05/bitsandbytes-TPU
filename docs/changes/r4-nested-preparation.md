# R4: Nested quantization experiment preparation

This change prepares 79 fixed cases for a separate Colab experiment.
It does not complete M4.
M4 also requires accepted records for saved nested state and restoration.
The reviewer accepted M3 before this device preparation.

The explicit `nested-v1` package adds three blockwise overloads to the four existing NF4 overloads.
It uses block size 64 for NF4 and block size 256 for nested scales.
The original public classes and methods remain in use.
The package retains the examined parameter transfer patch and fixed runtime.
Each device process selects precision `highest` before graph construction.

The reference uses the pinned default Python function bodies.
The prepare phase calculates new reference data in the qualified Linux runtime.
The reference does not use the native CPU lookup implementation.
Native CPU and CUDA byte comparisons remain untested.
The local fixture supplies synthetic test records only.

The matrix includes boundary values, partial blocks, zero values, subnormal numbers, and known scale codes.
It also includes eight public module cases with both compute types, ranks, and bias options.
The fixed gates compare packed bytes, scale codes, offsets, second scales, codebooks, decoded weights, outputs, and FP32 gradients.
The output overloads must retain their specified identity or documented transpose view.
BF16 gradients remain outside this experiment.

The reviewer repeated 19 controls in 23.370 seconds.
The same run repeated 79 local CPU cases and 540 numerical gates.
All passed, and both CPU records matched the worker records exactly.
This macOS PyTorch 2.14.1 result does not qualify Linux or TPU execution.
Incorrect source, missing data, invalid numerical values, and recorded CPU fallback cannot produce a successful result.
Finite numerical failures remain distinct from execution errors.

The reviewer repeated 310 cloud and state controls in 67.71 seconds; all passed.
These controls used synthetic service records and made no provider calls.
The cloud path retains one allocation, bounded phases, complete archive recovery, and exact resource closure.
The source contract binds all six installed Python files and seven operator schemas.
It also binds the built plugin wheel and installed source before reference calculation.
An incomplete archive or unsuccessful cleanup prevents acceptance.

The reference GEMM still constructs dense weights.
This preparation makes no Pallas, training, memory, or performance claim.
The nested saved-state supplement remains a separate required experiment.

The reviewer derived the 44-member packet independently from the accepted M2 packet.
Its 1,025,947 bytes matched the worker packet exactly.
Refer to [the preparation record](../../experiments/2026-10-04-bitsandbytes-tpu/results/nested-preparation.json).
