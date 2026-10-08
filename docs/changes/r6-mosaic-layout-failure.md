# Native decode layout failure

A fresh Colab V5E1 experiment connected on 2026-10-09, Korea time.
The marker, fixed runtime, source admission, and fresh Linux CPU oracle passed inspection.
The CPU oracle contained five cases and 164,096 output values.

The first native case failed during TPU compilation.
The compiler reported `infer-vector-layout: unsupported shape cast` in `decode_tile`.
The failed operation converted `vector<128x64x2xi32>` to `vector<128x128xi32>`.
No native numerical case passed.
The diagnostic retained the first error and marked the remaining four cases `NOT_RUN`.
It linked each unexecuted case to the original failure.

## Conversion and result inspection

The retained first call contains the original version-8 payload and converted version-7 payload.
The reviewer independently repeated the official conversion with the pinned JAX 0.7.1 CPU interpreter.
Both payloads produced the same current intermediate representation.
The new compiler error occurred after the earlier version rejection.
Successful serialization does not establish a supported TPU layout.

The unchanged result verifier rejected the run with `CHILD_TERMINAL`.
The reviewer verified all 110 result archive members and all 11 CPU archive members.
Six isolated audit controls passed before this inspection.
The original numerical criteria, kernel, and runtime remained unchanged.

## Resource closure

All 47 CLI process groups and all 14 remote steps closed.
The scientific child, local verifier, and host owner also closed.
The complete lifecycle used 457.25 seconds of its original 3,600-second limit.
The reviewer closed the temporary browser tab.
Fresh CLI observations then showed no active session and zero active usage.
The separate payload audit also closed its owned group.

Prepare a decode layout that preserves NF4 nibble order and scaling.
Retain this failure and require new source admission, CPU records, and actual TPU records for that change.

The [selected result](../../experiments/2026-10-04-bitsandbytes-tpu/results/native-mosaic-layout-failure.json) binds the error, archives, conversion audit, and closure records.
Raw records remain private under `.work`.
M6 remains unqualified, and accepted milestones remain three of eight.
