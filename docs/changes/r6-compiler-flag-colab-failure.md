# Compiler flag failure on Colab

Date: 2026-10-09.

The compiler experiment used one Colab V5E1 runtime and the accepted native dependency.
The fixed Linux runtime, source controls, and five CPU reference cases passed.
The independent CPU gate verified 164,096 reference output values.

The native child stopped before numerical execution.
The installed `libtpu 0.0.21` rejected `--xla_dump_hlo_unoptimized_snapshots=false` in `XLA_FLAGS`.
All five numerical cases remain `NOT_RUN` for this experiment.
The earlier accepted native result remains unchanged.

The reviewer verified all 110 scientific archive members and all 11 CPU archive members.
The separate compiler archive contained four control records and no raw compiler dumps.
The production verifier rejected the incomplete parent with `PARENT_TERMINAL`.
No compiler content, memory result, or new native result was accepted.

Nine independent controls examined valid and incorrect archive, array, closure, CPU gate, and failure records.
All 54 owner CLI groups, 15 remote steps, and the native child closed.
The host, verifier, independent reader, and two final inspection groups also closed.
The reviewer observed an empty server list, zero active usage, and the disconnected notebook.
The temporary browser tab closed.
The original deadline remained unchanged.
The complete allocation lifecycle took 385.51 seconds.

Refer to the [selected result](../../experiments/2026-10-04-bitsandbytes-tpu/results/compiler-flag-colab-failure.json).
Raw provider records remain outside Git.

Before another compiler experiment, inspect the flags that the fixed runtime supports.
Use a new source generation and repeat the qualified CPU gate.
M6 remains unqualified.
