# Mosaic version conversion preparation

The previous TPU run rejected Mosaic version 8 because its compiler accepted versions through 7.
The new recorder uses the official JAX 0.7.1 serialization passes to produce version 7.
It retains the original and converted modules with their hashes.
It changes only the module body and JSON encoding.
The ordered tensor objects and native binding remain unchanged.

The converter first reads the original module into the current intermediate representation.
It then writes version 7 and reads that module again.
The current representations must match exactly.
Unsupported features fail before the native call.
The fixed kernel passes this requirement; other valid modules can still fail this conservative comparison.
The pinned [JAX serializer](https://github.com/jax-ml/jax/blob/5712de44e97c455faed1fd45532e821ca66d025a/jax/_src/tpu_custom_call.py#L66)
and [TPU serialization rules](https://github.com/jax-ml/jax/blob/5712de44e97c455faed1fd45532e821ca66d025a/jaxlib/mosaic/dialect/tpu/transforms/serde.cc#L99)
define the version conversion.

The recovered-record verifier uses a separate, explicit JAX 0.7.1 interpreter.
It independently repeats the conversion and compares the complete module and configuration bytes.
It recomputes the audit fields and retains the existing graph, execution, and numerical requirements.
A stored success flag cannot replace this inspection.

The host checks the interpreter before service allocation.
The fixed Linux runtime, libtpu, kernel, M2 source, and numerical criteria remain unchanged.
After the first runtime error, later cases remain `NOT_RUN` and refer to that error.
The diagnostic makes no later XLA calls in that process.

## Validation

The reviewer repeated 56 focused controls; all passed.
They include genuine CPU conversions, independent parser checks, and isolated incorrect execution records.
The seven externally owned control groups closed without cleanup errors.
The separate interpreter inspection also passed and closed its process group.

The reviewer inspected all 73 members of the fresh execution packet.
Its 27 native sources match the accepted file map.
The packet contains 1,057,511 bytes.
Its SHA-256 is `6749df71f35a315593ef2cee86c801b3f59c7a2785946ae34d4c06396ff3668c`.

The [selected result](../../experiments/2026-10-04-bitsandbytes-tpu/results/native-mosaic-preparation.json) identifies the inspected sources and controls.
A fresh qualified Linux CPU oracle and actual TPU execution remain required.
Executable identity, compiler memory, public integration, and backward execution remain unqualified.
M6 remains unqualified, and accepted milestones remain three of eight.
