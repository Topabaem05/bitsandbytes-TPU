# bitsandbytes-TPU

A TPU backend for the upstream bitsandbytes API.

## Project status

The package, actual TPU execution, and module state milestones are completed.
Accepted milestones: **3 of 8**.
The first target is NF4 quantization and `bitsandbytes.nn.Linear4bit` on a TPU.
The M2 Colab V6E1 run passed all 42 API cases and four public CPU-to-XLA transfer cases.
The reviewer independently compared 196 numerical gates and verified all 273 archive members.
The runtime stopped with an empty server list and zero active usage.
Refer to [the accepted M2 result](docs/changes/r2-transfer-colab.md).
The experiment used native precision `highest` before graph construction.
The earlier `default` precision failures remain in [the precision result](docs/changes/r1-precision-colab.md).
BF16 forward gates passed; BF16 gradients remain untested.
The accepted M3 Colab run passed all eight saved-state cases after restoration in a new TPU process.
The reviewer compared 136 numerical gates and all eight checkpoint pairs.
Refer to [the accepted M3 result](docs/changes/r3-state-colab.md), including the separate CLI exit correction.
The latest M4 Linux CPU run passed 79 cases and 540 array checks.
Its TPU audit reported two passes, 12 failures, and 65 errors across the 79 required cases.
Thirty cases have no returned record after the evidence limit stopped execution.
Saved nested state did not run.
Refer to [the latest M4 result](docs/changes/r4-functional-v4-failure-20261011.md).
The focused [integer diagnostic](docs/changes/m4-word-diagnostic-v2-actual.md) returned all 24 TPU probes without execution errors.
Explicit Tensor shifts and the flat tail reduction axis made all five selected mathematical families match the CPU oracle.
Their graphs still contain S64 comparison operands from Python integer scalars.
The reviewed [S32 source preparation](docs/changes/m4-s32-lowering-v3-preparation.md) uses explicit Tensor comparison operands.
Fresh CPU79, TPU79, and saved state8 must test this generation.
The original graph criteria remain unchanged.

The [M5 record reader](docs/changes/m5-reader-v4-preparation.md) passed independent resource and lifecycle checks.
Its training math and checkpoint criteria remain unchanged.
Genuine M4 acceptance, a separate plugin rebase, and actual TPU training remain required.

Five native compiler diagnostic cases passed numerical checks on Colab.
The [compiler readback](experiments/2026-10-04-bitsandbytes-tpu/results/resume-actual-20261011/compiler-independent-readback.json) records their exact scope.
The [content diagnosis](experiments/2026-10-04-bitsandbytes-tpu/results/resume-actual-20261011/compiler-content-diagnosis.json) did not identify a valid selected executable chain.
Compiler memory, public integration, and performance claims remain unqualified.
The [passive PJRT source](docs/changes/pjrt-passive-v4-preparation.md) passed local compiled and independent review controls.
A fresh Linux and TPU integration experiment remains required.
M4 through M8 remain unqualified.

The [October 8 source inspection](docs/research/2026-10-08/README.md) covers each design area before further implementation.
It contains current documentation, original papers, fixed source revisions, and explicit version limits.
The [decision table](docs/research/2026-10-08/decisions.md) defines the next work and its acceptance requirements.
The inspection did not change the runtime, code, or numerical tolerances.

Earlier failure records remain available:

- [Allocation timeouts](docs/changes/r2-colab-allocation-repeat.md), [web diagnostic](docs/changes/r2-web-allocation-diagnostic.md), and [transport correction](docs/changes/r2-allocation-transport.md).
- [Device route comparison](docs/changes/task2-colab-routes.md).
- [First Colab run](docs/changes/task2-colab-first.md).
- [Functionalization repeat](docs/changes/task2-colab-functionalized.md).
- [Third Colab run](docs/changes/task2-colab-native.md).
- [Zero-block correction and 81 local tests](docs/changes/task1-zero-normalization.md).

The project name is **bitsandbytes-TPU**.
The Python package name is `bitsandbytes-tpu`.
The Python module name is `bitsandbytes_tpu`.

## Research goal

Keep the upstream bitsandbytes classes and public API.
Use PyTorch/XLA to execute their operations on a TPU.
Then use Pallas to reduce the time and memory necessary for NF4 matrix multiplication.

The required results are:

1. An installable backend with automatic registration.
2. Correct NF4 data and upstream `QuantState` data.
3. Correct `Linear4bit` output, gradients, and saved model data.
4. Correct nested quantization for `compress_statistics=True`.
5. A QLoRA test with a new process for model restoration.
6. A Pallas kernel with measured time and memory.
7. Repeatable results on Colab and Kaggle.

A CPU test does not show TPU operation.
A reference implementation does not show a reduction in memory.
Each result needs its own test records.

## Initial configuration

| Item | Initial target |
| --- | --- |
| Upstream bitsandbytes | Commit `833649043474794b8fe7a4136e0c40faf077b2e0` |
| Device interface | Native PyTorch/XLA |
| Quantization | NF4, block size 64 |
| Packed storage | `torch.uint8` |
| Input data types | FP32 and BF16 |
| First test | Quantization without nested quantization |
| First usable milestone | Nested quantization and the upstream `Linear4bit` API |

Inputs must be finite.
Scales must be finite and nonnegative.
The caller is responsible for these value preconditions.
The initial configuration excludes FP4, FP16, and other block sizes.
The backend must give an explicit error for an option that it cannot execute.

## Documents

- [Artifact storage](docs/artifact-storage.md)
- [Design](docs/design.md)
- [Research plan](docs/research-plan.md)
- [Current design sources and decisions](docs/research/2026-10-08/README.md)
- [Kaggle M2 repetition preparation](experiments/2026-10-04-bitsandbytes-tpu/m8/README.md)
- [Test requirements](docs/validation.md)
- [Writing guide](docs/writing-guide.md)
- [Technical terms](docs/terms.md)
- [Change log](CHANGELOG.md)

## Development

The package source is in `packages/bitsandbytes-tpu`.
The experiment source is in `experiments`.
Refer to [the package document](packages/bitsandbytes-tpu/README.md) for the current scope and local tests.
Use [the fixed runtime procedure](docs/changes/task2-runtime.md) for the planned Colab tests.

Project documents use ASD-STE100 Issue 9 as their writing standard.
Do not change code identifiers.
Refer to the writing guide for the inspection procedure and its limits.

## License

The project license is [Apache License 2.0](LICENSE).
Third-party source keeps its applicable license and notices.
