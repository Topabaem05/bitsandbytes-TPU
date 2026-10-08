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
The next milestone requires nested quantization with the upstream default statistics option.
The [first actual nested result](docs/changes/r4-nested-colab-failure.md) passed 52 cases and failed 27 cases under unchanged criteria.
Arithmetic and state representation repairs remain necessary before saved-state execution.
M4 and later milestones remain unqualified.

The [bounded Pallas experiment](docs/changes/r6-bf16-colab.md) passed five native and reference diagnostic cases on Colab.
The [public API preparation](docs/changes/r6-public-cloud-preparation.md) passed 118 canonical controls for its separate 15-case experiment.
Actual public integration and compiler memory records remain required for M6.
The [compiler flag failure](docs/changes/r6-compiler-flag-colab-failure.md) remains preserved with all five device cases unexecuted.

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
