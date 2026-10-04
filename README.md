# bitsandbytes-TPU

A TPU backend for the upstream bitsandbytes API.

## Project status

The initial package milestone is completed.
The reviewer repeated 71 CPU and package tests.
Accepted milestones: **1 of 8**.
The first target is NF4 quantization and `bitsandbytes.nn.Linear4bit` on a TPU.
The project does not yet have an accepted TPU result for this backend.
The first Colab run passed the TPU runtime probe and calculated all 42 CPU reference cases.
NF4 execution stopped at a PyTorch functionalization assertion before the first case completed.
Refer to [the first Colab result](docs/changes/task2-colab-first.md).
The repeat with the first functionalization patch stopped at tensor creation.
It also completed zero NF4 cases.
Refer to [the repeat result](docs/changes/task2-colab-functionalized.md).

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
