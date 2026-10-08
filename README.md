# bitsandbytes-TPU

A TPU backend for the upstream bitsandbytes API.

## Project status

The initial package milestone is completed.
The reviewer repeated 71 CPU and package tests.
Accepted milestones: **1 of 8**.
The first target is NF4 quantization and `bitsandbytes.nn.Linear4bit` on a TPU.
The latest Colab comparison passed its fixed FP32 gates with native precision `high` and `highest`.
The `default` mode reproduced the earlier forward and input-gradient failures.
The next correctness experiments will use `highest` before graph construction.
The reviewed CPU-to-XLA transfer correction passed local controls and still requires device tests.
Two Colab allocation requests ended with response timeouts before the transfer experiment could execute.
BF16 forward controls passed; BF16 gradients remain untested.
Four executed module routes passed reconstruction within the same process.
This does not establish restoration in a new process.
The full 42-case API test and milestones M2 and M3 remain unqualified.
Refer to [the precision result](docs/changes/r1-precision-colab.md).
Refer to [the allocation repeat](docs/changes/r2-colab-allocation-repeat.md) for the current execution blocker.
The [web diagnostic](docs/changes/r2-web-allocation-diagnostic.md) then obtained and closed a TPU runtime.
A longer bounded CLI allocation wait requires a separate transport review.

The [October 8 source inspection](docs/research/2026-10-08/README.md) covers each design area before further implementation.
It contains current documentation, original papers, fixed source revisions, and explicit version limits.
The [decision table](docs/research/2026-10-08/decisions.md) defines the next work and its acceptance requirements.
The inspection did not change the runtime, code, or numerical tolerances.

Earlier failure records remain available:

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
