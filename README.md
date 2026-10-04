# bitsandbytes-TPU

A TPU backend for the original bitsandbytes API.

## Project status

This project is at the initial implementation stage.
The first target is NF4 quantization and `bitsandbytes.nn.Linear4bit` on a TPU.
The project does not yet have an accepted TPU result for this backend.

The project name is **bitsandbytes-TPU**.
The Python package name is `bitsandbytes-tpu`.
The Python module name is `bitsandbytes_tpu`.

## Research goal

Keep the original bitsandbytes classes and public API.
Use PyTorch/XLA to execute their operations on a TPU.
Then use Pallas to reduce the time and memory necessary for NF4 matrix multiplication.

The required results are:

1. An installable backend with automatic registration.
2. Correct NF4 data and original `QuantState` data.
3. Correct `Linear4bit` output, gradients, and saved model data.
4. Correct nested quantization for `compress_statistics=True`.
5. A QLoRA test with a new process for model restoration.
6. A Pallas kernel with measured time and memory.
7. Repeatable results on Colab and Kaggle.

A CPU test does not establish TPU operation.
A reference implementation does not establish a reduction in memory.
Each result needs its own test evidence.

## Initial configuration

| Item | Initial target |
| --- | --- |
| Upstream bitsandbytes | Commit `833649043474794b8fe7a4136e0c40faf077b2e0` |
| Device interface | Native PyTorch/XLA |
| Quantization | NF4, block size 64 |
| Packed storage | `torch.uint8` |
| Input data types | FP32 and BF16 |
| First test | Quantization without nested quantization |
| First usable milestone | Nested quantization and the original `Linear4bit` API |

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
Installation instructions will follow the first accepted package test.

Project documents use ASD-STE100 Issue 9 as their writing standard.
Code identifiers retain their original spelling.
Refer to the writing guide for the review procedure and its limits.

## License

The project license is [Apache License 2.0](LICENSE).
Third-party source retains its applicable license and notices.
