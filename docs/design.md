# Design

## Required behavior

An application imports the upstream `bitsandbytes` package.
The backend adds TPU implementations for the existing PyTorch operators.
The application keeps `Linear4bit`, `Params4bit`, and `QuantState`.

The first device path uses native PyTorch/XLA.
The backend does not replace the application with a JAX program.

## Components

| Component | Function |
| --- | --- |
| `registration.py` | Add implementations for the XLA dispatch key. |
| `compatibility.py` | Compare the upstream version and operator schemas with the required values. |
| `reference.py` | Execute NF4 operations with PyTorch tensors on the selected device. |
| Pallas bridge | Connect a subsequent Pallas kernel to a PyTorch/XLA custom call. |
| Runtime resolver | Select fixed Linux wheels and record their SHA-256 values. |
| Experiment probe | Compare public API results with sealed reference data. |

The package uses the `bitsandbytes.backends` entry point.
Package import must not initialize a TPU client.
Registration must not replace CPU or CUDA implementations.
A second registration must not change the first registration.

## Operator boundary

The first implementation uses these existing operators:

- `bitsandbytes::quantize_4bit`
- `bitsandbytes::dequantize_4bit`
- `bitsandbytes::gemm_4bit`.

The backend must keep each operator schema, including output arguments.
The backend must keep the upstream packed format.
Each byte contains two NF4 codes, with the first code in the high nibble.
The implementation must keep zero blocks, partial blocks, and the upstream quantization boundaries.

The first tests use NF4, `uint8` storage, and block size 64.
They use finite FP32 and BF16 input data.
The caller must supply finite inputs and finite nonnegative scales.
The backend does not promise value validation for data outside these preconditions.
Incorrect structures still cause explicit errors.
The experiment keeps its numerical tolerances and rejects CPU fallback.
Nested quantization is a separate implementation step.

`Linear4bit` uses `compress_statistics=True` by default.
Thus, the first test without nested quantization is not the first usable milestone.
The milestone requires the upstream default for this option.
The application must select `quant_type="nf4"` explicitly.

## Reference path

The reference implementation keeps tensor operations on the selected device.
It can restore the full weight matrix before matrix multiplication.
This path shows numerical behavior.
It does not show the intended memory reduction.

The tests must detect an unintended CPU execution path.
An operator registration result alone does not show TPU execution.

## Pallas path

The subsequent kernel reads packed weights from TPU memory.
It restores a weight tile, applies its scales, and executes matrix multiplication.
It must not store a full restored weight matrix in TPU memory.

The first tile candidate has dimensions 128 by 128 by 128.
This candidate has no measured performance claim.
Partial tiles and other layouts need explicit tests or an explicit reference path.
Each result must identify the selected path.

The source inspection proposes packed layout `[Q,N,128]`, where Q=K/256.
This layout moves the squeezed group axis outside the final two block dimensions.
It satisfies one examined compiler rule but has not compiled.
The packed transpose can require additional storage and execution time.
Include those costs in the complete public call.
Refer to [the Pallas source inspection](research/2026-10-08/pallas.md).

PyTorch/XLA owns the TPU client.
The bridge must apply the XLA JAX import guard before JAX import.
The bridge uses `torch_xla.experimental.custom_kernel.make_kernel_from_pallas`.
The initial runtime does not require TorchTPU, Helion, or Qwix.

## Runtime candidate

| Package | Version |
| --- | --- |
| Python | 3.12, Linux x86-64 |
| PyTorch | 2.9.0+cpu |
| PyTorch/XLA | 2.9.0 |
| libtpu | 0.0.21 |
| JAX | 0.7.1 |
| jaxlib | 0.7.1 |

The fixed runtime passed the complete M2 and M3 experiments on Colab V6E1.
M2 passed 42 API cases and four public CPU-to-XLA transfers with `highest` precision.
M3 passed eight saved-state cases after restoration in a new TPU process.
Refer to [the transfer result](changes/r2-transfer-colab.md) and [the state result](changes/r3-state-colab.md).
The [earlier route failures](changes/task2-colab-routes.md) remain available.
Use the base JAX packages.
Do not install the JAX TPU extra for this configuration.
Its libtpu requirement differs from the PyTorch/XLA requirement.

Current package releases do not define a qualified replacement runtime.
The source inspection keeps published releases, development revisions, and the fixed experiment configuration separate.
Set native XLA precision to `highest` before graph construction for dependent correctness experiments.
The fixed numerical requirements remain unchanged.
Refer to [the XLA source inspection](research/2026-10-08/xla.md).

## Sources

- [Pinned bitsandbytes source](https://github.com/bitsandbytes-foundation/bitsandbytes/tree/833649043474794b8fe7a4136e0c40faf077b2e0)
- [PyTorch/XLA 2.9.0 package configuration](https://github.com/pytorch/xla/blob/v2.9.0/setup.py)
- [PyTorch/XLA Pallas bridge](https://github.com/pytorch/xla/blob/v2.9.0/torch_xla/experimental/custom_kernel.py)
- [PyTorch/XLA JAX import guard](https://github.com/pytorch/xla/blob/v2.9.0/torch_xla/_internal/jax_workarounds.py)

The [2026-10-08 source collection](research/2026-10-08/README.md) expands these references for all design areas.
It records source dates, exact revisions, limits, and required follow-up tests.
