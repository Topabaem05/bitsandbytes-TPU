# bitsandbytes API and NF4 research

Inspection date: 2026-10-08. Scope: M1, M3, M4, and M5.
This record contains source inspection. It contains no new device tests or performance measurements.
Use [the source manifest](bitsandbytes-sources.json) for source dates, revisions, limitations, and local evidence hashes.

## Current versions

| Item | Observable value | Actual source date |
| --- | --- | --- |
| Latest GitHub stable release | `0.50.2` | Published 2026-08-27, 00:13:56 UTC |
| Latest PyPI package | `0.50.2` | First current wheel uploaded 2026-08-27, 00:10:48 UTC |
| Current `main` | `833649043474794b8fe7a4136e0c40faf077b2e0` | Commit dated 2026-09-03, 18:12:53 UTC |
| Project pin | Same as current `main`; `0.50.3.dev0` | Same source commit |

GitHub and PyPI give the same latest stable version.
The inspection date is later than these publication dates.
The continuous wheel uses a fixed filename version that does not identify the installed source version.
Sources: [BNB-01: release](https://github.com/bitsandbytes-foundation/bitsandbytes/releases/tag/0.50.2), [BNB-02: PyPI](https://pypi.org/project/bitsandbytes/0.50.2/), and [current source commit](https://github.com/bitsandbytes-foundation/bitsandbytes/commit/833649043474794b8fe7a4136e0c40faf077b2e0).

The downloaded `nn/modules.py` files from stable and `main` have equal SHA-256 values.
Both keep `self.data = w_4bit` in `Params4bit._quantize`.
Thus, the examined releases provide no source change that resolves the recorded transfer failure.
This conclusion concerns the examined source; it does not certify another runtime.
Sources: [BNB-06: current module](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/nn/modules.py#L381) and [stable module](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/nn/modules.py#L381).

## Public API and registration: M1

The official API keeps `Linear4bit`, `Params4bit`, and `LinearNF4`.
`Linear4bit` defaults to `compress_statistics=True`, `quant_type="fp4"`, and `quant_storage=torch.uint8`.
Select NF4 explicitly for this project.
The documented transfer example uses CUDA; it does not establish XLA compatibility.
Source: [BNB-03: official API](https://huggingface.co/docs/bitsandbytes/main/en/reference/nn/linear4bit).

Package import loads and calls each `bitsandbytes.backends` entry point.
The declared device set does not include `xla`.
Plugin discovery therefore supplies a registration mechanism, without a TPU support guarantee.
Source: [BNB-04: package import](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/__init__.py#L48).

Keep the existing NF4 schemas and their output overloads.
Nested scales also need `quantize_blockwise.default`, `dequantize_blockwise.default`, and `dequantize_blockwise.out`.
The blockwise output overload mutates `out` and returns no tensor.
GEMM supplies optional arguments for scale codes, the map, and the offset.
Fake implementations describe structures; they do not show device execution.
Source: [BNB-05: operator definitions](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/_ops.py#L239).

## NF4 codes and the reference boundary: M1 and M4

The default Python quantizer converts inputs to FP32 and uses midpoint boundaries.
Equal midpoint inputs select the lower code.
It packs the first code in the high nibble and the second code in the low nibble.
An odd input count adds a zero input before code selection.
Complete zero blocks retain scale zero; partial zero blocks return a scale clamped to `1e-38`.
The NF4 map represents zero with code 7, minus one with code 0, and one with code 15.
Sources: [NF4 codebook](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/backends/utils.py#L14) and [BNB-08: Python default](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/backends/default/ops.py#L225).

The CUDA NF4 tree uses fixed thresholds and strict greater-than comparisons.
Use its exact FP32 constants for an independent boundary comparison.
Do not assume that its eight-bit blockwise quantizer has the same tie rule.
Source: [BNB-08: CUDA boundaries](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/csrc/kernels.cu#L110).

The native CPU blockwise quantizer first rounds normalized values into a 65,536-entry lookup table.
Its zero blocks return code zero and scale zero.
This path differs from the Python default algorithm.
A CPU tensor alone does not identify the reference implementation.
Source: [BNB-09: native CPU](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/csrc/cpu_ops.cpp#L496).

Design consequence: Record the native library and dispatch table before reference generation.
Keep Python default, native CPU, and CUDA results separate.
Keep midpoint neighbors, zero blocks, partial blocks, and odd packing counts in the fixed tests.

## Default nested state: M4

The functional NF4 quantizer defaults to `compress_statistics=False`.
With compression enabled, it subtracts the mean first-level scale and quantizes centered scales in blocks of 256.
The second quantizer uses the signed dynamic map, with 256 entries.
This map differs from the 16-entry NF4 map.
`state.absmax` then contains uint8 scale codes; `state2.absmax` contains second-level FP32 scales.
The offset restores the mean after second-level decoding.
Source: [BNB-07: functional path](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/functional.py#L938).

Nested GEMM accepts second-level block size 256.
Its default implementation decodes scales, adds the offset, and restores weights before the linear operation.
Sources: [BNB-10: GEMM call](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/autograd/_functions.py#L303) and [BNB-08: default GEMM](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/backends/default/ops.py#L323).

Design consequence: The initial uncompressed experiment does not cover the default module configuration.
The existing nested proposal remains a proposal.
The local prototype remains unadopted.
These statements describe project status, not upstream results.

## Transfer, saved state, and gradients: M3

For an unquantized parameter, `.to(device)` calls `_quantize`.
That method moves source data, quantizes it, and assigns the packed tensor to the existing parameter data.
`from_prequantized` instead creates a parameter subclass from data already moved to the requested device.
Source: [BNB-06: parameter paths](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/nn/modules.py#L355).

The [third TPU run](../../changes/task2-colab-native.md) reports incompatible tensor types at the data assignment.
The source is consistent with that failure location.
Source inspection alone does not prove target-device construction or public restoration.
The [latest route diagnostic](../../changes/task2-colab-routes.md) records actual execution of both alternatives on October 4.
Its [JSON report](../../../experiments/2026-10-04-bitsandbytes-tpu/results/routes-colab.json) separates execution, numerical gates, and reconstruction.

| Route | FP32 numerical gates | BF16 numerical gates |
| --- | --- | --- |
| `target_constructor` | Forward and input gradient: `FAIL` | `PASS` |
| `from_prequantized` | Forward and input gradient: `FAIL` | `PASS` |

Both alternatives executed for both data types.
The `from_prequantized` route used the sealed CPU checkpoint; it does not show TPU weight quantization.
All four alternatives passed reconstruction within the same process.
Restoration in a new process did not execute.
Both CPU-to-XLA transfer cases retained the incompatible tensor type error.
The full API42 matrix and M3 remain unqualified.
The FP32 failure cause remains unknown, and the fixed tolerances remain unchanged.
Keep CPU-to-XLA transfer open until its own test passes.

`Linear4bit` saves packed `QuantState` components with the weight and bias.
`QuantState` supports packed and unpacked dictionaries, including nested map, scales, block size, dtype, and offset.
The packed key includes `quant_state.bitsandbytes__nf4` for NF4.
Sources: [BNB-06: module save](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/nn/modules.py#L593) and [BNB-07: state dictionaries](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/functional.py#L494).

Serialization extracts the offset scalar.
Metadata packing uses host JSON bytes; metadata unpacking uses a CPU NumPy conversion.
These operations need a separate checkpoint boundary in the device probe.
Sources: [BNB-07: scalar extraction](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/functional.py#L565) and [metadata conversion](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/utils.py#L166).

`MatMul4Bit.backward` restores weights for the input gradient.
It returns no packed-weight gradient and computes a bias gradient when required.
Gradient requirements select the autograd path.
Source: [BNB-10: autograd](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/autograd/_functions.py#L332).

Design consequence: Restore through the public parameter and state APIs, then compare outputs in a new process.
Do separate input-gradient and bias-gradient comparisons for rank-two and rank-three inputs.
Do not use state object equality as the only restoration result.

## QLoRA meaning: M5

*QLoRA: Efficient Finetuning of Quantized LLMs*, arXiv:2305.14314v1, defines NF4, double quantization, and adapter training through a frozen quantized base.
The [current arXiv record](https://arxiv.org/abs/2305.14314) lists only v1, submitted on 2023-05-23.
Thus, v1 remains the latest observable arXiv revision on this inspection date.
Its NF4 construction uses normal-distribution quantiles and an exact zero.
Double quantization compresses centered scales, with blocks of 64 weights and 256 scales in the described configuration.
The stored weights and computation data type serve different purposes.
Base weights remain fixed, but input gradients must pass through their dequantized values to train adapters.
Source: [BNB-11: QLoRA, Sections 2 and 3](https://arxiv.org/html/2305.14314v1).

*8-bit Optimizers via Block-wise Quantization*, arXiv:2110.02861v2, supplies background for independent scales and a nonlinear dynamic map.
Its optimizer results do not establish nested NF4 byte behavior.
Source: [BNB-12: original paper](https://arxiv.org/abs/2110.02861v2).

Design consequence: Keep the planned FP32, SGD, twenty-step test as a small API behavior test.
It does not reproduce the paper's large-model experiments or paged optimizer behavior.
Compare base bytes and nested state before and after all steps.
Restore adapters, base state, and optimizer state after step ten.
Compare the resumed run with the uninterrupted run under the existing fixed criteria.

## Checks made and remaining limits

The research used official documentation, release metadata, immutable source links, and original papers.
GitHub REST metadata and PyPI metadata were downloaded into the local research cache.
Source files were downloaded, hashed, and examined.
Stable and pinned module files were compared by bytes.
The source manifest was parsed and its required fields were examined.
Markdown sentence lengths were examined as a limited writing check.
This check does not establish full ASD-STE100 conformity.

No package installation, backend mutation, cloud allocation, or new numerical experiment occurred.
This research added no behavior tests.
The referenced October 4 diagnostic contains earlier CPU and actual TPU tests.
The sources do not establish resumed training, TPU memory reduction, or measured performance.
