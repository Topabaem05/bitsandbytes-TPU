# bitsandbytes-tpu nested-v1 candidate

This is the explicit plugin variant for the `nested-79` experiment. Its six Python source files register seven existing bitsandbytes operators for XLA. The canonical non-nested package is a separate source variant.

The candidate supports NF4 packed weights with block size 64 and uint8 storage. Nested statistics use float32 centered scales, a block size of 256, a dynamic code map, a scalar offset, and a second `QuantState`. Compute types are float32 and bfloat16. GEMM accepts rank-two and rank-three inputs. The default NF4 module uses `compress_statistics=True`.

The caller must supply finite input, weight, and bias values. Normalization scales and restored NF4 scales must be finite and nonnegative. The scalar offset must be finite. The dynamic code map must have 256 finite, sorted float32 entries in a contiguous tensor. Centered scale values can be signed. The packed weights must use the pinned contiguous column layout. These are caller value preconditions. The kernels do not promise value validation or errors for values outside these preconditions. They reject unsupported structural options, shapes, layouts, types, devices, and scale counts.

The `bitsandbytes.backends` entry point registers these XLA overloads:

- `quantize_blockwise`
- `dequantize_blockwise`
- `dequantize_blockwise.out`
- `quantize_4bit`
- `dequantize_4bit`
- `dequantize_4bit.out`
- `gemm_4bit`

Registration keeps the upstream classes, methods, and operator schemas. It does not replace CPU or CUDA kernels. The two mutable dequantization overloads retain output identity and their None result through functionalization.

The experiment uses upstream revision `833649043474794b8fe7a4136e0c40faf077b2e0`, the exact [parameter transfer patch](../../../patches/params4bit-xla-v1.json), and the fixed [runtime lock](../runtime/requirements.lock.json). It requires Torch `2.9.0+cpu`. The packet admits the complete patched upstream inventory, all six plugin source files, the packaging files, and the seven [operator schemas](../nested-schemas.json). The source manifest fixes the admitted bytes. The backend does not apply patches during import or replace class methods during execution.

The CPU oracle executes bodies from the pinned upstream default Python backend. It does not use native CPU dispatch. This oracle supplies the byte reference for packed weights, scale codes, maps, offset, and second scales. It is not a native CPU or CUDA golden result. Decoded values, forward values, and FP32 gradients use the unchanged numerical gates. BF16 gradients do not execute.

M2 and M3 have accepted device records for the earlier non-nested source variant. Actual `nested-79` TPU execution for this candidate is pending. The 79-case probe does not test saved-state serialization. It cannot complete M4 by itself. Refer to the [project status](../../../README.md) and [research plan](../../../docs/research-plan.md) for acceptance scope.

The kernels use Torch operations on the input device. Reference GEMM creates a dense weight tensor. No fused execution, memory benefit, performance result, or training result is claimed.
