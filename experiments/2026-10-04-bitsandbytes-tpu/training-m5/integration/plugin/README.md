# Private nested-exact backend generation

This candidate supplies seven upstream XLA backend bodies and two explicit composite operators. It uses NF4 blocks of 64 and nested scale blocks of 256. Finite word arithmetic preserves the pinned default Python operation order, including complete-block reciprocal/multiply and partial-block division. Native CPU quantization is a separate reference.

The composite operators own mean/centering and restored-scale arithmetic. Static source branches call them from the original upstream functions. The original CPU branches, public classes and QuantState format remain unchanged. No public method is replaced at runtime. General nested blockwise quantization and decoding on XLA are outside this generation and reject. FP32 and BF16 weights are supported; BF16 gradients are not qualified.

Callers must admit finite inputs and intermediates, the fixed sorted finite maps, positive floor-clamped normalizers, and complete source maps. Mean input is contiguous rank1 FP32 with 1..513 first scales. The fixed corpus meets these bounds. Nonfinite values and arbitrary larger reductions need a separate reviewed contract. Registration requires the exact static upstream patch, accepted M2 module overlay and Torch 2.9.0+cpu.

Install only in a fresh admitted runtime with the matching patched upstream wheel. This variant uses the bitsandbytes_tpu module name and must not coexist with another distribution providing that module. The canonical plugin and prior nested-v1 generation remain unchanged. Source/wheel gates must run before import.

This is preparation only: NOT_ADOPTED, TPU NOT_RUN, M4 NOT_QUALIFIED. Dense GEMM remains a Torch reference path. No fused-kernel, memory or performance result is claimed.
