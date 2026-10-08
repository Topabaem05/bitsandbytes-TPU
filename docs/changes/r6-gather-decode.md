# Native decode layout correction

The reviewer accepted this preparation on 2026-10-09, Korea time.
The correction addresses the [actual shape-cast failure](r6-mosaic-layout-failure.md).
Actual TPU compilation and numerical results remain required.
M6 remains unqualified.

The previous decoder stacked high and low nibbles before a three-dimensional reshape.
The TPU compiler rejected that reshape.
The new decoder uses a two-dimensional gather.
The codebook, tile dimensions, grouped operands, dot precision, and numerical criteria remain unchanged.

For output column `j`, the gather selects packed byte `j // 2`.
Even columns select the high nibble, and odd columns select the low nibble.
The source contains two copies of the selected 64-byte row.
All gather indices are between zero and 63, so they select only the first copy.
Scale column `j // 64` supplies the factor for each output column.
This mapping preserves the original byte, nibble, and scale order.

The pinned [JAX lowering](https://github.com/jax-ml/jax/blob/5712de44e97c455faed1fd45532e821ca66d025a/jax/_src/pallas/mosaic/lowering.py#L2413) supports this gather pattern.
The source, indices, and result have shape `[128,128]` and 32-bit elements.
The local layout passes produce matching layouts with zero offsets and a native gather extent of 128.
These properties match the pinned [layout constraints](https://github.com/jax-ml/jax/blob/5712de44e97c455faed1fd45532e821ca66d025a/jaxlib/mosaic/dialect/tpu/transforms/apply_vector_layout.cc#L3303).
The local CPU wheel does not establish acceptance by the installed TPU compiler.

The reviewer independently repeated all 52 controls.
They cover CPU interpretation, nibble order, scale order, source admission, conversion, and local layout passes.
The controls include FP32, BF16, multiple groups, both halves, all byte values, and the actual failed case inputs.
Incorrect nibble and scale order fail the fixed numerical gate.
The old kernel reproduces its shape-cast error in the local layout passes.
The corrected kernel passes those passes for four frontend cases.
All six control process groups closed.

An earlier repeat-based candidate remains rejected.
Its CPU interpretation and pinned compiler canonicalization use different element orders.
The retained CPU success therefore does not qualify that candidate.

The accepted generation is `mosaic-serde7-gather-v1`.
Seven files contain the decoder change and its required source bindings.
The other 23 native sources remain unchanged.
The canonical packet contains 73 members and 1,057,741 bytes.
It matches the inspected private packet exactly and passed the pinned frontend preflight.
The builder binds the accepted M4 cloud baseline explicitly.

Generate a fresh qualified Linux CPU oracle before the next native TPU phase.
Inspect the actual graph, execution records, and output values.
Refresh the separate compiler and public integration preparations for this kernel generation.
Memory use, public backward integration, and performance remain unqualified.

Refer to the [selected preparation record](../../experiments/2026-10-04-bitsandbytes-tpu/results/native-gather-preparation.json).
