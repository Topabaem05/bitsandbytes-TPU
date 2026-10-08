# BF16 matrix operands

The actual gather experiment passed both FP32 native cases.
Its BF16 matrix operation failed compiler type validation.
The operation supplied BF16 operands with FP32 contract precision.

The corrected kernel retains decoded BF16 weight rounding.
It then converts both matrix operands to FP32.
It retains `highest` precision, FP32 accumulation, and the original output data type.
The fixed runtime, inputs, and numerical criteria remain unchanged.

The pinned [JAX lowering](https://github.com/jax-ml/jax/blob/5712de44e97c455faed1fd45532e821ca66d025a/jax/_src/pallas/mosaic/lowering.py#L2188-L2217) maps `HIGHEST` to FP32 contract precision.
Its [conversion rule](https://github.com/jax-ml/jax/blob/5712de44e97c455faed1fd45532e821ca66d025a/jax/_src/pallas/mosaic/lowering.py#L2279-L2295) extends BF16 operands to FP32.
These sources support the type correction but do not establish actual libtpu acceptance.

The new generation changes seven production files.
The adapter binds the corrected kernel hash.
The device diagnostic binds the corrected source-pins hash.
The native manifest and cloud contract bind all dependent sources.
The reviewer rejected an initial packet that omitted two internal pin updates.
That rejected packet remains separate from the corrected packet.

The reviewer repeated 56 controls and verified 82 frozen source files.
The controls include CPU interpretation, typed lowering, layout passes, exact rounding, source admission, and packet integrity.
All four retained native input sets produced exactly the original CPU interpreter output bytes.
The unchanged numerical criteria also passed for these inputs.
The incorrect rounding, nibble order, scale order, internal pins, and archive controls failed as required.
All six local control groups closed.

Two genuine private builds produced identical 73-member packets.
The canonical build after adoption produced the same bytes.
Its frontend preflight and complete source readback passed.
The packet contains 1,057,813 bytes.
Its SHA-256 is `469f232f533807c772cd421a9d3a154cd0c43dbf715e12b8a62b7e2d471bd709`.

The next experiment requires a fresh qualified Linux CPU oracle and actual TPU results.
Native acceptance remains absent, and M6 remains unqualified.
Compiler memory records, public integration, and performance measurements remain required.
Refer to [the selected preparation result](../../experiments/2026-10-04-bitsandbytes-tpu/results/native-bf16-preparation.json).

The reviewer inspected sentence length, document links, and technical terms.
The complete standard dictionary was unavailable.
This review does not establish independent ASD-STE100 conformity.
