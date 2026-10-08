# Pallas source inspection for M6 and M7

Checked date: 2026-10-08. Status: SOURCE REVIEW ONLY.
M6 and M7 remain NOT_RUN in this inspection.
No kernel compilation, TPU execution, or performance measurement occurred.

## Decision

Keep the current PyTorch/XLA bridge and runtime for the first Pallas qualification.
The source supports bounded tile decoding as a design candidate.
It does not establish numerical correctness, memory reduction, or faster execution.
The leading group axis remains an uncompiled candidate.

Use the public bitsandbytes API as the boundary.
Keep the current milestone order, numerical tolerances, and test inputs.
Complete M5 before the M7 comparison.

The reviewer supplied the common wiki search result from revision 648.
That search found no applicable Pallas source record.
This worker did not repeat the search or write wiki records.

## Version boundary

The qualified candidate uses PyTorch/XLA 2.9.0, libtpu 0.0.21, and JAX/jaxlib 0.7.1.
This source inspection does not change those versions.

| Item | Observable source | Date and meaning |
| --- | --- | --- |
| PyTorch/XLA source | `5fab7053df86c8d503b98d9e7202ca8b8d4978c7` | Commit time: 2025-11-17 UTC. |
| PyTorch/XLA latest release endpoint | `v2.9.0` | Publication time: 2025-11-17 UTC. This is the endpoint result, not a claim about nightly builds. |
| JAX 0.7.1 source | `5712de44e97c455faed1fd45532e821ca66d025a` | Tag commit time: 2025-08-19 UTC. |
| JAX/jaxlib 0.7.1 packages | PyPI release files | First upload date: 2025-08-20 UTC. |
| JAX/jaxlib latest stable packages | `0.11.2` | PyPI upload date: 2026-09-17 UTC. |
| JAX observed main | `0f1b83b4dd27cc259caa54d6df3ee9dfb81404e8` | Commit time: 2026-10-08 04:37:38 UTC. This is a source snapshot. |

The [JAX change log](https://docs.jax.dev/en/latest/changelog.html#jax-0-11-2-september-17-2026) confirms the 0.11.2 release date.
[JAX package metadata](https://pypi.org/pypi/jax/json) and [jaxlib package metadata](https://pypi.org/pypi/jaxlib/json) supply the upload times.
The [PyTorch/XLA release endpoint](https://api.github.com/repos/pytorch/xla/releases/latest) supplies its publication time.
The source manifest records the separate commit metadata URLs.

Current documentation is useful for design inspection.
It does not prove that new interfaces operate with JAX 0.7.1 or libtpu 0.0.21.
JAX 0.11.2 package metadata requires `libtpu==0.0.48.*` for its TPU extra.
That requirement also differs from the qualified runtime.
For example, observed main supports an additional `lax.Precision.HIGH` lowering.
The pinned lowering rejects that precision value.
Keep `lax.Precision.HIGHEST` for the proposed FP32 arithmetic. See PAL-03.

## Bridge and client ownership

[PAL-01: pinned bridge](https://github.com/pytorch/xla/blob/5fab7053df86c8d503b98d9e7202ca8b8d4978c7/torch_xla/experimental/custom_kernel.py#L211) supplies `make_kernel_from_pallas`.
Its trace function replaces positional Torch tensors with storage-free `jax.ShapeDtypeStruct` objects.
It retains the Torch tensors as custom-call operands.
The output callback returns a list of shape and dtype pairs.
Tensor containers and tensor keyword arguments do not have this operand handling.
The source rejects enabled `XLA_USE_BF16`.

The bridge extracts a backend configuration from a `stablehlo.custom_call` operation.
It does not execute an arbitrary surrounding JAX graph.
Use one Pallas call with the intended tensor operand order.
Keep operand preparation in the Torch/XLA graph.
Bind the extracted configuration to the executed Torch/XLA call during qualification.
A matching call name alone cannot prove that binding.

The [pinned import guard](https://github.com/pytorch/xla/blob/5fab7053df86c8d503b98d9e7202ca8b8d4978c7/torch_xla/_internal/jax_workarounds.py#L9) initializes the PyTorch/XLA computation client.
Call `jax_import_guard()` before JAX import.
Keep the bridge's `requires_jax` and `jax_env_context` behavior.
The source supports PyTorch/XLA client ownership.
It does not establish compatibility with the backend's functionalization wrapper.
That interface needs a direct integration test.

## Packed layout and tile arithmetic

[PAL-02: pinned BlockSpec source](https://github.com/jax-ml/jax/blob/5712de44e97c455faed1fd45532e821ca66d025a/jax/_src/pallas/core.py#L356) maps `None` to a squeezed axis.
The kernel Ref shape omits that axis.
However, [PAL-03: pinned Mosaic lowering](https://github.com/jax-ml/jax/blob/5712de44e97c455faed1fd45532e821ca66d025a/jax/_src/pallas/mosaic/lowering.py#L590) tests the complete block shape.
It treats the squeezed axis as size one.
The final two block dimensions must meet the 8/128 divisibility rule or cover their complete array dimensions.

Thus, `[N,Q,128]` with `(128,None,128)` fails this source rule when Q is greater than one.
The proposed `[Q,N,128]` layout moves that axis outside the final two dimensions.
Here Q=K/256, and N is the number of output rows.
The scales candidate uses `[Q,N,4]` with `(None,128,4)`.
Both candidates satisfy this particular source rule.
This conclusion is source analysis, not successful compilation.

The public packed format needs a grouped transpose for these candidates.
The compiler can require a complete packed buffer for that operation.
Include that buffer in the memory record.
Measure its preparation cost independently and within the complete public call.

The same pinned lowering maps `lax.Precision.HIGHEST` to `#tpu.contract_precision<fp32>`.
It also limits supported gather patterns.
Use the fixed NF4 values and a examined selection method.
Do not assume that an arbitrary dynamic codebook gather will compile.
The decode, narrow-type slices, casts, and dot operation still need compiler tests.

[PAL-04: pinned TPU matmul example](https://github.com/jax-ml/jax/blob/5712de44e97c455faed1fd45532e821ca66d025a/jax/experimental/pallas/ops/tpu/matmul.py) provides a useful accumulator pattern.
It uses FP32 accumulator storage and sequential K traversal.
The example does not decode NF4 or preserve the bitsandbytes packed format.
Its source contains integer dtype branches; those branches do not establish NF4 arithmetic.

[PAL-05: current TPU restrictions](https://docs.jax.dev/en/latest/pallas/tpu/details.html) explain HBM-to-VMEM transfers and the same block dimension rule.
They warn that FP32 dot inputs can use BF16 precision unless FP32 precision is requested.
These current restrictions support the design inspection.
The pinned source remains the implementation authority for the qualified runtime.

[PAL-06: current matmul guide](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html#fused-right-hand-side-transpose) shows a fused right operand transpose with `lax.dot_general`.
This can avoid a separate transpose of the decoded tile.
The guide also excludes operand relayout from some kernel timings.
That benchmark boundary cannot establish the cost of this backend's packed preparation.
The 128 tile candidate remains a starting point.
The guide's tile and pipeline discussion requires a later measured size comparison.

## Related implementations

[PAL-07: PyTorch/XLA quantized matmul source](https://github.com/pytorch/xla/blob/5fab7053df86c8d503b98d9e7202ca8b8d4978c7/torch_xla/experimental/pallas_kernels/quantized_matmul_kernel.py) supplies a first-party TPU example.
It uses tiled operands, scales, scratch storage, and selected block sizes.
Its wrapper can select the XLA quantized path when tuned sizes are unavailable.
The kernel rejects `quant_block_size` in the examined entry.
It does not implement the NF4 codebook or canonical high-nibble-first format.
Use its structure as evidence for tile design.
Keep its integer arithmetic and path selection outside the NF4 correctness claim.

[PAL-11: Qwix NF4 numerics](https://github.com/google/qwix/blob/457417660d01099aeef35a2053d9c44ee88c87ef/qwix/_src/core/numerics.py) support NF4 conversion.
The [QLoRA example](https://github.com/google/qwix/blob/457417660d01099aeef35a2053d9c44ee88c87ef/docs/source/lora.md) uses JAX/Flax state and `QArray` values.
The numerics source excludes NF4 from decoding after the dot operation.
NF4's nonlinear codebook requires decoding before the dot operation.
The [contributed TPU matmul](https://github.com/google/qwix/blob/457417660d01099aeef35a2053d9c44ee88c87ef/qwix/contrib/kernels/quantized_matmul.py) multiplies low-precision dot results by scales.
It has no NF4 codebook decode in the examined kernel body.
Thus, Qwix provides related numerics and pipeline examples.
It does not establish compatibility with bitsandbytes bytes, state, or operator schemas.
Its QLoRA support does not qualify this backend's M5.

[PAL-12: Tokamax architecture](https://github.com/openxla/tokamax/blob/d4fe571944a30734fc267f4c02c382f803a5b518/README.md) supports explicit implementation selection and serialized tuning results.
Its operation classes separate the reference implementation from accelerator implementations.
This is useful for selected-path records and fixed tuning configurations.
Its [current package configuration](https://github.com/openxla/tokamax/blob/d4fe571944a30734fc267f4c02c382f803a5b518/pyproject.toml) requires JAX/jaxlib 0.11.0 or later.
That requirement conflicts with the qualified 0.7.1 runtime.
No examined Tokamax source establishes a drop-in bitsandbytes NF4 operator.
Do not add Tokamax or Qwix as dependencies for this candidate.

## Compiler evidence, memory, and timing

[PAL-09: OpenXLA dump documentation](https://openxla.org/xla/hlo_dumps) describes HLO dumps and Mosaic dumps.
The page identifies before/after optimization and buffer assignment records.
Its Mosaic flag is `--xla_mosaic_dump_to`.
The page's last update date is 2026-05-16 UTC.
Flag acceptance and artifact completeness remain unverified for libtpu 0.0.21.

Keep four independent evidence records:

1. JAX lowering and the exact extracted custom-call configuration.
2. Executed Torch/XLA HLO and operand layout preparation.
3. Mosaic body and buffer assignment, including bounded decode storage.
4. Device memory samples with their measurement interval and scope.

An outer HLO custom call can hide its internal storage.
Buffer assignment alone therefore cannot establish bounded internal decode storage.
Examine the kernel body and the outer allocations together.
Reject a complete floating-point `[N,K]` forward weight buffer.
Use a dense decode reference as an incorrect memory fixture.
Keep compiler memory estimates separate from device memory samples.

[PAL-08: pinned Torch/XLA memory and synchronization source](https://github.com/pytorch/xla/blob/5fab7053df86c8d503b98d9e7202ca8b8d4978c7/torch_xla/core/xla_model.py#L1508) supplies `get_memory_info`.
The [pinned binding](https://github.com/pytorch/xla/blob/5fab7053df86c8d503b98d9e7202ca8b8d4978c7/torch_xla/csrc/init_python_bindings.cpp#L978) returns `bytes_used`, `bytes_limit`, and `peak_bytes_used`.
The wrapper has no interval reset argument.
The source does not define these samples as kernel VMEM measurements.
The device client's peak scope still needs runtime verification.
Use new processes for comparisons and retain setup allocations in the stated measurement scope.
Do not infer a target peak from two current-usage samples.

`wait_device_ops` waits for asynchronous device operations.
It does not replace submission of a pending lazy graph.
Submit the graph and wait for completion before the timer ends.
Record the actual synchronization method and compilation interval.
Keep five warm-up iterations and three groups of 30 measured iterations from the research plan.
Retain every raw sample and the selected path.
Report complete-call latency separately from isolated kernel latency.
Include packed preparation and host tracing in the appropriate complete-call boundary.

[PAL-10: pinned upstream autograd source](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/autograd/_functions.py#L374) defines the gradient boundary.
When activations require gradients, backward calls `dequantize_4bit` before matrix multiplication.
That path restores the complete weight matrix.
Forward memory evidence cannot establish training memory reduction.
Test the public backward path with Pallas forward selected.
A JAX gradient result alone cannot establish that behavior.

## Remaining qualification questions

| Question | Required observable result |
| --- | --- |
| Does the grouped candidate compile? | Successful fixed-runtime TPU compilation, including uint8 slices, NF4 selection, scales, and precision. |
| Does the bridge preserve the operator contract? | Bound configuration, operand order, TPU execution, and functionalization test records. |
| Do results meet the fixed tolerances? | FP32/BF16 comparisons against the sealed CPU and same-device references. |
| Is decode storage bounded? | Complete kernel and compiler records across more than one N/K size. |
| What does packed preparation cost? | Separate preparation records and complete public-call timings on the same TPU. |
| Does upstream backward remain correct? | Public activation, bias, and adapter gradient records. |
| Is a memory peak specific to the target interval? | A qualified profiler method, or an explicit process-scope limitation. |

## Checks performed

The source manifest contains 12 source groups with dates, versions, decisions, and limits.
Source groups can contain related files at the same revision.
Their commit dates are snapshot dates, not each file's last modification date.
Rolling documentation with no publication date uses a null source date.

Four downloaded source hashes match the prior wheel source record.
They cover the bridge, import guard, pinned Mosaic lowering, and pinned TPU matmul example.
JSON parsing, required fields, unique source identifiers, and prose sentence length received local checks.
All 22 primary source links returned HTTP 200 during a read-only link check.
These checks establish document structure and source consistency.
They do not establish full ASD-STE100 conformity or TPU behavior.
