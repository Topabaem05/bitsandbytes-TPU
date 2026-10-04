# Task 1: XLA primitive operations

Replace `torch.bucketize` with strict comparison and integer reduction for NF4 code selection.
A reduction adds comparison results to get the code index.
A precondition is a requirement that the caller must satisfy before execution.
CPU fallback occurs when an operation uses the CPU instead of the selected device.
Keep the midpoint ties, normalization, scale rules, tail padding, and packing order unchanged.

Remove added tensor-value assertions from each device path.
The pinned bitsandbytes API does not promise a NaN or infinity rejection error.
The pinned XLA source has no identified native implementation for these assertions.
A CPU fallback would add host synchronization and fail the device acceptance criteria.

Require finite input, weight, and bias values as caller preconditions.
Require finite nonnegative scales as a caller precondition.
Do not promise validation or an error for values outside these preconditions.
Keep structural option, shape, layout, data type, device, scale-count, and nested-argument errors.

Keep all finite numerical expectations and the requirement for zero CPU fallback.
Keep the M1 test reports.
Replace three rejection cases for outside-domain values with three finite-input operator controls.
Those earlier rejection cases used NaN, infinity, and a negative scale.
The new controls reject calls to bucketize or device assertions on finite data.
Existing controls reject explicit scalar extraction and CPU transfer calls.

Actual TPU execution is untested.
This change does not show a performance, memory, or training result.

Primary sources:

- [Torch 2.9 native dispatch](https://github.com/pytorch/pytorch/blob/0fabc3ba44823f257e70ce397d989c8de5e362c1/aten/src/ATen/native/native_functions.yaml#L168) lists CPU and CUDA implementations for `_assert_async`.
- [XLA CPU fallback](https://github.com/pytorch/xla/blob/5fab7053df86c8d503b98d9e7202ca8b8d4978c7/torch_xla/csrc/aten_fallback.cpp#L37) counts operations before CPU execution.
- [XLA decomposition registration](https://github.com/pytorch/xla/blob/5fab7053df86c8d503b98d9e7202ca8b8d4978c7/torch_xla/_internal/decomp_registration.py) registers only trilinear upsampling in this file.
- [Bitsandbytes quantization API](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/functional.py#L884) documents data type errors, with no NaN or infinity rejection guarantee.
