# Task 1: Local functionalization context

Use a local functionalization context for each full XLA kernel.
Functionalization replaces intermediate views and mutations with tensor operations.
The context wraps kernel inputs, removes intermediate views, and keeps input mutations that the caller can see.
Restore the previous dispatch context on completion or error.

The first TPU probe failed at NF4 code-table slicing before its first result.
PyTorch excludes functionalization when its custom-operator fallback enters a backend kernel.
XLA can create new functional tensors inside that kernel.
Its reentrant composite path requires functionalization to be enabled again.
The wrapper addresses the full Python kernel; it does not change one slice or disable functionalization globally.

Add a `Functionalize` implementation for the mutable dequantization overload.
Call the existing default dequantization operator to keep device dispatch.
Copy the full result into the supplied output and return None.
Keep output views, input aliases, and structural errors.
Do not replace CPU, CUDA, or default backend kernels.
Extend collision, repeated-registration, and rollback controls to this registration.

Keep the finite-input contract, NF4 arithmetic, 42 fixed inputs, and numerical limits unchanged.
A CPU float example had a maximum absolute difference of `2.384185791015625e-7` under functionalization.
It passed the fixed forward limits; an endpoint example passed with identical values.
Keep the first failed comparison in local test records.

The local controls use PyTorch 2.14.1 on CPU.
The reviewer repeated all 77 package tests; all passed in 22.07 seconds.
Actual Torch/XLA 2.9 execution and zero CPU fallback are unconfirmed for this change.
This change makes no performance, memory, or training claim.

Primary sources:

- [Custom-operator functionalization fallback](https://github.com/pytorch/pytorch/blob/0fabc3ba44823f257e70ce397d989c8de5e362c1/aten/src/ATen/FunctionalizeFallbackKernel.cpp#L58) rejects alias annotations and excludes functionalization before backend dispatch.
- [Reentrant composite context](https://github.com/pytorch/pytorch/blob/0fabc3ba44823f257e70ce397d989c8de5e362c1/aten/src/ATen/FunctionalTensorWrapper.cpp#L831) wraps inputs and locally restores the dispatch key.
- [Python functionalization transform](https://github.com/pytorch/pytorch/blob/0fabc3ba44823f257e70ce397d989c8de5e362c1/torch/_functorch/eager_transforms.py#L1633) keeps input mutations and removes its context after errors.
- [XLA reentrant composite](https://github.com/pytorch/xla/blob/5fab7053df86c8d503b98d9e7202ca8b8d4978c7/torch_xla/csrc/aten_xla_type.cpp#L4280) uses the same local dispatch-key restoration pattern.
- [XLA tensor wrapping](https://github.com/pytorch/xla/blob/5fab7053df86c8d503b98d9e7202ca8b8d4978c7/torch_xla/csrc/torch_util.cpp#L67) makes functional tensors unless the disabling flag is set.
