# Task 1: Native functional tensor context

The second TPU probe failed before its first NF4 result.
Its XLA factory returned a native functional tensor inside a functorch transform.
The transform then tried to add another wrapper and raised an internal assertion.
Keep the failed probe and its raw error records.

Replace the functorch transform with native tensor wrappers for each full backend kernel.
Keep the caller's included dispatch keys.
Remove only the local functionalization exclusion during the Python kernel.
Synchronize and unwrap results, then apply fixed-size input data updates.
Reject functional input tensors and input metadata changes at this backend boundary.
Restore the dispatch context on success or error.
Do not add a global functionalization mode or change the NF4 arithmetic.

An isolated CPU process gives `aten::lift_fresh` the factory behavior found in XLA.
The previous transform raises the same assertion; the native context completes all four kernels.
The controls include output views, repeated input identities, an input/output alias, and errors.
Pure quantization, dequantization, and GEMM inputs stay unchanged.
Keep the fixed numerical limits for the float GEMM comparison.
Keep the endpoint comparison that requires identical values.

The mutable `Functionalize` overload and registration rules stay unchanged.
The 42 fixed inputs, finite-input contract, and CPU fallback limit stay unchanged.
These local tests use CPU PyTorch 2.14.1.
The reviewer repeated all 79 package tests; all passed in 20.08 seconds.
Actual PyTorch/XLA 2.9 execution is untested for this change.

Primary sources:

- [Native composite helper](https://github.com/pytorch/pytorch/blob/0fabc3ba44823f257e70ce397d989c8de5e362c1/aten/src/ATen/FunctionalTensorWrapper.cpp#L837) wraps inputs, keeps included keys, removes exclusion, and unwraps synchronized results.
- [Native Python bindings](https://github.com/pytorch/pytorch/blob/0fabc3ba44823f257e70ce397d989c8de5e362c1/torch/csrc/autograd/python_torch_functions_manual.cpp#L652) expose synchronization and tensor wrapper operations.
- [Input update test helper](https://github.com/pytorch/pytorch/blob/0fabc3ba44823f257e70ce397d989c8de5e362c1/test/test_functionalization.py#L42) copies synchronized input data back to the caller.
- [XLA factory](https://github.com/pytorch/xla/blob/5fab7053df86c8d503b98d9e7202ca8b8d4978c7/torch_xla/csrc/aten_xla_type.cpp#L2153) can return a native functional tensor.
