# PyTorch/XLA source inspection

Source access date: **2026-10-08**.
Scope: M2 and M3 runtime, functionalization, parameter conversion, and numerical precision.
M6 and M7 use the precision findings as future validation requirements.

This inspection adds source evidence only.
It does not change the runtime or complete a milestone.
No package installation, TPU allocation, implementation, or numerical experiment occurred during this inspection.
The [source register](xla-sources.json) records source dates, revisions, claims, and limits.
A source snapshot date does not identify the last change to each file.

## Release and runtime comparison

| Component | Fixed project runtime | Latest observable record | Meaning |
| --- | --- | --- | --- |
| PyTorch | `2.9.0+cpu` | Stable `2.14.1`, published 2026-09-30 | Compatibility with XLA 2.9.0 is unproved. |
| PyTorch/XLA | `2.9.0` | Stable `2.9.0`, published 2025-11-17 | The fixed XLA release remains the latest published release. |
| XLA master | Not selected | `41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf`, dated 2026-04-23 | This source is separate from the stable wheel. |
| libtpu | `0.0.21` | XLA master requires `0.0.24.dev20250929+nightly` | This is a source dependency declaration. |
| JAX and jaxlib | `0.7.1` | XLA master requires `0.8.0.dev20251001` | This is a source dependency declaration. |

GitHub release metadata and PyPI metadata agree on the latest stable PyTorch and PyTorch/XLA versions.
See [XLA-01](https://github.com/pytorch/xla/releases/tag/v2.9.0) and [XLA-02](https://github.com/pytorch/pytorch/releases/tag/v2.14.1).
The observed PyTorch main revision is `b15cc7a846a1385805d4499a53eecc5fe30ba2f5`, dated 2026-10-08.
The observed PyTorch 2.14.1 tag resolves to `5c4886908584029761b579af026dcfb627c84070`.

The [published XLA metadata](https://pypi.org/pypi/torch-xla/2.9.0/json) requires `libtpu==0.0.21` for its `tpu` extra.
Its `pallas` extra requires `jax==0.7.1` and `jaxlib==0.7.1`.
Its CPython 3.12 Linux x86-64 wheel has a `manylinux_2_28` tag.
Its requirements do not specify `torch`.
Thus, package dependency resolution alone cannot establish PyTorch/XLA binary compatibility.

The [pinned setup source](https://github.com/pytorch/xla/blob/5fab7053df86c8d503b98d9e7202ca8b8d4978c7/setup.py#L113) selects stable dependencies.
The [observed master source](https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/setup.py#L117) selects nightly dependencies.
These declarations do not establish wheel availability or a qualified replacement runtime.
Keep the fixed runtime until a separate inspection accepts a replacement.

## Functionalization and output arguments

[XLA-03: `torch.func.functionalize`](https://docs.pytorch.org/docs/2.14/generated/torch.func.functionalize.html) describes removal of intermediate mutations and optional removal of views.
Input mutation can require a final `copy_` operation.
The transform has limits for direct `.backward()` calls and non-local state.
These general semantics do not establish support for the existing mutable bitsandbytes operators.

[XLA-04: pinned native fallback](https://github.com/pytorch/pytorch/blob/0fabc3ba44823f257e70ce397d989c8de5e362c1/aten/src/ATen/FunctionalizeFallbackKernel.cpp#L58) rejects schemas with any alias information.
Its error text discusses output aliases, but its condition calls `schema.hasAnyAliasInfo()`.
The [observed current fallback](https://github.com/pytorch/pytorch/blob/b15cc7a846a1385805d4499a53eecc5fe30ba2f5/aten/src/ATen/FunctionalizeFallbackKernel.cpp#L56) keeps this condition.
A generic fallback cannot establish correct support for an operator with mutable output arguments.
The design requires the existing schemas and their alias behavior.
Deleting an alias annotation to bypass this guard would not meet that requirement.

The fallback synchronizes functional inputs before it unwraps them.
It wraps results when any tensor input is functional or when the operator has no tensor inputs.
The [pinned XLA `lift` and `lift_fresh`](https://github.com/pytorch/xla/blob/5fab7053df86c8d503b98d9e7202ca8b8d4978c7/torch_xla/csrc/aten_xla_type.cpp#L2153) can call `MaybeWrapTensorToFunctional`.
The [observed master source](https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/torch_xla/csrc/aten_xla_type.cpp#L2180) retains that path.
A Python transform around an XLA kernel must account for native wrappers.

The [native composite helper](https://github.com/pytorch/pytorch/blob/0fabc3ba44823f257e70ce397d989c8de5e362c1/aten/src/ATen/FunctionalTensorWrapper.cpp#L837) supplies source evidence for synchronization and dispatch context handling.
These internal helpers have no public stability guarantee in the examined sources.
[XLA code generation](https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/codegen/xla_native_functions.yaml#L392) identifies composite operations that must re-enable functionalization.
The same file says new functionality must not depend on operations provided only to disable functionalization.

The [existing native-context record](../../changes/task1-native-functionalization.md) preserves the earlier wrapper assertion and isolated CPU controls.
Those controls do not establish real XLA behavior for every output argument and alias combination.
The October 4 [device route record](../../changes/task2-colab-routes.md) supplies actual TPU evidence for its tested cases.

## CPU-to-XLA parameter conversion

[XLA-05: `Params4bit._quantize`](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/nn/modules.py#L381) transfers dense data, quantizes it, then assigns the packed result to `self.data`.
The unquantized `Params4bit.to` route calls this method before it returns.
The [pinned PyTorch guard](https://github.com/pytorch/pytorch/blob/0fabc3ba44823f257e70ce397d989c8de5e362c1/torch/csrc/autograd/variable.cpp#L473) requires compatible tensor types for `set_data`.
The [pinned compatibility list](https://github.com/pytorch/pytorch/blob/0fabc3ba44823f257e70ce397d989c8de5e362c1/c10/core/TensorImpl.h#L2072) includes CPU and CUDA dense tensors but excludes XLA.
These sources explain the location and condition of the recorded CPU-to-XLA assignment error.
They do not prove a correction.

[XLA-06: stable `Module._apply`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/nn/modules/module.py#L930) calls the conversion function before it chooses a swap, data assignment, or replacement.
Thus, a module conversion flag alone cannot bypass the earlier `Params4bit._quantize` error.
The replacement and swap branches construct a `Parameter` from the conversion result.
The [custom Parameter constructor](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/nn/parameter.py#L51) requires `detach()` to return the same Python type.
Returning a new subclass parameter therefore requires inspection of `detach` behavior and retained state.

The pinned, latest stable, and observed main sources retain these requirements.
Their `Parameter.__new__` methods have matching abstract syntax trees.
Stable and main `_apply` methods match each other.
The pinned `_apply` differs, including the later exclusion of `FakeTensor` from the data assignment decision.
This comparison establishes source behavior only.

## Public tensor swap

[XLA-07: official swap recipe](https://docs.pytorch.org/tutorials/recipes/recipes/swap_tensors.html) was created and updated on 2024-04-19.
Its verification field says `Not Verified`.
It describes a public method that preserves Python object identity while it exchanges tensor content and Python attributes.
It also describes module conversion and state loading extension points.

The [stable Python implementation](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/utils/__init__.py#L35) rejects Python weak references, different slots, and unsupported tensor use counts.
It exchanges Python classes, dictionaries, and slots before it calls the C++ exchange.
The [stable C++ implementation](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/csrc/Module.cpp#L390) then tests C++ weak references before it exchanges TensorImpl content.
The public recipe gives no exception-atomicity guarantee.
A failure after Python attribute exchange can require separate state recovery.

Pinned, stable, and main Python swap methods have matching abstract syntax trees.
The C++ implementation changed its final Python pointer storage between pinned and current source.
A recovery argument tied to the pinned implementation must not be generalized to every PyTorch release.
The existing private swap proposal remains unadopted and unqualified on TPU.
Its local controls do not complete M2 or M3.

## FP32 precision controls

[XLA-08: precision tutorial](https://docs.pytorch.org/xla/master/tutorials/precision_tutorial.html) records creation and modification on 2025-05-15.
The [public API source](https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/torch_xla/backends/__init__.py#L41) supplies these TPU precision descriptions:

| Setting | Documented computation | Limit |
| --- | --- | --- |
| `default` | FP32 operands downcast to BF16 before multiplication | Lowest stated precision |
| `high` | Three passes, approximately 14 bits of precision | No bitwise CPU equality promise |
| `highest` | Six passes, approximately 22 bits of precision | Final-bit differences can remain |

The API source is identical at the pinned and observed master revisions.
Use `torch_xla.backends.set_mat_mul_precision` to select the setting.
Read it with `torch_xla.backends.get_mat_mul_precision`.
The tutorial recommends one setting at the start of a script.
It also shows a final-bit difference between `highest` and CPU FP32 multiplication.
Its introductory statement about BF16 results must not be treated as a guaranteed output data type.

[XLA-09: pinned source default](https://github.com/pytorch/xla/blob/5fab7053df86c8d503b98d9e7202ca8b8d4978c7/torch_xla/csrc/helpers.cpp#L76) initializes `PrecisionConfig::DEFAULT`.
The [observed master default](https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/torch_xla/csrc/helpers.cpp#L277) is the same.
The [native dot lowering](https://github.com/pytorch/xla/blob/5fab7053df86c8d503b98d9e7202ca8b8d4978c7/torch_xla/csrc/xla_lower_util.cpp#L476) uses the shared setting to build operand precision.
The [current lowering](https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/torch_xla/csrc/xla_lower_util.cpp#L477) keeps that behavior.
The [binding](https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/torch_xla/csrc/init_python_bindings.cpp#L2516) reads and writes this shared setting.
A source default does not prove the effective setting in an earlier process.

[XLA-10: PyTorch FP32 control](https://docs.pytorch.org/docs/2.14/generated/torch.set_float32_matmul_precision.html) documents a CUDA-only native device effect.
Its TF32 controls describe NVIDIA computation.
They do not establish TPU precision in the native PyTorch/XLA path.
The historical [XLA 2.6 release note](https://github.com/pytorch/xla/releases/tag/v2.6.0) marks `XLA_USE_BF16` as deprecated.
Old environment-variable examples do not qualify a current precision control.
No supported precision environment variable was established by this inspection.

## Rounding and subnormal numbers

[XLA-11: Cloud TPU conversion documentation](https://docs.cloud.google.com/tpu/docs/bfloat16) shows a last update of 2026-10-07 UTC.
It states BF16 multiplication with FP32 accumulation as the TPU default.
It states round to nearest even for FP32-to-BF16 conversion.
It states that Cloud TPU BF16 conversion flushes subnormal values to zero.
This conversion statement does not establish flushing for every FP32 operation.
It also does not identify the compiler path used by this project.

The existing scalar witness exactly matches recorded FP32 forward and input-gradient values after BF16 operand rounding.
Its scope is `READ_ONLY_SCALAR_FP64_WITNESS_NOT_DEVICE_CAUSE_PROOF`.
The witness file is `.work/task1-precision/operand-rounding-witness.json`.
It supports a precision hypothesis, but it does not prove the device cause.
The separate subnormal hypothesis requires evidence at each conversion or arithmetic boundary.

## TorchTPU availability

[XLA-12: official Google announcement](https://developers.googleblog.com/torchtpu-running-pytorch-natively-on-tpus-at-google-scale/) was published on 2026-04-07.
It describes a native `PrivateUse1` backend, eager modes, XLA compilation, and Pallas/JAX custom kernels.
It lists public GitHub launch as future work.
The [observed XLA README](https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/README.md#L4) says TorchTPU will replace PyTorch/XLA once public.

A [PyTorch discussion category](https://dev-discuss.pytorch.org/t/about-the-tpu-category/3463), published 2026-10-07, addresses the native `torch_tpu` backend.
It gives no backend installation instructions.
The [current Google `tpu-sync` source](https://github.com/google/tpu-sync/blob/0d6efd49c7c62c24c74b79dc9cf5d78278492ec8/README.md#L158) says its `torch_tpu` dependency is not on a public index.
The observed [PyPI record](https://pypi.org/pypi/torch-tpu/json) contains version `0.0.0` with summary `Placeholder`.
It does not establish a usable Google backend wheel.

The observed `google-pytorch/torch_tpu` GitHub API returned HTTP 404.
The [current public PyTorch CI pin](https://github.com/pytorch/pytorch/blob/b15cc7a846a1385805d4499a53eecc5fe30ba2f5/.github/ci_commit_pins/torch_tpu.txt) contains `676d795ab231f97103c52d1b7261d6b6487f7c48`.
A CI pin does not supply accessible backend source or a compatible installation.
Public source availability and public installability remain **NOT VERIFIED**.
This limited result does not establish the absence of a private repository.

## Existing results and future gates

The [October 4 route record](../../changes/task2-colab-routes.md) reports 34 nonlinear numerical passes and two CPU-to-XLA transfer errors.
Both executed FP32 module routes failed forward and input-gradient gates.
The maximum absolute errors were `0.001275897` for forward and `0.001057386` for the input gradient.
Both BF16 module routes passed their numerical gates.
These results do not qualify the full API42 matrix or milestones M2 and M3.
This inspection does not change their limits or status.

Before a future precision test:

1. Keep the fixed runtime, source identities, inputs, and tolerances.
2. Use a fresh process for each precision setting.
3. Select one setting before graph construction or matrix multiplication.
4. Record the precision readback, relevant environment values, and compiler settings.
5. Compare plain FP32 matrix multiplication, module forward, and input gradients.
6. Include `default`, `high`, and `highest` with the existing BF16 control.
7. Examine device HLO operand precision and record CPU fallback counters.
8. Require the unchanged numerical gates for each required route.

Before a future conversion test:

1. Examine the correction and admit its complete source identity.
2. Test direct `Params4bit.to` and CPU-created `Linear4bit.to` independently.
3. Include a compatible CPU control and an incompatible conversion control in isolated fixtures.
4. Test rejected swap conditions without changes to production records.
5. Require original class identity, state aliases, bias placement, and frozen base weights.
6. Compare forward and gradients with the unchanged limits.
7. Save and restore state in a new process.
8. Keep constructor and prequantized routes as separate records.

Before a future output-argument test:

1. Keep each upstream schema and its alias annotations.
2. Compare functional and mutable overloads on the real XLA backend.
3. Include repeated inputs, input/output aliases, output views, and invalid structures.
4. Require correct mutations, returned values, dispatch context restoration, and zero unintended CPU fallback.

Before a future subnormal test:

1. Compare BF16 conversion, FP32 multiplication, division, scale application, and quantization separately.
2. Include normal values, exact zero blocks, and adjacent subnormal values.
3. Record values before and after each device boundary.
4. Keep conversion flushing separate from an arithmetic flushing claim.

These are proposed validation gates.
No gate was executed during this source inspection.
Source metadata and selected code methods were read back locally.
JSON structure, source identifiers, revision links, and sentence lengths received local checks.
Those checks do not certify full ASD-STE100 conformity.
