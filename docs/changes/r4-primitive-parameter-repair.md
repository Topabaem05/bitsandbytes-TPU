# Arithmetic parameter correction

The reviewer accepted this correction on 2026-10-09, Korea time.
The correction prepares a fresh Colab diagnostic and does not complete M4.

The [previous experiment](r4-primitive-parameter-failure.md) stopped at the third builder case.
The retained graph contains two scalar parameters for the lower and upper clamp limits.
The pinned Torch/XLA source creates these parameters from the supplied minimum and the finite FP32 maximum.
The previous record did not retain their values.
This correction does not reconstruct those missing observations.

The collector now identifies both parameters from the typed clamp operands.
It requires distinct identifiers and a complete parameter inventory.
Copy operations must preserve the type and shape.
Reshape operations must preserve the type and element count.
The collector rejects cyclic alias paths.
Other diagnostic rows retain their previous parameter requirements.

Two additional gates compare the observed scalar bytes with the fixed source values.
The values are FP32 `1e-38` and the finite maximum with bit pattern `0x7f7fffff`.
A byte difference gives `FAIL`.
This inspection does not validate all possible compiler graph semantics.

The collector writes observed values before graph validation.
If validation fails, it retains an `ERROR` record and the available evidence.
If JSON cannot represent a value, it retains a separate text record with a SHA-256 hash.
An error record cannot qualify the diagnostic.
Inputs, arithmetic kernels, tolerances, and the runtime remain unchanged.

The reviewer independently repeated 67 controls against the final private candidate.
The 44 focused controls also passed after adoption.
All owned process groups closed.
The new packet contains 56 members and 1,084,675 bytes.
Every member matches the inspected private packet exactly.
Seven isolated incorrect packets or dependencies failed admission.
The previous actual CPU oracle also failed the new source requirement, as required.

Generate a fresh qualified Linux CPU oracle before the TPU phase.
Retain all actual scalar values and the remaining mean observations.
Accepted milestones remain three of eight.

Refer to the [selected verification record](../../experiments/2026-10-04-bitsandbytes-tpu/results/primitive-parameter-repair.json).
The pinned [scalar construction](https://github.com/pytorch/xla/blob/5fab7053df86c8d503b98d9e7202ca8b8d4978c7/torch_xla/csrc/tensor_methods.cpp) supplies the source basis.
The pinned [parameter mapping](https://github.com/pytorch/xla/blob/5fab7053df86c8d503b98d9e7202ca8b8d4978c7/torch_xla/csrc/init_python_bindings.cpp) defines the observation API.
