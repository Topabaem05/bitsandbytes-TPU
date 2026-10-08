# R2: Public transfer and API42 preparation

## Source correction

The original CPU-to-XLA transfer failed when `Params4bit._quantize` assigned data with an incompatible tensor type.
The examined patch keeps the compatible assignment path.
For incompatible types, it uses `torch.utils.swap_tensors` to retain the original parameter object.
It preserves custom attributes and binds the resulting state to the parameter and module.

The patch changes one method in one upstream file.
The other 46 Python files remain unchanged.
The original revision, complete package inventory, patch, and resulting inventory have separate hash records.
Refer to [the patch manifest](../../patches/params4bit-xla-v1.json).
The original experiment profile, inputs, and numerical gates remain unchanged.

The incompatible path rejects existing gradients, enabled module conversion flags, and unsupported parameter subclasses.
These conditions are examined before the swap.
Recovery from the examined synchronous errors depends on the pinned PyTorch 2.9.0 implementation.
The plugin permits the exact patched variant only under PyTorch `2.9.0+cpu`.
The patch does not promise recovery from concurrent use, asynchronous interruption, or a later bias conversion error.

The builder applies the patch to a fresh copy of the original archive.
It preserves the original archive and builds the modified upstream wheel separately.
It never changes an installed class or method during execution.
The original source remains a separate permitted plugin variant.

## Device experiment

The new `transfer-api42` experiment uses native precision `highest` once before graph construction.
The [R1 device comparison](r1-precision-colab.md) supplies the basis for this setting.

The experiment first repeats 18 source controls under the fixed Linux runtime.
It also examines the built wheel and installed Python source before scientific execution.
The CPU reference contains all 42 fixed cases from the pinned default Python implementation.
Its source record explicitly identifies the patch without changing the original profile.

The TPU process then executes these independent record groups:

1. Direct public `Params4bit.to` for FP32 and BF16 weights.
2. Public `Linear4bit.to` from a CPU module for FP32 and BF16 weights.
3. All 42 cases from the fixed API matrix.

The transfer records examine parameter identity, class identity, attributes, placement, packed bytes, and `QuantState` data.
The module records also examine bias, module state aliases, and frozen base weights.
The FP32 module transfer includes input and bias gradients.
BF16 gradients remain `NOT_RUN`.

Every case retains its output or explicit error.
A numerical failure remains a failure even when the record passes validation.
API42 passes only when all 42 cases and all four transfer records pass.
Restoration in a new process remains outside this experiment.

## Inspection and limits

The reviewer repeated 90 package tests; all passed in 28.92 seconds.
The reviewer also repeated all 18 portable source controls.
These tests used macOS, Python 3.12.14, and PyTorch 2.14.1.
They do not qualify the Linux runtime or TPU behavior of the patch.

The source controls include compatible CPU conversion, incompatible type fixtures, and rejected conversions with unchanged parameter state.
Two controls compare genuine CPU forward and gradient values and restore saved state in a separate CPU process.
The Meta fixtures examine type mechanics only.
They do not supply numerical device results.

The reviewer repeated 100 scientific verifier controls; all passed in 5.72 seconds.
They include 40 transfer controls, 38 precision controls, and 22 retained route controls.
They exercise correct records, numerical failures, missing cases, incorrect source, altered methods, and incorrect references.
The fixtures do not execute TPU calculations.

The rank-two FP32 HLO observer requires contributing matrix operations with the selected precision.
Rank-three cases retain their HLO without imposing the rank-two observer.
Their numerical gates, device placement, and execution requirements still apply.
HLO records describe graphs before compiler optimization.

The reviewer repeated 191 cloud controls; all passed in 14.30 seconds.
These controls use simulated service operations, real source archives, retained arrays, and local verifier processes.
They reject incorrect wheel source, incomplete control records, changed phase records, missing results, incorrect process groups, damaged archives, and closure failures.
The final packet contains 36 archive members.
Its independent readback passed before allocation.

Actual device results must complete before R2 or M2 acceptance.
M3, nested quantization, training, and performance remain separate requirements.
