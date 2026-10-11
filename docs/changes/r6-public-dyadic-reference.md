# Public CPU reference preparation

## Cause and scope

The first public experiment stopped at its Linux CPU gate.
The host FP32 matrix result did not match the retained Linux result exactly.
All 15 public TPU cases remain unexecuted.
The [failed experiment](r6-public-cpu-colab-failure.md) remains a failed experiment.

PyTorch does not guarantee identical floating-point output bits across platforms.
The [PyTorch 2.9 accuracy note](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html) describes this limit.
Source inspection occurred on 2026-10-11.
This statement supports a separate reference protocol.
It does not establish the cause of each observed bit difference.

## Reference protocol

The generation is `public-pallas-device-mosaic7-gather-bf16-fp32-dyadic-v2`.
The original 15 Linux CPU rows remain the primary results.
The original `F.linear` and autograd calculations remain unchanged.
The device verifier compares device outputs with those primary rows under the original tolerances.
The fixed inputs, cases, runtime, first-error rule, and kernel remain unchanged.

Each case also has an exact integer reference result.
The reference multiplies dyadic operands and sums their products with Python integers.
It rounds each exact result once to the specified output data type.
It preserves high-first NF4 nibble order and the original FP32 scale multiplication.
It rounds decoded BF16 weights before matrix multiplication.
It calculates each repeated gradient as twice the first rounded FP32 gradient.
It does not substitute FP64 arithmetic for the original calculations.

The CPU seal binds 15 primary rows and 15 supplements.
The host verifier recomputes each supplement exactly from the fixed inputs and pinned upstream codebook.
It compares the original decoded weight bits with the integer decoder bits.
It applies the unchanged numerical gates to each primary row and its supplement.
An old CPU seal cannot authorize this generation.
A fresh qualified Linux CPU seal remains required before device execution.

## Composition and controls

The private composition uses committed revision `5dbfd05128fe741ba19c513d43a87c09b7cca017`.
Only two cloud files change: `public_contract.py` and `build_public_packet.py`.
The accepted native sources and corrected compiler sources remain byte-identical.
Runtime, source, wheel, process ownership, deadline, and archive gates remain in effect.

Fresh local controls passed 33 science cases, five saved-record cases, and six archive cases.
The compiler phase passed 18 controls, and the ordinary native phase passed 23 controls.
Three fresh source packets passed their mode-specific preflight.
These controls use unqualified macOS Torch and synthetic records.
The source-only wheels are not installed.
The 39-member CPU archive contains 59,171,154 uncompressed bytes.
The original 64 MiB archive limit remains unchanged.

The first complete cloud control attempt reached its 220-second host deadline.
All 17 recorded process groups closed.
The timeout remains in the private review records.
A separate run passed all 79 cloud controls in 226.88 seconds.
Its finite host bound was 600 seconds, and all 19 recorded process groups closed.
The numerical criteria and scientific deadlines remain unchanged.

The private source and adoption maps identify each changed file and its committed base hash.
The compact review retains control results, closure observations, and hashes before fixture removal.
The worker used a temporary directory and did not modify the canonical checkout.
The reviewer independently repeated all 33 science controls, five saved-record controls, and six archive controls.
All 44 controls passed, and their owned process groups closed.
The reviewer accepted this source preparation on 2026-10-11.
The [review record](../../experiments/2026-10-04-bitsandbytes-tpu/results/public-dyadic-preparation.json) preserves the distinction between source preparation and device acceptance.

## Remaining requirements

A qualified Linux CPU run has not executed this protocol.
The 15 actual public TPU cases remain unexecuted for this generation.
Compiler memory acceptance and public device acceptance remain separate requirements.
M6 remains unqualified, and accepted milestones remain three of eight.
