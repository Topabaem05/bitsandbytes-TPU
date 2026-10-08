# Preparation of the native Pallas diagnostic

Date: 2026-10-08.

The reviewer accepted preparation of the bounded native diagnostic.
The experiment uses the accepted M2 source and fixed runtime.
It does not require or qualify the failed M4 nested implementation.
M6 remains unqualified until its required actual results exist.

## Fixed experiment

The native manifest identifies 23 source files.
Its SHA-256 is `b25c5d2bf128b5164538c06267ec3a2f55a001a1a7f86eb63142153842eb3909`.
The admitted variant is `M2_CANONICAL_NON_NESTED_R2_PATCH`.
All 23 files retain their reviewed bytes.

The fixed cases contain four native calls and one reference case for a partial tile.
The native matrix dimensions are M=128, N=256, and K=512.
The native cases use FP32 and BF16 inputs through direct and functional interfaces.
The reference case uses M=129.
The protocol retains the existing numerical tolerances and `highest` precision.

The host must recover the complete qualified Linux CPU oracle before device execution.
It verifies source inventories, runtime identity, actual process identity, and process closure.
It then uploads the separate CPU oracle gate.
The device parent starts a new child in its owned process group.
The owner retrieves all records and closes the exact Colab session.

## Native records

The recorder retains the native payload, Mosaic body, context protobuf, printer HLO, and actual input parameters.
The verifier compares graph content across both protobuf records and the printed HLO.
It independently compares complete outputs with the recovered CPU reference.
The execution interval submits and waits for the actual native output.
It excludes reference graph execution from that interval.

The context text API returns protobuf text, which differs from HLO printer text.
The corrected recorder retains both representations separately.
Graph names and identifiers do not establish the selected executable.

Compiler dumps are off for this bounded diagnostic.
Selected executable identity and allocator peak remain `UNKNOWN`.
Compiler memory, public backend integration, backward execution, and performance need separate results.
This preparation does not complete M6.

## Inspection

The reviewer previously repeated 55 recorder and recovery controls; all passed.
Two fresh JAX 0.7.1 parser cases matched the retained printer fixtures.
Five fresh CPU cases matched 164,096 output values.
Those CPU tests used macOS and PyTorch 2.14.1, which is an unqualified runtime.

The reviewer independently repeated 33 cloud controls; all passed.
These tests include complete simulated owner sequences, incorrect records, and three actual local bootstrap processes.
The bootstrap processes unpack source files only.
They do not initialize a device runtime.
The worker also passed 399 existing cloud controls.

The first portable test omitted the saved host gate from its selected fixture files.
The phase test stopped before its simulated device request.
The corrected fixture retains that exact gate and the original failure records.
No production source or numerical tolerance changed for this correction.

The [result record](../../experiments/2026-10-04-bitsandbytes-tpu/results/native-cloud-preparation.json) gives packet hashes and review scope.
The [native directory](../../experiments/2026-10-04-bitsandbytes-tpu/native/README.md) gives the source contract.
Accepted milestones remain **3 of 8**.
