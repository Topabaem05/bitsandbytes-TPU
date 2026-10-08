# R1: XLA precision diagnostic

## Purpose

The previous Colab run failed the FP32 forward and input-gradient gates on two module paths.
A CPU calculation with BF16 operand rounding reproduced those outputs.
This observation supports a precision hypothesis.
It does not identify the effective precision of that device run.

This diagnostic compares native XLA precision with fixed inputs and numerical requirements.
Refer to [the source inspection](../research/2026-10-08/xla.md).

## Procedure

The parent starts three separate processes for `default`, `high`, and `highest`.
Each process sets native XLA precision once before it creates a TPU graph.
Each process records the precision readback, HLO, execution metrics, and fallback counters.
The verifier requires FP32 dots that contribute to the HLO root outputs.
It compares operand shapes, contraction dimensions, and result shapes.
It also examines the numerical sample format in execution metrics.

Each mode uses these calculations:

1. Plain FP32 matrix multiplication with the sealed decoded weights.
2. FP32 `Linear4bit` forward, input gradient, and bias gradient on two public paths.
3. BF16 forward controls on the same two public paths.

The module paths use a target-device constructor or `Params4bit.from_prequantized`.
The diagnostic identifies the quantization source for each path.
It checks packed weights, state aliases, and frozen base weights with the accepted route helpers.
BF16 gradients remain `NOT_RUN`.

The diagnostic keeps the original profile, inputs, backend probe, route probe, runtime, and package source.
The CPU reference remains the pinned default Python implementation in the qualified Linux runtime.
The FP32 forward and gradient gates remain `atol=1e-5` and `rtol=1e-4`.
The BF16 gates remain `atol=0.02`, `rtol=0.02`, and `NRMSE<=0.02`.
Packed bytes must match exactly.

## Records and closure

The parent binds each child PID, process token, source hash, output inventory, and exit result.
The verifier compares retained arrays with the sealed CPU reference and independent scalar calculations.
It keeps a valid numerical failure separate from an invalid record.

The Colab owner permits one allocation under the existing time limits.
It retrieves the full archive and stops the owned session.
The final inspection requires an empty server and zero active usage.
`PASS_PRECISION_RECORDS` means that record validation and resource closure passed.
It does not mean that all precision modes met the numerical gates.

## Limits

The HLO record describes the graph before compiler optimization.
It does not prove a specific hardware instruction sequence or execution cost.
The diagnostic measures no performance or memory benefit.

The fresh processes isolate precision settings.
They do not test checkpoint restoration across processes.
The CPU-to-XLA transfer requirement, full API42, and M3 remain open.
CUDA comparison remains `NOT_RUN`.

## Local inspection

The reviewer repeated 164 tests; all passed in 6.83 seconds.
These tests include 38 precision controls, 22 retained route controls, and 104 cloud controls.
They use synthetic records, local child processes, and simulated service operations.
They do not execute TPU calculations.

The negative controls rejected incorrect source, inputs, parent identity, precision, HLO, fallback, archive bytes, and closure records.
A false reference with matching output values fails the independent scalar check.
Normal child completion and a timeout both have process-reaping controls.

The source inspection and packet preflight passed without a cloud allocation.
The original probes, input profile, runtime lock, and backend source remain unchanged.
Actual Colab execution is the next step.
