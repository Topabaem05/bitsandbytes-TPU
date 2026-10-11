# Exact word diagnostic preparation

Date: 2026-10-11.

The new diagnostic compares the original word implementation with two isolated variants.
One variant uses explicit Tensor bit shifts.
The other also specifies dimension zero for the flat tail maximum.
The main implementation remains unchanged.

The 24 probes cover raw and functional bitcasts, maximum reductions, shifts, scalar predicates, arithmetic, midpoints, zero codes, and short tails.
Each output retains its dtype, shape, and exact storage words.
The Linux CPU and TPU stages use the same source and input bytes.
HLO observation errors remain separate from numerical errors.

The worker passed 14 preparation controls.
The reviewer independently passed 24 local CPU probes and rejected nine incorrect controls.
The adviser independently rejected six incorrect CPU results.
Three simulated observation failures preserved the numerical results.
These local and simulated checks have no actual TPU qualification.

All 75 members of the complete preparation archive passed full readback.
The reviewer adopted 65 selected source and recipe files.
The unchanged execution packet has 58 members and 123,166 bytes.
Its SHA-256 is `a73c5b044ae981752e04a0a8b67706d1d21b8d9e94ac6044610bc1dc0ef984fc`.

The original browser intent starts a 600-second diagnostic deadline.
The final 120 seconds are reserved for recovery and closure.
The driver owns the exact assigned runtime before marker verification, so a failed marker still triggers exact runtime termination.
The owner checks the original intent deadline.

The [adopted source](../../experiments/2026-10-04-bitsandbytes-tpu/cloud/word-diagnostic-v1/ADOPTION.md)
and [review record](../../experiments/2026-10-04-bitsandbytes-tpu/results/m4-word-diagnostic-v1-preparation/root-review.json) contain the reproduction details.
Pinned Linux CPU and actual TPU execution remain required.
M4 remains unqualified.
