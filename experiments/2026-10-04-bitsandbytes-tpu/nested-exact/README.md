# Exact nested M4 experiment

This directory contains the reviewed 19-member scientific preparation, its complete admitted source closure, and offline control scripts. The dedicated cloud implementation is `../cloud/exact-nested`.

The fixed matrix has 79 cases and 540 CPU output arrays. It uses the original exact gates, numerical tolerances, public methods, and first-scale lengths 1 through 513. The eight saved-state cases use separate owned CPU, save, and restore processes. M5 lengths of 1,024 first scales are outside this experiment.

The fresh CPU process must use CPython 3.12.14, Torch 2.9.0+cpu, the admitted Torch source, AVX2, and one thread. It must not import XLA or JAX. The device process must use the admitted Linux runtime, actual TPU execution, all nine schemas, admitted loaded code, and highest precision before graph construction.

The record store applies lossless gzip and byte-identical content deduplication. It preserves graph, record, checkpoint, and numerical bytes. Selected host recovery removes each recovered graph group after validation. The complete archive retains the original 96 MiB inner limit, 100 MiB physical archive limit, and 2,000-file limit.

The driver recovers and verifies actual79 before it waits for root acceptance. It cannot generate acceptance. Root uses the explicit host command after record review. State CPU begins only when the original allocation has enough time for the state stage and resource closure. State device execution also needs the separate actual state CPU launch audit. The accepted79 schema does not change.

`REPRODUCE.md` gives the additive materializer, offline replay, packet, CLI, browser adoption, acceptance, and disposal commands. `CHANGES-PROPOSAL.md` records the proposed logical change. Historical source records retain their original dates and limits. This preparation contains no qualified new Linux or TPU result.
