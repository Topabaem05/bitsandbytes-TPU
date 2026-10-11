# Compiler debug trace preparation

The new diagnostic reader connects complete debug records to complete module bytes through the pinned lazy hash.
It retains compilation and execution input and output counts.
Compilation counts refer to the submitted module.
Execution counts refer to the optimized module with the matching hash.
The counts can differ after compilation.
Tuple output counts include only the top-level children.

Offline review can occur after the original deadline.
The recorded execution must still finish within that deadline and before review.
The live process owner retains its original deadline.

The reviewer passed 59 CPU controls and 15 retained source excerpt checks.
Astra Max independently passed 21 additional memory-only controls.
The reviewer verified all 31 source archive members before adoption.
The existing compiler analyzer and allocation auditors remain unchanged.

This diagnostic preparation does not qualify M6.
The supplied provenance, synchronization, and readback statements still need independent runtime proof.
The direct PJRT recorder remains unimplemented.
Actual TPU verification has not run.
Loaded handle identity, binary identity, allocator peak, and internal Mosaic storage remain unknown.

Refer to the [source and scope](../../experiments/2026-10-04-bitsandbytes-tpu/compiler-analysis/debug-trace-v2/README.md).
The [root review](../../experiments/2026-10-04-bitsandbytes-tpu/results/m6-debug-trace-v2-preparation/root-review.json) records the independent checks.

The prose received sentence-length and technical-term inspection.
The complete ASD-STE100 dictionary inspection remains incomplete.
