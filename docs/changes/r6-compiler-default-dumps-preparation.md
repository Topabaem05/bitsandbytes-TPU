# Compiler default dump preparation

The next compiler capture generation removes one additional pipeline dump request.
The exact request now contains 13 tokens, including its private dump directory.
Text, proto, module metadata, the 32-module cap, and all five native science cases remain unchanged.
The limits remain 1,024 entries, 128 MiB total, and 16 MiB per file.
The original monitor, precision rules, process ownership, deadlines, and closure remain unchanged.

The generation is `compiler-mosaic-serde7-gather-bf16-fp32-libtpu021-default-dumps-v2`.
Its manifest, policy, and cloud pins differ from the previous request.
The packet builder reads the approved `bf7c843` cloud base, which includes the adopted public dyadic protocol.
Both public cloud files remain byte-identical to that base.
The new [source and adoption maps](../../experiments/2026-10-04-bitsandbytes-tpu/results/compiler-default-dumps-preparation/adoption-map.json) explicitly supersede the historical 187-file compiler map with 194 files.
They reconstruct the complete control tree without a restored private work directory.
They do not replace the additive compiler allocation reader.

The [previous actual failure](r6-compiler-dump-limit-colab.md) stopped at 1,025 visited entries.
The monitor's byte figures describe only the visited directory prefix.
No raw compiler dumps were recovered from that failed capture.
The pinned [pipeline source](https://github.com/openxla/xla/blob/31eb6029e8973337ef6c99810c27e1d807791c11/xla/hlo/pass/hlo_pass_pipeline.cc) and [dump source](https://github.com/openxla/xla/blob/31eb6029e8973337ef6c99810c27e1d807791c11/xla/service/dump.cc) distinguish requested pipeline boundaries from default compiler dumps.
This supports the proposed scope reduction; it does not establish a loaded libtpu revision or predict the next dump distribution.

The isolated preparation passed 185 distinct positive and negative controls.
These include exact request admission, old request rejection, 1,024/1,025 entry behavior, cloud contracts, accepted native dependencies, and portable reconstruction.
Genuine compiler, ordinary native, and public packets passed byte inspection.
All final controls ran from the reconstructed source tree with explicit interpreter and repository paths.
The [replay recipe](../../experiments/2026-10-04-bitsandbytes-tpu/cloud/tests/compiler-boundary/REPRODUCE.md) uses bounded local process groups.
The [dispatch recipe](../../experiments/2026-10-04-bitsandbytes-tpu/cloud/tests/compiler-boundary/DISPATCH.md) requires a new root gate and fresh qualified Linux CPU references.

The new generation has not executed on a TPU.
The reviewer independently passed 157 relevant controls and reproduced the exact packet twice.
All 11 observed reviewer groups closed, and 355,922,165 disposable bytes were removed after compact evidence capture.
The [review record](../../experiments/2026-10-04-bitsandbytes-tpu/results/compiler-default-dumps-preparation/root-review.json) preserves hashes and closure observations.
Local JAX CPU parser acceptance does not establish libtpu parser acceptance.
Missing compiler metadata must produce unsupported or failed analysis.
M6 remains unqualified. Accepted milestones remain three of eight.
