# Version 2 static allocation bounds

The actual compiler records contain integer and predicate storage.
The original auditor cannot inspect those types.
A separate version 2 auditor adds PRED, S32, S64, and U32 with natural storage widths.
The original auditor and its parser remain unchanged.
Explicit nonnatural packing and other unsupported layouts remain unsupported.

The reviewer independently passed all 65 original controls and 93 new controls.
The actual reader parsed 55 HloProto records and selected 11 buffer assignments through outer field 3.
All 11 plans passed static array, logical buffer, slice, alias, and allocation bounds.
All input sizes and hashes remained unchanged.
The reviewer also confirmed that the complete Auditor class has the original abstract syntax tree.

These results describe compiler allocation plans.
They do not measure runtime peak memory or internal Mosaic storage.
The selected executable link remains unknown, and M6 remains unqualified.

Read [the source basis and commands](../../experiments/2026-10-04-bitsandbytes-tpu/compiler-analysis/ALLOCATION_V2.md).
Inspect [the reviewer record](../../experiments/2026-10-04-bitsandbytes-tpu/results/compiler-allocation-v2/root-review.json)
and [the actual static audit](../../experiments/2026-10-04-bitsandbytes-tpu/results/compiler-allocation-v2/actual-static-audit.json).
The schema reference is a public XLA revision; the internal libtpu revision remains unknown.

The document uses the project clarity rules.
Full STE dictionary review remains incomplete.
