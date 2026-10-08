# R4: Nested saved-state experiment preparation

The `nested-state-8` experiment uses the eight module cases from the fixed 79-case nested input set.
It requires a separately accepted 79-case TPU result with matching source, runtime, profile, and reference records.
An isolated synthetic dependency supports local tests only.
It does not authorize a device experiment.

The packet creates a fresh qualified Linux reference for all 79 cases.
It then creates a separate state reference through the original public serialization and restoration methods.
The owner retrieves and binds both reference seals before device execution.

The device coordinator starts a save process and waits for its complete exit.
It then starts a fresh restore process in the same owned process group.
Restore constructs the original public module on the target device before dtype conversion.
It uses `QuantState.from_dict`, `Params4bit.from_prequantized`, and strict loading with `assign=False`.

All checkpoint files, saved tensors, restored tensors, and packed metadata must match exactly.
The comparison includes scale codes, the offset, second scale state, and both maps.
Outputs use the existing fixed numerical tolerances.
FP32 input and bias gradients remain in scope; BF16 gradients remain outside scope.

The verifier requires every case, both process identities, complete archives, and observed resource closure.
It retains finite numerical failures and case errors.
An exception after execution invalidates an earlier complete receipt.

The reviewer repeated 23 scientific controls in 32.426 seconds.
Eight original-public CPU cases passed 90 output comparisons and exact state comparisons.
These controls used unqualified macOS PyTorch 2.14.1.
The reviewer also repeated all 391 cloud controls in 119.291 seconds, with no skipped tests.
Those controls include generated notebook entry snippets and incorrect source, process, checkpoint, archive, and closure records.

No qualified Linux reference or TPU state experiment has executed for this supplement.
M4 requires both the 79-case result and this saved-state result.
Refer to [the preparation record](../../experiments/2026-10-04-bitsandbytes-tpu/results/nested-state-preparation.json).
