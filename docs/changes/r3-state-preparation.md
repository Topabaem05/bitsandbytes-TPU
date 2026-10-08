# R3: State restoration experiment preparation

The M3 experiment uses the accepted M2 package source, runtime, inputs, and native precision `highest`.
It covers all eight fixed linear cases.
The cases include FP32, BF16, rank two, rank three, and both bias options.

A coordinator starts separate save and restore processes.
Each process has a distinct PID and token within the externally owned process group.
The coordinator records each actual launch before it waits.
It confirms child exit and absence before the next launch.
The outer owner can close the complete group if the coordinator cannot finish its cleanup.

The save process uses the original public module constructor and writes plain CPU tensors.
The restore process uses fresh public objects, `Params4bit.from_prequantized`, and strict `load_state_dict(assign=False)`.
The verifier requires exact checkpoint bytes, tensor state, and output agreement between save and restore.
It also compares both phases with the fixed CPU oracle under the existing numerical gates.
FP32 cases include input gradients and applicable bias gradients.
BF16 gradients remain outside this scope.

The records bind source, inputs, precision, device placement, HLO, execution counters, and actual process identities.
The verifier requires complete numerical gates and rejects a successful row that contains error metadata.
A case exception remains an error, including an exception after structural validation.
The owner recovers valid numerical failures for review and retains the existing closure requirements.

Validation: The reviewer repeated 239 portable controls; all passed in 26.08 seconds.
The controls include source admission, archive recovery, checkpoint corruption, numerical failures, process identity, and cleanup failures.
The selected CPU fixture contains tensor records; tests create checkpoint files only in temporary directories.
Synthetic device records do not show TPU execution.

Two earlier macOS cleanup fixtures reported a permission error during a group probe after an immediate coordinator kill.
Those original records remain unverified; later observations found the processes absent.
The corrected positive fixture first observes the running child's PID, parent PID, and group.
A separate permission-error control requires an unverified cleanup result.
The production cleanup implementation is unchanged; the immediate-launch behavior remains unresolved.

The reviewer inspected all 40 packet members and the installed CLI identity.
The new packet retains 34 files from the accepted M2 packet without changes.
Refer to [the preparation record](../../experiments/2026-10-04-bitsandbytes-tpu/results/state-preparation.json).

The actual TPU experiment remains required.
M3 remains unqualified, and accepted milestones remain **2 of 8**.
