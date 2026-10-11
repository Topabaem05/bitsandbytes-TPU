# Actual integer diagnostic

The reviewed V2 diagnostic ran on Colab V5E1 with the fixed Linux runtime.
All 24 CPU probes passed their independent references.
All 24 TPU probes returned without execution errors, and all graph observations completed.
This diagnostic does not qualify M4.
Accepted milestones remain three of eight.

The diagnostic normalized 11 native functional TPU inputs before the frozen backend call.
Their data types, shapes, strides, and devices were preserved.
The frozen arithmetic and selected inputs remained unchanged.

The returned records show these results:

| Probe | Actual result |
| --- | --- |
| Four bitcast cases | Exact CPU and TPU agreement |
| Flat reductions of length 1 and 255 | Explicit dimensions 0 and -1 agree; omitted dimensions disagree |
| Left shift | Legacy and explicit Tensor forms agree |
| Right shift | Explicit Tensor form agrees; legacy form disagrees |
| Eight scalar comparison and selection probes | Exact numerical agreement |
| Five original mathematical families | All disagree |
| Five families with explicit Tensor shifts | Four agree; tail scale remains incorrect |
| Five families with explicit Tensor shifts and flat tail dimension 0 | All agree, including tail codes and scales |

Fifteen of the 24 diagnostic rows agree exactly.
The other rows include deliberate legacy controls.
This fraction is not a milestone completion rate.

The five corrected graphs still contain S64 scalar promotion, constants, broadcasts, and parameters used in comparisons.
They do not establish the required S32 graph.
The reviewer kept the original trace formats, parameter roles, and no-S64 graph witness unchanged.
The next source generation must use checked S32 Tensor constants for the affected comparisons.
Fresh CPU79, TPU79, root acceptance, and saved state8 remain required.

The experiment completed 209.44 seconds after its original intent began.
The original deadline was 600 seconds.
The host made 55 official CLI commands and no CLI allocation attempts.
All 62 owned local groups were absent at closure.
A fresh CLI inspection showed no sessions and zero active assignments.
The browser showed a disconnected TPU runtime.
Usage statements cover only values displayed by the CLI.

The reviewer independently checked 48 row files and 48 compressed graph objects.
The graph objects contain 22,903,304 uncompressed bytes.
Their sizes, hashes, single gzip members, and complete stream endings passed inspection.
The recovered archive also passed complete member readback.

The complete records are in the owner-only [private archive](https://drive.google.com/file/d/1oXlYrN3ADCJoL2gWER8nXvM2c6-5td3R/view).
It contains 507 members and has 7,534,576 compressed bytes.
Its SHA-256 is `a9a54f4f794f73565ca637dfafbef1c7661cc50d7b4a377d576b43abe07e7092`.
An independent download passed the complete archive hash, member hashes, sizes, and CRC checks.

The first deletion attempt correctly rejected source drift before deleting any root.
The adviser had removed copied graph lines already preserved in the private archive and added two compact review records.
The reviewer verified those three exact changes and preserved both new records in Git.
All other archive snapshot files were unchanged.
Seven completed local roots and the temporary archive copy were then deleted.
No new output was added to `.work`.

Selected [root review](../../experiments/2026-10-04-bitsandbytes-tpu/results/m4-word-diagnostic-v2-actual/root-review.json),
[independent review](../../experiments/2026-10-04-bitsandbytes-tpu/results/m4-word-diagnostic-v2-actual/adviser-independent-review.json),
and [disposal record](../../experiments/2026-10-04-bitsandbytes-tpu/results/m4-word-diagnostic-v2-actual/source-disposal.json)
preserve the exact scope and limitations.
This document uses short technical English sentences.
A complete STE dictionary review was not available.
