# Research plan

## Goal

Make the upstream bitsandbytes NF4 API operate correctly on a TPU.
Then measure a Pallas implementation against the correct reference path.

The plan has eight required milestones.
A milestone is completed only when its tests give `PASS` results and the reviewer accepts its test records.
Work that does not meet all requirements does not count as a completed milestone.
Port2TPU results do not count for these backend milestones.

## Milestones

| ID | Required result | Required test records | Status |
| --- | --- | --- | --- |
| M1 | Installable backend | Wheel installation, automatic registration, schema tests, and CPU differential tests | Completed: local CPU and package tests |
| M2 | Actual TPU execution | Fixed runtime, TPU device test records, and upstream API output on Colab | Completed: 42 API cases and four public transfers on Colab V6E1 |
| M3 | Upstream module state | Bias, gradients, saved state, and restoration in a new process | Completed: eight fixed cases restored in a new TPU process |
| M4 | Nested quantization | Upstream default statistics option and matching saved nested state | In progress: actual 79-case run failed 27 cases; arithmetic repair and saved-state results required |
| M5 | QLoRA execution | Twenty steps, frozen base weights, and restoration after step ten | In progress: private CPU preparation; TPU test required |
| M6 | Pallas NF4 kernel | Actual custom call, numerical tests, and compiler test records for memory use | In progress: two actual FP32 native cases passed; BF16 operand correction and compiler memory records required |
| M7 | Measured performance | Raw time samples and memory measurements on the same TPU | In progress: local collector controls passed; actual measurements require accepted M6 |
| M8 | Repeated cloud results | Matching source and inputs on Colab and Kaggle, with resource closure | In progress: bounded M2 repetition preparation passed local controls; actual device repetition and closure required |

Accepted milestones: **3 of 8**.
The reviewer accepted M1 on 2026-10-04.
The reviewer repeated 71 package tests on macOS with Python 3.12.14 and PyTorch 2.14.1.
This result does not show Linux PyTorch/XLA 2.9.0 or TPU execution.
The reviewer accepted M2 on 2026-10-08 after 46 Colab cases and 196 independent numerical comparisons passed.
The [M2 result](changes/r2-transfer-colab.md) includes the fixed Linux runtime, source inspection, complete archive, and resource closure.
The reviewer accepted M3 on 2026-10-08 after all eight cases passed restoration in a new TPU process.
The [M3 result](changes/r3-state-colab.md) includes 136 independent numerical comparisons, exact checkpoint pairs, and complete resource closure.
Update this count only after inspection of the test records.

## Source inspection before implementation

The user requested current sources for each design before further implementation.
The [2026-10-08 collection](research/2026-10-08/README.md) supplies 48 primary source groups and five technical notes.
The reviewer examined the collection and its version limits.
Source inspection does not complete an implementation milestone.

Use the [decision table](research/2026-10-08/decisions.md) before assigning the next worker task.
Keep the existing runtime until a separate compatibility inspection accepts a replacement.
Keep source-supported facts separate from hypotheses and proposed tests.

## Current blockers

The reviewer accepted the R1 Colab precision comparison on 2026-10-08.
Native precision `high` and `highest` passed all fixed gates in this diagnostic.
The `default` mode reproduced the earlier FP32 forward and input-gradient failures.
Use `highest` before graph construction for the next correctness experiments.
Refer to [the precision result](changes/r1-precision-colab.md).

The corrected CPU-to-XLA transfers and complete API42 matrix passed on Colab V6E1.
The experiment used the [bounded transport](changes/r2-allocation-transport.md) after the retained allocation failures and web diagnostic.
Refer to [the accepted transfer result](changes/r2-transfer-colab.md).
Do not replace the upstream classes or methods dynamically during execution.
M3 restoration in a new process passed its separate device experiment.
The next device requirement is M4 nested quantization with the default upstream statistics option.
The [nested preparation](changes/r4-nested-preparation.md) provides 79 fixed cases and an explicit plugin variant.
Those cases do not test saved nested state; M4 also requires that separate result.
The [state supplement](changes/r4-nested-state-preparation.md) passed 23 scientific controls and 391 cloud controls during preparation.
It requires an accepted 79-case TPU result before its own device execution.
The [first nested Colab attempt](changes/r4-bootstrap-failure.md) stopped at a bootstrap import before numerical work.
The [entry correction](changes/r4-bootstrap-repair.md) passed 15 local controls and is ready for a fresh Colab attempt.
Its [first retry](changes/r4-allocation-failure.md) received an allocation server error before payload execution.
The [browser reuse mode](changes/r4-browser-adoption.md) passed 44 local controls after the repeated CLI allocation failure.
Actual marker identification passed, and the [79-case run](changes/r4-nested-colab-failure.md) completed with 52 passes and 27 failures.
The exact nested statistics and code failures require an arithmetic repair under unchanged criteria.
The [arithmetic diagnostic](changes/r4-primitive-preparation.md) supplies an explicit mode to inspect reduction, division, and subnormal behavior.
Its focused preparation controls passed, with the separate macOS process-group test failure retained.
It requires a fresh qualified Linux CPU oracle before actual TPU execution.
Its [first browser allocation](changes/r4-primitive-allocation-wait.md) exceeded the planned wait before the marker or scientific code executed.
The reviewer closed the temporary tab and verified an empty server list and zero active usage.
The [actual V5E1 arithmetic run](changes/r4-primitive-v5-record-failure.md) passed its qualified CPU gate.
The native tensor view used fallback, and the first builder record failed metric serialization.
All 113 archive members and complete resource closure passed inspection.
The [metric correction](changes/r4-primitive-metric-repair.md) passed 24 controls before and after adoption.
Its fresh 56-member packet requires a new qualified CPU oracle and actual arithmetic observations.
The [second arithmetic run](changes/r4-primitive-parameter-failure.md) passed its new CPU gate and retained two builder rows.
All four output and input byte comparisons passed inspection.
The third case failed the parameter map requirement before it wrote its JSON record.
The reviewer verified all 120 archive members and complete resource closure.
Additional scalar parameters require source inspection and exact value observations before further arithmetic conclusions.
The [parameter correction](changes/r4-primitive-parameter-repair.md) admits the two typed clamp parameters and retains failed observations.
Its 67 independent controls and 44 focused controls after adoption passed.
The new 56-member packet requires fresh qualified CPU and actual TPU observations.
The [complete arithmetic experiment](changes/r4-arithmetic-colab.md) retained all 32 rows and passed record validation.
Its 265 byte comparisons include 229 passes and 36 failures; both clamp scalar source gates passed.
The reviewer verified all 406 archive members and complete resource closure.
The observed output bits now supply the basis for the arithmetic correction.
The failed result cannot authorize the saved-state supplement or an accepted M4 dependency.
The private CPU preparations for M4, M5, and M6 do not qualify those device milestones.
The [bounded native preparation](changes/r6-native-cloud-preparation.md) uses accepted M2 sources and can execute independently of the M4 repair.
It requires actual native graph and output records before a separate compiler memory experiment.
The [first native attempt](changes/r6-native-archive-failure.md) stopped at a remote archive import after five CPU reference cases completed.
The native device phase did not execute; a fresh execution remains required.
The [archive entry correction](changes/r6-native-archive-repair.md) passed 12 new controls and seven existing bootstrap controls.
Its fresh packet retains all 23 native files and is ready for another Colab run.
The [actual retry](changes/r6-native-colab-errors.md) completed with four native case errors and one reference case error.
The source, archive, and resource closure records passed inspection.
Pallas lowering, tensor wrapping, and metric serialization require separate corrections before another native qualification attempt.
The [combined correction](changes/r6-native-repair.md) passed local source, diagnostic, and Pallas conversion controls.
It requires a fresh qualified CPU oracle and actual TPU execution under the new source manifest.
Its fresh V6E1 allocation exceeded the planned wait before any code executed; resource closure passed inspection.
The [V5E1 connection diagnostic](changes/colab-v5-connection.md) then connected in a new empty notebook.
The reviewer terminated that runtime and verified zero active usage without scientific execution.
The [explicit hardware admission](changes/colab-v5-adoption.md) passed 48 local controls after adoption.
A fresh V5E1 experiment requires its new packet, exact runtime record, marker, and qualified CPU oracle.
The [actual V5E1 run](changes/r6-native-v5-mosaic.md) passed its CPU gate but failed during Mosaic module loading.
The compiler supported versions through 7; the first native payload used version 8.
Four subsequent cases failed during their initial XLA synchronization.
All records were recovered, and resource closure passed inspection.
Inspect the pinned serialization passes before the next native attempt.
The [Mosaic conversion preparation](changes/r6-mosaic-preparation.md) passed 56 independent controls.
The official versioned passes preserve the fixed kernel's current intermediate representation exactly.
The reviewer verified all 73 packet members and the explicit JAX 0.7.1 CPU interpreter.
The new generation requires fresh Linux CPU and actual TPU records.
The [portable conversion controls](changes/r6-mosaic-portable-controls.md) passed all 35 cases before and after adoption.
The corrected rejection helper also rejects a fake successful verifier; production sources remain unchanged.
Its [V5E1 allocation attempt](changes/r6-mosaic-allocation-wait.md) remained pending for 703.63 seconds without scientific execution.
The reviewer closed the temporary tab and verified six closed CLI groups, no active session, and zero active usage.
The [fresh V5E1 experiment](changes/r6-mosaic-layout-failure.md) passed its five-case CPU gate and reached TPU layout compilation.
The first case failed an unsupported shape cast; the remaining four cases were not executed.
Independent conversion replay, all 110 archive members, and complete resource closure passed inspection.
The [gather decode correction](changes/r6-gather-decode.md) passed 52 independent controls and the canonical packet preflight.
Its source map preserves the original numerical criteria and requires a fresh qualified CPU oracle.
The [actual gather experiment](changes/r6-gather-colab.md) passed both FP32 native cases after its five-case CPU gate.
The BF16 matrix operation failed operand type validation; two later cases remained unexecuted.
Independent FP32 output and graph inspection, conversion replay, all 130 archive members, and complete resource closure passed.
Correct the BF16 operand types under the unchanged precision and numerical requirements.
The complete native dependency remains unaccepted.
The [measurement preparation](changes/r7-measurement-preparation.md) passed 64 local controls and nine independent retained-record checks.
Actual M7 collection requires accepted M6 results.
BF16 gradients remain outside the fixed diagnostic scope.

The [Kaggle readiness inspection](changes/r8-kaggle-readiness.md) verified CLI 2.2.4 access on 2026-10-09.
The reviewer repeated 16 offline SDK version controls; all passed.
That readiness inspection did not submit a job or establish actual device capacity.
The [M2 repetition package](changes/r8-kaggle-m2-preparation.md) passed 39 canonical controls after adoption.
Its fresh private candidate requires exact source, version, runtime, output, and independent closure records.
The batch CPU gate precedes device execution; root review occurs after execution.
An accepted M2 repetition supplies partial M8 evidence only.
The [first actual submission](changes/r8-kaggle-submission-failure.md) returned HTTP 400 after the server saved a draft.
The draft session was off, and exact `v1` source and session requests returned HTTP 404.
No accepted execution identity or scientific result exists for that request.
All five local submission and inspection groups closed; the owner preserves its unconfirmed remote state.
The [error record correction](changes/r8-error-retention.md) passed 80 independent controls.
All 14 portable controls also passed after adoption.
Future submissions can retain bounded response classifications without changing the original SDK error.
The previous HTTP 400 body remains unavailable, and its cause remains unknown.
The [V5E8 request generation](changes/r8-v5-request.md) passed 62 independent controls.
Nine portable request controls passed before and after adoption.
It requires exact saved-version hardware readback and a new experiment identity.
The [V5E8 submission](changes/r8-v5-source-size-failure.md) returned HTTP 400 with a retained `SOURCE_SIZE` classification.
The exact draft remained inactive, both `v1` requests returned HTTP 404, and all five local groups closed.
A smaller transport script must preserve the identical scientific ZIP before another reviewed submission.
The [public payload transport](changes/r8-public-payload.md) passed 40 controls before and after adoption.
It admits the complete wrapper and adds one transport record to the unchanged scientific archive.
The actual public ZIP readback passed, and Kaggle saved the exact private version.
The [saved-version continuation](changes/r8-saved-version-continuation.md) passed 41 controls before and after adoption.
It retrieves that version within the original deadline and requires independent provider closure.
Actual scientific acceptance remains required.
Actual immutable download, provider execution, and resource closure remain required.
M8 requires an identified version, the accepted runtime and scientific results, and independent resource closure.
The earlier [cloud inspection](research/2026-10-08/cloud.md) remains a dated source record.

## Work order

1. Keep the accepted source inspection and R1 precision records as the basis for the next changes.
2. Keep the accepted public transfer correction and its qualified Linux source controls.
3. Use the accepted M2 source, runtime, inputs, and `highest` precision for dependent experiments.
4. Keep the accepted M3 state protocol, exact checkpoint requirements, and process ownership for dependent work.
5. Complete M4 nested quantization with the identified reference implementation.
6. Complete M5 QLoRA execution and checkpoint restoration.
7. Qualify M6, then measure M7 on the same TPU as the correct reference path.
8. Repeat the accepted payload on Kaggle for M8, with independent version and closure records.

The bounded M6 native diagnostic can proceed during M4 repair because it uses the separate accepted M2 source variant.
This independent work does not qualify nested execution or change the M5 dependency.

Task identifiers R1 through R8 in the decision table define the remaining outputs and acceptance conditions.
Use GPT-6.1 sol with high reasoning for worker tasks.
The reviewer accepts each result before dependent work starts.
Commit and push each accepted logical change with its change log entry.

## First experiment

Use NF4 with block size 64 and `uint8` storage.
Do the FP32 tests and BF16 tests independently.
Set `compress_statistics=False` for this first experiment.
Use the upstream CPU implementation to calculate reference data in the qualified Linux runtime.
Record the source, inputs, reference data, tolerances, and their hashes before the TPU test.

Record CUDA comparison as `NOT_RUN` until an actual CUDA test exists.
Do not describe CPU reference data as CUDA test records.

## QLoRA experiment

Use two linear layers with width 256 and adapter rank 4.
Use a batch size of 8, FP32 data, and mean squared error.
Use SGD with a learning rate of 0.01 for 20 steps.
Save the model after step 10.
Restore the model in a new process.
Compare the restored result with the uninterrupted result.

## Performance experiment

Use five warm-up iterations.
Then measure three groups of 30 iterations.
Record compilation time independently from execution time.
Record the TPU type, compiler version, selected kernel, and all raw samples.
Compare both paths on the same hardware with the same data.

Report compiler memory estimates independently from allocator measurements.
Do not report an unmeasured peak as a measured result.

## Limits

The initial plan excludes FP4, eight-bit optimizers, and full bitsandbytes feature parity.
Automatic Hugging Face device placement and multiple-host execution need separate requirements and tests.
A failure stays a failure until a corrected implementation meets the unchanged test criteria.
