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
| M2 | Actual TPU execution | Fixed runtime, TPU device test records, and upstream API output on Colab | In progress: parameter transfer and full API42; precision comparison accepted |
| M3 | Upstream module state | Bias, gradients, saved state, and restoration in a new process | In progress: local preparation and reconstruction within one process |
| M4 | Nested quantization | Upstream default statistics option and matching nested state | In progress: source inspection and local reference data |
| M5 | QLoRA execution | Twenty steps, frozen base weights, and restoration after step ten | Not started |
| M6 | Pallas NF4 kernel | Actual custom call, numerical tests, and compiler test records for memory use | In progress: source inspection; layout candidate has not compiled |
| M7 | Measured performance | Raw time samples and memory measurements on the same TPU | Not started |
| M8 | Repeated cloud results | Matching source and inputs on Colab and Kaggle, with resource closure | Not started |

Accepted milestones: **1 of 8**.
The reviewer accepted M1 on 2026-10-04.
The reviewer repeated 71 package tests on macOS with Python 3.12.14 and PyTorch 2.14.1.
This result does not show Linux PyTorch/XLA 2.9.0 or TPU execution.
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

The original CPU-to-XLA transfers failed during parameter data assignment in the earlier run.
The reviewer accepted the explicit source correction and its local controls for a separate device experiment.
Two Colab requests ended during allocation before this experiment could execute.
The same payload failed with `requests.ReadTimeout` during each assignment POST.
Both owners confirmed empty server lists, zero active usage, and closed local processes.
Examine the allocation request path before another attempt.
Refer to [the allocation repeat](changes/r2-colab-allocation-repeat.md).
The subsequent [web diagnostic](changes/r2-web-allocation-diagnostic.md) obtained a V6E1 runtime after more than 175 seconds.
The reviewer closed that runtime and confirmed empty session lists and zero CLI usage.
The [bounded transport](changes/r2-allocation-transport.md) passed local review with one assignment POST and a 360-second command limit.
Use this transport for the next Colab experiment with the same scientific payload.
Keep that transfer requirement open until its own test passes.
Do not replace the upstream classes or methods dynamically during execution.
The next transfer experiment must also execute the complete fixed API42 matrix.
The full API42 matrix and M3 remain unqualified.
Restoration in a new process requires separate device records.
BF16 gradients remain outside the fixed diagnostic scope.

The previous Kaggle inspection found no usable CLI credentials.
This source inspection did not repeat account authentication.
M8 needs CLI access and independently identified run records before acceptance.
The current CLI development source reports version-selection corrections that are absent from release 2.2.4.
Refer to [the cloud inspection](research/2026-10-08/cloud.md).

## Work order

1. Keep the accepted source inspection and R1 precision records as the basis for the next changes.
2. Examine the public CPU-to-XLA transfer correction with compatible and rejected-conversion controls.
3. Execute the transfer controls and complete M2 API matrix with `highest` on Colab.
4. Complete M3 restoration in a new process with the accepted source and runtime.
5. Complete M4 nested quantization with the identified reference implementation.
6. Complete M5 QLoRA execution and checkpoint restoration.
7. Qualify M6, then measure M7 on the same TPU as the correct reference path.
8. Repeat the accepted payload on Kaggle for M8, with independent version and closure records.

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
