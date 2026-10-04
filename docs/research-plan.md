# Research plan

## Goal

Make the original bitsandbytes NF4 API operate correctly on a TPU.
Then measure a Pallas implementation against the correct reference path.

The plan has eight required milestones.
A milestone is complete only when its tests pass and the reviewer accepts its evidence.
Partial work does not count as a complete milestone.
Previous Port2TPU results do not establish completion for this backend.

## Milestones

| ID | Required result | Completion evidence | Status |
| --- | --- | --- | --- |
| M1 | Installable backend | Wheel installation, automatic registration, schema tests, and CPU differential tests | Complete: local CPU and package tests |
| M2 | Actual TPU execution | Exact runtime, TPU device evidence, and original API output on Colab | In progress |
| M3 | Original module state | Bias, gradients, saved state, and restoration in a new process | Not started |
| M4 | Nested quantization | Original default statistics option and matching nested state | Not started |
| M5 | QLoRA execution | Twenty steps, frozen base weights, and restoration after step ten | Not started |
| M6 | Pallas NF4 kernel | Actual custom call, numerical tests, and compiler evidence for memory use | Not started |
| M7 | Measured performance | Raw time samples and memory measurements on the same TPU | Not started |
| M8 | Repeated cloud results | Matching source and inputs on Colab and Kaggle, with resource closure | Not started |

Accepted milestones: **1 of 8**.
M1 passed review on 2026-10-04.
The reviewer repeated 71 package tests on macOS with Python 3.12.14 and PyTorch 2.14.1.
This result does not establish Linux PyTorch/XLA 2.9.0 or TPU execution.
Update this count only after evidence review.

## Work order

1. Complete the package and CPU tests for M1.
2. Resolve the runtime wheels and prepare the public API probe for M2.
3. Review the local results before TPU allocation.
4. Execute the smallest public API probe on Colab.
5. Complete M3 and M4 before the QLoRA test.
6. Complete M5 before the Pallas performance comparison.
7. Complete M6 and M7 with the same input data and TPU type.
8. Repeat the accepted configuration on Kaggle for M8.

The runtime resolver and the package can advance in parallel.
The reviewer accepts each result before dependent work starts.

## First experiment

Use NF4 with block size 64 and `uint8` storage.
Test FP32 and BF16 separately.
Set `compress_statistics=False` for this first experiment.
Use the original CPU implementation to produce reference data in the qualified Linux runtime.
Seal the source, inputs, reference data, and tolerances before the TPU test.

Record CUDA comparison as `NOT_RUN` until an actual CUDA test exists.
Do not describe CPU reference data as CUDA evidence.

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
Record compilation time separately from execution time.
Record the TPU type, compiler version, selected kernel, and all raw samples.
Compare both paths on the same hardware with the same data.

Report compiler memory estimates separately from allocator measurements.
Do not report an unmeasured peak as a measured result.

## Limits

The initial plan excludes FP4, eight-bit optimizers, and full bitsandbytes feature parity.
Automatic Hugging Face device placement and multiple-host execution need separate requirements and tests.
A failure remains a failure until a corrected implementation passes the unchanged test criteria.
