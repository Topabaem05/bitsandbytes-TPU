# Test requirements

## Rules for test records

Record the source commit, runtime lock hash, input hash, and test profile for each experiment.
Set tolerances before the experiment.
Keep failed results with their actual status.

Use these result labels:

| Label | Meaning |
| --- | --- |
| `PASS` | The required test gave a `PASS` result with full test records. |
| `FAIL` | The test completed but did not meet a requirement. |
| `BLOCKED` | A necessary dependency prevented the test. |
| `NOT_RUN` | The test did not execute. |

Use synthetic fixtures to examine the result reader.
They do not show actual device behavior.

## Package tests

1. Build the wheel.
2. Install the wheel in an isolated environment.
3. Import the upstream bitsandbytes package in a new process.
4. Make sure that the entry point adds the required XLA implementations.
5. Make sure that a second registration has no more effect.
6. Make sure that import does not initialize an accelerator client.
7. Use an incorrect upstream version and an incorrect operator schema for the tests.
8. Make sure that registration keeps existing implementations.

## Numerical tests

Do tests of all 16 NF4 codes and both positions in each packed byte.
Use input lengths 1, 2, 63, 64, 65, 127, 128, and 129 for the tests.
Include zero blocks, partial blocks, quantization boundaries, and adjacent values.
Compare packed data, scales, restored data, matrix output, and bias.
Use two-dimensional and three-dimensional input tensors for the tests.

For nested quantization, compare the offset, second state, scales, and saved state.
For gradients, compare the input, bias, and adapter gradients.
Make sure that packed base weights do not change.

## Actual TPU tests

1. Install the fixed runtime from the wheel lock.
2. Record the installed package versions and device type.
3. Calculate the CPU reference data and write its hashes.
4. Start the TPU probe in a separate process.
5. Make sure that the device hardware is a TPU.
6. Record the loaded backend source and selected operator implementations.
7. Execute the upstream public API calls.
8. Save the raw output arrays and XLA metrics.
9. Make sure that the test did not use an unintended CPU path.
10. Compare the output with the sealed reference data.

## State and restoration

Save the upstream `state_dict` and quantization metadata.
Restore them in a new process.
Compare the model output and all required state fields.
Use an incorrect state file to make sure that the result reader rejects it.

## Resource closure

Limit each cloud experiment to one allocation.
Reserve time for test records retrieval and resource closure.
Use an overall time limit of 3600 seconds.
Limit installation to 1200 seconds and scientific execution to 1800 seconds within that overall limit.

For Colab, stop the runtime and get an empty server list.
For Kaggle, record the fixed terminal version and make sure that Draft is off.
Do not accept an experiment until its test records include resource closure.
