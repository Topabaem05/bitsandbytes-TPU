# Test requirements

## Evidence rules

Record the source commit, runtime lock hash, input hash, and test profile for each experiment.
Set tolerances before the experiment.
Retain failed results with their actual status.

Use these result labels:

| Label | Meaning |
| --- | --- |
| `PASS` | The required test passed with complete evidence. |
| `FAIL` | The test completed but did not meet a requirement. |
| `BLOCKED` | A necessary dependency prevented the test. |
| `NOT_RUN` | The test did not execute. |

Synthetic fixtures test the result reader.
They do not establish actual device behavior.

## Package tests

1. Build the wheel.
2. Install the wheel in an isolated environment.
3. Import the original bitsandbytes package in a new process.
4. Make sure that the entry point adds the required XLA implementations.
5. Make sure that a second registration has no additional effect.
6. Make sure that import does not initialize an accelerator client.
7. Test an incorrect upstream version and an incorrect operator schema.
8. Make sure that registration preserves existing implementations.

## Numerical tests

Test all 16 NF4 codes and both positions in each packed byte.
Test input lengths 1, 2, 63, 64, 65, 127, 128, and 129.
Include zero blocks, partial blocks, quantization boundaries, and adjacent values.
Compare packed data, scales, restored data, matrix output, and bias.
Test two-dimensional and three-dimensional input tensors.

For nested quantization, compare the offset, second state, scales, and saved state.
For gradients, compare the input, bias, and adapter gradients.
Make sure that packed base weights do not change.

## Actual TPU tests

1. Install the exact runtime from the wheel lock.
2. Record the installed package versions and device type.
3. Produce and seal the original CPU reference data.
4. Start the TPU probe in a separate process.
5. Make sure that the device hardware is a TPU.
6. Record the loaded backend source and selected operator implementations.
7. Execute the original public API calls.
8. Save the raw output arrays and XLA metrics.
9. Make sure that the test did not use an unintended CPU path.
10. Compare the output with the sealed reference data.

## State and restoration

Save the original `state_dict` and quantization metadata.
Restore them in a new process.
Compare the model output and all required state fields.
Test an incorrect state file to make sure that the result reader rejects it.

## Resource closure

Limit each cloud experiment to one allocation.
Reserve time for evidence retrieval and resource closure.
Use an overall time limit of 3600 seconds.
Limit installation to 1200 seconds and scientific execution to 1800 seconds within that overall limit.

For Colab, stop the runtime and obtain an empty server list.
For Kaggle, record the exact terminal version and make sure that Draft is off.
An experiment is incomplete until the evidence includes resource closure.
