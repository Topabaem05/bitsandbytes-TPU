# Task 2: Device route diagnostic

## Purpose

The third Colab run found incorrect zero codes and a CPU-to-TPU parameter transfer error.
This diagnostic tests the zero-block correction and three public module paths independently.
It does not replace the required 42-case API test.
It cannot complete M2 or M3.

## Cases

The diagnostic uses the 34 quantization and decode cases from the fixed API test.
It keeps the FP32 and BF16 inputs, CPU reference process, and numerical tolerances.

It also uses one fixed linear case for each data type with these paths:

1. Create the module on the CPU and call `.to(device)`.
2. Create the module on the TPU and quantize it on that device.
3. Restore the CPU reference checkpoint with public `Params4bit.from_prequantized`.

The third path tests checkpoint restoration.
It does not test TPU quantization of those weights.
The first path keeps its actual error if CPU-to-TPU transfer fails.
An error in another stage does not count as the known transfer error.

The linear cases include bias, saved state, and restoration in the same process.
The FP32 cases also include input and bias gradients.
An independent calculation compares the gradient reference with the sealed decoded weights.
Packed weights and saved state must match exactly.
Floating-point outputs use the existing numerical tolerances.
M3 still requires restoration in a new process.

## Execution records

Each of the 40 cases produces a result or an error record.
One case error does not remove the other cases.
The diagnostic keeps numerical failures separate from checks of the records.
`PASS_DEVICE_ROUTE_RECORDS` means that the records and resource closure passed their checks.
It does not mean that all module paths or numerical comparisons passed.

The owner keeps the fixed runtime and independent CPU reference process.
It allows one Colab allocation with the existing time limits.
It retrieves the full result archive and stops the owned session.
The source records identify the plugin, upstream package, diagnostic, inputs, and runtime.

## Limits

The diagnostic keeps the upstream classes and source unchanged.
The numerical correction changes only the plugin.
Local synthetic records do not show actual TPU behavior.
Actual execution is still required.

## Local inspection

The reviewer repeated 22 diagnostic tests and 49 owner tests independently.
The tests passed in 1.61 seconds and 1.50 seconds, respectively.
The reviewer then ran all 71 tests from the adopted paths; all passed in 3.10 seconds.
The owner tests use a simulated service and synthetic data.
They exercise actual archive recovery, the local record verifier, and local process cleanup.
Incorrect source records, gradient references, restored weights, transport bytes, and cleanup results cannot produce `PASS_DEVICE_ROUTE_RECORDS`.
A numerical failure stays a failure when its records pass their checks.

The diagnostic source SHA-256 is `feae751734e57c741b1bdade004ff7ca3c041ee7eb3b7086b531bc6be433038e`.
The fixed 42-case probe, profile, inputs, and runtime lock stay unchanged.
