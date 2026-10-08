# Measurement preparation

This directory contains the M7 collector and local controls.
Actual TPU measurements have not executed.
M7 remains unqualified.

The protocol requires accepted M6 results and the same identified TPU for both paths.
Each path uses a new process, five warm-up iterations, and three groups of 30 measurements.
Each measurement includes the operator call and completion of its actual output.
Input setup and the first call have separate time records.
Backend compilation metrics have a separate scope and unit conversion.

The collector retains raw allocator readings outside the measured intervals.
A reported high-water value has an unverified measurement scope.
It does not establish peak memory for the selected operator.
An unavailable reading remains `UNKNOWN`.

The recovery verifier compares the phase, sequence, synchronization, raw metrics, and calculated fields.
It rejects the four incorrect records found during the first review.
The [change record](../../../docs/changes/r7-measurement-preparation.md) gives the sources, test results, and remaining requirements.
The [source list](sources.json) gives exact source revisions and hashes.
Source bodies are external references and are not included here.

Run the controls from the repository root:

```sh
python -B experiments/2026-10-04-bitsandbytes-tpu/measurement/run_controls.py --output .work/measurement-controls
```

Use a new output directory outside this source directory.
The command copies the reviewed sources before it generates test records.
It executes 64 local controls with a simulated clock, allocator, and device interface.
It does not allocate a TPU.
