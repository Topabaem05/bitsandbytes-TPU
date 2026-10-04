# Change log

## 2026-10-04 — Public API experiment preparation

- Add 42 fixed NF4 cases and a separate CPU reference procedure.
- Add source and runtime identities to the experiment records.
- Preserve the original BF16 module dtype and small-tensor transpose behavior.
- Add result checks for missing data, CPU execution, and numerical differences.

Validation: The reviewer repeated 25 isolated tests; all passed in 1.38 seconds.
These tests use synthetic records and do not establish actual device behavior.
The CPU reference experiment and TPU experiment remain `NOT_RUN`.
Refer to [the probe change record](docs/changes/task2-probe.md).

## 2026-10-04 — Initial NF4 backend package

- Add the installable `bitsandbytes-tpu` package and automatic backend registration.
- Preserve the original classes and four operator schemas.
- Add NF4 reference operations without nested quantization.
- Add source checks, registration conflict checks, and package installation tests.
- Include the Apache-2.0 and upstream MIT license texts in the wheel.

Validation: The reviewer repeated 71 tests; all passed in 24.00 seconds.
The test environment used macOS, Python 3.12.14, and PyTorch 2.14.1.
M1 is complete. Actual TPU execution remains `NOT_RUN`.
Refer to [the package change record](docs/changes/task1.md).

## 2026-10-04 — Fixed Linux runtime files

- Add an exact wheel lock for the initial PyTorch/XLA runtime.
- Add Linux dependency checks and complete wheel integrity checks.
- Add a separate TPU runtime probe.
- Add 16 tests for valid and incorrect runtime files.

Validation: The reviewer repeated all 16 tests and the complete checks for 35 wheels.
The wheel files contain 572,327,598 bytes.
Actual TPU execution remains `NOT_RUN`.
Refer to [the runtime change record](docs/changes/task2-runtime.md).

## 2026-10-04 — Research baseline

- Set the project name to bitsandbytes-TPU.
- Set the product boundary to the original bitsandbytes NF4 API.
- Define eight milestones with separate completion evidence.
- Define the initial PyTorch/XLA runtime candidate.
- Add the design, test requirements, writing guide, and technical glossary.
- Add project instructions and exclusions for local files.

Validation: The reviewer examined the source references, milestone requirements, and document links.
Implementation status: No backend milestone is complete at this baseline.
