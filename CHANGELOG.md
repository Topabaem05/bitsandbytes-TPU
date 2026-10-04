# Change log

## 2026-10-04 — Packet and attempt record corrections

- Compare the loose manifest with the archive before allocation.
- Reject a symbolic link or directory in place of that manifest.
- Record a TPU attempt before the child starts.
- Keep numerical failures separate from execution and cleanup failures.

Validation: The reviewer repeated 24 isolated tests; all passed in 0.17 seconds.
The first Colab records stayed unchanged.
Refer to [the owner correction record](docs/changes/task2-cloud-owner-fixes.md).

## 2026-10-04 — First actual Colab result

- Record successful installation and the TPU runtime probe.
- Record all 42 completed CPU reference cases.
- Keep the NF4 TPU assertion as a blocked result with zero completed cases.
- Record full result retrieval and resource closure after the failure.

Validation: The reviewer checked the full archive and all 42 sealed CPU records.
The archive has 120 members and 82,927 bytes; all hashes and recovered bytes matched.
The session stopped, the server was empty, and active usage was zero.
M2 is not complete. No numerical tolerance changed.
Refer to [the first Colab result](docs/changes/task2-colab-first.md).

## 2026-10-04 — Colab experiment owner

- Add one Colab allocation with fixed source and runtime files.
- Keep runtime, CPU reference, and TPU computation in separate processes.
- Retrieve the full result archive, including nested receipts and partial records.
- Attempt resource closure after failures and expired work limits.

Validation: The reviewer repeated 12 isolated tests and the fixed packet preflight; all passed.
The installed CLI identity also passed inspection without allocation.
The actual experiment stays `NOT_RUN` at this preparation step.
Refer to [the cloud change record](docs/changes/task2-cloud-preparation.md).

## 2026-10-04 — State and gradient experiment preparation

- Add state reconstruction through the upstream public methods.
- Add separate CPU, TPU save, and TPU restore phases.
- Keep eight fixed linear cases and the existing numerical tolerances.
- Check FP32 input and bias gradients with frozen base weights.

Validation: The reviewer repeated 22 tests and eight subtests; all passed in 6.28 seconds.
The Mac helper run passed all eight cases with no numerical or state differences.
Qualified Linux and TPU execution stay `NOT_RUN`; M3 is not complete.
Refer to [the state change record](docs/changes/task3-state-preparation.md).

## 2026-10-04 — Controlled document terms

- Replace general words with simpler words from the writing standard.
- Define more computer terms in the project glossary.
- State the finite-input preconditions in the main design and README.
- Record the local work for the module state milestone.

Validation: The reviewer examined meanings, sentence lengths, and document links.
No program source, fixed input, runtime lock, or numerical tolerance changed.
This inspection does not give independent ASD-STE100 certification.

## 2026-10-04 — NF4 operations for XLA

- Replace bucket selection with strict comparisons and an integer sum.
- Keep the finite-input numerical results and midpoint rules.
- Remove added value assertions that have no identified native XLA implementation.
- Require finite inputs and finite nonnegative scales as caller preconditions.
- Keep structural errors and the prohibition on CPU fallback.

Validation: The reviewer repeated 71 tests; all passed in 24.82 seconds.
The fixed TPU test profile and numerical tolerances did not change.
Actual TPU execution stays `NOT_RUN`.
Refer to [the primitive change record](docs/changes/task1-xla-primitives.md).

## 2026-10-04 — Public API experiment preparation

- Add 42 fixed NF4 cases and a separate CPU reference procedure.
- Add source and runtime identities to the experiment records.
- Keep the upstream BF16 module dtype and small-tensor transpose behavior.
- Add result checks for missing data, CPU execution, and numerical differences.

Validation: The reviewer repeated 25 isolated tests; all passed in 1.38 seconds.
These tests use synthetic records and do not show actual device behavior.
The CPU reference experiment and TPU experiment stay `NOT_RUN`.
Refer to [the probe change record](docs/changes/task2-probe.md).

## 2026-10-04 — Initial NF4 backend package

- Add the installable `bitsandbytes-tpu` package and automatic backend registration.
- Keep the upstream classes and four operator schemas.
- Add NF4 reference operations without nested quantization.
- Add source checks, registration conflict checks, and package installation tests.
- Include the Apache-2.0 and upstream MIT license texts in the wheel.

Validation: The reviewer repeated 71 tests; all passed in 24.00 seconds.
The test environment used macOS, Python 3.12.14, and PyTorch 2.14.1.
M1 is complete. Actual TPU execution stays `NOT_RUN`.
Refer to [the package change record](docs/changes/task1.md).

## 2026-10-04 — Fixed Linux runtime files

- Add a fixed wheel lock for the initial PyTorch/XLA runtime.
- Add Linux dependency checks and complete wheel integrity checks.
- Add a separate TPU runtime probe.
- Add 16 tests for valid and incorrect runtime files.

Validation: The reviewer repeated all 16 tests and the complete checks for 35 wheels.
The wheel files contain 572,327,598 bytes.
Actual TPU execution stays `NOT_RUN`.
Refer to [the runtime change record](docs/changes/task2-runtime.md).

## 2026-10-04 — Research baseline

- Set the project name to bitsandbytes-TPU.
- Set the product boundary to the upstream bitsandbytes NF4 API.
- Define eight milestones with separate completion test records.
- Define the initial PyTorch/XLA runtime candidate.
- Add the design, test requirements, writing guide, and technical glossary.
- Add project instructions and exclusions for local files.

Validation: The reviewer examined the source references, milestone requirements, and document links.
Implementation status: No backend milestone is complete at this baseline.
