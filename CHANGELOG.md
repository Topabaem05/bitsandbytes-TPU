# Change log

## 2026-10-04 — Device route Colab result

- Record 34 nonlinear TPU cases that passed the fixed numerical gates.
- Keep two CPU-to-XLA transfer errors.
- Record FP32 forward and input-gradient failures on two alternative paths.
- Record two BF16 path passes and four reconstruction passes within the same process.
- Keep full API42, M2, and M3 unqualified.

Validation: All 40 diagnostic records passed record validation; the numerical result was `FAIL`.
The 238-member archive passed hash and byte checks.
The session stopped, the server was empty, and active usage was zero.
The reviewer repeated archive and report readback on 2026-10-08.
Refer to [the device route result](docs/changes/task2-colab-routes.md).

## 2026-10-04 — Device route diagnostic preparation

- Add 34 fixed nonlinear cases and six module path cases.
- Keep the CPU transfer error separate from alternative path results.
- Identify the CPU checkpoint used by the restoration path.
- Add independent gradient checks and the existing floating-point restoration tolerances.
- Keep diagnostic record acceptance separate from M2 and M3 completion.

Validation: The reviewer repeated 22 diagnostic tests and 49 owner tests; all passed.
The tests include correct records, incorrect records, simulated cloud operations, archive recovery, and process cleanup.
Actual TPU execution of this diagnostic is still required.
Refer to [the diagnostic change record](docs/changes/task2-device-route-preparation.md).

## 2026-10-04 — Zero-block normalization correction

- Use a safe denominator for zero blocks and partial zero blocks.
- Keep zero elements and odd padding at NF4 code `7`.
- Add a failure fixture and isolated CPU subnormal controls.
- Keep the fixed device cases and numerical tolerances unchanged.

Validation: The reviewer repeated 81 package tests; all passed in 22.05 seconds.
The previous calculation failed the zero control, and the corrected calculation passed it.
The TPU repeat and CPU-to-TPU parameter transfer correction are still required.
Refer to [the zero-block change record](docs/changes/task1-zero-normalization.md).

## 2026-10-04 — Native wrapper device result

- Record the successful TPU runtime probe and all 42 CPU reference cases.
- Record 17 executed TPU cases, with nine partial passes and eight packed-code failures.
- Keep the incompatible tensor type failure during `Params4bit._quantize`.
- Keep the full 42-case result and M2 incomplete.

Validation: Independent saved-array comparisons used the same fixed gates and gave matching results.
All eight all-zero quantization cases failed packed-code equality.
The 171-member result archive passed hash and byte checks.
The session stopped, the server was empty, and active usage was zero.
Refer to [the native wrapper result](docs/changes/task2-colab-native.md).

## 2026-10-04 — Native tensor wrapper correction

- Replace the functorch transform with native functional tensor wrappers.
- Keep the caller's included dispatch keys and restore the local context.
- Apply fixed-size input updates and reject metadata changes.
- Add an isolated factory test that reproduces the second TPU assertion.

Validation: The reviewer repeated 79 package tests; all passed in 20.08 seconds.
The old transform failed the factory control, and the native wrapper passed all four kernel controls.
The arithmetic, fixed inputs, and numerical tolerances stayed unchanged.
An actual TPU repeat is still required.
Refer to [the native wrapper change record](docs/changes/task1-native-functionalization.md).

## 2026-10-04 — Functionalization patch device result

- Record the second successful runtime probe and 42 CPU reference cases.
- Record a different assertion during NF4 tensor creation.
- Keep the attempted TPU result as blocked, with zero completed cases.
- Keep both raw runs and their resource closure records.

Validation: The reviewer checked all 120 archive members and the sealed CPU records.
The session stopped, the server was empty, and active usage was zero.
The first functionalization patch did not qualify the TPU backend.
M2 is not complete; the diagnosis must account for the XLA tensor wrapper.
Refer to [the repeat result](docs/changes/task2-colab-functionalized.md).

## 2026-10-04 — XLA functionalization correction

- Apply a local functionalization context to each Python XLA kernel.
- Add a functionalization path for the mutable dequantization overload.
- Keep output mutations and the selected device backend.
- Extend registration conflict and rollback tests.

Validation: The reviewer repeated 77 package tests; all passed in 22.07 seconds.
The original NF4 arithmetic, 42 inputs, and numerical tolerances stayed unchanged.
The actual TPU repeat is still required.
Refer to [the functionalization change record](docs/changes/task1-functionalization.md).

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
