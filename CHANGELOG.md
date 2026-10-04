# Change log

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
