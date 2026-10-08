# Native archive entry correction

Date: 2026-10-09.

The remote module now loads the admitted `transport.py` file from its exact sibling path.
The load does not depend on the notebook module search path or an existing `transport` module.
This change corrects the import failure in the [first native attempt](r6-native-archive-failure.md).

The owner first verifies the complete recovered archive.
It then determines whether the required native records exist.
If they are absent, the owner records `NOT_RUN_INCOMPLETE_NATIVE_RECORDS` and keeps the original error.
It also keeps an unsuccessful resource closure status.
Complete native records still require the original numerical and process verification.

## Tests

The reviewer repeated 12 new local controls and seven existing bootstrap controls; all passed.
These tests use synthetic science and service records with actual local processes.
They do not allocate a cloud resource or execute TPU science.

Three new processes use the actual `runpy.run_path` entry from an unrelated directory.
The original source reproduces the missing transport error.
The corrected source creates the archive parts.
A conflicting cached module does not change the corrected result.

Nine owner controls cover early command failure, a blocked remote receipt, cleanup failure, and a missing native parent.
They also cover complete native records, incorrect archive bytes, and an incorrect native protobuf.
The original blocked receipt reproduces the secondary missing-oracle error.
The corrected owner preserves the original failure and skips unavailable native verification.

The preparation retained earlier fixture and relocation failures.
Those failures included omitted source files and the build requirements lock file in an isolated source layout.
The reviewer repeated the final tests in the normal repository layout.
The production correction did not change during those fixture repairs.

## Fixed experiment

Only `cloud/remote.py` and `cloud/owner.py` changed in the next experiment payload.
All 23 native files, the M2 source variant, inputs, runtime, and numerical criteria retain their accepted hashes.
The reviewer inspected all 68 ZIP members.
The new packet contains 1,039,634 bytes.
Its SHA-256 is `138e9ac03520f6e2e643a62bc6dac77788bf4e0a19f34dc3aa570192fbfc7d10`.

The [result record](../../experiments/2026-10-04-bitsandbytes-tpu/results/native-archive-repair.json) supplies source hashes and test limits.
The corrected experiment is ready for a fresh Colab run.
M6 remains unqualified, and accepted milestones remain **3 of 8**.
