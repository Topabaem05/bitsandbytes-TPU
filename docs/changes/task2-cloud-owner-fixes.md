# Task 2: Packet admission and TPU attempt records

## Change

Before it creates a CLI instance, the owner examines the loose `manifest.json` file.
The file must be a regular file, not a symbolic link.
Its bytes must match the archive manifest bytes.
A change to the loose file alone prevents allocation.

Before the TPU child starts, the remote receipt records `tpu_attempted: true` and `tpu_status: RUNNING`.
An unqualified exit, child exception, or cleanup failure gives `ATTEMPTED_BLOCKED` and a `BLOCKED` receipt.
The receipt keeps the child record and cleanup errors.
Exit code `0` with a clean child result still requires local review.
Exit code `2` with a clean child result still records the numerical result as `FAIL`.
A child error cannot give a pass, even if its exit code is `0`.

## Checks

The reviewer repeated all 24 focused cloud tests; all passed in 0.17 seconds.
The tests include a matching packet, loose manifest byte changes, a symbolic link, and a directory.
Each rejected loose manifest prevented CLI construction and output creation.

The TPU fixtures include exit codes `0`, `1`, and `2`, an exception, and cleanup failures.
They examine the saved running receipt before the simulated child returns.
These fixtures do not allocate hardware or execute NF4.

The first Colab test records stay unchanged.
Their saved receipt still says `NOT_RUN`, as recorded during that attempt.
The first-test report explains that a TPU attempt occurred before the failure.

No new packet or cloud run was made.
The scientific source, input data, tolerances, runtime lock, and package source manifest were not changed by this work.
