# Project instructions

## Scope

Use the original bitsandbytes public API as the product boundary.
Read `docs/research-plan.md` before implementation work.
Do not treat previous Port2TPU results as evidence for this backend.

## Documents

Write project documents in English under ASD-STE100 Issue 9.
Follow `docs/writing-guide.md` and `docs/terms.md`.
Keep source identifiers unchanged.
Record each logical change in `CHANGELOG.md`.

## Execution

Use the worker model and review process that the user requested.
Workers must keep their changes within the assigned files.
Workers must not commit or push unless the reviewer assigns that action.
The reviewer must examine the change and its test evidence.
Then commit and push each accepted logical change.

Keep local environments, wheel bodies, credentials, and raw provider logs outside Git.
Use `.work` for local test output.
Use small, selected fixtures for tests in Git.

## Claims

Separate CPU tests, actual TPU tests, and performance measurements.
Retain failures and tests that did not execute.
Do not change a tolerance after an experiment to obtain a pass.
Do not report completion without the required evidence.
