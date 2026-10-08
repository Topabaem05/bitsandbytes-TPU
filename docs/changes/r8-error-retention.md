# Kaggle error record correction

The submission worker can now retain a bounded error classification before the SDK raises its original error.
The previous HTTP 400 response body remains unavailable.
Its cause and remote execution state remain unknown.

The recorder examines only response bytes that the HTTP library has already stored.
It examines a maximum of 16 KiB and does not read a response stream.
It retains status codes, byte counts, a prefix hash, and fixed constraint classes.
It does not retain messages, headers, addresses, or credentials.
A classification identifies possible constraints; it does not establish their cause.

The worker verifies the recorder's exact source before authentication.
A record failure preserves the original response or exception.
The one-submission limit, TLS verification, deadline, and resource ownership remain unchanged.
The scientific ZIP and wrapper match the previous generation for identical inputs.
The recorder executes on the host and is absent from the scientific ZIP.

## Validation

The reviewer repeated 25 focused controls, two integration controls, and 39 existing controls.
All 66 passed with isolated service fixtures.
The reviewer also ran 14 portable controls before and after adoption; all passed.
These controls cover correct records, changed source, missing source, and record failures.
They made no service calls and read no credentials.

Run the portable controls from the repository root:

```sh
python -B experiments/2026-10-04-bitsandbytes-tpu/m8/tests_error_retention.py
```

The [selected result](../../experiments/2026-10-04-bitsandbytes-tpu/results/kaggle-error-retention.json) records the inspected sources and test summaries.
Actual Kaggle execution remains required.
M8 is unqualified, and accepted milestones remain three of eight.
