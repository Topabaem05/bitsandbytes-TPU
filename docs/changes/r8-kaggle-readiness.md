# Kaggle access and version selection

The reviewer verified installed Kaggle CLI 2.2.4 access on 2026-10-09.
The owned notebook list and quota requests completed successfully.
No additional login was necessary for these requests.
No job was submitted, and no credential changed.

The quota response reported 17.93 TPU hours remaining.
Its refresh value was `2026-10-10T00:00:00`, without a time zone.
Quota does not establish current device capacity or an accepted runtime.
M8 remains unqualified.

## Source and request inspection

The [released CLI implementation](https://github.com/Kaggle/kaggle-cli/blob/f0afa32699d28c97f82691728ada3ed8c16c5abf/src/kaggle/api/kaggle_api_extended.py) omits parsed version fields from status and output requests.
Those helpers cannot establish the requested version.
The installed [SDK 0.1.37](https://pypi.org/project/kagglesdk/0.1.37/) supplies typed version fields.
Eight relevant installed SDK files matched the published wheel.

| Operation | Required version field |
| --- | --- |
| Source, status, and output list | `version_label='v<N>'` |
| Output download | `version_number=N`, as an integer |

The [official development correction](https://github.com/Kaggle/kaggle-cli/blob/9c57dd74c29904c40ff7879eed33644be89d2e2f/src/kaggle/api/kaggle_api_extended.py) documents the `v<N>` form.
It is not the installed release.
The reviewer repeated 16 offline request controls with the installed SDK; all passed.
These controls made no network or credential-read attempt.
They do not prove server behavior.

The original push function sends one code file and can request `sessionTimeoutSeconds=3600`.
The SDK documents server termination after that runtime limit.
Live enforcement and queue duration remain unverified.

## Remaining work

Use a dedicated private version with fixed source, runtime, inputs, and a complete result archive.
Keep one host deadline and bounded client calls.
Do not repeat a submission after an ambiguous response.
Retain exact request versions beside their responses.

The inspected batch responses do not expose the session ID required by the cancellation API.
An exact-version terminal status alone does not prove TPU resource absence.
Require an independent observation of the identified run before accepting resource closure.

The [readiness result](../../experiments/2026-10-04-bitsandbytes-tpu/results/kaggle-readiness.json) records verification scope and source hashes.
Private source inspection and control records remain under `.work`.
