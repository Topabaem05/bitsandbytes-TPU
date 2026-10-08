# Kaggle submission failure

The reviewer made one private M2 submission on 2026-10-09, Korea time.
The exact script, metadata, 45-member package, and original deadline passed inspection before submission.
The request used the admitted SDK, `TpuV6E8`, and a 3,600-second session limit.

The official `SaveKernel` request returned HTTP 400.
The worker retained the exception, but it did not retain the response body.
The root cause remains unknown.
The owner correctly retained `SUBMISSION_UNKNOWN` without an accepted kernel identity.
It did not submit again.

## Independent observations

The browser showed a saved draft at the exact requested reference.
Its title, nonce, and deadline matched the submitted script.
The draft session was off.
The version panel showed `Starting Fresh` with no saved execution version.
Separate read-only requests for the exact source and session version `v1` both returned HTTP 404.

Thus, HTTP 400 did not prevent all persistent changes.
These observations do not establish an accepted execution version or scientific result.
The owner's closure field remains `UNCONFIRMED` because no original submission identity was returned.
The reviewer did not create a substitute identity or use the successful closure entry.

## Local closure and next work

The submission's internal owner and SDK worker both closed.
The separate read-only owner and its two SDK workers also closed.
Each retained record shows child reaping and process-group absence.
The temporary browser tab closed after the reviewer saved the observation.

Retain a bounded error record for future HTTP failures.
Inspect request constraints and preserve the existing draft.
Do not repeat this ambiguous submission automatically.
Require a separate reviewed candidate before another execution request.

The [selected result](../../experiments/2026-10-04-bitsandbytes-tpu/results/kaggle-m2-submission-failure.json) contains the request, source, and observation hashes.
Raw provider records remain private under `.work`.
M8 remains unqualified, and accepted milestones remain three of eight.
