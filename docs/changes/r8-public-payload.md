# Kaggle payload download preparation

The reviewer accepted this preparation on 2026-10-09, Korea time.
The preparation addresses the [source size rejection](r8-v5-source-size-failure.md).
The provider's exact source size limit remains unknown.

The original wrapper embedded a 983,676-byte ZIP and contained 1,314,296 bytes.
The new offline wrapper contains 38,747 bytes.
It downloads the identical ZIP from a full commit in the project repository.
This size reduction does not establish provider acceptance.

The explicit download mode admits the complete generated wrapper before provider supervision.
It uses verified HTTPS, one request, an exact byte limit, and a fixed SHA-256 hash.
It rejects redirects and alternate source locations.
A separate child deadline limits the request even if network operations block.
The original deadline and 660-second reserve remain unchanged.

The original 298 scientific members remain byte-identical during repackaging.
The additional transport record must match the source, deadline, wrapper, and closed child process.
Only the original scientific members enter the unchanged verifier.
The runtime, precision, inputs, 46 cases, and 196 numerical comparisons remain fixed.
A valid numerical failure remains a failure.

Inspection of the first candidate found incomplete wrapper admission and an exception path that could leave an unfinished receipt.
The first candidate remains retained and unaccepted.
The corrected candidate uses complete deterministic rendering and records cleanup uncertainty.
It also closes a rejected redirect response without replacing the original rejection.

The reviewer repeated 40 controls before adoption and 40 controls after adoption.
All controls passed without provider calls or credential access.
The controls cover exact download bytes, changed sources, deadlines, cleanup errors, and complete scientific recovery.
They use synthetic scientific records and do not establish device results.

The repository retains the exact ZIP for immutable publication.
Generate a fresh candidate with its full publication commit before an actual submission.
Use the explicit download owner so transport admission remains mandatory.
Retain separate provider closure observations.
M8 remains unqualified, and accepted milestones remain three of eight.

Refer to the [transport instructions](../../experiments/2026-10-04-bitsandbytes-tpu/m8/PAYLOAD_DOWNLOAD_README.md) and [selected verification record](../../experiments/2026-10-04-bitsandbytes-tpu/results/kaggle-public-payload-preparation.json).
