# Remote control source directories

The owner previously prepared only the main control and ownership directories.
A nested source upload failed when its parent directory was absent.
The new owner derives all directories and upload destinations from the same admitted source map.
It checks each ancestor with `lstat` and reads every directory before upload.
It rejects invalid paths, existing files, and symbolic links.

Readiness must return one final completion record with the exact nonce and directory list.
Missing records, duplicate records, duplicate JSON fields, and later output cause rejection.
CLI process success alone cannot satisfy this check.

The reviewer independently passed 42 candidate tests.
Another 21 tests passed against the current canonical source inventory.
Those tests included all 42 admitted cloud Python files.
The earlier candidate inventory contained 41 files before the separate M4 kernel helper was adopted.
The correction omits no source file.
Scientific code, numerical criteria, original deadlines, and exact session cleanup remain unchanged.

The tests also reproduced the original missing directory failure.
They checked shared parents, invalid paths, incomplete execution, deadline expiry, and a failed upload child.
The failed child closed its owned process group.
All temporary fixtures were deleted.

The existing CLI test callback now accepts its two existing keyword arguments.
Both arguments must remain `None` in those cases.
This test repair changes no production behavior.

Actual PUBLIC execution remains required.
M6 remains unqualified.
Refer to the [root review](../../experiments/2026-10-04-bitsandbytes-tpu/results/cloud-control-parents-preparation/root-review.json).

The prose received sentence-length and technical-term inspection.
The complete ASD-STE100 dictionary inspection remains incomplete.
