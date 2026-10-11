# Completed work disposal

Date: 2026-10-11.

The reviewer archived 24 completed experiment, preparation, and review roots.
Eleven roots were under `.work`.
The other roots were temporary review directories and selected disposal receipts.
Current workers confirmed that they did not need these historical roots.

The private ZIP has 4,469 members, including its inventory.
It contains 4,468 source files with 167,529,817 logical bytes and no symbolic links.
The compressed archive has 42,262,201 bytes.
Its SHA-256 is `730d5f2d734a838fadd2e65ce6ecd2537d01b8a5a3352a6368f3f7c5359829ea`.

The [private archive](https://drive.google.com/file/d/1n7MgdpD821UUyEo3wF3Kzjg_qgtNwwlD/view) permits only the owner.
An independent download matched the full archive hash.
Every member passed its full hash and ZIP integrity check.
The reviewer checked the unchanged source inventory and confirmed that no active process used the recorded roots.

Removal stopped at one sealed source directory because the owner had no write permission.
The reviewer checked all remaining source bytes again, restored owner write permission to seven directories, and completed removal.
All 24 recorded roots and the temporary archive copy are absent.

A separate completed download cache contained 194,359,942 logical bytes.
The reviewer removed this derived cache before the archive was created.
The remote archive data remains available.

Measured `.work` storage fell from 1,467,140 KiB to 1,104,224 KiB.
The reduction is 362,916 KiB.
Required JAX environments, installation wheels, CLI identity files, and pinned upstream source remain available.
New test output must use `/tmp` and must be removed after required records are preserved.

The [disposal receipt](../../experiments/2026-10-04-bitsandbytes-tpu/results/completed-work-disposal-20261011/root-disposal.json),
[download proof](../../experiments/2026-10-04-bitsandbytes-tpu/results/completed-work-disposal-20261011/private-download-proof.json),
and [cache receipt](../../experiments/2026-10-04-bitsandbytes-tpu/results/completed-work-disposal-20261011/derived-cache-disposal.json) contain the selected verification records.
Artifact disposal does not qualify a scientific milestone.
