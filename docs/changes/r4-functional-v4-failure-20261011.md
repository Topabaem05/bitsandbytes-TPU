# M4 functional output experiment failure

The new pinned Linux CPU run passed 79 cases and 540 array checks.
The TPU run retained 49 case records before its evidence limit stopped execution.
These records contain 35 execution errors and 14 returned results.
The independent numerical audit reports two passes, 12 failures, and 65 errors across all 79 required cases.
Thirty cases have no returned record.
State save and restore did not run.
M4 remains unqualified.

The previous double functional wrapper assertion is absent.
All 28 retained successful bitcast frames preserve shape and functional wrappers.
The tail maximum uses an omitted dimension, which the pinned XLA lowering treats as an empty reduction axis list.
The retained HLO therefore keeps the tail shape and produces incorrect later dimensions.
An explicit axis is a focused repair candidate.

The numerical mismatch has a separate unresolved cause.
For a zero block, the CPU expected code is 127 and the TPU returned code is 19.
The pinned legacy shifts lower through power and multiplication or division.
Astra Max advised a small comparison with explicit bitwise shifts and typed scalar intermediates before another full run.
These observations do not prove that shifts explain every mismatch.

A required textproto had 37,924,102 bytes, above the unchanged 32 MiB object limit.
The run stopped after retaining the affected row.
It did not accept truncated evidence or renew its original deadline.
The runtime lifecycle lasted 742.85 seconds.

The root verified 191 lossless logical files, including 131,235,819 raw bytes.
Astra independently verified their hashes, sizes, and single-member gzip closure.
All 70 local process groups were absent during fresh review.
Fresh server records showed no active session or assignment and a displayed usage rate of 0.00 per hour.
The browser showed a disconnected runtime, and the owned tab was closed.

The [private archive](https://drive.google.com/file/d/14J4jLVytGo4ApevWsucyvPzL6agT5hrT/view) contains complete preparation and experiment records.
An independent download verified all 974 archive members before local disposal.
No new output was added to `.work`.

The full ASD-STE100 dictionary was unavailable; no certified conformity is claimed.
