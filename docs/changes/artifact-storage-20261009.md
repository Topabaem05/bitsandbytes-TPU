# Verified artifact storage

Historical research output used most of the available local storage.
The reviewed source and documents remain in GitHub.
A private Google Drive archive now contains the historical research output, including failures and incomplete preparations.

## Archive verification

The restic snapshot contains 2,039,703 regular files and 128,613,676,037 source bytes.
The backup reported no unreadable source files.
The stored data used 4,188,249,418 bytes after compression and duplicate removal.
The final permission inspection found one owner and no other access permissions.
The archive has no separate password; Google Drive permissions protect access.

The complete stored-data inspection read all 123 packs without errors.
Four restored files matched their original sizes and SHA-256 hashes.
These files included source code, a binary packet, and a failure record.
A subsequent comparison found all 2,039,703 regular files unchanged and no new archive data.
All 345 source roots were present in the snapshot.

## Local removal

The cleanup removed exactly 326 historical paths and retained 20 active paths.
The retained paths include runtime environments, wheel files, and current research preparations.
The first removal attempt stopped at a read-only directory after 108 completed paths.
The recovery added owner-write permission only to owned directories within the remaining removal scope.
The recovery continued from the exact completion journal.
Isolated controls rejected invalid paths and journals and preserved an external symbolic link target.
The remote archive retains the original directory permissions.

| Measurement | Before | After |
| --- | ---: | ---: |
| Local `.work` allocated bytes | 130,770,849,792 | 3,840,954,368 |
| Volume available bytes | 17,872,486,400 | 145,864,171,520 |

The `.work` allocation decreased by 126,929,895,424 bytes.
Other system activity can affect the volume measurement.
Both retained Python runtime commands returned Python 3.12.14.
The Git working tree remained clean after removal.

## Records and limits

The private archive folder contains `RESTORE.md`, `COMPLETION-20261009.json`, and `VERIFICATION-20261009.zip`.
The verification archive contains the original logs, failed removal record, and successful recovery records.
Downloads of the guide, completion receipt, and verification archive matched their local SHA-256 hashes.
Historical preparations can require restoration of archived dependencies before execution.
This storage operation does not change research acceptance or establish TPU performance.

Refer to the [selected result](../../experiments/2026-10-04-bitsandbytes-tpu/results/artifact-storage-20261009.json)
and the [storage procedure](../artifact-storage.md).
