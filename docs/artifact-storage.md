# Artifact storage

GitHub contains the reviewed source, project documents, and selected result records.
The private Google Drive archive contains raw experiments, failed controls, and unfinished research preparations.
Archival does not change the acceptance status of a result.

Local `.work` directories contain temporary output and active research files.
Keep runtime environments and the files necessary for the next experiment locally.
Archive completed historical directories before their local removal.
Keep authentication credentials outside Git and research archives.

## Archive requirements

1. Stop writers before the backup starts.
2. Record the source scope and excluded files.
3. Verify that the destination folder permits only the owner.
4. Require a complete backup with no unreadable source files.
5. Read all stored archive data during integrity verification.
6. Restore selected source and binary records into a new directory.
7. Compare their exact hashes with the local originals.
8. Verify that local sources have not changed since the backup.
9. Remove only the recorded historical paths.
10. Record the snapshot identifier and measured storage reduction.

The archive uses restic with its rclone Google Drive backend.
The private folder contains restoration instructions and a completion receipt.
The receipt identifies the exact snapshot and verification results.
Google Drive authentication remains necessary for restoration.
The archive has no separate password.
Its access protection depends on the private Google Drive permissions.

Use `check --read-data` for complete stored-data verification.
The default restic integrity inspection does not read every data object.
Refer to the [official verification instructions](https://restic.readthedocs.io/en/stable/045_working_with_repos.html).

Restore historical dependencies before executing a preparation that refers to archived `.work` paths.
Use a new destination and compare the original source maps before execution.
Refer to the [official restoration instructions](https://restic.readthedocs.io/en/stable/050_restore.html).
