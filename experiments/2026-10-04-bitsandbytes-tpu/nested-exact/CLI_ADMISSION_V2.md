# CLI admission correction, version 2

The configured CLI Python executable is a symlink.
The original helper passed this link to a hash function that requires a regular file.
It therefore rejected the admitted CLI before any provider operation.

Resolve only the executable path for the interpreter content hash.
Keep the original environment path for `pyvenv.cfg`, installed CLI files, and invocation.
Keep the original interpreter and distribution identity pins.
Reject symlinks in environment directories and CLI dependencies.
The executable link may point outside the environment if its resolved bytes match the admitted interpreter.

The actual configured interpreter and existing CLI identity passed read-only inspection.
No CLI command or provider operation executed during this inspection.
All 24 identity controls, 11 integration controls, and five driver controls passed.
The driver controls contain nine synthetic scenarios.

The science19 manifest and all 19 scientific files remain byte-identical.
The inner integration admission, patch, and adoption maps remain byte-identical.
The outer portable map and adoption manifest identify the corrected closure.
The new source archive supersedes the original archive for future adoption.
The original frozen archive remains evidence and must not be changed.

This correction does not qualify M4 or supply a CPU or TPU scientific result.
