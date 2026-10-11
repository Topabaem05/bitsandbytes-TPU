# Default compiler dumps

This generation removes only `--xla_dump_hlo_pass_re=pipeline-start|pipeline-end` from `compiler_flags.py`.
The exact request has 13 tokens, including the dump directory.
The prior exact 14-token request must fail admission.
The five scientific cases and original numerical criteria remain unchanged.
Text, proto, module metadata, and the 32-module cap remain required.
The limits remain 1,024 entries, 128 MiB total, and 16 MiB per file.
The deadline, observation interval, monitor, and closure procedures remain unchanged.

The previous actual capture stopped at 1,025 entries.
Its observed total was 15,559,534 bytes, and its largest file contained 530,344 bytes.
The first two FP32 cases passed; three later cases did not execute.
The raw dump inventory was omitted after the limit failure.
This failure remains a failure.

The pinned OpenXLA sources describe extra per-pass dumps when a pass filter matches.
Removing that filter is a proposed reduction in dump count.
It does not prove that the next capture fits the limits.
The internal libtpu revision and loaded executable identity remain unknown.
The independent reader must reject absent or unsupported metadata.
Compiler allocation analysis remains a separate procedure.

The new source and adoption maps supersede the old 187-file map for this generation.
They preserve both public cloud files at their exact `bf7c843` bytes.
Use `REPRODUCE.md` for bounded local controls and packet construction.
Use `DISPATCH.md` for the separate actual experiment requirements.
