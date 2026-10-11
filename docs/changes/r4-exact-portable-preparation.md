# Portable exact nested experiment

The exact nested experiment now has a separate complete source tree.
It includes the upstream package, plugin variant, fixed science, build files, and runtime locks.
All 19 scientific source members retain their reviewed bytes.
The science manifest SHA256 is `9e71efc73b0f2b3a210f8131c10972569a5e2d04cf298fcf5a8da59772cf4f5a`.
The experiment preserves the original 79-case criteria and the separate eight saved-state cases.
M5 retains its separate 1,024-scale requirement.

The owner recovers and audits the fresh CPU oracle before device execution.
It pauses after actual79 recovery until the root reviewer supplies an accepted result.
A separate CPU state audit precedes the device save and restore processes.
Missing, incorrect, old, or failed acceptance records stop the state phase.
Every phase uses the original allocation deadline.
The complete archive includes physical metadata within 2,000 files and 100 MiB.
The inner record limit is 96 MiB.
An evidence-limit failure does not qualify the result.

The CLI check resolves the interpreter only to compare its content hash.
It retains the original virtual environment path for installed files and command invocation.
Incorrect interpreter bytes and symlink escapes in dependency files are rejected.
This correction passed read-only inspection of the installed CLI without an account command.

The reviewer independently passed 24 CLI identity controls and 11 integration controls.
Five driver tests covered nine valid and invalid scenarios through an isolated fake CLI.
The earlier portable science replay passed all 23 controls.
The reviewer checked every source archive member and all 143 adopted files.
The official CLI class and inner integration maps retain their reviewed bytes.
These results qualify the preparation only.
Fresh qualified Linux CPU, actual79, and actual state8 records remain required for M4.

Read the [portable commands](../../experiments/2026-10-04-bitsandbytes-tpu/nested-exact/REPRODUCE.md)
and the [independent reviewer record](../../experiments/2026-10-04-bitsandbytes-tpu/results/nested-exact-preparation/root-source-review.json).
The document uses the project clarity rules.
Full STE dictionary review remains incomplete.
