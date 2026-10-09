# Compiler flag correction for libtpu 0.0.21

The previous Colab attempt rejected one dump flag before any native case executed.
The new compiler request removes `--xla_dump_hlo_unoptimized_snapshots=false`.
The failed experiment remains preserved.

The fixed libtpu library differs from the JAX frontend registry.
The JAX CPU parser accepts the rejected flag.
That result does not establish support in the TPU library.
The retained source inspection identifies the exact wheel, library, frontend, and requested options.

Before numerical execution, a separate process initializes the TPU backend with the corrected request.
This process constructs no tensors.
Its private dump directory remains outside the scientific record directories.
The owner requires the correct library, report, arguments, successful exit, and complete process closure.
The native parent cannot start when this preliminary requirement fails.
The recovered records must contain the same report and process evidence.

The change retains the accepted native dependency and all 27 ordinary native files.
Kernel operations, input data, numerical criteria, and precision remain unchanged.
Public and ordinary native phase bodies remain unchanged.

The reviewer verified 187 selected files, including 54 changes and 133 exact prerequisites.
All 250 distinct local controls passed before and after adoption:

- 67 affected controls.
- 41 accepted dependency controls.
- 23 ordinary native controls.
- 35 flag controls.
- Three public preservation controls.
- Two portable layout controls.
- 79 public regression controls.

The two canonical compiler packets matched the reviewed private packets exactly.
Each packet contains 93 members and 1,114,172 bytes.
The packet SHA-256 is `5992724099a519a82a265ff823b09a8d2e1524789f17b3ee69b10be977b32884`.
The reviewer verified closure of 59 process groups from canonical construction and controls.

The first portable replay lacked two map files in its reconstructed directory.
The corrected instructions copy both exact metadata files before execution.
The final metadata revision passed both portable controls in a new process group.
That group also closed, which brings the inspected canonical group count to 60.
Three earlier root commands used incorrect relative paths and failed before test execution.
Their corrected absolute commands passed; the original errors remain preserved.

The final documentation revisions changed no executable source body.
The reviewer inspected sentence length, document links, and technical terms.
The complete standard dictionary was unavailable.
This inspection does not establish independent ASD-STE100 conformity.

The collector retains its 16 MiB file limit and 128 MiB monitored total.
It observes at most 1,024 entries at 0.01-second intervals.
These observations do not provide a hard disk quota.
Runtime memory use and selected executable identity remain unknown.

The corrected TPU request has not executed during this preparation.
Fresh qualified Linux CPU records and a new Colab experiment remain required.
M6 remains unqualified, and three of eight milestones remain accepted.

Refer to the [selected result](../../experiments/2026-10-04-bitsandbytes-tpu/results/compiler-libtpu021-preparation.json).
Use the [replay instructions](../../experiments/2026-10-04-bitsandbytes-tpu/cloud/tests/compiler-accepted/REPRODUCE.md) to reconstruct the controls from repository files.
