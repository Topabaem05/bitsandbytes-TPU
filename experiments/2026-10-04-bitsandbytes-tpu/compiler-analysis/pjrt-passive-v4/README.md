# Passive PJRT recorder preparation

This local packet implements `M6_DIRECT_PJRT_TRACE_V4`.
It records calls through the actual PJRT C API table.
The local controls use a fake shared provider through `dlopen`.
They do not establish actual TPU execution.

The only target claim is `DIRECT_LOADED_HANDLE_EXECUTE_CALL`.
A successful execute call does not establish asynchronous device completion.
This packet does not establish graph identity, binary selection, memory peak, or Mosaic behavior.

## Source and ABI

The OpenXLA revision is `92f7b5952dd585c5be17c9a5caad27407005b513`.
The Torch/XLA revision is `5fab7053df86c8d503b98d9e7202ca8b8d4978c7`.
The proof directory retains selected source ranges and full source hashes.
The source check fetched all five pinned primary bodies and verified each selected range.
It then discarded the downloaded bodies.

The shim loads one explicit provider and verifies the entry-point file identity.
It rejects self-recursion.
It copies the runtime table's exact `struct_size`, including an unknown tail.
It replaces only present, non-null hook slots.
It preserves version fields, extension pointers, and all other table bytes.
It retains the provider handle until process exit.

The hooks cover compile, executable lookup, optimized-program capture, execute, deserialize, destroy, and delete.
Each hook calls its original function once with the same argument pointer.
The hook returns the original error pointer and preserves original output values.
It never destroys a caller error.
The recorder makes no additional observer API calls.
It never holds the state mutex across a vendor call.

## Capture and identity

The journal uses nonce, handle kind, and creation sequence for handle identity.
Raw addresses do not appear in output records.
Address reuse creates a new generation after destruction.
Duplicate live addresses and overlapping operations make the affected identities ambiguous.
Deserialized handles retain a separate origin and cannot establish the compile-origin target claim.

The compile hook saves the full submitted program and format bytes.
The optimized-program hook reads only `code` before a first size query.
It does not read uninitialized query size or format fields.
A successful query records the expected size and format for that executable generation.
A later fill must match that query before the recorder copies the payload.
An absent query, changed length, changed format, or invalid identity leaves the claim unknown.
Every optimized-program BEGIN supersedes the earlier payload for that executable generation.
This rule also applies when arguments are short or the provider returns an error.
The recorder snapshots and consumes the preceding completed query at BEGIN.
A fill can publish new evidence only if its epoch remains current through END.

The verifier requires each executable parent to match the GetExecutable input handle.
That loaded generation must exist before BEGIN and remain valid through END.
The verifier selects completed payload candidates at Execute BEGIN.
A later optimized BEGIN during Execute invalidates the affected candidate.
An unrelated executable query does not invalidate another loaded handle's candidates.
A query after Execute END does not cancel a completed historical claim.
Destroying an executable wrapper preserves its historical payload link to a valid loaded handle.

The recorder preserves the full `hlo_with_config` envelope and its SHA-256 hash.
The independent verifier reports separate field offsets, lengths, and hashes.
Field 1 contains the HLO message; field 2 contains the configuration message.
These bytes do not establish equality with Torch/XLA's reconstructed `ToProto()` bytes.
The verifier does not reconstruct or normalize protobuf messages.

## Bounds and closure

The hard limits are 1,024 entries, 128 MiB total, and 16 MiB per file.
The recorder reserves 3 MiB for the journal, footer, and closed inventory.
It reserves each payload's quota before file creation.
The recorder budget is separate from the compiler-dump budget.
The combined experiment must reserve both budgets before execution.
The host helper requires an explicit recorder byte budget.
The caller must subtract this reservation from the original experiment capture budget.
The caller supplies the original `CLOCK_MONOTONIC` deadline in nanoseconds.
The recorder does not create or renew that deadline.
An expired deadline stops capture while the original API call still proceeds.

`M6TraceFinalize()` rejects closure while a wrapped call remains active.
A completed finalizer writes `API_FINALIZED`; this state alone does not establish closure.
The host reads every file and verifies its full size and SHA-256 hash.
The host then writes the bounded `CLOSED` inventory.
The verifier rejects unlisted files, altered files, missing sequences, and an absent closed inventory.
Capture failures produce an unknown claim and do not replace original API errors.

## Repeat the local controls

Run these commands from the packet directory:

```sh
python3 source/verify_sources.py
python3 source/run_controls.py --result /tmp/m6-pjrt-controls-result.json
python3 source/reproduce_peer.py /tmp/m6-pjrt-peer-regression.json
```

The runner downloads the exact header into a temporary directory.
It builds the shim, fake provider, and harness with the host C++ compiler.
It deletes each private fixture after validation.
It deletes all build files before it publishes the result.
The selected controls and retained results are in `controls.json`.
The runner checks 46 provider scenarios, 12 lineage or mutation controls, and nine sanitizer scenarios.
The peer regression uses exact frozen v3 fake-provider and harness inputs.
It changes only the recorder and verifier under test.
The same compiled harness passes the cold case and returns UNKNOWN for the replacement-query case.
The retained peer regression script targets macOS.
The main control runner also supplies Linux build commands, but Linux execution remains NOT_RUN.

ASan and UBSan passed the selected controls on this macOS arm64 host.
MemorySanitizer is unavailable for this host target.
The successful sanitizer results do not prove the absence of uninitialized reads.
The query fixture leaves output fields uninitialized and uses the actual two-call protocol.

## Prepare a separately authorized runtime

Build the shim on the target host:

```sh
python3 source/build_recorder.py --output /absolute/path/recorder.so
```

Use an explicit provider path and a new output directory.
Import `source/recorder_host.py` before importing Torch/XLA.
Construct `PassiveRecorder` with the shim, provider, nonce, output, original deadline, and reserved recorder byte budget.
The helper identifies the recorder with its actual library SHA-256 hash.
It sets `TPU_LIBRARY_PATH` to the shim.
Set a bounded public phase label before each caller phase.
Complete the planned API calls before finalization.
Call `finalize()` and retain the complete closed inventory with the experiment archive.
An `INFLIGHT` result cannot authorize closure or a target claim.

No actual TPU provider or TPU operation ran during this preparation.
No canonical repository, Git, or wiki write occurred.
The parent reviewer must inspect the source and repeat the controls before adoption.

## Writing inspection

The sentence-length check passed for this document.
The full ASD-STE100 dictionary was unavailable for this preparation.
This packet does not claim dictionary conformity or independent certification.
