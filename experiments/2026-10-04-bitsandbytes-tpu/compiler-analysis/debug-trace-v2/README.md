# M6 debug and hash preparation, version 2, arity revision 1

This directory contains isolated CPU preparation. Actual TPU verification did not run. M6 remains `NOT_QUALIFIED`.

The repository and its scientific criteria remain unchanged. The frozen compiler analyzer remains unchanged.

## Source facts

`proof/sources.json` identifies fifteen pinned primary source bodies. Each complete body passed SHA-256 and exact expression checks before removal.

`proof/verified-full-bodies.json` retains those results. Selected line excerpts retain their original line numbers and hashes.

The pinned Torch/XLA `WORKSPACE` selects OpenXLA revision `92f7b5952dd585c5be17c9a5caad27407005b513`. The internal libtpu revision remains unknown.

Compilation and execution analysis print `coll.hash`. This value identifies a cache key. It does not identify a loaded executable handle.

Post-compilation analysis hashes the complete deterministic `HloModuleProto`. The PJRT client obtains that module from `CompileAndLoad` and `GetHloModules()[0]`.

The serializer does not remove fields. Its deterministic output preserves the complete supplied module.

`HashToString` leaves the high 64 bits unpadded. It pads the low 64 bits to sixteen hexadecimal digits.

`_get_graph_hash` returns native object bytes. Their hexadecimal representation differs from `HashToString`.

## Corrected verification

The input kind is `M6_DEBUG_HASH_TRACE_V2_ARITY_REV1`. The preceding candidate remains frozen and unchanged.

Offline review can occur after the experiment deadline. The verifier requires `started <= finished <= deadline <= parent_deadline` and `finished <= now`.

The live process owner still enforces its original deadline. This preparation does not modify that owner.

Compilation and execution records retain exact input and output counts. Each cold compilation requires independently captured submitted module bytes and their SHA-256.

Compilation counts must match that submitted module's `host_program_shape`. Execution counts must match the full optimized module identified by the computation hash.

Submitted and optimized counts can differ. The controls include a valid two-input submitted module and a one-input optimized module.

Output counts use top-level tuple children. Nested tuple leaves and internal helper parameters do not increase the count.

`module_arity_v2.py` reads the pinned protobuf fields without changing module bytes. It does not validate executable HLO semantics.

Missing shapes, duplicate singular fields, unsupported wire forms, and excessive reader bounds return `UNSUPPORTED` errors.

Scope, capture provenance, synchronization, and readback remain caller attestations. The future recorder must independently establish those facts.

## Verification scope

`lazy_hash_v2.py` implements the pinned lazy hash for little-endian storage. The independent C++ reference uses native 128-bit arithmetic.

`debug_trace_v2.py` accepts complete preparation, target, and reference windows. It requires contiguous offsets, sealed bytes, synchronization, and readback records.

The parser accepts a cold compilation followed by execution analysis. It also accepts warm execution after an observed matching cold compilation.

Each compilation requires the exact hash of the supplied full module bytes. Missing, truncated, ambiguous, or interleaved evidence fails admission.

The fixtures use minimal synthetic `HloModuleProto` messages. Their fields follow the pinned protobuf schema. They are not executable HLO modules. They test hash and parser behavior. They do not prove protobuf provenance or actual runtime behavior.

Scope and completion fields remain caller attestations. An independent recorder must establish their provenance before actual evidence can qualify.

`recorder-interface-v2.json` defines a future recorder contract. It contains no TPU implementation or completed actual observation.

## Evidence levels

`DEBUG_SOURCE_LINK` connects observed debug hashes through the pinned source control flow. Execution analysis occurs before asynchronous execution completes.

A completed synchronization and output readback are required. This preparation cannot independently establish those facts from caller booleans.

`PJRT_LOADED_HANDLE_LINK` requires future direct observation. The required chain is `Compile -> loaded L -> GetExecutable E -> OptimizedProgram -> Execute L`.

That observer must record a process nonce and lifetime generation. It must prevent pointer reuse from creating a false identity match.

The pinned API defines these entry points. Their availability and behavior in the actual libtpu remain unverified.

Neither level proves binary identity, allocator peak, or internal Mosaic storage. A module hash match cannot establish those results.

## Reproduction

Run `python3 -B controls/run_controls.py` from this directory. The command builds the C++ reference in a temporary child directory.

The command removes its binary after readback. All 59 CPU controls passed in this generation. It retains only the compact test report.

Run `python3 -B proof/verify_sources_v2.py` to verify retained excerpts. This command does not access the network.

To repeat complete body checks, download each pinned URL into its named `proof/<name>.source` file. Then run the source verifier.

Delete those downloaded bodies after readback. Preserve their verified hashes and selected excerpts.

## Adoption limits

The root reviewer must inspect this preparation before repository adoption. No Git, provider, browser, account, or wiki mutation occurred.

The existing limits remain 1,024 entries, 16 MiB per file, and 128 MiB in total. Combine tracer records with existing compiler records.

Use the original parent deadline. This parser does not enforce a filesystem quota or supervise a device process. Submitted modules share the existing combined budget.

Raw debug output must use a fresh dedicated file. The strict parser does not accept unrelated stderr output.

English clarity checks are limited. Full dictionary review remains incomplete. This preparation makes no certified STE claim.
