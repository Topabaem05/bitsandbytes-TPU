# M4 functional output preparation v4

The previous actual 79-case TPU run failed at the output wrapper in `exact.py`.
The output already had a functional wrapper.
The fixed XLA bridge can return a functional tensor.
The PyTorch constructor rejects a tensor that already has this wrapper.
The [source records](FUNCTIONAL_OUTPUT_SOURCE_BASIS.json) identify the complete pinned sources and inspected lines.

`bitcast` now adds one wrapper when its input is functional and its output is raw.
It returns an existing functional output as the same object.
It also preserves both output states when its input is raw.
Input synchronization, input extraction, builder SHA-256, output count, data type, device, and all mathematical operations are unchanged.
The change does not alter dispatch TLS or a functionalization environment setting.

The source generation is `nested-exact-v2-functional-output`.
This generation has a new source admission, integration map, and science manifest.
Exactly three of the 19 science members change: `exact_admission.py`, `generation_gate.py`, and `integration-admission.json`.
Their changes bind the new source identity.
The other 16 members are identical, including the input cases, numerical criteria, public API observations, graph checks, record checks, state8 scope, and CPU gate.
All mathematical function syntax trees outside `bitcast` and all four integer helper files are identical.
The upstream source patch is identical.

The historical science manifest has SHA-256 `9e71efc73b0f2b3a210f8131c10972569a5e2d04cf298fcf5a8da59772cf4f5a`.
See the [historical map](history/functional-output-v1-20261011/source-map.json).
It reconstructs 19 members from three retained identity files and the 16 unchanged current files.
This is a reconstruction map, not a claim that the old inventory occupies one physical directory.
The new manifest has SHA-256 `29beb28f4320acc5e07ce6822f2120eaff846809bfc54bd123ac37f501700c06`.
The new source admission has SHA-256 `8b5aa5f0c86346db982664cfad3b1e242dc8a67a4e49865584ccc01c8620f4bf`.
The new integration map has SHA-256 `bfd7ce45a900fee3bc5ecfed54524c17931e2964ef5b7295fca6e6038883729a`.
The v3 kernel helper remains SHA-256 `e9c7a5b7b943cea08dfb8986205e47e008ceac37553a2f8576687527a58f9d5f`.

Local macOS PyTorch 2.14.1 passed 23 boundary controls.
They use real CPU functional tensors with a stub XLA builder and a device facade.
They reproduce the old constructor assertion and check all four input/output wrapper states in both bitcast directions.
They preserve signed zero, subnormal values, finite extrema, pending input mutations, and copied output values.
They reject incorrect input/output data types, output devices, output counts, and builder hashes.
They check unchanged TLS after success and exceptions.
These controls do not execute XLA and do not qualify the pinned Linux runtime.

Eight source controls passed.
They check the complete new source, old identity rejection, the exact science delta, unchanged mathematics, historical reconstruction, altered plugin rejection, and symbolic link rejection.
The existing 11 integration controls, 29 v3 kernel controls, and five driver tests covering nine scenarios also passed.
All temporary fixture trees were deleted after each suite.
No provider calls or canonical writes were made by this worker.

A root review must precede adoption and a fresh packet.
The replacement map records every source file, each changed preimage, and each replacement.
The adoption tool first checks the complete closed source and all destination bytes.
It rejects a different destination file before it changes any file.
Run its check mode on the canonical repository and review the exact replacements before applying it.
The source archive requires full member and byte readback before use.
The ordinary `root_records.py` source review and `build_packet.py` packet procedure then use the new static pins.
Do not change admitted literals during runtime execution.

Fresh qualified Linux CPU79, actual TPU79, external root acceptance, and CPU/state8 execution remain required.
The original allocation deadline, reserves, archive limits, and resource closure checks remain in force.
M4 remains `NOT_QUALIFIED`.
The frozen M5 candidate still contains the prior plugin generation.
It requires a separate explicit propagation change after review; this preparation does not qualify or modify M5.

The prose received a project clarity review.
It has no claim of complete ASD-STE100 dictionary compliance.
