# Fixed libtpu 0.0.21 compiler flag correction

This private preparation removes `--xla_dump_hlo_unoptimized_snapshots=false` from the compiler request.
The actual compiler child rejected that option before any native case.
The failed run and the accepted native records remain unchanged.

The new generation is `compiler-mosaic-serde7-gather-bf16-fp32-libtpu021-flags-v1`.
Its 36-file manifest hash is `f2a3e62f3a8fc7790fdeff12973af71a2be831f2a057199812b8f2f4c6676a88`.
Its policy hash is `131637cedccf2ed86c4d800a42782bb082329c64a62e06ef4f1c8c713709b932`.
The policy binds ten source files.
The compiler source tree retains 30 of the prior 34 source files without a byte change.
The changes add the flag contract and probe.
They bind the preflight in the contract, parent, verifier, and policy.
Kernel, adapter, native case body, input recipes, five CPU oracles, numerical gates, and precision remain unchanged.
All 27 ordinary native files remain unchanged.

The official [libtpu 0.0.21 wheel](https://pypi.org/project/libtpu/0.0.21/) has the exact runtime-lock checksum.
Its `libtpu.so` hash is `cdb7980d4332097b8e16576568e138ef54cf9fb8ed5aae2aeefbd0c5a425e94a`.
That hash matches the actual failed runtime.
Static inspection finds the 14 retained option-name suffixes, including the dump destination.
It finds no bytes for the rejected option name.
String presence does not prove a complete flag registry or supported values.

The failed request had 15 tokens: one destination and 14 other options.
The error lists one unknown token.
The [pinned upstream parser](https://github.com/openxla/xla/blob/31eb6029e8973337ef6c99810c27e1d807791c11/xla/parse_flags_from_env.cc) consumes known options before it reports all remaining tokens.
This supports an inference that the actual parser recognized the other 14 requested names.
It does not prove that the corrected request passes.
The [JAX 0.7.1 upstream registry](https://github.com/openxla/xla/blob/31eb6029e8973337ef6c99810c27e1d807791c11/xla/debug_options_flags.cc) registers the rejected option.
Local JAX 0.7.1 CPU initialization also accepts it.
That source revision is frontend context.
The internal libtpu build revision and its complete registry remain unknown.

The corrected request has 14 tokens: one destination and 13 other options.
Before science, the remote owner starts a separate child with the exact fixed library and Torch/XLA frontend hashes.
The child initializes the TPU backend and parses the request.
It constructs no tensors and runs no native case.
Its scope is parser and backend initialization only.
The preflight dump directory is `/content/bnb-tpu-first/compiler-flag-preflight-private`.
That directory is outside the scientific and compiler inventories.
The owner requires the exact report, token, argv, source bindings, and successful exit.
It also requires a reaped leader and an absent group.
The bounded terminal observation must occur after group cleanup.
A blocked or timed-out preflight cannot launch the native parent.
The independent result reader requires the same report and owned step from the recovered science archive.
Preflight success cannot satisfy a native numerical gate.

The cloud base is the root-adopted public revision `5ada90eeef3f8a9ec3913a2a1a83d877be709f65`.
The six source hashes are in `evidence/adopted-public-cloud/composition.json`.
Only the compiler branch, optional observation in `run_step`, and exact helper import differ from that remote source.
Public and ordinary native phase bodies contain the same bytes.
A fresh PUBLIC15 packet passes preflight with all public scientific source bytes unchanged.

The accepted native record is `c64fca47f57374fab691d9870005cf614fc6eebbc8a02706ed70781f9dd1042a`.
The actual result is `fd15cb724a398b9079b0b97c19f66bd858fe3f7027e9ae9e1a0ff8983eb1a291`.
Its adoption revision is `a084a4d578362f9733453595ab208385cf70afba`.
The native manifest is `30b820b90a4404a89da6e6c7b6fc91dbec47afbe06ae322eb5835b9eb0e70901`.
The native generation is `mosaic-serde7-gather-bf16-fp32-v1`.
All six exact native dependency bindings remain required.
The runtime lock remains `323371ff61c5fbcc4f79fd6a358cf2ba17cb72b907382a5ceaac07f91dc66ed6`.

The final local suites contain 171 distinct controls:

- 67 affected controls.
- 41 accepted dependency controls.
- 23 ordinary native controls.
- 35 flag controls.
- Three public preservation controls.
- Two portable layout controls.

Each compiler packet has 93 genuine members.
The two final packets contain the same bytes.
The flag suite includes synthetic accepted and rejected reports.
It also includes actual local owner/monitor controls and actual JAX CPU parser controls.
None is an actual corrected libtpu or TPU scientific acceptance.
Historical fixture adaptation, invocation, and metadata-sealing failures remain sealed.

The collector uses these limits:

- Per-file limit: 16 MiB.
- Monitored total: 128 MiB.
- Named-file entry limit: 1,024.
- Observation interval: 0.01 seconds.
- Minimum free space: 256 MiB.
- Scientific record budget: 8 MiB.

These are bounded collector limits.
There is no hard aggregate filesystem quota.
A fast writer can overshoot between observations.
Physical, allocator, executable, and internal backend memory remain unknown.
M6 remains unqualified.
This preparation performs no provider action, installation, canonical write, or compiler dispatch.

Use `REPRODUCE.md` for portable canonical replay.
The prior ordinary-native repair basis and old analyzer fixtures are historical context.
The prior root review and source hashes cover the unchanged analyzer and topology controls.
Their replay does not increase this preparation's control count.
