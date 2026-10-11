# Public exact-reference controls

These controls use explicit source and fixture paths.
They do not install wheels or call a cloud provider.
They cannot qualify Linux CPU records or TPU execution.
Use a new temporary output directory for each run.
Keep completed summaries and hashes before large fixture removal.

Run `run_owned.py` with `--public-source`, `--base-packet`, `--ownership-source`, and `--output`.
The base packet must contain the original public generation, its baseline sources, and `patched-upstream.tar`.
The expected result is 33 passed controls.

Run `run_device_flow.py` after the science controls finish.
Pass their `controls` directory as `--cpu-fixture`.
Also pass the public source, original base packet, ownership source, and new output directory.
Pass the canonical `cloud/tests/public-mode` directory as `--fixture-source`.
Pass the pinned JAX 0.7.1 interpreter as `--jax071-python`.
The expected result is five passed controls.

Run `run_archive.py` with `--cloud`, `--cpu-fixture`, `--ownership-source`, and a new `--output` directory.
The expected result is six passed controls.
The original 64 MiB uncompressed CPU archive limit remains in effect.

Use `run_packet.py` for a finite, owned source packet build.
Pass the pinned upstream Git checkout as `--upstream`.
Also pass `--cloud`, `--repo-root`, `--public-source`, `--ownership-source`, and a new `--output` directory.

The unchanged `cloud/tests/public-mode/run_controls.py` supplies the complete cloud control suite.
Use `--seconds 600` for the repeated exact-reference computations.
The original 220-second host-control run timed out and remains a failed control run.
This larger host bound does not change a scientific deadline or numerical gate.

The current compiler control map is `results/compiler-libtpu021-preparation`.
Use both of its source and adoption maps with `materialize_controls.py`.
The older `compiler-accepted-preparation` map does not describe all current canonical files.
Use the exact accepted M2 dependency document for mode packet builds.
The accepted M2 result document has a different schema.
