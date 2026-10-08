# R4: Colab bootstrap import correction

The notebook entry point now loads `nested_contract.py` from its admitted control directory.
The loader uses the file path directly.
It does not select or replace a module from the installed module cache.
The owner verifies all control source hashes before this import.

The reviewer repeated 15 local tests in an isolated source copy.
Four tests executed the actual generated notebook entry snippet.
The old source reproduced the original missing-module error.
The corrected source unpacked the admitted packet from a neutral working directory.
A conflicting cached module did not affect the result.
Incorrect control bytes stopped execution before import.

Each entry test also verified the missing-receipt outcome.
A zero CLI exit code did not authorize CPU or TPU work without the installation receipt.
The remaining tests verified owner CLI exit status behavior.

The scientific probe, plugin files, runtime, and numerical tolerances did not change.
These local tests do not qualify a TPU milestone.
Refer to [the repair record](../../experiments/2026-10-04-bitsandbytes-tpu/results/nested-bootstrap-repair.json).
The [first failed attempt](r4-bootstrap-failure.md) remains in the project history.
