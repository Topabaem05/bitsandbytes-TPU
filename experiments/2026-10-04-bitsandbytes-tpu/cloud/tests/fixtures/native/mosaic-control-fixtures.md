These fixtures test the native27 `mosaic-serde7-v1` verifier on CPU. They do not contain actual TPU execution records.

The archive has 17 regular files. Its compressed size is 172,225 bytes. The inventory gives the exact expanded size and SHA-256 for each file. The runner checks the inventory, archive and each member before extraction. The archive is extracted once per test run.

The CPU oracle files are retained macOS ARM64 Torch 2.14.1 records. They are unqualified for the fixed Linux runtime. Tests use them only with `actual_device=False`. The numerical gates and input values are unchanged.

The original version-8 payloads came from retained CPU frontend records. The fixture preparation used the pinned JAX 0.7.1 official deserialize pass. It printed the current IR without debug locations, parsed that text, and checked exact current-IR equality. It then used official serialization to version 8 and the reviewed converter to version 7. This removes private source paths from debug locations. It does not edit a version attribute. `provenance.json` binds the retained and normalized payload hashes.

The HLO text and proto files were parsed and printed with JAX 0.7.1. This operation did not compile or execute a TPU program. The strict-ordering fixture is valid input whose requested version-7 conversion is unsupported.

Run `mosaic_controls.py` with explicit native, host and JAX interpreter paths. The frozen native27 manifest must have SHA-256 `d294b189011f8200e039761f054c5cc96a15275b679efeb7bdde1ef67b414ffa`. Use a non-0.7.1 interpreter for the `--ambient-python` negative control. No installation or provider operation occurs in the runner.
