# Portable Mosaic controls

The repository now contains the native27 conversion and recovery controls.
The runner uses explicit source and interpreter paths with fresh output directories.
It uses the pinned JAX 0.7.1 CPU frontend.
No production source, kernel, runtime, or numerical criterion changed.

The compressed fixture archive contains 17 files and 172,225 bytes.
Its inventory binds every expanded member by size and SHA-256.
The fixtures contain unqualified macOS CPU records and synthetic native records.
They contain no actual TPU result.
The preparation removed private debug paths through the official deserialize and serialize passes.
It preserved the current intermediate representation exactly.

## Test harness correction

The earlier private rejection helper could catch its own assertion after an incorrect record unexpectedly passed.
The corrected helper places that assertion outside the exception handler.
A new control supplies a fake successful verifier and requires a failed test without a pass record.
The reviewer inspected all nine original negative records.
Each showed a genuine `ValueError`, rather than the helper's assertion.
All ten negative records in the adopted suite also showed genuine `ValueError` results.
The original records remain unchanged.

## Independent results

The reviewer repeated all 35 controls before and after adoption.
Each run closed its two owned process groups without cleanup errors.

| Control group | Passed |
| --- | ---: |
| Independent conversion audit | 13 |
| Record recovery | 14 |
| First-error handling | 3 |
| Test harness rejection | 1 |
| Official serializer | 4 |

Run the following command with existing absolute interpreter paths and a fresh output directory.
The ambient interpreter must use a JAX version other than 0.7.1.

```sh
"$HOST_PYTHON" -B experiments/2026-10-04-bitsandbytes-tpu/cloud/tests/mosaic_controls.py \
  --native experiments/2026-10-04-bitsandbytes-tpu/native \
  --host-python "$HOST_PYTHON" --jax-python "$JAX071_PYTHON" \
  --ambient-python "$WRONG_JAX_PYTHON" --output "$RESULTS"
```

The [selected result](../../experiments/2026-10-04-bitsandbytes-tpu/results/native-mosaic-portable-controls.json) binds both independent runs and the fixture archive.
The local controls do not establish qualified Linux CPU or TPU execution.
M6 remains unqualified, and accepted milestones remain three of eight.
