# Kaggle M2 repetition

This program can repeat the accepted M2 test on Kaggle. The test has 42 API cases and four public transfer cases. A complete result has 196 numerical comparisons. This result supplies partial M8 evidence. It cannot complete M8.

## Fixed inputs

The builder requires a project source directory and an upstream archive. It does not obtain inputs from an accepted result archive. `payload-source-pins.json` identifies all 40 required source files. The builder contains its exact SHA-256 hash.

The source map admits two reviewed hardware changes from revision `838f193589a640addddf27b11d98a70693a6164e`. The other 38 files keep their previous hashes. The numerical inputs, source correction and runtime remain fixed. The Kaggle requests remain fixed.

| Input | Required value |
| --- | --- |
| Upstream revision | `833649043474794b8fe7a4136e0c40faf077b2e0` |
| Upstream archive SHA-256 | `41e1fc76ddd4a25915f8f4f804d881be9fc1a89ab88834db4284f9971ae7769e` |
| Source map SHA-256 | `f7b39c3b1647d05a7d48f10c166c723a021fd7780c0fcc14366a5e98cd32f617` |
| Source admission SHA-256 | `8b2b3e26e2e7c98c0fde52f51663bed84999ac733dcbc39573ef8b3aeca4f9a1` |
| Host CLI | Kaggle 2.2.4 |
| Host SDK | kagglesdk 0.1.37 |

The builder applies the reviewed patch to a new archive directory. It examines the complete package maps before and after that patch. Only `bitsandbytes/nn/modules.py` can change. It requires the exact corrected archive hash and source admission hash.

## Offline regression

Use a new output directory. Supply the original scientific fixture files in the project source directory. Supply the fixed Kaggle Python interpreter. Run this command with the applicable paths:

```sh
python -B experiments/2026-10-04-bitsandbytes-tpu/m8/tests.py \
  --project-source /path/to/bitsandbytes-TPU \
  --base-archive /path/to/reviewed-upstream.tar \
  --kaggle-python /path/to/pinned-kaggle-python \
  --output /path/to/new-offline-controls
```

The tests use synthetic provider and device records. The ownership tests use real local processes. The SDK tests examine installed metadata and source hashes. They do not import the authentication entry point. These tests cannot qualify Kaggle or TPU execution.

## Candidate preparation

Select the identified account owner and a new dedicated slug. Select a nonce and one absolute deadline. Use matching title and slug values. Use a new output directory. Run this command with the selected values:

```sh
python -B experiments/2026-10-04-bitsandbytes-tpu/m8/build_candidate.py \
  --project-source /path/to/bitsandbytes-TPU \
  --base-archive /path/to/reviewed-upstream.tar \
  --out /path/to/new-candidate \
  --owner IDENTIFIED_OWNER --slug NEW_DEDICATED_SLUG --title 'Matching Title' \
  --nonce 32_LOWERCASE_HEX_DIGITS --deadline-epoch ORIGINAL_ABSOLUTE_DEADLINE
```

The builder makes no provider request. It writes one script with the complete ZIP package. It also writes the metadata, plan and source maps. The plan requires 298 output members. The original scientific cases and fixed process steps determine this set.

Examine the exact script, metadata, plan and source maps before submission. Do not submit a synthetic test candidate.

## Execution and recovery

The provider entry is `owner_cli.py execute`. It requires `--live-provider`, `--candidate`, `--output` and `--kaggle-python`. Supply a new output directory. Start this entry only after root authorization.

The entry starts an external supervisor. The supervisor starts the internal owner and each SDK worker as direct child processes. Its registry records each worker before the provider waits. It does not use an old registry to stop processes.

The installed SDK paths come from the selected interpreter's metadata. The worker requires the fixed versions and all 11 source hashes. It does not require a specific personal installation directory.

Root recovery repeats the original scientific verifier. `recovery.py` can examine downloaded records without a provider request. It requires explicit candidate, evidence and output paths. The Kaggle script runs its CPU gate before its device phase. Root review occurs after device execution. This differs from the Colab host gate.

The `owner_cli.py close` entry uses the same candidate and output directory. It requires a root observation for the exact returned kernel and version. It can record closure after scientific failure. It preserves that failure and keeps M8 unqualified.

A terminal API status does not prove resource closure. An ambiguous submission without an exact identity remains unconfirmed. This entry has no established API session mapping or cancellation operation. Root must observe actual hardware and resource closure independently.

The supervisor cannot guarantee cleanup after its own SIGKILL or host failure. Local process closure cannot prove Kaggle resource closure. See [DESIGN.md](DESIGN.md) for the fixed order, limits and result rules.
