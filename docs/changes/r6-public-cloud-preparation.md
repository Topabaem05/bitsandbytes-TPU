# Public API preparation after native acceptance

The new cloud mode executes 15 fixed public API cases.
Eleven cases use the Pallas path, and four cases use the reference path.
The cases include FP32 gradients and BF16 outputs.
They retain the original upstream classes, inputs, and numerical criteria.
Nested quantization, BF16 gradients, memory, and speed remain outside this scope.

The packet requires the accepted five-case native result.
That dependency does not establish public API correctness.
A new qualified Linux CPU process must first produce all 15 reference rows.
The host independently calculates those rows and verifies their source, runtime, launch arguments, and process closure.
Only then can the owner start the device process.
The device probe executes directly as the leader of its process group.
The first runtime error stops later cases and preserves their unexecuted state.

The reviewer verified 90 adopted files and 19 unchanged prerequisites.
The adopted files contain six cloud changes, 37 scientific files, and 47 control or document files.
The initial adoption map incorrectly included four generated cache files.
The corrected maps exclude those files without changing the source bodies.
The original maps and failed preparation records remain preserved.

The reviewer repeated all 118 controls before and after adoption.
Each repetition passed 79 public controls, 16 compiler controls, and 23 ordinary native controls.
The reviewer also verified all 29 actual process groups from the canonical controls and packet construction.
These groups were absent after their recorded closure.
Synthetic records in these controls do not establish qualified Linux or TPU execution.

All three canonical packets matched the isolated builds exactly.
The public packet contains 122 members and 1,207,176 bytes.
Its SHA-256 is `4220f8b34d2775c1b81369c9e0302ae5cd999af0cf171166f80ac03d4b9b45dd`.
The packet requires the explicit JAX 0.7.1 CPU interpreter for Mosaic inspection.
It rejects the unrelated host JAX 0.11.2 interpreter.

Public result archives can contain at most 256 MiB and 2,000 members, including the receipt.
This limit applies to total uncompressed content.
The remote packager and host recovery both enforce the limit.
The source manifest and phase gates contain the same limits.
The other modes retain their 100 MiB archive limit.
CPU records retain their separate 64 MiB limit.
These archive limits do not provide a live disk quota.

The public mode requests no compiler dump flags.
The [actual compiler failure](r6-compiler-flag-colab-failure.md) remains a separate result.
The compiler mode in this composition retains that failed generation until a separate correction passes review.
No new qualified Linux CPU or public TPU execution occurred during this preparation.
M6 remains unqualified, and three of eight milestones remain accepted.
Refer to [the selected preparation result](../../experiments/2026-10-04-bitsandbytes-tpu/results/public-cloud-preparation.json).

## Replay from repository files

Use the adopted cloud directory and public directory for the public controls and packet construction.
Use the [portable control instructions](../../experiments/2026-10-04-bitsandbytes-tpu/cloud/tests/public-mode/REPRODUCE.md) with those paths.
Those preserved instructions describe the original private preparation.
The compiler fixture materializer requires the older accepted source revision.
The current cloud source has different owner and remote files.
Do not change the historical source map to accept those differences.

Reconstruct the fixture base from its accepted revision:

```sh
repo_root=/Users/topamini/code/bitsandbytes-TPU
host_python=/Users/topamini/code/Post2TPU/.venv/bin/python
replay_root="$repo_root/.work/public-replay-new"
mkdir -p "$replay_root/compiler-base"
git -C "$repo_root" archive 0b9f79a5a94e9342817582b4fb786ec0c73fc104 \
  experiments/2026-10-04-bitsandbytes-tpu | tar -x -C "$replay_root/compiler-base"
fixture_base="$replay_root/compiler-base/experiments/2026-10-04-bitsandbytes-tpu"
"$host_python" -B "$fixture_base/cloud/tests/compiler-accepted/materialize_controls.py" \
  --root "$replay_root/compiler-base" \
  --source-map "$fixture_base/results/compiler-accepted-preparation/source-map.json" \
  --adoption-map "$fixture_base/results/compiler-accepted-preparation/adoption-map.json" \
  --output "$replay_root/compiler-controls"
```

Use a new output directory for each replay.
The materializer must verify all 147 files before it writes the fixture tree.
Pass this tree as `--control-tree` to the adopted `run_existing_regression.py`.
Set `--cloud` to the current adopted cloud directory.
Use the existing bounded ownership runner for each control command.
The reviewer verified this reconstruction from the pinned repository revision.

The reviewer inspected sentence length, document links, and technical terms.
The complete standard dictionary was unavailable.
This review does not establish independent ASD-STE100 conformity.
