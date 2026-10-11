# Portable replay

Use a fresh temporary results directory and the existing fixed interpreters.
Do not install packages or call a provider.
The replacement maps reconstruct the complete source tree from canonical files.
The old 187-file map remains historical and cannot reconstruct this generation.

```sh
REPO=/Users/topamini/code/bitsandbytes-TPU
export BNB_TPU_TEST_REPO_ROOT="$REPO"
HOST_PYTHON=/Users/topamini/code/Post2TPU/.venv/bin/python
JAX_PYTHON=/Users/topamini/code/bitsandbytes-TPU/.work/r6-pallas-preparation/venv-jax071/bin/python
UPSTREAM=/Users/topamini/code/bitsandbytes-TPU/.work/upstream-bnb-8336490
RESULTS=$(mktemp -d /tmp/bnb-tpu-compiler-default-replay-XXXXXXXX)
STAGE="$RESULTS/source"
MAPS="$REPO/experiments/2026-10-04-bitsandbytes-tpu/results/compiler-default-dumps-preparation"
"$HOST_PYTHON" -B "$REPO/experiments/2026-10-04-bitsandbytes-tpu/cloud/tests/compiler-accepted/materialize_controls.py" --root "$REPO" --source-map "$MAPS/source-map.json" --adoption-map "$MAPS/adoption-map.json" --output "$STAGE"
cp "$MAPS/source-map.json" "$STAGE/source-map.json"
cp "$MAPS/adoption-map.json" "$STAGE/adoption-map.json"
"$HOST_PYTHON" -B "$STAGE/run_owned.py" --output "$RESULTS/build1-owner" --seconds 100 "$HOST_PYTHON" -B "$STAGE/build_and_preflight.py" --upstream "$UPSTREAM" --out "$RESULTS/packet1" --result "$RESULTS/packet1.json"
"$HOST_PYTHON" -B "$STAGE/run_owned.py" --output "$RESULTS/build2-owner" --seconds 100 "$HOST_PYTHON" -B "$STAGE/build_and_preflight.py" --upstream "$UPSTREAM" --out "$RESULTS/packet2" --result "$RESULTS/packet2.json"
"$HOST_PYTHON" -B "$STAGE/run_owned.py" --output "$RESULTS/boundary-owner" --seconds 100 "$HOST_PYTHON" -B "$STAGE/boundary_controls.py" --output "$RESULTS/boundary"
"$HOST_PYTHON" -B "$STAGE/run_owned.py" --output "$RESULTS/affected-owner" --seconds 260 "$HOST_PYTHON" -B "$STAGE/run_affected.py" --packet "$RESULTS/packet1" --jax-python "$JAX_PYTHON" --output "$RESULTS/affected"
"$HOST_PYTHON" -B "$STAGE/run_owned.py" --output "$RESULTS/accepted-owner" --seconds 100 "$HOST_PYTHON" -B "$STAGE/accepted_controls.py" --packet "$RESULTS/packet1" --second-packet "$RESULTS/packet2" --jax-python "$JAX_PYTHON" --output "$RESULTS/accepted"
"$HOST_PYTHON" -B "$STAGE/run_owned.py" --output "$RESULTS/flags-owner" --seconds 100 "$HOST_PYTHON" -B "$STAGE/flag_controls.py" --jax-python "$JAX_PYTHON" --output "$RESULTS/flags"
"$HOST_PYTHON" -B "$STAGE/run_owned.py" --output "$RESULTS/plain-build-owner" --seconds 100 "$HOST_PYTHON" -B "$STAGE/build_plain_native.py" --upstream "$UPSTREAM" --out "$RESULTS/plain-packet" --result "$RESULTS/plain-packet.json"
"$HOST_PYTHON" -B "$STAGE/run_owned.py" --output "$RESULTS/plain-owner" --seconds 100 "$HOST_PYTHON" -B "$STAGE/plain_native_controls.py" --packet "$RESULTS/plain-packet" --output "$RESULTS/plain"
"$HOST_PYTHON" -B "$STAGE/run_owned.py" --output "$RESULTS/public-owner" --seconds 100 "$HOST_PYTHON" -B "$STAGE/public_branch_controls.py" --upstream "$UPSTREAM" --output "$RESULTS/public"
"$HOST_PYTHON" -B "$STAGE/run_owned.py" --output "$RESULTS/portable-owner" --seconds 100 "$HOST_PYTHON" -B "$STAGE/portable_layout_controls.py" --output "$RESULTS/portable"
```

Keep compact summaries, closure records, and inventory hashes before removal of completed large fixtures.
Use a fresh qualified Linux CPU gate for the actual compiler experiment.
Local JAX CPU parser success does not qualify libtpu or TPU execution.
