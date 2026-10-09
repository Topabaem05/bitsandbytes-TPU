# Portable replay

Use a fresh output directory.
Use an existing Python 3.12.14 interpreter and the fixed upstream Git checkout.
Do not install a runtime.
Set `REPO`, `HOST_PYTHON`, `UPSTREAM`, and `RESULTS` to absolute paths.
The source and adoption maps reconstruct the complete test tree after root adoption.
They use exact canonical destinations.
No private prepared fixture is required.

```sh
REPO=/Users/topamini/code/bitsandbytes-TPU
HOST_PYTHON=/Users/topamini/code/Post2TPU/.venv/bin/python
UPSTREAM=/Users/topamini/code/Post2TPU/.cache/bitsandbytes-repo
RESULTS="$REPO/.work/root-compiler-libtpu021-controls"
STAGE="$RESULTS/source"
MAPS="$REPO/experiments/2026-10-04-bitsandbytes-tpu/results/compiler-libtpu021-preparation"
"$HOST_PYTHON" -B "$REPO/experiments/2026-10-04-bitsandbytes-tpu/cloud/tests/compiler-accepted/materialize_controls.py" --root "$REPO" --source-map "$MAPS/source-map.json" --adoption-map "$MAPS/adoption-map.json" --output "$STAGE"
cp "$MAPS/source-map.json" "$STAGE/source-map.json"
cp "$MAPS/adoption-map.json" "$STAGE/adoption-map.json"
"$HOST_PYTHON" -B "$STAGE/run_owned.py" --output "$RESULTS/build1-owner" --seconds 100 "$HOST_PYTHON" -B "$STAGE/build_and_preflight.py" --upstream "$UPSTREAM" --out "$RESULTS/packet1" --result "$RESULTS/packet1.json"
"$HOST_PYTHON" -B "$STAGE/run_owned.py" --output "$RESULTS/build2-owner" --seconds 100 "$HOST_PYTHON" -B "$STAGE/build_and_preflight.py" --upstream "$UPSTREAM" --out "$RESULTS/packet2" --result "$RESULTS/packet2.json"
"$HOST_PYTHON" -B "$STAGE/run_owned.py" --output "$RESULTS/affected-owner" --seconds 260 "$HOST_PYTHON" -B "$STAGE/run_affected.py" --packet "$RESULTS/packet1" --output "$RESULTS/affected"
"$HOST_PYTHON" -B "$STAGE/run_owned.py" --output "$RESULTS/accepted-owner" --seconds 100 "$HOST_PYTHON" -B "$STAGE/accepted_controls.py" --packet "$RESULTS/packet1" --second-packet "$RESULTS/packet2" --output "$RESULTS/accepted"
"$HOST_PYTHON" -B "$STAGE/run_owned.py" --output "$RESULTS/flags-owner" --seconds 100 "$HOST_PYTHON" -B "$STAGE/flag_controls.py" --output "$RESULTS/flags"
"$HOST_PYTHON" -B "$STAGE/run_owned.py" --output "$RESULTS/plain-build-owner" --seconds 100 "$HOST_PYTHON" -B "$STAGE/build_plain_native.py" --upstream "$UPSTREAM" --out "$RESULTS/plain-packet" --result "$RESULTS/plain-packet.json"
"$HOST_PYTHON" -B "$STAGE/run_owned.py" --output "$RESULTS/plain-owner" --seconds 100 "$HOST_PYTHON" -B "$STAGE/plain_native_controls.py" --packet "$RESULTS/plain-packet" --output "$RESULTS/plain"
"$HOST_PYTHON" -B "$STAGE/run_owned.py" --output "$RESULTS/public-owner" --seconds 100 "$HOST_PYTHON" -B "$STAGE/public_branch_controls.py" --upstream "$UPSTREAM" --output "$RESULTS/public"
"$HOST_PYTHON" -B "$STAGE/run_owned.py" --output "$RESULTS/portable-owner" --seconds 100 "$HOST_PYTHON" -B "$STAGE/portable_layout_controls.py" --output "$RESULTS/portable"
```

The existing JAX 0.7.1 audit interpreter remains an explicit local prerequisite at `.work/r6-pallas-preparation/venv-jax071/bin/python`.
Use fresh Linux CPU oracles for all five cases before actual dispatch.
Archive and recover the oracles.
Qualify the oracles independently.
Local frontend CPU parse success does not qualify libtpu.
Adopt this exact generation before an actual experiment.
Use a new exact dispatch gate.
The actual backend flag preflight is part of the new compiler remote path.
Make sure the preflight passes before the unchanged science parent starts.
Do not reuse the prior failed dump inventory or native receipt.
