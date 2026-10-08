# Canonical test replay

Root must first adopt the exact source and adoption maps.
Root must save `source-map.json` and `adoption-map.json` under
`experiments/2026-10-04-bitsandbytes-tpu/results/compiler-accepted-preparation`.
These two review metadata files are outside the source inventory to avoid a self-referential hash.
The adoption map records their proposed destinations separately.

Run the commands below from the repository root after adoption.
Use fresh absolute output directories.
The commands read canonical adopted files only.
Keep the existing host and isolated JAX/JAXlib 0.7.1 interpreters.
Do not resolve the isolated interpreter symlink to its base Python.
The upstream repository must contain fixed revision `833649043474794b8fe7a4136e0c40faf077b2e0`.

```sh
ROOT=/Users/topamini/code/bitsandbytes-TPU
SCI="$ROOT/experiments/2026-10-04-bitsandbytes-tpu"
HOST_PYTHON=/Users/topamini/code/Post2TPU/.venv/bin/python
UPSTREAM=/Users/topamini/code/Post2TPU/.cache/bitsandbytes-repo
MAPS="$SCI/results/compiler-accepted-preparation"
STAGE="$ROOT/.work/root-compiler-canonical-controls-source"
RESULTS="$ROOT/.work/root-compiler-canonical-controls-results"
"$HOST_PYTHON" -B "$SCI/cloud/tests/compiler-accepted/materialize_controls.py" --root "$ROOT" --source-map "$MAPS/source-map.json" --adoption-map "$MAPS/adoption-map.json" --output "$STAGE"
mkdir "$RESULTS"
"$HOST_PYTHON" -B "$STAGE/run_owned.py" --output "$RESULTS/build1-owner" --seconds 60 "$HOST_PYTHON" -B "$STAGE/build_and_preflight.py" --upstream "$UPSTREAM" --out "$RESULTS/packet1" --result "$RESULTS/packet1.json"
"$HOST_PYTHON" -B "$STAGE/run_owned.py" --output "$RESULTS/build2-owner" --seconds 60 "$HOST_PYTHON" -B "$STAGE/build_and_preflight.py" --upstream "$UPSTREAM" --out "$RESULTS/packet2" --result "$RESULTS/packet2.json"
"$HOST_PYTHON" -B "$STAGE/run_affected.py" --packet "$RESULTS/packet1" --output "$RESULTS/affected"
"$HOST_PYTHON" -B "$STAGE/run_owned.py" --output "$RESULTS/accepted-owner" --seconds 60 "$HOST_PYTHON" -B "$STAGE/accepted_controls.py" --packet "$RESULTS/packet1" --second-packet "$RESULTS/packet2" --output "$RESULTS/accepted"
"$HOST_PYTHON" -B "$STAGE/run_owned.py" --output "$RESULTS/plain-build-owner" --seconds 60 "$HOST_PYTHON" -B "$STAGE/build_plain_native.py" --upstream "$UPSTREAM" --out "$RESULTS/plain-packet" --result "$RESULTS/plain-packet.json"
"$HOST_PYTHON" -B "$STAGE/run_owned.py" --output "$RESULTS/plain-owner" --seconds 60 "$HOST_PYTHON" -B "$STAGE/plain_native_controls.py" --packet "$RESULTS/plain-packet" --output "$RESULTS/plain"
```

Expect 65 affected, 41 accepted dependency, and 23 ordinary native controls.
Expect the two compiler archives to have identical bytes.
The affected metadata must identify the actual accepted native hash and revision.
It must report actual compiler TPU execution as `NOT_RUN`.
The canonical replay above requires no private prepared source or fixture.

For a separate review of the unchanged analyzer and topology controls, use the same staged source tree.
These optional commands do not run a compiler or use a provider.

```sh
"$HOST_PYTHON" -B "$STAGE/run_owned.py" --output "$RESULTS/analyzer-owner" --seconds 120 "$HOST_PYTHON" -B "$STAGE/portable_controls.py" --output "$RESULTS/analyzer"
"$HOST_PYTHON" -B "$STAGE/run_owned.py" --output "$RESULTS/topology-owner" --seconds 120 "$HOST_PYTHON" -B "$STAGE/topology_controls.py" --output "$RESULTS/topology"
```

The earlier root integration review covered 44 analyzer, 18 source, and 26 topology controls.
This repair does not count those historical controls as new runs.
Root must inspect the exact new map and packet before it creates a compiler dispatch gate.
The separate actual experiment must obtain a fresh qualified Linux CPU archive and gate.
