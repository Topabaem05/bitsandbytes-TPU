Use the existing host Python, explicit JAX0.7.1 CPU environment, official CLI0.7.4 source and unchanged runtime wheel cache. Do not install or call a provider. Each output path must be new.

Set these paths for the private review. After adoption, set cloud_source to the canonical cloud directory. Set public_source to the admitted public scientific directory. The portable controls have no fixed private fixture path.

```sh
repo_root=/Users/topamini/code/bitsandbytes-TPU
prep_root="$repo_root/.work/r6-public-bf16-cloud-preparation/merged"
cloud_source="$prep_root/cloud"
public_source="$repo_root/.work/r6-public-bf16-device-preparation"
host_python=/Users/topamini/code/Post2TPU/.venv/bin/python
jax071_python="$repo_root/.work/r6-pallas-preparation/venv-jax071/bin/python"
cli_python=/Users/topamini/code/Post2TPU/.runtime/cloud-cli/bin/python
runtime_cache="$repo_root/.work/cache/bitsandbytes-tpu-wheelhouse"
cd "$repo_root"
"$host_python" -B "$prep_root/tests/run_controls.py" \
  --cloud "$cloud_source" --packet "$prep_root/packet-v1" \
  --runtime-wheels "$runtime_cache" --jax071-python "$jax071_python" \
  --cli-python "$cli_python" --output "$repo_root/.work/root-public79-review" --seconds 180
```

Expected result: 79 pytest controls and a PASS review.json. Every reported actual group must be absent. The original local CPU record remains unqualified. Qualified-shaped test records and source-only wheels are synthetic and are never installed.

Build a fresh public source packet with the admitted builder. This creates no allocation and does not install a runtime.

```sh
"$host_python" -B "$cloud_source/build_public_packet.py" \
  --upstream "$repo_root/.work/upstream-bnb-8336490" \
  --out "$repo_root/.work/root-public15-packet" --repo-root "$repo_root" \
  --public-source "$public_source" \
  --public-manifest-sha256 131cdd25935b3216858b823c59732a30f3e5e43c25bbd8cc2eb9f284108e12bf
```

The two fresh private public archives are byte-identical: 122 members, 1207176 bytes, SHA4220f8b34d2775c1b81369c9e0302ae5cd999af0cf171166f80ac03d4b9b45dd. The manifest SHA is 58ac0f17266484e420320e4c6bc76fc27d13c73d1b0fad90758d486f0c9c327b. Adoption and a committed packet will require a fresh source readback.

To replay the preserved compiler and plain-native predicates, first use canonical cloud/tests/compiler-accepted/materialize_controls.py. Its canonical source and adoption maps are under results/compiler-accepted-preparation. The materializer must verify all 147 exact files. Pass that new tree as --control-tree to tests/run_existing_regression.py, with --cloud set to this composition. Use --mode compiler and its fresh compiler packet for 16 controls. Use --mode native and its fresh native packet for 23 controls. Launch each through an existing bounded Ownership runner. The retained final commands are in compiler-regression-owned-v2/launch.json and native-regression-owned-v2/launch.json.

For complete source packet replay, tests/build_mode_packets.py accepts explicit --cloud, --repo-root, --upstream, --public-source, --m2-dependency and --output. It builds and preflights the three modes. The M2 dependency must be the exact previously accepted document, not a generated acceptance. Use a bounded owned runner. The exact completed command is in mode-packets-owned-v1/launch.json.

Root owns actual dispatch. The owner requires an exact ACTUAL_DISPATCH_AUTHORIZED acceptance, CLI identity, browser adoption if used, original deadline and explicit unchanged runtime cache. Before the device phase, it must receive a fresh qualified Linux15 oracle and independently replay it. After the device phase, it must recover the complete records and prove child and host closure. Source packet checks alone cannot meet these requirements.
