# Reproduce the exact nested preparation

These commands use the existing local Python. They do not install a local runtime. The root reviewer owns source adoption, account activity, resource allocation, result acceptance, and Git operations. Source admission does not qualify M4.

Set the local paths:

```sh
m4_python=/Users/topamini/code/Post2TPU/.venv/bin/python
m4_repo=/Users/topamini/code/bitsandbytes-TPU
m4_bundle=/private/tmp/bnb-tpu-m4-cli-admission-v2-20261011-3_fpudng
m4_experiment=experiments/2026-10-04-bitsandbytes-tpu
m4_source="$m4_bundle/canonical/$m4_experiment"
```

## Inspect and materialize the additive source

Read the bundle review and verify the source archive SHA256 from the independent root readback. The adoption manifest lists every new file. The materializer checks the full source closure before it writes. It permits an absent destination or identical bytes. It rejects different canonical bytes and symlinks.

```sh
m4_manifest_sha=$($m4_python -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["adoption_manifest_sha256"])' "$m4_bundle/review.json")
$m4_python -B "$m4_source/nested-exact/materialize.py" \
  --bundle-root "$m4_bundle/canonical" \
  --manifest "$m4_bundle/adoption-manifest.json" \
  --manifest-sha256 "$m4_manifest_sha" --repository "$m4_repo" --check-only
```

After root source review, run the same command without `--check-only`. This action adds only `nested-exact` and `cloud/exact-nested`. It does not replace the shared owner, existing plugin, or accepted records. Read back the resulting materializer report. Add the logical change in `CHANGES-PROPOSAL.md` to the shared changelog during root adoption.

## Replay local controls

The independent integration controls require no archive or account connection. They create isolated correct and incorrect fixtures, retain a compact review, and remove their fixture bodies.

```sh
m4_controls=$(mktemp -d /tmp/bnb-tpu-m4-controls-XXXXXXXX)
$m4_python -B "$m4_source/cloud/exact-nested/tests/controls.py" --output "$m4_controls/integration"
$m4_python -B "$m4_source/cloud/exact-nested/tests/driver_controls.py" --output "$m4_controls/driver"
$m4_python -B "$m4_source/cloud/exact-nested/tests/cli_identity_controls.py" \
  --actual-python /Users/topamini/code/Post2TPU/.runtime/cloud-cli/bin/python \
  --actual-identity "$m4_repo/.work/task2-cloud/cli-identity.json" \
  --output "$m4_controls/cli-identity.json"
```

The driver controls execute the actual driver and remote stage branches through a fake OfficialCLI. Their source reviews, CPU/device records, acceptances, and audit results are synthetic protocol fixtures. They prove branch ordering and closure behavior. They do not prove a scientific result. Every fixture directory is removed after its compact record is read.

The original 16 controls and the seven state/codec controls need four historical selections. Restore only these selections from the existing private archive. Do not initialize or reset the archive. Do not restore runtime caches or wheel bodies.

```sh
m4_restore=$(mktemp -d /tmp/bnb-tpu-m4-readonly-XXXXXXXX)
restic --insecure-no-password -r rclone:gdrive-backup:restic \
  -o 'rclone.args=serve restic --stdio --drive-root-folder-id 1Wj_aNLgxquksWtyHthccE7bpTnT5Z4Vz --tpslimit 4' \
  --cache-dir "$m4_repo/.work/artifact-storage-20261009/cache" \
  restore a06abedfe14c1e8451d8ad500b48b1bf82104de7a314cba4c4886b396a034a86 \
  --target "$m4_restore" --verify \
  --include /Users/topamini/code/bitsandbytes-TPU/.work/cloud-20261008-nested-browser-adopted/recovered/cpu-oracle \
  --include /Users/topamini/code/bitsandbytes-TPU/.work/cloud-20261008-nested-browser-adopted/recovered/nested/receipt.json \
  --include /Users/topamini/code/bitsandbytes-TPU/.work/r4-actual-arithmetic-analysis/readback-map.json \
  --include /Users/topamini/code/bitsandbytes-TPU/.work/r4-exact-muldiv-preparation/fraction_reference.py
$m4_python -B "$m4_source/nested-exact/run_controls.py" \
  --historical-root "$m4_restore/Users/topamini/code/bitsandbytes-TPU/.work" \
  --output "$m4_controls/science"
```

Check `science/result.json`: original 16, new seven, PASS, temporary directory removed, and historical hashes unchanged. The controls check all 79 raw rows against readback-map SHA256 `7799ac334c5bffcec9c348217ba33ce80a433664bd82dbb621ada77d4079a52c`. Record the restore inventory and hashes, then remove only the newly created `$m4_restore` directory. The historical CPU data remains a comparison fixture. It is never resealed as a new qualified oracle.

## Build a source packet

After root review, root writes the source-review record with this explicit command. Set `M4_SOURCE_REVIEW` to a new external file:

```sh
$m4_python -B "$m4_source/cloud/exact-nested/root_records.py" source-review \
  --source "$m4_source" --output "$M4_SOURCE_REVIEW" \
  --reviewer-decision ACCEPT_SOURCE_AFTER_ROOT_REVIEW
```

The record has these fixed fields:

```json
{
  "kind": "ROOT_NESTED_EXACT_SOURCE_REVIEW",
  "status": "SOURCE_REVIEW_ACCEPTED_FOR_EXPERIMENT",
  "integration_map_sha256": "cc237461aa97a4b33ebca3d794cba0b158139c137ce2323cbb9b2ebda0cd50b6",
  "source_admission_sha256": "75790f1d4613e20daa06d99111288463ef24b05aaa29922a7580512078f9912f",
  "science_map_sha256": "9e71efc73b0f2b3a210f8131c10972569a5e2d04cf298fcf5a8da59772cf4f5a",
  "device_acceptance": null
}
```

Set `M4_SOURCE_REVIEW` to that reviewed file. This is a root decision; the preparation does not create it.

```sh
m4_packet_parent=$(mktemp -d /tmp/bnb-tpu-m4-packet-XXXXXXXX)
m4_review_sha=$($m4_python -B -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$M4_SOURCE_REVIEW")
$m4_python -B "$m4_source/cloud/exact-nested/build_packet.py" \
  --source "$m4_source" --source-review "$M4_SOURCE_REVIEW" \
  --source-review-sha256 "$m4_review_sha" --output "$m4_packet_parent/packet" \
  > "$m4_packet_parent/build.json"
m4_packet_sha=$($m4_python -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["packet_sha256"])' "$m4_packet_parent/build.json")
m4_driver_sha=$($m4_python -B -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$m4_source/cloud/exact-nested/driver.py")
```

The builder verifies the frozen 19 science members, 48 upstream members, plugin closure, helper bytes, and root review. The driver independently recovers the whole admitted packet and compares every loose file before its first account call.

## Root dispatch and CLI execution

Root supplies the existing admitted CLI Python and identity record as `M4_CLI_PYTHON` and `M4_CLI_IDENTITY`. The driver accepts only google-colab-cli 0.7.4 and reads every installed admitted CLI Python file. It does not start OAuth. Root sets `M4_DISPATCH` to a new external file after it checks the reviewed packet, intended session, and budget, then runs:

```sh
$m4_python -B "$m4_source/cloud/exact-nested/root_records.py" dispatch \
  --packet "$m4_packet_parent/packet" --packet-zip "$m4_packet_parent/packet.zip" \
  --packet-sha256 "$m4_packet_sha" --cli-python "$M4_CLI_PYTHON" \
  --cli-identity "$M4_CLI_IDENTITY" --output "$M4_DISPATCH" \
  --reviewer-decision AUTHORIZE_ONE_ALLOCATION_AFTER_ROOT_REVIEW
```

The record has this schema, with the three measured SHA256 values filled by the command:

```json
{
  "kind": "ROOT_M4_EXACT_NESTED_DISPATCH",
  "status": "ACTUAL_DISPATCH_AUTHORIZED",
  "packet_sha256": "the exact build.json packet_sha256",
  "driver_sha256": "the exact current driver.py SHA256",
  "cli_identity_sha256": "the exact admitted CLI identity record SHA256",
  "science_manifest_sha256": "9e71efc73b0f2b3a210f8131c10972569a5e2d04cf298fcf5a8da59772cf4f5a",
  "allocation_seconds": 3600,
  "allocation_attempts": 1
}
```

These three SHA256 values are measured above or read from the existing admitted identity file. Root must supply them; an offline fixture must not be used. The allocation-attempt field is the maximum. Browser adoption makes zero allocation calls.

Use a fresh root-selected session name. Set `M4_SESSION` to that name, `M4_DISPATCH` to the explicit dispatch record, and `M4_RUN_PARENT` to a fresh local run directory. The `owned` child directory must not exist. Run the driver in the root's managed command session so root can inspect the review request while it waits:

```sh
$m4_python -B "$m4_source/cloud/exact-nested/driver.py" \
  --packet "$m4_packet_parent/packet" --packet-zip "$m4_packet_parent/packet.zip" \
  --packet-sha256 "$m4_packet_sha" --cli-python "$M4_CLI_PYTHON" \
  --cli-identity "$M4_CLI_IDENTITY" --dispatch "$M4_DISPATCH" \
  --session "$M4_SESSION" --output "$M4_RUN_PARENT/owned" \
  --accept79-file "$M4_RUN_PARENT/root-accepted79.json"
```

The CLI path requires an empty server before its one allocation attempt. Each CLI process and remote child has a bounded ownership record and verified process-group closure. The driver budget starts before the allocation request. The browser path uses the previously recorded conservative allocation start. Every science child receives the same absolute `birth + 3600 - 120` deadline. No state phase resets it.

The remote install uses the pinned CPython bootstrap and runtime/build locks. It builds the two source wheels without native CMake, installs without bytecode, and reads all admitted installed bytes before the fresh CPU process. CPU79 explicitly sets `ATEN_CPU_CAPABILITY=avx2` and one-thread environment variables. It verifies the loaded capability and thread count before arithmetic. The host recovers CPU79 and audits the actual argv, PID, PGID, token, wheel bytes, numerical matrix, and reaped launch before it uploads the CPU gate.

The driver then runs actual79, recovers the complete physical archive, and runs the scientific host verifier. A failed operation attempts bounded recovery before exact-session closure. A returned numerical row remains available when graph observation fails; that record cannot qualify execution.

## Browser session adoption

Root must create the browser runtime. The preparation makes no browser or allocation call. Record a conservative budget start immediately before the root's one UI allocation action:

```sh
$m4_python -B "$m4_source/cloud/exact-nested/browser_record.py" prepare \
  --packet-sha256 "$m4_packet_sha" --driver "$m4_source/cloud/exact-nested/driver.py" \
  --cli-identity "$M4_CLI_IDENTITY" --session "$M4_SESSION" \
  --output "$M4_RUN_PARENT/browser-intent"
```

Root creates one TPU runtime in its authenticated browser. Root pastes the generated `browser-intent/first-cell.py` into that runtime's first cell and checks the exact marker hash in its output. Root obtains the actual endpoint and hardware from its observed assignment. Set `M4_ENDPOINT` and `M4_HARDWARE` to those observed values (`V5E1` or `V6E1`). Do not substitute the time of a later cell for the allocation start.

```sh
$m4_python -B "$m4_source/cloud/exact-nested/browser_record.py" finalize \
  --intent "$M4_RUN_PARENT/browser-intent/intent.json" \
  --endpoint "$M4_ENDPOINT" --hardware "$M4_HARDWARE" \
  --output "$M4_RUN_PARENT/browser-adoption.json"
m4_adoption_sha=$($m4_python -B -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$M4_RUN_PARENT/browser-adoption.json")
```

Use the same driver command, replace `--session "$M4_SESSION"` with these arguments:

```sh
--browser-adoption "$M4_RUN_PARENT/browser-adoption.json" \
--browser-adoption-sha256 "$m4_adoption_sha"
```

The driver checks fresh record timestamps, all packet/driver/CLI bindings, one exact official assignment, hardware, endpoint, and the marker bytes before it marks the runtime as owned. It makes no allocation call in this mode. It removes an unproved provisional local registration instead of stopping an unproved runtime.

## Root actual79 acceptance and state8 boundary

Wait for `owned/root-review-request.json`. The request identifies the recovered records, actual79 completion time, host audit SHA256, original deadline, and acceptance file. Root checks the archived physical inventory, actual launch closure, fresh CPU gate, all 79 rows, graph witnesses, exact gates, and remaining allocation time. M4 is still unqualified.

Set `m4_records` and `m4_remote_base` from that actual request and `owned/driver.json`:

```sh
m4_records=$($m4_python -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["records"])' "$M4_RUN_PARENT/owned/root-review-request.json")
m4_remote_base=$($m4_python -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["remote_base"])' "$M4_RUN_PARENT/owned/driver.json")
```

After root review, this explicit command reruns the scientific verifier and writes the unchanged accepted79 schema atomically. The running driver does not call this command:

```sh
$m4_python -B "$m4_source/cloud/exact-nested/host_audit.py" accept79 \
  --packet "$m4_packet_parent/packet" --records "$m4_records" \
  --remote-base "$m4_remote_base" --packet-sha256 "$m4_packet_sha" \
  --reviewer-decision ACCEPT_AFTER_ROOT_RECORD_REVIEW \
  --output "$M4_RUN_PARENT/root-accepted79.json"
```

Missing acceptance waits only within the original deadline. Wrong, old-packet, failed, or mismatched acceptance rejects before state CPU. The state start needs at least 780 seconds left in the original allocation, including the 120-second closure reserve. This minimum is an admission bound, not a measured runtime or a promise that a late stage will finish. The driver records the actual external acceptance boundary and remaining budget.

State CPU runs in a distinct owned process. The host independently validates its original deadline, token, PID/PGID, separate reaped launch, public checkpoint bytes, and link to the fresh CPU79 oracle. It uploads a separate state CPU launch-audit record. State device execution requires this record and the accepted79 bytes. It uses fresh save and restore children in the parent group and retains exact checkpoint pairs. The host then recovers and verifies the state records. Root must still review the required actual result before milestone acceptance.

To end after actual79, add `--stop-after-79` and omit `--accept79-file`. The driver recovers actual79 and closes that allocation. A later driver run starts a new bounded experiment; it cannot resume or extend the closed allocation.

## Closure, limits, and disposal

Each recovery covers every physical record and counts archive metadata within the 2,000-file and 96 MiB inner limits. The compressed outer archive must be at most 100 MiB. The lossless scientific record store retains original graph and numerical bytes. No actual compressed size or scientific pass is claimed by the offline tests. An evidence-limit result blocks acceptance.

After complete host byte readback, the driver deletes transfer parts and remote transfer archives. It removes a previous recovered record tree only after its verified replacement contains the cumulative record set. Compact hashes and disposal records remain. The final recovered record tree remains for root review and the private archive; delete it after that archive is independently read back.

Closure requires an exact-session stop, an empty server listing, one parsed `Active assignments: 0` row, one parsed zero `Usage rate` row, no cleanup errors, and elapsed allocation time no greater than 3,600 seconds. CLI exit zero alone is insufficient. A failed closure is `BLOCKED_CLEANUP`. Signals trigger bounded exact-session cleanup. Keep the actual owner and child closure records with the private result.

After local reviews, keep only the compact review JSON, source hashes, required diagnostic evidence, and the reviewed source bundle. Remove generated packets, restored historical data, checkpoints, graph fixtures, and transfer parts. The supplied controls already remove their disposable fixture bodies.
