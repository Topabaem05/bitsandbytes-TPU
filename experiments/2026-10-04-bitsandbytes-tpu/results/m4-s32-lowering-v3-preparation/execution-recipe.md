Use the live acceptance path for a run that may continue to state8.
The driver waits after fresh CPU79 and TPU79 recovery. Root then accepts those exact records.
The driver validates the current packet, device receipt, fresh CPU oracle, source generation, and remaining original time before state8.

`--stop-after-79` returns through the driver's cleanup block and stops the runtime.
It cannot later resume state8 from that runtime.
A separate state allocation must rerun fresh CPU79 and TPU79, then accept that allocation's current records.
Historical acceptance is insufficient.

Run the following only after root adopts and verifies the repaired source and driver.
The canonical path currently carries the prior generation until adoption.
All paths below must be fresh absolute paths. Use the actual retained CLI interpreter and identity paths.
This recipe does not authorize or launch cloud work.

```sh
PY=/Users/topamini/code/Post2TPU/.venv/bin/python
BASE=/Users/topamini/code/bitsandbytes-TPU/experiments/2026-10-04-bitsandbytes-tpu
CONTROL="$BASE/cloud/exact-nested"
RUN=/tmp/ROOT_SELECTED_FRESH_M4_RUN
REVIEW="$RUN/source-review.json"
PACKET="$RUN/packet"
PACKET_ZIP="$RUN/packet.zip"
CLI_PY=/ABSOLUTE_RETAINED_OFFICIAL_CLI_PYTHON
CLI_ID=/ABSOLUTE_RETAINED_VERIFIED_CLI_IDENTITY_JSON
DISPATCH="$RUN/dispatch.json"
BROWSER="$RUN/browser"
ADOPTION="$RUN/browser-adoption.json"
DRIVER_OUT="$RUN/actual"
ACCEPT_STAGE="$RUN/accepted79-reviewed.json"
ACCEPT_PATH="$RUN/accepted79.json"
SESSION=ROOT_SELECTED_FRESH_EXACT_SESSION
```

Complete root source review and packet building before starting the browser clock.
The records below express explicit root decisions. They contain no device acceptance.

```sh
$PY -B "$CONTROL/root_records.py" source-review --source "$BASE" --output "$REVIEW" --reviewer-decision ACCEPT_SOURCE_AFTER_ROOT_REVIEW
$PY -B "$CONTROL/build_packet.py" --source "$BASE" --source-review "$REVIEW" --source-review-sha256 EXACT_RETURNED_REVIEW_SHA --output "$PACKET"
$PY -B "$CONTROL/root_records.py" dispatch --packet "$PACKET" --packet-zip "$PACKET_ZIP" --packet-sha256 EXACT_RETURNED_PACKET_SHA --cli-python "$CLI_PY" --cli-identity "$CLI_ID" --output "$DISPATCH" --reviewer-decision AUTHORIZE_ONE_ALLOCATION_AFTER_ROOT_REVIEW
$PY -B "$CONTROL/browser_record.py" prepare --packet-sha256 EXACT_RETURNED_PACKET_SHA --driver "$CONTROL/driver.py" --cli-identity "$CLI_ID" --session "$SESSION" --output "$BROWSER"
```

`browser_record.py prepare` starts the original 3600-second clock immediately.
Create one free browser TPU runtime and run the generated `first-cell.py` there.
Observe the actual endpoint and V5E1 or V6E1 hardware before finalization.
Do not invent endpoint identifiers or restart the intent clock.

```sh
$PY -B "$CONTROL/browser_record.py" finalize --intent "$BROWSER/intent.json" --endpoint ACTUAL_OBSERVED_ENDPOINT --hardware V5E1 --output "$ADOPTION"
```

Start the driver under root's approved owned outer launcher.
Its outer timeout must preserve the same original birth and allow exact closure.
The short worker helper's 300-second limit is unsuitable for the whole 3600-second scientific run.
Pass the following driver arguments to that launcher:

```sh
$PY -B "$CONTROL/driver.py" --packet "$PACKET" --packet-zip "$PACKET_ZIP" --packet-sha256 EXACT_RETURNED_PACKET_SHA --cli-python "$CLI_PY" --cli-identity "$CLI_ID" --dispatch "$DISPATCH" --browser-adoption "$ADOPTION" --browser-adoption-sha256 EXACT_RETURNED_ADOPTION_SHA --output "$DRIVER_OUT" --accept79-file "$ACCEPT_PATH"
```

Do not pass `--stop-after-79` when same-allocation state8 remains possible.
The acceptance path must not exist initially.
The driver performs installation, fresh qualified CPU79, CPU host audit, TPU79, and device host audit before requesting root review.
Failed scientific gates trigger recovery and closure; they do not request acceptance.
The root request appears at `actual/root-review-request.json`.
Read its actual `records` path and the driver's actual `remote_base`; do not substitute old recovery paths.
The current records include both CPU79 and TPU79 evidence.

Run independent host readbacks while the driver waits:

```sh
$PY -B "$CONTROL/host_audit.py" cpu79 --packet "$PACKET" --records ACTUAL_REQUEST_RECORDS_PATH --remote-base ACTUAL_DRIVER_REMOTE_BASE --packet-sha256 EXACT_RETURNED_PACKET_SHA --output "$RUN/root-cpu79-audit.json"
$PY -B "$CONTROL/host_audit.py" device79 --packet "$PACKET" --records ACTUAL_REQUEST_RECORDS_PATH --remote-base ACTUAL_DRIVER_REMOTE_BASE --packet-sha256 EXACT_RETURNED_PACKET_SHA --output "$RUN/root-device79-audit.json"
```

Require qualified CPU identity, all 79 cases and 540 arrays, full record validity, exact numerics, and the unchanged graph gates.
The fresh graph must exclude S64 nodes and operands under the original witness rules.
Inspect physical archives, runtime/source identities, ownership, closure records, and actual arrays before accepting.
The live state boundary needs more than 780 seconds remaining from the original birth.
Withholding acceptance causes the existing deadline reserve to close the runtime.
Do not reset the birth or reuse a historical acceptance record.

After genuine root record review, create the acceptance in a staging file:

```sh
$PY -B "$CONTROL/host_audit.py" accept79 --packet "$PACKET" --records ACTUAL_REQUEST_RECORDS_PATH --remote-base ACTUAL_DRIVER_REMOTE_BASE --packet-sha256 EXACT_RETURNED_PACKET_SHA --output "$ACCEPT_STAGE" --reviewer-decision ACCEPT_AFTER_ROOT_RECORD_REVIEW
```

Publish atomically so the waiting driver cannot read a partially written JSON file:

```sh
$PY -B - "$ACCEPT_STAGE" "$ACCEPT_PATH" <<'PY'
import os,sys
from pathlib import Path
source,target=map(Path,sys.argv[1:])
assert source.is_file() and not source.is_symlink()
assert source.parent.resolve()==target.parent.resolve()
assert not target.exists() and not target.is_symlink()
os.replace(source,target)
PY
```

The accepted record binds these exact fields:

| Field | Required binding |
| --- | --- |
| kind | ROOT_NESTED_EXACT79_DEVICE_ACCEPTANCE |
| status | ACCEPTED_NESTED_EXACT79_DEVICE_RECORDS |
| record_validation / numerical_status | PASS / PASS |
| case_count | 79 |
| source_variant | nested-exact-v3-s32-tensor-lowering |
| source_admission_sha256 / integration_map_sha256 / science_map_sha256 | Current reviewed generation pins |
| inputs_sha256 / profile_sha256 / runtime_lock_sha256 | Current actual device receipt |
| packet_sha256 | Current packet archive |
| actual_receipt_sha256 | Current device79/receipt.json bytes |
| fresh_cpu_oracle_sha256 | Current cpu79/oracle-seal.json bytes |
| saved_state_status / m4_status | NOT_RUN / NOT_QUALIFIED |

The driver validates this dependency record, writes its timed review boundary, and uploads both records.
It then runs a separate fresh CPU state gate and saved-state TPU stages.
Both stages use the original clock and current oracle dependencies.
The driver's later recoveries contain the earlier records and replace its preceding recovery copies after successful readback.
Do not delete a recovery directory while its host audit is using it.

After state recovery, independently audit the final records path reported by the driver:

```sh
$PY -B "$CONTROL/host_audit.py" cpustate --packet "$PACKET" --records ACTUAL_FINAL_RECORDS_PATH --remote-base ACTUAL_DRIVER_REMOTE_BASE --packet-sha256 EXACT_RETURNED_PACKET_SHA --output "$RUN/root-state-cpu-audit.json"
$PY -B "$CONTROL/host_audit.py" devicestate --packet "$PACKET" --records ACTUAL_FINAL_RECORDS_PATH --remote-base ACTUAL_DRIVER_REMOTE_BASE --packet-sha256 EXACT_RETURNED_PACKET_SHA --output "$RUN/root-state-device-audit.json"
```

Require complete state receipts, record and numerical gates, source/runtime identities, and exact resource closure before claiming M4 qualification.
The driver retains 3600 seconds total, a 120-second closure reserve, and a 780-second state-start reserve.
No command here weakens a scientific criterion or supplies synthetic acceptance.
