M5 reader v4 checks saved official CLI output and complete process lifecycle records. It fixes raw closure and unowned-child false positives from reader v3. This is an isolated source proposal. Root has not adopted it. M5 remains unqualified.

The complete frozen v3 source archive is retained as the closed preimage. The source delta changes only the two independent reader files and two generation declarations. The shared kernel helper, canonical driver, phase builders, math, training, checkpoint, reference, and plugin bytes remain unchanged. The generation is `nested-exact-m5-1024-scope-v4`.

The reader parses the installed official CLI 0.7.4 format, bound to three source hashes. Sessions must show the exact empty marker after the exact optional stale-local pruning line. Closure usage requires three unique ordered labels, zero active assignments, and the displayed rate `0.00/hr`. That rate is rounded display evidence. It is not proof of an unrounded zero billing rate. Authentication warnings, extra lines, active rows, duplicate labels, and contradictions fail validation.

Every physical CLI result must match the admitted argv, ownership, cleanup, command summary, and complete file inventory. Every lifecycle start must match an admitted child generation and its cleanup. Extra, missing, duplicate, orphaned, reversed, signalled, or unknown events fail validation. The current CLI and host wrappers each launch one child per owner directory. The generic ledger supports multiple admitted children and distinct reused PID generations. UTC fields must parse; event sequence determines ordering because wall time can move backward. The driver closure text and flags must match the physical raw output. The two fresh closure CLI directories receive the same checks. Full independent inspection includes whole recovered archive readback before numerical review.

Ten phases remain mandatory. Wheel-cache is the eleventh phase only when admitted host argv includes `--runtime-wheels`. Submitted scripts, physical CLI execs, and proof rows must agree exactly. The reader regenerates every submitted script from canonical phase contracts and the unique observed nonce. Completion and artifact download paths, hashes, expected fields, and independent export summary bytes remain bound.

Kernel entries disable bytecode before sealed imports. The exact shared helper SHA remains `e9c7a5b7b943cea08dfb8986205e47e008ceac37553a2f8576687527a58f9d5f`. Completion requires one fresh success marker as the last nonempty line. Independent completion and artifact downloads remain mandatory. CLI exit zero alone cannot establish kernel completion.

The protocol and inputs retain SHA `bd221d5dcd8a06a3080ff30345f80b8b99d8ca609fadcb813ea8da09964d007f` and `f16b79f90556d3d2eb01c4340af03be5c561e5b70591d6754ffdd990fb14002e`. Training retains 20 updates, save after update 10, and fresh-process resume for updates 11–20. The model retains two width-256 Linear4bit layers, rank-4 adapters, batch eight, FP32 MSE, and SGD at 0.01. The frozen base, optimizer, final model, continuation, and dense-reference comparisons remain unchanged. The 1024 extension remains separate from M4 acceptance. Sizes 514–1023 and 1025 remain rejected.

The historical v2 local smoke completed 20 updates and fresh resume. Its 200 continuation and 400 dense-reference comparisons passed. It used Darwin arm64, Python 3.12.14, and Torch 2.14.1. These are unqualified historical results. No local training was repeated because the closed delta verifies unchanged scientific bytes. All disposable training records were deleted after compact evidence was retained.

Blocking dependencies remain external. Root must supply genuine accepted M4 exact79 plus state8, independent readback, closure, and provenance. `source/admission.py:m4_dependency` remains a proposed strict consumer schema. Root must reconcile it with the final accepted M4 schema before adoption. Synthetic tests do not create acceptance. This reader revision also retains the inherited plugin with the known actual M4 functional-tensor error. A repaired plugin requires a separate reviewed M5 generation. No current M4 repair is silently propagated here.

Qualified execution requires the locked Linux x86_64 runtime and fresh qualified Linux CPU primary. The existing 35-wheel cache is read-only preparation input. Only root's reviewed cloud flow may install it. No provider, qualified Linux training, or TPU execution ran during this preparation.

Offline replay writes outside this frozen candidate. Each test deletes disposable fixtures, including on failure.

```sh
M5=/tmp/bnb-tpu-m5-reader-v4-20261011-pccc37y3
PY=/Users/topamini/code/Post2TPU/.venv/bin/python
OUT=$(mktemp -d /tmp/bnb-tpu-root-m5-reader-v4-review-20261011-XXXXXX)
"$PY" -B "$M5/tests/frozen_science.py" "$OUT/frozen-science-replay.json"
"$PY" -B "$M5/tests/scope_controls.py" "$OUT/scope-replay.json"
"$PY" -B "$M5/tests/cloud_controls.py" "$OUT/cloud-replay.json"
"$PY" -B "$M5/tests/replay_offline.py" "$OUT/all-offline"
```

The replay covers scope, source admission, frozen science, shared kernel controls, synthetic driver scenarios, and reader consistency controls. Full record fixtures stub only the numerical audit and explicitly remain unqualified. They exercise physical closure records, source checks, lifecycle inventories, and archive readback. They do not establish actual scientific results. No local training repeat is required for this reader-only delta.

Root preparation recipe follows. These commands are blocked until source review, a reconciled genuine M4 acceptance, and the separate scientific source repair are available. Root must supply every required external value. No old acceptance, CPU seal, packet, or allocation clock can substitute.

```sh
CLI=/Users/topamini/code/Post2TPU/.runtime/cloud-cli/bin/python
IDENTITY=/Users/topamini/code/bitsandbytes-TPU/.work/task2-cloud/cli-identity.json
WHEELS=/Users/topamini/code/bitsandbytes-TPU/.work/cache/bitsandbytes-tpu-wheelhouse
: "${M4_ACCEPTANCE:?Supply the genuine accepted M4 79+8 record path}"
: "${M4_SHA:?Supply its independently established SHA-256}"
: "${REVIEWED_BASE_COMMIT:?Supply the reviewed canonical base commit}"
: "${ROOT_REVIEW:?Supply a fresh absolute root review directory path}"
"$PY" -B "$M5/control/root_records.py" source-review --source "$M5" \
  --reviewed-base-commit "$REVIEWED_BASE_COMMIT" \
  --reviewer-decision ACCEPT_SOURCE_AFTER_ROOT_REVIEW \
  --output "$ROOT_REVIEW/source-review.json"
REVIEW_SHA=$("$PY" -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$ROOT_REVIEW/source-review.json")
"$PY" -B "$M5/control/build_packet.py" --source "$M5" \
  --source-review "$ROOT_REVIEW/source-review.json" --source-review-sha256 "$REVIEW_SHA" \
  --m4-acceptance "$M4_ACCEPTANCE" --m4-acceptance-sha256 "$M4_SHA" \
  --output "$ROOT_REVIEW/packet"
PACKET_SHA=$("$PY" -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$ROOT_REVIEW/packet.zip")
"$PY" -B "$M5/control/root_records.py" check --packet "$ROOT_REVIEW/packet" \
  --packet-zip "$ROOT_REVIEW/packet.zip" --packet-sha256 "$PACKET_SHA" \
  --cli-python "$CLI" --cli-identity "$IDENTITY" --runtime-wheels "$WHEELS"
```

Root owns one actual allocation. Record the original clock immediately before Connect. Its absolute deadline is Connect plus 3600 seconds. Installation, science, retrieval, and cleanup share that clock. Science stops 660 seconds before the deadline. The CPU phase is bounded to 300 seconds; device training to 1500. Each operation has a finite timeout. No retry resets the original clock.

Source and recovered envelopes retain the 100 MiB and 2000-file limits, with 96 MiB inside. CPU recovery retains 64 MiB. Evidence limits fail closed. Partial records are recovered after operation failure where possible. Closure stops only the exact adopted runtime. No allocation retry is authorized.

```sh
root_record() {
  "$PY" -B "$M5/control/root_records.py" "$1" \
    --packet "$ROOT_REVIEW/packet" --packet-zip "$ROOT_REVIEW/packet.zip" \
    --packet-sha256 "$PACKET_SHA" --review "$ROOT_REVIEW/connect" \
    --output "$ROOT_REVIEW/actual" --cli-python "$CLI" --cli-identity "$IDENTITY" \
    --runtime-wheels "$WHEELS" "${@:2}"
}
root_record preconnect
root_record request
```

Preconnect must observe an empty server and displayed zero usage. Root opens an empty notebook with one V5E1 assignment and performs Connect once. Run the generated browser marker cell. Root must observe its exact output and hardware within 180 seconds.

```sh
: "${MARKER_OUTPUT:?Supply the exact marker stdout observed by root in the notebook UI}"
root_record observe --marker-output "$MARKER_OUTPUT"
root_record dispatch --reviewer-decision AUTHORIZE_ONE_ALLOCATION_AFTER_ROOT_REVIEW
"$PY" -B "$M5/control/host_run.py" --config "$ROOT_REVIEW/connect/dispatch/host-config.json"
```

Dispatch binds that fresh observation to the original request, marker, packet, and endpoint. The driver makes no allocation attempts. Fresh qualified Linux CPU training must recover and pass independent review before device training starts. The complete CPU archive, source, wheels, ownership, and all comparisons must pass. The first scientific failure blocks further science and enters recovery and closure.

After exact-runtime closure, root closes its owned browser tab and observes absence. Then root collects fresh physical CLI closure records.

```sh
root_record fresh-closure
: "${OWNED_TAB_ID:?Supply the browser tab ID root closed and observed absent}"
root_record tab-closure --tab-id "$OWNED_TAB_ID" --tab-absent
"$PY" -B "$M5/control/inspect_actual.py" --packet "$ROOT_REVIEW/packet" \
  --packet-zip "$ROOT_REVIEW/packet.zip" --packet-sha256 "$PACKET_SHA" \
  --actual "$ROOT_REVIEW/actual" --review "$ROOT_REVIEW/connect" \
  --output "$ROOT_REVIEW/independent-readback.json"
```

Independent inspection repeats archive and scientific checks, and checks physical runtime closure. It always leaves the final milestone decision to root. Actual TPU numerics, graph-envelope fit, timing, execution selection, runtime memory, and Mosaic internals remain unknown. This preparation makes no M6 or performance claim.

Additional focused replay commands follow.

```sh
PYTHONDONTWRITEBYTECODE=1 "$PY" -B "$M5/tests/generation_delta.py" "$OUT/generation-delta.json"
PYTHONDONTWRITEBYTECODE=1 "$PY" -B "$M5/tests/kernel_controls.py" "$OUT/kernel-controls.json"
PYTHONDONTWRITEBYTECODE=1 "$PY" -B "$M5/tests/driver_controls.py" "$OUT/driver-controls.json"
```

The interactive controls reproduce the old bytecode failure and test corrected unpack, readmission, inherited child settings, cached definitions, and completion readback. Driver fixtures use synthetic transport and numerical artifacts. All acceptance fixtures are explicitly synthetic and deleted. Tests use the existing installed CLI identity without invoking a provider or installing dependencies.

Complete source archive readback uses an independently established SHA.

```sh
: "${SOURCE_TAR_SHA:?Supply the independently established source tar SHA-256}"
"$PY" -B "$M5/tests/bundle_readback.py" --source "$M5" \
  --archive "$M5/m5-reader-v4-preparation-source.tar.gz" --expected-sha256 "$SOURCE_TAR_SHA" \
  --output "$OUT/source-bundle-readback.json"
"$PY" -B "$M5/tests/bundle_controls.py" "$OUT/bundle-controls.json"
```

The source delta verifies every member of the complete historical v3 archive. The final source archive receives independent complete member, SHA, size, and source-byte readback. Replay outputs belong outside the frozen candidate. Bulk fixtures are deleted after compact summaries.

English prose sentences were checked against a 25-word limit. The full STE dictionary was unavailable. No vocabulary conformity claim is made.
