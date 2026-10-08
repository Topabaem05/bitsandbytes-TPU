# Kaggle M2 execution design

## Scope and source

This program repeats accepted M2 only. It has 46 cases and 196 comparisons. It cannot qualify state restoration, nested quantization, QLoRA, custom calls or performance measurements. A successful result supplies partial M8 evidence.

The scientific source, input arrays, profile and tolerances remain fixed. `source-admission.json` requires the accepted upstream correction and original plugin map. The plugin manifest SHA-256 is `18af6ca3152b9b9012dda369daf7a50ecc16023f6161ac70421ebf8f81b6f5e6`.

The wrapper contains one complete ZIP package. It examines the package hash and all member types, paths, sizes and hashes before code import. It does not obtain scientific source during execution. Runtime downloads use fixed public URLs and byte hashes. Each download runs in an owned process with a fixed limit.

The wrapper creates a separate interpreter and environment. It does not replace the Kaggle system interpreter.

| Runtime item | Required value |
| --- | --- |
| Operating system | Linux x86_64; glibc 2.31 or later |
| Python | 3.12.14 |
| torch | 2.9.0+cpu |
| torch-xla | 2.9.0 |
| libtpu | 0.0.21 |
| JAX and jaxlib | 0.7.1 |
| Matrix multiplication precision | `highest` |

The built source wheels and installed Python maps must match the admission. The output includes both built source wheels. Root recovery independently examines their Python file bytes.

## CPU gate and device order

Colab used host review before the device phase. The Kaggle script uses a local CPU gate before that phase. Root receives and reviews the CPU evidence after device execution. This is an explicit orchestration change.

The receipt records `BATCH_LOCAL_CPU_GATE_HOST_REVIEW_AFTER_DEVICE` and `host_pre_device_review=false`. The scientific criteria do not change.

The script uses this order:

1. Install the fixed runtime and source wheels.
2. Run the original 18 Linux CPU source controls.
3. Generate a fresh CPU oracle with the original default Python bodies.
4. Run the original oracle verifier in a new owned process.
5. Compare the selected FP32 outputs with the existing scalar reference.
6. Run the original TPU runtime probe in a separate process.
7. Run the original transfer probe.
8. Run the original scientific verifier in another owned process.
9. Export all retained records and raw child logs.

All CPU steps finish before the TPU runtime probe. A CPU failure prevents device execution. The transfer probe sets native precision to `highest` once before graph construction. Its numerical criteria remain fixed.

## Time limits

The wrapper and plan contain the same absolute owner deadline. No phase changes that deadline. The requested Kaggle session limit is 3600 seconds. This request does not prove server enforcement.

The work deadline occurs 660 seconds before the owner deadline. This reserves 600 seconds for retrieval and 60 seconds for cleanup. The adapter sets `allocation_epoch` to `owner_deadline - 3600`. This value is not an observed Kaggle allocation time.

| Phase | Maximum seconds |
| --- | ---: |
| Installation | 1200 |
| Science | 1800 |
| CPU source controls | 300 |
| CPU oracle preparation | 300 |
| Local CPU gate | 60 |
| TPU runtime probe | 120 |
| Transfer child | 1500 |
| Local final verification | 60 |

Each phase also uses the remaining original deadline. Insufficient time blocks execution. The program does not retry or request another allocation.

## Provider identity

The original one-shot CLI push requests a new private Python script. Its exact `machine_shape` is `TpuV5E8`. Its requested session limit is 3600 seconds. The exact-version source readback must return the planned machine shape. The program requires the expected ref, positive kernel ID and version 1.

The typed SDK requests `v1` for source, status and output records. Output download requests integer `versionNumber=1`. The source readback must match the exact submitted script hash and identity. The program rejects a different ref or version.

A latest-only status cannot prove the requested version. The status response does not independently identify its version. The program retains the exact request and source readback. It does not describe the status response as a separate version proof.

The program does not retry an ambiguous submission. A timed-out local request can leave an unknown server result. The program does not infer remote cancellation.

## Host process ownership

The external supervisor starts the internal owner in a new process group. It also starts each SDK worker as a direct child. The owner sends requests through an inherited local socket. The owner does not create SDK workers.

The provider registers the actual child process before it waits. The registry records its PID, process group, command hashes, token and owner identity. The supervisor retains the actual process object in memory. Cleanup uses this object and its owned group.

The supervisor does not load an old registry to select live processes. It does not signal a completed, closed group again. A worker registration during owner closure causes rejection and cleanup of that new child.

If the internal owner receives SIGKILL, the supervisor closes its registered worker groups. It sends TERM first. It sends KILL if an owned group remains. It reaps its direct children and requires observed group absence. It does not claim direct ownership of each descendant.

The supervisor uses the original absolute deadline. It stops normal work three seconds before that deadline. Cleanup uses the remaining time. Each SDK request retains its original call limit. The supervisor restores its catchable signal handlers after cleanup. It saves raw owner logs with file mode 0600.

Socket closure alone does not prove process exit. The supervisor waits for observed child exit within the original deadline.

The supervisor cannot guarantee cleanup after its own SIGKILL, host failure or power loss. Restart does not cause cleanup from an old registry. Root must examine that interruption separately. Local cleanup does not prove provider resource closure.

## Archive and numerical results

The two output files are `m8-evidence-manifest.json` and `m8-evidence.tar`. The manifest identifies the nonce, package, runtime, profile, inputs and plugin manifest. Every member has a byte count and SHA-256 hash. The plan contains the complete 298-member set.

The final TAR limit is 100 MiB, with at most 2000 members. The byte limit includes headers and padding. The script requires this limit before it writes the output manifest. Partial or malformed output cannot pass recovery.

Root recovery repeats the original oracle and scientific verifiers. It examines source maps, qualified CPU controls, runtime records, launch records and local process closure. It requires all 46 cases.

Numerical results outside the fixed tolerances remain `FAIL`. A retained case exception remains `ERROR`. Malformed records are invalid. A final verifier exception cannot produce `PASS`. No empty comparison set can qualify this test.

The original probes generate all actual outputs. The execution script contains no fixture outcomes. Offline test records contain `SYNTHETIC_FIXTURE`. Actual recovery rejects this label. Offline tests cannot qualify Linux, TPU or Kaggle execution.

## Root closure and acceptance limits

Root must observe closure for the exact returned kernel and version. A terminal API status is insufficient. This path has no established API session mapping. It has no inferred cancellation or deletion operation.

Closure recording requires a known identity from the original successful submission. It can retain a root closure observation after invalid scientific records. It does not change `INVALID_RECORDS`, numerical status or the original error. An ambiguous submission without an admitted identity remains unconfirmed.

Closure time must follow submission. If the program observed a terminal status, closure time must also follow that observation. The program rejects future times and repeated closure observations. It cannot authenticate a human statement.

Even after closure, a successful owner result is `PARTIAL_M2_READY_FOR_ROOT_REVIEW`. M8 remains `NOT_QUALIFIED`.

The hardware request does not prove the allocated topology. Root must observe actual hardware and runtime for the exact version. The runtime probe records its observed TPU backend and fixed test results.

Kaggle can restrict downloads, interpreter execution, wheel installation or TPU discovery. Queue delay consumes the original owner deadline. Added output files can fail the exact archive contract. Actual Kaggle execution must establish these conditions.
