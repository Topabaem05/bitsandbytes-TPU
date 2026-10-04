# Task 2: Functionalization patch Colab test

## Result

One new V6E1 session tested the reviewed functionalization patch on October 4, 2026.
The result is `BLOCKED`.
The patch did not qualify the TPU NF4 backend.
Milestone M2 is not complete.

The fixed runtime again passed its independent TPU probe.
It returned loss `5.0` and gradient `[2.0, 4.0]`.
The CPU reference process completed all 42 cases.
The fixed verifier passed the saved CPU seal, source admission, and records.
These records are not CUDA golden results.

The TPU NF4 process completed zero cases and exited with code `1`.
The trace enters the functionalization wrapper at `bitsandbytes_tpu/functionalization.py:20`.
It then identifies code-table tensor creation at `bitsandbytes_tpu/reference.py:30`.
PyTorch raised an internal assertion at `FunctionalizeInterpreter.cpp:10`.
This is a different assertion location from the first attempt.
No TPU numerical comparison or CPU fallback gate completed.

The remote receipt now records `tpu_attempted: true` and `tpu_status: ATTEMPTED_BLOCKED`.
The failed child receipt says `FAILED`.
No post-execution source seal was recorded after the failure.
Both raw runs stay unchanged.

## Retrieval and cleanup

The full result ZIP has 120 members and 82,980 bytes.
Its SHA256 is `93c23e0bcf642debd779f8d22aef6f9cf7b94bd89e9fac0aa97fe0b9cef25fb6`.
An independent readback passed each member hash, size, recovered file, archive part, and common receipt field.

The lifecycle took 120.366308 seconds.
The owned session stopped.
The server was empty, and active usage was zero.
All 36 owned host process groups were absent during readback.
The 12 remote cleanup records reported absent groups and restored signal handlers and timers.
No cleanup error was recorded.
There was no retry or second allocation within this admission.

The [JSON report](../../experiments/2026-10-04-bitsandbytes-tpu/results/functionalized-colab.json) binds the packet, sources, runtime, seals, archive, owner, and process records by hash.
Raw records stay in the local cache outside public Git.
This result makes no performance, Pallas, QLoRA, or optimizer claim.
