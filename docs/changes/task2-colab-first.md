# Task 2: First Colab test

## Result

One V6E1 session started on October 4, 2026.
The result is `BLOCKED`.
Milestone M2 is not complete.

The installed runtime used Python 3.12.14, PyTorch 2.9.0+cpu, PyTorch/XLA 2.9.0, and libtpu 0.0.21.
The independent TPU runtime probe passed.
It returned loss `5.0` and gradient `[2.0, 4.0]`.
This probe did not execute NF4 or a Pallas kernel.

The CPU reference process completed all 42 cases.
The fixed verifier passed the saved CPU seal, source admission, and all 42 records.
These records are not CUDA golden results.

The TPU NF4 process started, but it completed zero cases.
It exited with code `1` during `quantize_4bit`.
The trace identifies code-table slicing at `bitsandbytes_tpu/reference.py:31`.
PyTorch raised an internal functionalization assertion at `FunctionalTensorWrapper.cpp:850`.
This failure does not show a numerical mismatch.
No TPU numerical comparison or CPU fallback gate completed.

The remote receipt still says `tpu_status: NOT_RUN`.
The report keeps that value and records the actual attempt as `TPU_ATTEMPTED_BLOCKED`.
The failed child receipt says `FAILED`.
No post-execution source seal was recorded after the failure.

## Retrieval and cleanup

The full result ZIP has 120 members and 82,927 bytes.
Its SHA256 is `a22250ccedc84ba792ab6b5db08617cc9cc483a074274458b78c0b928ff8b070`.
An independent readback passed each member hash, size, recovered file, archive part, and common receipt field.

The lifecycle took 142.520069 seconds.
The owned session stopped.
The server was empty, and active usage was zero.
All 36 owned host process groups were absent during readback.
The 12 remote cleanup records reported absent groups and restored signal handlers and timers.
No cleanup error was recorded.
There was no retry or second allocation.

The [JSON report](../../experiments/2026-10-04-bitsandbytes-tpu/results/first-colab.json) contains the packet, source, runtime, seal, archive, owner, and process-record hashes.
Raw records stay in the local cache outside public Git.
This result makes no performance, QLoRA, or optimizer claim.
