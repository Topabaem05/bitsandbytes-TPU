# Task 2: Native wrapper Colab test

## Result

One new V6E1 session tested the reviewed native wrapper on October 4, 2026.
The result is `BLOCKED`.
The patch did not pass all TPU NF4 tests.
Milestone M2 is not complete.

The fixed runtime passed its independent TPU probe.
It returned loss `5.0` and gradient `[2.0, 4.0]`.
The CPU reference process completed all 42 cases.
The fixed verifier passed the saved CPU seal, source admission, and records.
These records are not CUDA golden results.

The TPU process executed 17 FP32 quantize and decode cases before it exited with code `1`.
An independent comparison used the same fixed gates on those saved arrays.
Nine cases passed; eight all-zero quantization cases failed packed-code equality.
The TPU used code `0` for valid zero elements; the CPU reference used code `7`.
The odd padding code stayed `7`.
Decoded zeros passed, and scale differences were within the fixed tolerance.
These partial results do not give a full numerical pass.

The next `Linear4bit.to(device)` operation failed.
The trace identifies `Params4bit._quantize` at `bitsandbytes/nn/modules.py:390`.
Its `self.data = w_4bit` assignment could not change the CPU parameter to packed XLA data.
PyTorch reported incompatible tensor types in `variable.set_data(tensor)`.
The remote receipt records `tpu_status: ATTEMPTED_BLOCKED`.
The failed child receipt says `FAILED`.
The full 42-case matrix and post-execution source seal are incomplete.

## Retrieval and cleanup

The full result ZIP has 171 members and 162,383 bytes.
Its SHA256 is `218093c58e2769e1bbf3c7d038efbb667aa7bbf6d7f5279cfe20bbe1fa352ce3`.
The independent readback found matching member hashes, byte counts, recovered files, archive parts, and common receipt fields.
The partial numerical comparison matched the reviewer's saved comparison.

The lifecycle took 184.062855 seconds.
The owned session stopped.
The server was empty, and active usage was zero.
All 36 owned host process groups were absent during readback.
The 12 remote cleanup records reported absent groups and restored signal handlers and timers.
No cleanup error was recorded.
There was no retry or second allocation within this admission.
Both earlier raw runs stay unchanged.

The [JSON report](../../experiments/2026-10-04-bitsandbytes-tpu/results/native-colab.json) binds the packet, sources, runtime, seals, archive, comparisons, owner, and process records by hash.
Raw records stay in the local cache outside public Git.
This result makes no performance, Pallas, QLoRA, or optimizer claim.
