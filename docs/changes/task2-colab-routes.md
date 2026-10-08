# Task 2: Device route Colab result

## Result

One new V6E1 session ran the fixed device route diagnostic on October 4, 2026.
All 40 records passed record validation.
The numerical result is `FAIL`.
`PASS_DEVICE_ROUTE_RECORDS` does not mean that all operations passed.
The full API42 matrix and milestones M2 and M3 are not qualified.

The fixed runtime passed its independent TPU probe.
It returned loss `5.0` and gradient `[2.0, 4.0]`.
The CPU reference process completed all 42 cases.
The TPU diagnostic passed the fixed numerical gates for all 34 nonlinear cases.
These CPU reference records are not CUDA golden results.

The six module path records kept these results:

| Path | FP32 result | BF16 result |
| --- | --- | --- |
| CPU module transfer to XLA | Incompatible tensor type error during construction | Incompatible tensor type error during construction |
| Module made on the target device | Executed; forward and input-gradient gates failed | Executed; numerical gates passed |
| `from_prequantized` into a target module | Executed; forward and input-gradient gates failed | Executed; numerical gates passed |

The two CPU transfer errors matched the fixed expected error and stage.
The CPU-to-XLA transfer requirement is still unresolved.
The `from_prequantized` path used the sealed CPU checkpoint.
It does not show TPU quantization of the module weight.

Both FP32 paths had the same failed forward and input-gradient gates.
The forward maximum absolute error was `0.001275897`.
The input-gradient maximum absolute error was `0.001057386`.
The cause is not known.
The fixed tolerances were not changed.
The BF16 routes did not test gradients.

All four paths that returned results passed their state and output reconstruction checks within the same process.
These checks do not show restoration in a new process.
The source seals before and after execution matched the required admission.

## Retrieval and cleanup

The full result ZIP has 238 members and 303,829 bytes.
Its SHA256 is `fd31e2c138cf2caab021107e5f5a440a99b18849b09e5dc158b8101d0a0ca684`.
Independent readback found matching member hashes, byte counts, recovered files, archive parts, and common receipt fields.

The session lifecycle took 141.335318 seconds.
The host driver closed after 143.710407 seconds.
The owned session stopped.
The server was empty, and active usage was zero.
All 36 owned host process groups were absent during readback.
The 12 remote cleanup records reported absent groups and restored signal handlers and timers.
No cleanup error was recorded.
There was no retry or second allocation.
The earlier failed runs stay unchanged.

The [JSON report](../../experiments/2026-10-04-bitsandbytes-tpu/results/routes-colab.json) binds the packet, sources, runtime, seals, archive, verifier, owner, and cleanup records by hash.
Raw records stay in the local cache outside public Git.

## Reviewer readback

On 2026-10-08, the reviewer compared all 238 archive members with the recovered files.
All 31 report bindings matched existing files by size and SHA256.
The stored numerical comparisons agree with the reported passes, failures, and errors.
This readback did not execute a new experiment.
