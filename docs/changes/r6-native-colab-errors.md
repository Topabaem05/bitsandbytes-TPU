# Actual native diagnostic errors

The reviewer inspected the Colab V6E1 records on 2026-10-09, Korea time.
The device diagnostic completed with five errors and no passing case.
M6 remains unqualified.

The run used the accepted M2 source variant and all 23 fixed native files.
The runtime used Python 3.12.14, PyTorch 2.9.0+cpu, PyTorch/XLA 2.9.0, libtpu 0.0.21, and JAX/JAXlib 0.7.1.
The requested precision was `highest`.
No numerical tolerance changed.

| Cases | Observed error | Required action |
| --- | --- | --- |
| Direct FP32 and BF16 | Pallas TPU lowering rejected `dynamic_slice` | Use supported kernel operations and repeat lowering tests. |
| Functional FP32 and BF16 | The original wrapper rejected the input tensor wrappers | Correct the diagnostic call boundary. |
| FP32 partial tile reference | Execution metric validation rejected the sample container | Correct metric serialization before validation. |

The native kernel did not produce a qualified numerical result.
The reference case retained execution metrics, but its record failed validation.
The independent native verifier rejected the device records with `CHILD_TERMINAL`.
These errors do not establish a numerical tolerance failure.

## CPU and archive inspection

The complete CPU oracle passed source, runtime, launch, and closure verification before the device phase.
The reviewer examined five CPU input recipes and 164,096 finite reference values with their declared types and shapes.
This inspection did not independently calculate those reference values again.

The complete archive contains 118 members and 383,583 bytes.
Its SHA-256 is `ae0b8772fe1baee31a151c8122c5860229feb7d017c9321a30bc1869f8df5fe5`.
The reviewer verified every recovered member against its original bytes and hash.
Six isolated controls accepted valid records and rejected incorrect archive, array, and closure records.

## Connection and resource closure

The first CLI connection failed during its initial working-directory command.
Its six local process groups closed, and its provisional local registration was removed.
No scientific code ran during that connection attempt.

One bounded connection retry used the same browser runtime and its original allocation deadline.
The marker matched before upload or installation.
The retry did not extend the deadline or allocate another runtime.

All 46 CLI process groups, 14 remote steps, and the host process closed.
The complete allocation interval was 1,062.60 seconds, below the 3,600-second limit.
The server listed no active session, and active usage was zero.
The browser showed `Reconnect`; the temporary tab then closed.

The [result record](../../experiments/2026-10-04-bitsandbytes-tpu/results/native-colab-errors.json) contains source, archive, error, and inspection hashes.
The original arrays, logs, and provider records remain private under `.work`.
Accepted milestones remain three of eight.
