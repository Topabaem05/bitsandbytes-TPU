# Native Mosaic version failure

The reviewer inspected the Colab V5E1 records on 2026-10-09, Korea time.
The corrected native diagnostic completed with five errors and no passing case.
M6 remains unqualified.

The first FP32 case reached the TPU compiler.
That compiler rejected Mosaic version 8 because its reader supported versions through 7.
The retained error contains the complete rejected module.
The previous dynamic-slice error did not occur in this case.

The next four cases failed during synchronization before their individual operations.
Each reported `Check failed: tensor_data:`.
The first failure possibly left invalid shared XLA state.
This explanation remains a hypothesis.
These records do not establish four independent kernel defects.
They also do not qualify the tensor wrapper or metric corrections.

The original verifier rejected the run with `CHILD_TERMINAL`.
No qualified native numerical comparison completed.
No numerical tolerance changed.

## Source and runtime

The run used the accepted M2 source variant and the corrected native source manifest.
The manifest SHA-256 is `c533a0b7ac754318ce4d6baa64e0e14d7961941983ffdf726d4b57177e865a14`.
The packet SHA-256 is `5a94d35b09288eb0724aeae5b40fb0f725bc997f4643d95006be330ac933bf32`.

The runtime used Python 3.12.14, PyTorch 2.9.0+cpu, PyTorch/XLA 2.9.0, libtpu 0.0.21, and JAX/JAXlib 0.7.1.
The requested precision was `highest`.
The runtime probe observed the TPU backend and `TPU:0`.
The CPU conversion controls from preparation did not test this compiler's module reader.

## CPU and archive inspection

The complete qualified CPU oracle passed source, runtime, launch, and closure verification before device execution.
The reviewer examined five input recipes and 164,096 finite reference values with their declared types and shapes.
This inspection did not independently calculate those reference values again.

The complete archive contains 118 members and 393,447 bytes.
Its SHA-256 is `153c0eeb4712926744d81efe923856d8f1dace38b8a113ca2b957558187cd437`.
The reviewer compared every recovered member with its original bytes and hash.
Six isolated audit controls accepted valid records and rejected incorrect archive, array, and closure records.

## Resource closure

All 47 CLI process groups, 14 remote steps, the native child, and the host process closed.
The allocation interval was 387.73 seconds, below the original 3,600-second limit.
The server listed no active session, and active usage was zero.
The browser showed a disconnected runtime; the temporary tab then closed.

## Next work

Inspect the pinned JAX serialization passes and their supported version conversion.
Keep the current runtime until a separate compatibility inspection accepts a change.
Require fresh CPU and TPU records after any native source change.
Continue the independent arithmetic diagnostic with its existing fixed criteria.

The [selected result](../../experiments/2026-10-04-bitsandbytes-tpu/results/native-v5-mosaic-failure.json) supplies the retained source, archive, error, and review hashes.
Full logs and arrays remain private under `.work`.
Accepted milestones remain three of eight.
