# Preparation of time and memory measurements

Date: 2026-10-08.

The reviewer accepted the local measurement preparation.
No actual TPU performance samples exist for this protocol.
M7 remains unqualified, and the accepted milestone count remains three of eight.

## Measurement boundary

The protocol requires a previously accepted M6 case and the same identified TPU allocation for both paths.
Each path uses a new owned process with fixed source, runtime, inputs, precision, and numerical requirements.
The collector retains five warm-up iterations and three groups of 30 measurements.
The measured interval contains the complete operator call and synchronization of its actual output.
Input setup and the first call have separate records.

The first call includes tracing, compilation, submission, and execution.
It is not a measurement of compilation alone.
The [pinned client](https://github.com/pytorch/xla/blob/5fab7053df86c8d503b98d9e7202ca8b8d4978c7/torch_xla/csrc/runtime/pjrt_computation_client.cpp) supplies separate backend compilation metrics.
The [metric definition](https://github.com/pytorch/xla/blob/5fab7053df86c8d503b98d9e7202ca8b8d4978c7/torch_xla/csrc/runtime/metrics.h) records durations in nanoseconds.
The Python binding reports sample timestamps in seconds.
The collector retains these units and calculates duration separately.
An absent compilation metric remains `NOT_OBSERVED`, with an `UNKNOWN` duration.

## Memory scope

The collector retains allocator readings before and after the specified stages.
Those reads occur outside the measured operator intervals.
The [client allocator interface](https://github.com/pytorch/xla/blob/5fab7053df86c8d503b98d9e7202ca8b8d4978c7/torch_xla/csrc/runtime/pjrt_computation_client.cpp#L1023) uses `PjRtDevice.GetAllocatorStats()`.
Its reported high-water value does not establish peak memory for the measured interval.
The inspected API does not expose a reset for that interval.
The collector retains `CLIENT_REPORTED_HIGH_WATER_SCOPE_UNVERIFIED` for a valid reported peak.
It retains `UNKNOWN` when the API or peak is unavailable.
Compiler allocation plans remain separate from allocator measurements.

## Inspection and controls

The first review found four incorrect records that the original verifier accepted.
The repaired verifier rejects a changed sample phase, a negative derived peak, an incorrect compilation duration, and incomplete setup synchronization.
It also compares the complete time sequence, raw metric fields, memory stages, and calculated summaries.
The original failures remain in the local review records.

The reviewer independently repeated all 64 controls; all passed.
Nine additional independent records passed the expected acceptance or rejection checks.
These records include valid unavailable-memory cases.
All tests used a simulated clock, allocator, and device interface.
They do not establish device execution, speed, or memory reduction.

The collector, runner, protocol, and two control modules retain their reviewed bytes.
An isolated command copies these files before it executes the controls.
The [result record](../../experiments/2026-10-04-bitsandbytes-tpu/results/measurement-preparation.json) gives their hashes and review scope.
The [measurement directory](../../experiments/2026-10-04-bitsandbytes-tpu/measurement/README.md) gives the command.

Accept M6 correctness and the required compiler records before actual performance collection.
Then retain both complete measurement paths, allocation identity, numerical results, and resource closure.
