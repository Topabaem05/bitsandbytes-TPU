# Exact word diagnostic: input boundary failure

Date: 2026-10-11.

The fixed Linux CPU runtime passed all 24 diagnostic probes.
The TPU returned four exact bitcast results.
These results cover scalar and vector inputs with raw and functional tensors.
The other 20 probes failed before their arithmetic bodies started.
Each failure reported: `Backend kernel inputs must be unwrapped tensors`.

The diagnostic synchronized its XLA inputs but retained the original wrapped input dictionary.
The serialization helper unwrapped only its local tensor argument.
The kernel correctly rejected the wrapped inputs.
The next diagnostic must normalize inputs at its caller boundary.
It must preserve the kernel rejection, arithmetic, inputs, runtime, and limits.

The reviewer repeated the retained record checks independently.
All 48 physical CPU and TPU row files matched their reports.
All eight graph objects passed compressed and raw hash checks.
The complete recovered ZIP passed member readback.
All 61 local process groups were absent.
Fresh CLI records showed no sessions and zero active assignments.
The browser showed a disconnected runtime.
The CLI usage display does not establish exact billing.

Execution and closure finished 184.29 seconds after the original browser intent.
The original diagnostic deadline was 600 seconds.
No CLI allocation attempt occurred.

The complete private archive has 455 members and 558,379 bytes.
An independent download passed its full archive hash, every member hash, and CRC checks.
Permission metadata showed one owner and no sharing.
The reviewer then deleted six completed temporary roots.
The temporary archive copy was also deleted after proof preservation.
No new output was added to `.work`.

Refer to the [selected evidence](../../experiments/2026-10-04-bitsandbytes-tpu/results/m4-word-diagnostic-v1-actual/root-review.json)
and [private archive](https://drive.google.com/file/d/1oy-C14kdB4re4U4L3xUKOTDLzxt6aYRz/view).
The 20 arithmetic outcomes remain unknown.
M4 remains unqualified; accepted milestones remain three of eight.

The English text follows the project clarity guide.
Full ASD-STE100 dictionary review remains unavailable.
