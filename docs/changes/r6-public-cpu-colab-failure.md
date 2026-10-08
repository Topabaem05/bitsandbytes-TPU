# Public CPU comparison stopped before device execution

The new public API experiment connected to one Colab V5E1 runtime.
It used revision `5ada90eeef3f8a9ec3913a2a1a83d877be709f65` and the reviewed 122-member packet.
All 13 remote setup and CPU steps completed successfully.
A new Linux process produced the 15 CPU rows.
Its runtime was Python 3.12.14 with PyTorch 2.9.0+cpu on x86_64.

The independent host gate rejected `direct-float32` at the exact row comparison.
The retained error is `CPU_INDEPENDENT_REPLAY:direct-float32`.
The owner preserved this rejection and stopped before public device execution.
All 15 public TPU cases remain unexecuted.
The CPU oracle has no root acceptance.

The reviewer independently compared the first 32,768 output values again.
The input digest, output shape, and output data type matched.
Of these values, 30,476 differed from the Mac PyTorch 2.14.1 calculation.
The maximum absolute difference was `3.2186508178710938e-6`.
The NRMSE was `4.984152727812131e-7`.
This comparison passed the existing device numerical criteria.
It failed the separate exact CPU row requirement.
No tolerance changed, and the original failed gate remains failed.

The complete Linux output matched the earlier accepted native CPU reference exactly.
The two Linux runs therefore reproduced this reference output.
The first mismatch occurs in the host dense FP32 calculation.
The retained records do not identify the actual Linux BLAS implementation.
A separate reference revision must address the CPU platform difference before another public experiment.
It must preserve the original numerical criteria and require fresh CPU records.

The reviewer verified all 103 members of the complete experiment archive.
The reviewer also verified all 24 members of the separate CPU archive.
Five positive and negative failure-record controls passed.
All 13 recorded remote groups closed.
All 61 local groups from execution, independent reads, and closure were absent.
The owner stopped the exact runtime within 440.55 seconds of the allocation request.
Fresh CLI observations showed no active session, no active assignment, and zero usage rate.
The browser showed a disconnected runtime, and the reviewer closed the experiment tab.
The remaining balance was 1.06 compute units.

The original archives, stderr, and closure records remain private and unchanged.
Refer to [the selected failure result](../../experiments/2026-10-04-bitsandbytes-tpu/results/public-cpu-colab-failure.json).
M6 remains unqualified, and three of eight milestones remain accepted.

The reviewer inspected sentence length, document links, and technical terms.
The complete standard dictionary was unavailable.
This review does not establish independent ASD-STE100 conformity.
