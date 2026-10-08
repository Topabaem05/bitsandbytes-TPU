# Native gather results on Colab

A fresh Colab V5E1 experiment executed the admitted gather kernel on 2026-10-09, Korea time.
The fixed runtime, source admission, and qualified Linux CPU oracle passed inspection.
The CPU oracle contained five cases and 164,096 output values.

The direct FP32 case and functional FP32 case passed.
Each case produced 32,768 output values.
Both cases had a maximum absolute error of 0.000002384185791015625 against the CPU oracle.
The reviewer independently verified the saved outputs, operand values, graph binding, and synchronized execution records.
The independent JAX 0.7.1 CPU audit reproduced both retained Mosaic conversions.
This result supplies two successful native boundary cases under the fixed numerical criteria.

The direct BF16 case failed during TPU compilation with `Bad lhs type`.
The retained `tpu.matmul` operation had BF16 operands, FP32 accumulation, and FP32 contract precision.
The diagnostic preserved this error and marked the remaining two cases `NOT_RUN`.
The unchanged complete-run verifier rejected the run with `CHILD_TERMINAL`.
The two successful FP32 cases do not authorize a complete native dependency.

The reviewer verified all 130 result archive members and all 11 CPU archive members.
Six isolated archive, array, and closure controls passed.
Six additional incorrect FP32 records failed the independent verifier.
These records changed an output, operand binding, or execution statement in an isolated copy.
The original records remained unchanged.

All 47 CLI process groups, 14 remote steps, and the scientific child closed.
The local verifier, host owner, and independent audit also closed.
The complete lifecycle used 380.95 seconds of its original 3,600-second limit.
The reviewer closed the temporary browser tab.
Two fresh CLI observations showed no active session and zero active usage.

Inspect the BF16 operand types under the fixed `highest` precision requirement.
Preserve BF16 input and output semantics, decode rounding, and FP32 accumulation in the proposed correction.
Require fresh CPU and TPU records for the corrected source generation.

The [selected result](../../experiments/2026-10-04-bitsandbytes-tpu/results/native-gather-colab.json) binds the partial success, retained failure, independent audit, and closure records.
Raw records remain private under `.work`.
Public integration, compiler memory, and performance remain unqualified.
M6 remains unqualified, and accepted milestones remain three of eight.
