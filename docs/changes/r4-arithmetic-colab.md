# Complete arithmetic observations on Colab

The reviewer inspected this V5E1 experiment on 2026-10-09, Korea time.
The corrected collector retained all 32 diagnostic rows.
Record validation passed, but numerical comparison failed.
M4 remains unqualified.

The experiment used the [parameter correction](r4-primitive-parameter-repair.md), the fixed runtime, and precision `highest`.
A fresh qualified Linux CPU oracle passed 32 rows and supplied 206 outputs.
Its 28 mean cases used the recorded AVX2 CPU capability.
The experiment did not reuse an earlier oracle.

The native tensor view remained unsupported because it used an ATen fallback.
The builder completed all 31 rows.
Twenty-two rows passed, and nine rows failed.
The complete diagnostic contains 265 byte comparisons: 229 passed and 36 failed.

| Observed case group | Failed output comparisons |
| --- | --- |
| Float arithmetic | `add_zero`, `clamp` |
| One-element mean, FP32 and BF16 inputs | `sum_div`, `sum_reciprocal`, `ordered_div8` |
| Length 16320, FP32 and BF16 inputs | `mean`, `sum_div`, `sum_reciprocal`, `ordered_div8` |
| Four zero-input cases | `mean`, `sum_div`, `sum_reciprocal`, `ordered_sum8`, `ordered_div8` |

Both clamp scalar parameters passed their exact source byte gates.
The lower parameter contained the FP32 representation of `1e-38`.
The upper parameter contained the finite FP32 maximum, with bit pattern `0x7f7fffff`.
These observations resolve the previous missing parameter record.
They do not resolve the numerical differences.

The reviewer independently inspected all 265 gate arrays and their graph bindings.
The recovered result archive contains 406 members and 408,135 bytes.
The separate CPU archive contains 18 members and 36,340 bytes.
All archive member names, byte counts, hashes, and recovered files passed inspection.
Four isolated incorrect reports failed admission.

All 46 CLI process groups, 14 remote steps, and two direct scientific children closed.
The local verifier and host process also closed.
Two fresh CLI observations confirmed an empty server list and zero active usage.
The reviewer closed the temporary browser tab and verified its absence.
The lifecycle took 359.08 seconds within the original deadline.

Use the retained output bits to design the arithmetic correction.
Keep the existing inputs and numerical criteria.
Repeat the complete 79-case nested test before the separate saved-state test.
Accepted milestones remain three of eight.

Refer to the [selected result](../../experiments/2026-10-04-bitsandbytes-tpu/results/arithmetic-colab.json).
