# Actual BF16 operand results

The corrected native diagnostic passed all five cases on Colab V5E1.
Both FP32 cases, both BF16 cases, and the explicit tail reference case passed.
The fresh qualified Linux CPU oracle supplied 164,096 reference output values.
The fixed runtime and numerical criteria remained unchanged.

Both BF16 cases matched their independent CPU outputs exactly.
The largest absolute FP32 difference was approximately `2.38419e-6`.
All four native cases passed custom-call, operand, layout, graph, and synchronized execution validation.
The tail case used the explicit device reference route.
Independent replay validated all four Mosaic conversions.

The reviewer verified all 148 result archive members and all 11 CPU archive members.
The result archive contained 1,359,337 bytes.
The independent reader compared input arrays, output bytes, source seals, and qualified runtime records.
Six isolated reader controls and 13 altered output, operand, or execution controls behaved as required.

All 47 CLI groups, 14 remote steps, and the scientific child closed.
The local verifier, process owner, and independent reader also closed.
A fresh server query found no active session.
A fresh usage query reported zero active assignments and zero usage rate.
The reviewer observed disconnection and closed the temporary Colab tab.

The reviewer accepts these five bounded native cases as a dependency for subsequent experiments.
The [acceptance record](../../experiments/2026-10-04-bitsandbytes-tpu/results/native-bf16-acceptance.json) binds the exact generation and result hash.
The [selected result](../../experiments/2026-10-04-bitsandbytes-tpu/results/native-bf16-colab.json) retains numerical, archive, and closure evidence.
The earlier failed experiments remain unchanged.

M6 remains unqualified.
Compiler memory evidence and public API integration remain required.
This result does not establish backward integration, allocator peaks, or performance.

The reviewer inspected sentence length, document links, and technical terms.
The complete standard dictionary was unavailable.
This review does not establish independent ASD-STE100 conformity.
