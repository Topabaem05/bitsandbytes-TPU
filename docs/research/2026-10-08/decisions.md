# Design decisions and required tests

Inspection date: 2026-10-08.
This table records decisions from the source inspection.
It does not claim implementation or device acceptance.
Source identifiers refer to the five registers in [the collection index](README.md).

## Design coverage

| Design | Milestones | Primary source groups | Decision | Required result before acceptance |
| --- | --- | --- | --- | --- |
| Backend discovery and schemas | M1, M2 | BNB-03/04/05; XLA-03/04 | Keep entry points, public classes, and mutable output schemas. | Actual XLA tests of output mutation, alias behavior, and dispatch restoration. |
| NF4 values and packed bytes | M2 | BNB-07/08/09; XLA-11 | Keep the upstream format and name the exact reference implementation. | Fixed packed-byte, scale, zero-block, boundary, and partial-block comparisons. |
| Runtime and FP32 precision | M2 | XLA-01/02/08/09/10/11 | Keep the fixed runtime. Compare native XLA precision settings separately. | Recorded precision, HLO, unchanged numerical gates, and CPU fallback counters. |
| Parameter transfer and state | M2, M3 | BNB-06/07/10; XLA-05/06/07 | Examine a source correction separately from alternative construction paths. | Public CPU-to-XLA transfer, identity and state retention, gradients, and restoration in a new process. |
| Nested quantization | M4 | BNB-05/07/08/09/11/12 | Add the second scale state and its blockwise operators after reference inspection. | Offset, scale codes, second state, packed serialization, and fixed numerical comparisons. |
| Small QLoRA experiment | M5 | BNB-06/07/10/11 | Keep the specified twenty-step test and frozen base. | Adapter gradients, unchanged base bytes, and step-ten restoration in a new process. |
| Pallas integration and layout | M6 | PAL-01/02/03/04/05/06/07 | Keep one bound custom call. Compile the leading-group layout candidate. | Fixed-runtime compilation, actual selected call, full numerical comparisons, and explicit handling of unsupported shapes. |
| Memory and performance | M6, M7 | PAL-06/08/09/10; CLOUD-08 | Include preparation and backward costs. Separate compiler estimates from device samples. | Bounded forward decode storage and synchronized measurements on the same TPU. |
| Cloud reproduction | M8 | CLOUD-01/02/03/04/05/06/07/08 | Use fixed payloads, independent result identity, and observed closure. | Matching Colab and Kaggle numerical records with identified versions and closed resources. |
| Documents and future tools | All | CLOUD-09; PAL-11/12; XLA-12; TOOL-01/02/03 | Keep STE Issue 9. Treat related tools as design references. | Reviewed documents; separate compatibility evidence before any new dependency. |

## Next work after this inspection

| Order | Task | Worker output | Reviewer acceptance condition |
| --- | --- | --- | --- |
| R1 | Isolate the FP32 precision hypothesis. | A fixed diagnostic with fresh processes for `default`, `high`, and `highest`. | Inputs and tolerances remain fixed; plain matmul, forward, and input-gradient records explain the comparison. |
| R2 | Resolve public parameter transfer. | A minimal, explicit source correction with compatible and rejected-conversion controls. | The original API and state survive success and failure; actual CPU-to-XLA transfer passes. |
| R3 | Complete M2 and M3. | Full API42 and saved-state records, including a new process. | All required cases pass; no diagnostic subset substitutes for the full matrix. |
| R4 | Complete M4. | Nested operator implementation and reference records from the identified upstream path. | Default statistics, offset, second state, and saved state pass. |
| R5 | Complete M5. | Twenty-step adapter run and step-ten checkpoint restoration. | Frozen base and resumed results meet the fixed criteria. |
| R6 | Qualify M6. | Pallas bridge, layout, compiler, and public API records. | Actual custom-call execution and bounded forward decode storage are established. |
| R7 | Measure M7. | Raw synchronized time samples and documented memory samples. | Both paths use the same TPU, data, precision, and stated measurement boundaries. |
| R8 | Complete M8. | Same-payload Kaggle reproduction with version and closure records. | Independent result identity, complete records, and resource closure pass. |

R1 and R2 require separate changes and separate records.
A constructor pass cannot close the CPU transfer requirement.
A precision setting change cannot close state restoration or nested quantization.
M6 source design can advance independently, but its accepted performance comparison follows M5.

Use GPT-6.1 sol with high reasoning for worker tasks.
The reviewer examines source changes and actual test records before acceptance.
Commit and push each accepted logical change with its change log entry.

## Open questions

The BF16 operand-rounding witness supports the FP32 precision hypothesis.
It does not prove the effective device precision in the earlier run.
Only a controlled device comparison can resolve that question.

The tensor swap proposal uses behavior from a fixed PyTorch implementation.
It still needs an inspection of failure recovery and actual TPU behavior.
Do not present it as a portable public-API guarantee.

The grouped Pallas layout satisfies one examined shape rule.
Gather lowering, narrow data types, scale access, and precision still require compilation.
An accepted forward kernel does not establish reduced training memory.

The Kaggle CLI version issue needs an independent result reader or an explicitly examined fixed client.
Kaggle authentication, actual accelerator access, and a future TorchTPU runtime remain unverified in this research.
