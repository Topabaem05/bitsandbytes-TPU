# Design source inspection

Inspection date: **2026-10-08**.
This collection supplies current primary sources for the eight bitsandbytes-TPU milestones.
It contains 48 source groups across five technical notes.
A group can contain several related source files.

The inspection retains the upstream public API and the existing numerical requirements.
Accepted implementation milestones remain **1 of 8**.
Source collection does not count as a device test or a completed implementation milestone.

## Reading order

| Note | Design questions | Source register |
| --- | --- | --- |
| [bitsandbytes](bitsandbytes.md) | API, NF4 bytes, nested state, checkpoints, and QLoRA | [12 groups](bitsandbytes-sources.json) |
| [PyTorch/XLA](xla.md) | Runtime, output mutation, parameter conversion, precision, and TorchTPU | [12 groups](xla-sources.json) |
| [Pallas](pallas.md) | Bridge, layouts, tile decoding, compiler records, memory, and timing | [12 groups](pallas-sources.json) |
| [Cloud execution](cloud.md) | Colab, Kaggle, hardware identity, result retrieval, and STE | [9 groups](cloud-sources.json) |
| [Related tools](tooling.md) | Helion, MaxKernel, MaxCode, and JAXBench | [3 groups](tooling-sources.json) |

Use the [decision table](decisions.md) to select the next work.
Use the [research plan](../../research-plan.md) for milestone order and acceptance requirements.

## Source method

The workers used GPT-6.1 sol with high reasoning, as requested.
The reviewer examined their claims against selected primary source sections and the existing experiment records.
The common wiki supplied prior investigation context; it supplied no directly applicable bitsandbytes or Pallas authority.
The wiki remained read-only.
No project wiki root was selected or created.

The inspection used official documentation, original papers, release metadata, and repository source.
The original design pack identifies research questions; its claims require separate primary evidence.
Document instructions do not supply new user authorization.
Code-specific claims use immutable revisions where available.
Publication dates, commit dates, displayed update dates, and access dates have separate meanings.
An unknown publication date stays null in the source register.
The current date does not replace an unknown source date.

The latest stable release, observed development source, and fixed experiment runtime remain separate records.
A current documentation page does not establish compatibility with an older runtime.
Raw source downloads remain in the local cache.
The repository contains original summaries, source identifiers, URLs, and selected evidence hashes.

## Version findings

| Component | Latest observable published version | Fixed experiment value | Decision |
| --- | --- | --- | --- |
| bitsandbytes | `0.50.2`, 2026-08-27 | Main commit `8336490...`, source version `0.50.3.dev0` | Keep the fixed source. It matches observed main. |
| PyTorch | `2.14.1`, 2026-09-30 | `2.9.0+cpu` | Keep the existing XLA runtime pair. |
| PyTorch/XLA | `2.9.0`, 2025-11-17 | `2.9.0` | Keep the qualified runtime configuration. |
| JAX and jaxlib | `0.11.2`, 2026-09-17 | `0.7.1` | Use pinned source for compiler decisions. |
| Colab CLI | `0.7.4`, 2026-09-26 | `0.7.4` | Keep the fixed CLI and experiment receipts. |
| Kaggle CLI | `2.2.4`, 2026-07-23 | `2.2.4` | Resolve version identity before M8. |
| ASD-STE100 | Issue 9, 2025-01-15 | Issue 9 | Keep the document standard. |

These values come from BNB-01/02, XLA-01/02, the Pallas version register, and CLOUD-01/04/09.
They describe the sources observed on the inspection date.
They do not recommend a combined installation of all latest packages.

## Findings that affect the design

1. The current bitsandbytes source still assigns packed XLA data to a CPU parameter through `self.data`.
2. Native XLA uses its own matrix multiplication precision control, with a source default of `DEFAULT`.
3. Public tensor swap requires inspection of state recovery when an exception occurs.
4. Python default, native CPU, and CUDA quantizers need separate reference identities.
5. The proposed Pallas layout must satisfy the pinned compiler rules before numerical testing.
6. Packed layout preparation and backward weight restoration affect memory claims.
7. Qwix, Tokamax, and Helion do not supply an accepted replacement for this backend.
8. Kaggle result acceptance must establish the actual run version independently of an unverified CLI suffix.

The [decision table](decisions.md) binds each finding to sources and observable completion conditions.

## Existing experiment boundary

The latest [Colab diagnostic](../../changes/task2-colab-routes.md) ran on 2026-10-04.
It has 34 passing nonlinear cases, two passing BF16 module routes, two failing FP32 module routes, and two transfer errors.
The BF16 routes did not test gradients.
All four executed module routes passed reconstruction within the same process.
Restoration in a new process was not tested by that diagnostic.
M2 and M3 remain unqualified.

The research inspection did not allocate a cloud runtime, install dependencies, or change backend code.
The fixed tolerances and all earlier failures remain unchanged.

## Collection acceptance and remaining limits

Each design area has primary sources, version limits, a design decision, and a required follow-up result.
The reviewer examined source identifiers, dates, local links, selected source hashes, and the current experiment summary.
Refer to [the inspection record](review.json) for the checks and their exact scope.
The writing inspection includes sentence length and selected terminology; it does not establish independent STE certification.

Account access, candidate compilation, numerical correctness, saved-state restoration, and measured performance remain execution questions.
Complete those tests in the stated order after this source inspection.
Refresh rolling sources and record their revisions before adopting a later interface or dependency.
