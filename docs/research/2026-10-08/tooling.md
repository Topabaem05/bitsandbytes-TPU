# Related kernel tools

Inspection date: 2026-10-08. Scope: M6 and M7 design references.
Use [the source manifest](tooling-sources.json) for source dates and revisions.

## Helion

The PyTorch article describes a Helion backend that generates Pallas code for TPUs.
It also states a dependency on TorchTPU.
Its reported attention results concern TPU v7 and do not measure bitsandbytes NF4 operations.
Source: [TOOL-01: Helion on TPU](https://pytorch.org/blog/helion-on-tpu-towards-hardware-heterogeneous-kernel-authoring/), published 2026-07-23.

The latest observable GitHub release is `v1.4.0`, published 2026-07-29.
The examined development revision is `fc0df0fca6a1f684feddf0de197b38807fc06433`, dated 2026-10-08.
These records do not qualify Helion with the project's PyTorch/XLA runtime.
Refer to [the XLA research](xla.md) for the separate TorchTPU availability question.

Design decision: Keep Helion as a future comparison candidate.
The initial implementation continues to use the existing PyTorch/XLA Pallas bridge.
No Helion dependency was installed or selected.

## MaxKernel and MaxCode

Accelerator Agents contains Pallas kernel tools and a separate model conversion tool.
The examined revision is `44ac97defdfe66a22231e4bbf4c83d216a68a5f4`, dated 2026-10-07.
Its root document describes MaxCode as a PyTorch-to-JAX conversion tool.
That conversion changes the product boundary of this bitsandbytes backend.
Source: [TOOL-02: repository overview](https://github.com/AI-Hypercomputer/accelerator-agents/blob/44ac97defdfe66a22231e4bbf4c83d216a68a5f4/README.md).

The current tree contains `maxkernel/adk` and `maxkernel/claude-code`.
The original paper links a different directory spelling.
Use the examined tree paths when retrieving code.
Sources: [ADK implementation](https://github.com/AI-Hypercomputer/accelerator-agents/tree/44ac97defdfe66a22231e4bbf4c83d216a68a5f4/maxkernel/adk), [Claude Code implementation](https://github.com/AI-Hypercomputer/accelerator-agents/tree/44ac97defdfe66a22231e4bbf4c83d216a68a5f4/maxkernel/claude-code), and [MaxKernel paper](https://arxiv.org/abs/2609.04523).

MaxKernel supplies procedures for kernel generation, compilation, numerical comparisons, tuning, and profiling.
These procedures can inform the experiment structure.
They do not establish this project's NF4 format, gradients, or memory reduction.

Design decision: Use documented engineering ideas after source inspection.
Keep the requested GPT-6.1 sol high workers and the project reviewer.
External agent services and optimization runs were not started.

## JAXBench

JAXBench supplies general JAX kernel workloads and an evaluation interface.
Its documented numerical comparison uses `atol=1e-2` and `rtol=1e-2`.
Those values differ from this project's fixed FP32 criteria.
Source: [TOOL-03: JAXBench interface](https://github.com/AI-Hypercomputer/accelerator-agents/blob/44ac97defdfe66a22231e4bbf4c83d216a68a5f4/JAXBench/README.md).

Design decision: Use its measurement structure only as a reference.
Keep the bitsandbytes inputs, tolerances, state requirements, and independent result reader.
A JAXBench pass cannot replace M2, M3, or M5 acceptance.

## Scope of the older design pack

The older pack also lists model conversion, application integration, and broad kernel search projects.
The current target remains the eight milestones in [the research plan](../../research-plan.md).
The source collection does not add a general model converter, inference server, or autonomous optimization service to that target.
Refer to [the Pallas research](pallas.md) for Qwix and Tokamax source comparisons.
