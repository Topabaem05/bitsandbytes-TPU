# Arithmetic metric correction

The reviewer accepted this correction on 2026-10-09, Korea time.
The correction prepares the arithmetic diagnostic for a fresh Colab experiment.
It does not complete M4.

The pinned Torch/XLA implementation returns tuples for the metric record, samples, and sample pairs.
The previous collector converted only the outer tuple.
The unchanged validator rejected the inner tuples before the builder wrote its first JSON record.
Refer to the [actual failure](r4-primitive-v5-record-failure.md).

The collector now converts the complete metric record to JSON before validation.
It preserves the counters, sample order, and numerical values.
Nonfinite values, invalid samples, negative counters, and positive fallback counters still fail.
The correction does not change kernels, inputs, tolerances, or the runtime.

The reviewer repeated 24 focused controls against the private candidate.
All 24 controls also passed after adoption into the project.
The controls reproduce the original failure and reject invalid observations.
They also reject the previous actual CPU oracle under the new probe hash.
The previous oracle and failed device records remain unchanged.

The new packet contains 56 members and 1,082,421 bytes.
Its only changed members are the collector source, scientific manifest, and cloud contract.
The reviewer verified every archive member against its file and manifest hash.
Two isolated incorrect packets failed admission.

Generate a fresh qualified CPU oracle before the next TPU phase.
Retain the fixed numerical criteria and inspect the complete recovered records.
Accepted milestones remain three of eight.

Refer to the [selected verification record](../../experiments/2026-10-04-bitsandbytes-tpu/results/primitive-metric-repair.json).
The primary sources are the pinned [metric API](https://github.com/pytorch/xla/blob/5fab7053df86c8d503b98d9e7202ca8b8d4978c7/torch_xla/debug/metrics.py#L30) and [tuple construction](https://github.com/pytorch/xla/blob/5fab7053df86c8d503b98d9e7202ca8b8d4978c7/torch_xla/csrc/init_python_bindings.cpp#L833).
