# Arithmetic diagnostic record failure

The reviewer inspected the actual Colab V5E1 run on 2026-10-09, Korea time.
The qualified CPU gate passed all 32 rows and 206 outputs, including 28 mean cases.
The device record remained incomplete, so M4 remains unqualified.

The native tensor view used `aten::view_copy.dtype` fallback twice.
Its record reported `OBSERVED_UNSUPPORTED` with no accepted output values.
The protocol permits this observation and still requires the separate builder path.

The first builder case retained its HLO and execution metrics after synchronization.
It then failed with `EXECUTION_METRIC_SAMPLES` before it wrote its numerical JSON record.
The metric collector converted the outer tuple only.
The unchanged validator required JSON lists for the samples and their pairs.
The remaining builder cases did not execute.

This failure does not establish builder correctness, a numerical difference, or subnormal behavior.
The parent retained `STATE_CHILD_TERMINAL` and the original child errors.
The host skipped final scientific verification because the required records were incomplete.

## Source and archive inspection

The packet retained the fixed arithmetic inputs, runtime, criteria, and source admission.
Its 56 members passed root inspection before execution.
The differences from the earlier primitive packet were three previously accepted cloud files.
They contain the two hardware changes and the separate native-contract correction.

The complete returned archive has 113 members and 128,899 bytes.
Its SHA-256 is `f63f24e26480f759f8f4be49ff9466003ee45d78c2851c9bfef98b5aaa2f7e2a`.
The reviewer compared every member with its recovered bytes and repeated the independent CPU gate.
The CPU archive has 18 members and 36,340 bytes.

## Resource closure and next work

All 46 CLI process groups, 14 remote steps, and two direct scientific children closed.
The host process also closed.
The allocation interval was 416.89 seconds, below the original 3,600-second limit.
The server listed no active session, and active usage was zero.
The browser showed `Reconnect`; the temporary tab then closed.

Correct the complete metric serialization before the unchanged validator.
Retain invalid samples and numerical failures as failures.
Use a fresh source manifest and CPU oracle for the next device attempt.

Refer to the [selected result](../../experiments/2026-10-04-bitsandbytes-tpu/results/primitive-v5-record-failure.json).
Raw provider logs and arrays remain private under `.work`.
Accepted milestones remain three of eight.
