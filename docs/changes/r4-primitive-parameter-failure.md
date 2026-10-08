# Arithmetic parameter observation failure

The reviewer inspected the second Colab V5E1 arithmetic run on 2026-10-09, Korea time.
The complete metric serialization passed for the two retained builder records.
The third builder case failed with `PARAMETER_VALUE_MATRIX`.
The diagnostic remains incomplete, and M4 remains unqualified.

The new qualified CPU gate passed 32 rows and 206 outputs.
The first two builder rows retained their synchronized outputs, input values, HLO, and metrics.
The reviewer repeated their byte comparisons and graph inspections.
All four comparisons passed: two output comparisons and two input comparisons.

The third case retained HLO and metrics but did not write its raw JSON record.
Its HLO contains more parameters than the declared input map permits.
The missing mapping and scalar values cannot be reconstructed as actual observations.
No complete arithmetic or subnormal behavior result exists.
The native tensor view again reported fallback and remained unsupported.

## Archive and resource inspection

The reviewer verified all 120 returned archive members against their recovered bytes.
The archive contains 135,978 bytes.
Its SHA-256 is `8ab553f514bd3f06d8f34a72625c5f7326273a1b8f9ca7abb48917e5a731eef8`.
The separate CPU archive contains 18 members and 36,346 bytes.

All 46 CLI process groups, 14 remote steps, and two scientific children closed.
The host process also closed.
The allocation interval was 391.73 seconds, below the original 3,600-second limit.
The server listed no active session, and active usage was zero.
The browser showed `Reconnect`; the reviewer then closed the temporary tab.

Inspect the pinned lowering implementation and its additional scalar parameters.
Require explicit roles and exact values for each permitted parameter.
Retain partial error records before future graph validation failures.
Keep the original failed records and numerical criteria unchanged.

Refer to the [selected result](../../experiments/2026-10-04-bitsandbytes-tpu/results/primitive-parameter-failure.json).
Accepted milestones remain three of eight.
