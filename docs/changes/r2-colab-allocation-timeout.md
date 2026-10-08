# R2: Colab allocation timeout

The first R2 allocation request ended with a provider response timeout on 2026-10-08.
The official CLI reported `requests.ReadTimeout` during the assignment request after 120 seconds.
No runtime session was observed.
Package installation, source controls, CPU reference calculation, and TPU science did not execute.

The owner retained the failure and attempted resource closure.
The server list was empty, and active usage was zero.
A later independent read-only query confirmed both conditions before another attempt.
All owned local processes closed without a cleanup error.
The owner lifecycle took 123.45 seconds.

This is an allocation failure, not a numerical test result.
R2, API42, and M3 remain unqualified.
Accepted milestones remain **1 of 8**.

The reviewer then permitted one bounded repeat with the same reviewed payload.
That repeat also ended during allocation; refer to [the repeat record](r2-colab-allocation-repeat.md).
Refer to [the machine-readable record](../../experiments/2026-10-04-bitsandbytes-tpu/results/transfer-allocation-timeout.json).
