# R4: Arithmetic allocation wait

The reviewer requested one Colab V6E1 runtime on 2026-10-09.
The browser still showed an allocation wait after 606.25 seconds.
The planned browser wait was 600 seconds.
The reviewer closed the temporary tab approximately 692.96 seconds after the request.

The browser marker did not execute.
No packet upload, package installation, CPU oracle, or TPU case executed.
The arithmetic result is `NOT_RUN`.
This attempt does not change the numerical status of the diagnostic.

The subsequent CLI inspection found no active server session.
It also showed zero active assignments and a zero usage rate.
All four CLI process groups from the initial and final inspections closed.
The original allocation time and deadline remain in the private journal.

The [retained record](../../experiments/2026-10-04-bitsandbytes-tpu/results/primitive-allocation-wait.json) gives the packet identity and inspection hashes.
The [arithmetic preparation](r4-primitive-preparation.md) remains available for a separately reviewed attempt.
M4 remains unqualified.
