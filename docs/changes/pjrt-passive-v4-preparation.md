# Passive PJRT source preparation

A previous recorder allowed an old payload to qualify an overlapping execute call.
Its verifier also accepted an executable parent that differed from the input handle.
The revised source invalidates old payloads at optimized-program BEGIN.
It checks the input parent generation and snapshots completed evidence at Execute BEGIN.
It preserves valid historical links and unrelated concurrent calls.

The root inspected the source and verified the complete sealed preparation.
The root compiled the original counterexample and checked both a correct and an incorrect case.
The correct case passed; the overlapping replacement case remained UNKNOWN.
Independent review passed 46 provider scenarios, 12 lineage controls, and nine sanitizer scenarios.

The source was added under `compiler-analysis/pjrt-passive-v4/`.
No scientific tolerance, numerical operation, or compiler flag changed.
A fresh integration generation must fit both capture budgets within the original aggregate limit.
Linux compilation and actual TPU observations remain required.
M6 remains unqualified.

The full ASD-STE100 dictionary was unavailable; no certified conformity is claimed.
