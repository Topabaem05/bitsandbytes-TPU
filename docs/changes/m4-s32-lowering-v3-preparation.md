# Explicit S32 lowering preparation

The reviewer adopted source generation `nested-exact-v3-s32-tensor-lowering` for a fresh experiment.
This is source acceptance only.
M4 remains unqualified, and accepted milestones remain three of eight.

The focused Colab diagnostic supports three source changes:

- Use explicit Tensor overloads at 44 integer shift sites.
- Use checked S32 Tensor constants at 79 comparison sites.
- Use dimension 0 for the flat partial tail reduction.

The scalar helper checks the reference tensor, its S32 data type, the Python integer type, and the signed 32-bit range.
It creates a scalar on the reference device.
The comparison changes preserve operand order.
Python metadata comparisons and Tensor-to-Tensor comparisons remain unchanged.
Forward and reverse AST inspection verified the complete mathematical delta.

Sixteen of the 19 scientific files retain exact parent bytes.
Only three identity files changed within that scientific set.
The trace format, parameter roles, no-S64 graph witness, runtime lock, 79 inputs, state8 criteria, and numerical tolerances remain unchanged.
Local CPU controls do not prove that the actual TPU graph excludes S64.

The inherited browser driver also required a transport correction.
It now acquires ownership immediately after matching the exact session, endpoint, hardware, and TPU variant.
Marker failure then stops that exact owned runtime.
Wrong assignments remain unowned.
Unique hardware and variant fields reject prefixes and duplicate fields while allowing valid additional status fields.

Independent review rejected an earlier ownership test fixture.
That fixture stopped before marker execution because two required source descriptors were absent.
Its reported PASS did not establish marker-path coverage.
The corrected fixture supplies those descriptors and requires the actual marker call and each expected branch result.
The old driver reproduces the missing exact-stop failure after marker execution.

The root repeated 11 word, 21 comparison, eight generation, and 15 corrected ownership controls.
Astra independently passed 21 mathematical controls and all 15 corrected ownership controls.
The worker also passed 23 functional boundary and seven packet preparation controls.
Local process groups were reaped and absent after these tests.
Disposable test output was deleted after compact proof preservation.

The final source archive contains 231 members and has 813,950 compressed bytes.
Its SHA-256 is `3c5970c07a5f0918156a15cb185cddf1ec5c0063ebb9f2f32e043118a2afb435`.
The closed replacement map has 173 files and 28 changes.
All archive members and canonical adopted files passed complete readback.
The owner-only [private source archive](https://drive.google.com/file/d/1khezhk_3BYGrsqbuj1kQLGwko-Gctt54/view) passed independent download and complete member verification.
The completed source and two adviser roots were then deleted.
The [root review](../../experiments/2026-10-04-bitsandbytes-tpu/results/m4-s32-lowering-v3-preparation/root-source-review.json)
and [adoption record](../../experiments/2026-10-04-bitsandbytes-tpu/results/m4-s32-lowering-v3-preparation/canonical-adoption-readback.json)
preserve the exact source identity.

Fresh qualified Linux CPU79 and TPU79 must pass before root acceptance of their current receipt and oracle.
The driver then requires a separate CPU state gate and actual saved state8.
The original 3600-second clock, 120-second closure reserve, and 780-second state-start reserve remain unchanged.
No historical acceptance substitutes for the new experiment.
No new output belongs in `.work`.

This document uses short technical English sentences.
A complete STE dictionary review was not available.
