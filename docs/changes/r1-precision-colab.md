# R1: Colab precision result

## Result

The reviewer accepted the controlled precision comparison on 2026-10-08.
Native XLA precision `high` and `highest` passed every fixed numerical gate in this diagnostic.
The `default` mode failed the FP32 forward and input-gradient gates on all three paths.
Its four module output records matched the corresponding October 4 records exactly.

Use `highest` for the next correctness experiments.
Set the precision once before graph construction in each new process.
This decision does not establish a performance or memory benefit.

The full API42 matrix, original CPU-to-XLA transfer, and restoration in a new process still need separate device tests.
Accepted milestones remain **1 of 8**.

## Fixed comparison

One Colab V6E1 allocation executed three separate processes.
Each process used the same source, runtime, inputs, CPU reference, and numerical gates.
The runtime used Linux, Python 3.12.14, PyTorch 2.9.0+cpu, PyTorch/XLA 2.9.0, and libtpu 0.0.21.
JAX and jaxlib remained at 0.7.1.
The CPU reference contained 42 cases from the pinned default Python implementation.
CUDA comparison remained `NOT_RUN`.

Each mode executed plain FP32 matrix multiplication and two FP32 module paths.
The module paths used a target-device constructor and `Params4bit.from_prequantized`.
Each mode also executed two BF16 forward controls.
BF16 gradients remained `NOT_RUN`.

The FP32 forward and gradient requirements remained `atol=1e-5` and `rtol=1e-4`.
The BF16 requirements remained `atol=0.02`, `rtol=0.02`, and `NRMSE<=0.02`.
Packed bytes required exact equality.
No tolerance changed after execution.

## Numerical records

The table gives the maximum absolute error against the sealed CPU reference.
All three FP32 paths gave identical output and gradient values within each mode.

| Native precision | Forward error | Input-gradient error | FP32 result |
| --- | --- | --- | --- |
| `default` | 0.0012758970260620117 | 0.0010573863983154297 | `FAIL` |
| `high` | 0.000003129243850708008 | 0.000001773238182067871 | `PASS` |
| `highest` | 0.00000011920928955078125 | 0.00000005960464477539063 | `PASS` |

Bias gradients matched the CPU reference exactly in every mode.
Packed bytes, scales, codebooks, and decoded weights matched exactly for every module path.
All six BF16 forward records matched the CPU reference exactly.

The comparison contains 15 records and 81 numerical gates.
Twelve records passed, and three records failed.
The six failed gates belong to the `default` FP32 forward and input-gradient calculations.
The complete numerical status remains `FAIL` to retain these negative results.

## Execution and inspection

Each process recorded one precision-setting call before graph construction and the matching precision readback.
The HLO records contained the expected precision on FP32 matrix operations that contributed to the output.
The execution records contained no positive `aten::` fallback counter.
The HLO records describe graphs before compiler optimization.
They do not identify a hardware instruction sequence.

The reviewer independently compared all 81 gates with retained arrays.
The reviewer also compared the four `default` module outputs with the October 4 records.
These comparisons support the precision explanation for the tested FP32 error.
They do not establish correct results for other shapes or the full API42 matrix.

The result archive contained 176 members and 199,795 bytes.
All archive members matched the recovered files and their recorded hashes.
The archive SHA-256 is `9a04a522d28b6be74caa8ba5479d4ef496eba6d0cbd973c230c00207eded365b`.
The Colab owner returned `PASS_PRECISION_RECORDS`.
The session stopped, the server was empty, and active usage was zero.
All owned process groups closed without a cleanup error.

Refer to [the machine-readable result](../../experiments/2026-10-04-bitsandbytes-tpu/results/precision-colab.json) for numerical records and 30 artifact bindings.
Refer to [the diagnostic preparation](r1-precision-preparation.md) for local controls and verifier limits.
The next device experiment will examine the explicit transfer correction and the full API42 matrix.
