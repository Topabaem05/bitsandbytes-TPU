# Task 2: First NF4 API probe

## Change

Add a probe for the original bitsandbytes public API.
Add fixed inputs, a test profile, and isolated artifact tests.
This preparation does not include actual CPU reference data or TPU results.
CUDA comparison remains `NOT_RUN`.

The profile contains 42 cases.
They use FP32 and BF16, NF4, block size 64, and `uint8` storage.
The cases include all 16 codes, zero blocks, partial blocks, bias, and two-dimensional and three-dimensional inputs.
The lengths are 1, 2, 63, 64, 65, 127, 128, and 129.

The probe uses the original `Linear4bit`, `Params4bit`, `QuantState`, and functional API.
It does not use Torchax or import JAX.
For BF16, it sets the CPU module dtype before weight assignment and quantization.
It preserves the original dequantization transpose when the packed tensor has one row.
Thus, the length-two case returns a tensor with shape `[2, 1]`.

## Source and runtime admission

Use bitsandbytes commit `833649043474794b8fe7a4136e0c40faf077b2e0`.
Use the reviewed backend source and the runtime wheel lock.
The required runtime is Linux x86-64, CPython 3.12, PyTorch `2.9.0+cpu`, PyTorch/XLA `2.9.0`, and libtpu `0.0.21`.
The installer must prove the wheel identities before the probe starts.
Installed version strings alone do not prove wheel identity.

Create the admission file from the reviewed installed source.
Use this structure:

```json
{
  "format": "bnb-tpu.probe-source-admission.v1",
  "runtime_lock_sha256": "323371ff61c5fbcc4f79fd6a358cf2ba17cb72b907382a5ceaac07f91dc66ed6",
  "bitsandbytes": {
    "commit": "833649043474794b8fe7a4136e0c40faf077b2e0",
    "files": {"installed-relative-file.py": "SHA256"}
  },
  "bitsandbytes_tpu": {
    "files": {"installed-relative-file.py": "SHA256"}
  }
}
```

Replace the example maps with all Python source files in each installed package.
The probe rejects additional source, changed source, symbolic links, and bytecode files.
The profile also binds seven required upstream source files.
The probe compares source before execution and after execution.

Start each process with `-B`.
This option prevents bytecode writes; it does not prevent bytecode reads.
The separate source admission rejects existing bytecode files.

## CPU reference procedure

Use a new process in the qualified Linux runtime.
Do not set `PJRT_DEVICE=TPU` for this process.
The process executes the pinned default quantization and dequantization function bodies on CPU tensors.
It removes only the quantization registration decorator in an isolated namespace.
It uses the undecorated dequantization body and preserves the public transpose rule.
It uses CPU `torch.nn.functional.linear` for matrix output.
It does not use the optional CPU packed format or claim CUDA equivalence.

The process saves all raw arrays and a source-bound seal.
Obtain the seal hash before TPU execution.
Do not replace reference data after the TPU result.
Missing, changed, or incomplete reference data prevents TPU execution.

The following commands are templates for the external experiment owner.
Use new output directories for each command.

```sh
python -B experiments/2026-10-04-bitsandbytes-tpu/probe_backend.py prepare \
  --admission source-admission.json --admission-sha256 "$ADMISSION_SHA" \
  --output cpu-oracle

PJRT_DEVICE=TPU python -B experiments/2026-10-04-bitsandbytes-tpu/probe_backend.py execute \
  --admission source-admission.json --admission-sha256 "$ADMISSION_SHA" \
  --oracle cpu-oracle --oracle-sha256 "$ORACLE_SHA" --output tpu-actual

python -B experiments/2026-10-04-bitsandbytes-tpu/probe_backend.py verify \
  --admission-sha256 "$ADMISSION_SHA" \
  --oracle cpu-oracle --oracle-sha256 "$ORACLE_SHA" --actual tpu-actual
```

The owner must bind the command, admission hash, runtime installation proof, and probe source hash.
The owner must impose the process deadline and close all resources.
This worker does not allocate a TPU or control transport.

## Retained evidence and result rules

Each CPU case produces one raw JSON file.
Each TPU case produces raw JSON, HLO text, and an XLA metrics report.
The seal binds every file by size and SHA-256.
The verifier requires all 42 cases and all required outputs.
It rejects an incomplete terminal record, incorrect identity, nonfinite output, CPU placement, or missing XLA registration.

The probe synchronizes execution before it reads counters and execution metrics.
It requires a positive `ExecuteTime` or `ExecuteReplicatedTime` sample.
It rejects every positive `aten::` counter without an exception list.
It retains unknown counters and execution failures.
Output serialization occurs after this measurement boundary.
An actual device review must examine these records with the external runtime proof.
Synthetic device records cannot establish physical TPU execution.

The verifier reports byte integrity separately from numerical results.
A complete numerical mismatch remains `FAIL`.
The process returns 0 for a numerical pass, 2 for a numerical failure, and 1 for an unqualified result.
Failures retain the traceback and available raw files.
The owner must retrieve partial evidence when execution stops before a seal exists.

The profile contains the original tolerances as complete values.
The previous profile hash is provenance only; execution does not need the previous repository.
Nested quantization, gradients, QLoRA, Pallas, saved state, and performance are outside this probe.

## Local validation

Run the isolated tests:

```sh
python3 -B -m unittest discover -s tests -p test_backend_probe.py -v
```

The suite has 25 tests.
The tests use synthetic arrays and device records.
The BF16 regression uses a labeled stub of constructor dtype and destination casting.
The length-two regression uses a labeled stub of the original transpose rule.
These tests establish artifact rejection and probe preparation only.
They do not establish CPU arithmetic, actual TPU execution, or a milestone pass.

Initial missing-implementation and regression failures remain in ignored `.work/task2-probe/` logs.
The final local suite passed all 25 tests.

## Primary sources

- [Pinned upstream functional API](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/functional.py)
- [Pinned upstream CPU math](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/backends/default/ops.py)
- [PyTorch/XLA 2.9 metrics API](https://github.com/pytorch/xla/blob/v2.9.0/torch_xla/debug/metrics.py)

Source review date: 2026-10-04.
