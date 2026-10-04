# Task 3: State and gradient preparation

## Change

Add a state probe and isolated tests.
The probe uses the fixed Task 2 source, inputs, runtime admission, and tolerances.
It uses all eight linear cases without new input generation.
No backend, runtime, or Task 2 file changed.

This preparation has no qualified Linux CPU reference data or TPU results.
It does not complete M3.
CUDA comparison stays `NOT_RUN`.

## Supported state workflow

`Linear4bit.state_dict()` saves the packed weight, bias, scales, codebook, and packed quantization metadata.
Its class has no load hook for a packed NF4 checkpoint.
A new float module cannot use plain `load_state_dict()` for that full mapping.
The weight shape differs, and the quantization metadata keys need reconstruction.

Use the public `Params4bit.from_prequantized()` method to reconstruct the weight and `QuantState`.
Pass the new module to that method.
Then use `load_state_dict(..., strict=True)` for the weight and optional bias.
The probe does not create substitute classes.
It does not quantize the saved packed weight again.

The reconstruction passes a new metadata mapping to `QuantState.from_dict()`.
That method removes and adds mapping entries.
The separate mapping prevents changes to the checkpoint mapping.
The new weight state must also be the new module state.
A missing state or an inconsistent state alias prevents qualification.

Before save, the module state can be absent when `weight.quant_state` is valid.
The pinned forward method permits this condition.
The restored module must have the state alias that `from_prequantized(..., module=module)` creates.

## Fixed test scope

| Data type | Cases | Required outputs |
| --- | --- | --- |
| FP32 | Two-dimensional and three-dimensional input, with and without bias | Output, dX, and db when bias exists |
| BF16 | Two-dimensional and three-dimensional input, with and without bias | State and output |

The four FP32 cases use the fixed upstream gradient `((i % 5) - 2) / 8` in output element order.
The probe keeps base weights frozen.
It requires a missing base weight gradient and unchanged packed bytes after backward execution.
The bias gradient must have shape `[19]`, including the three-dimensional input cases.

BF16 gradients stay `NOT_RUN`.
The fixed Task 2 profile has no BF16 training tolerance.
This preparation does not create that requirement or claim that result.
Nested quantization, optimizer steps, and QLoRA stay outside this probe.

## CPU reference procedure

Use a new process in the qualified Linux runtime.
The required packages are PyTorch `2.9.0+cpu`, PyTorch/XLA `2.9.0`, and libtpu `0.0.21`.
The process uses the same source admission as Task 2.
The external installer must prove the wheel identities.

The CPU preparation uses the pinned default quantization function body from Task 2.
It reconstructs a CPU module through the public state workflow.
It runs the bitsandbytes forward and FP32 backward methods in training mode.
Training mode prevents the optional CPU inference packing path.
The pinned packing condition also requires an input without gradients.
The probe requires `packing_format_for_cpu=False` in each raw record.
A Linux CPU can select native dequantization operators that the Mac test did not exercise.
Missing native symbols, unsupported shapes, or different numerical results can still prevent the qualified CPU reference from passing.
Do not add a packing workaround or change source to hide that failure.

The process also computes a dense CPU reference with `torch.nn.functional.linear` and its autograd.
Both paths use the same restored weight, bias, input, and fixed upstream gradient.
The process keeps both sets of raw results.
It uses the unchanged Task 2 tolerances for the comparison.
A CPU reference failure stays `FAIL` and prevents TPU execution.
The seal stays available with all full raw results.

## Three-process experiment

The external owner must start separate CPU, TPU save, and TPU restore processes.
The following commands are templates.
They do not authorize allocation or execution.

```sh
python -B experiments/2026-10-04-bitsandbytes-tpu/probe_state.py prepare \
  --admission source-admission.json --admission-sha256 "$ADMISSION_SHA" \
  --output state-cpu-oracle

PJRT_DEVICE=TPU python -B experiments/2026-10-04-bitsandbytes-tpu/probe_state.py save \
  --admission source-admission.json --admission-sha256 "$ADMISSION_SHA" \
  --oracle state-cpu-oracle --oracle-sha256 "$STATE_ORACLE_SHA" \
  --output state-tpu-save

PJRT_DEVICE=TPU python -B experiments/2026-10-04-bitsandbytes-tpu/probe_state.py restore \
  --admission source-admission.json --admission-sha256 "$ADMISSION_SHA" \
  --oracle state-cpu-oracle --oracle-sha256 "$STATE_ORACLE_SHA" \
  --saved state-tpu-save --saved-sha256 "$SAVED_SEAL_SHA" \
  --output state-tpu-restore

python -B experiments/2026-10-04-bitsandbytes-tpu/probe_state.py verify \
  --admission-sha256 "$ADMISSION_SHA" \
  --oracle state-cpu-oracle --oracle-sha256 "$STATE_ORACLE_SHA" \
  --saved state-tpu-save --saved-sha256 "$SAVED_SEAL_SHA" \
  --restored state-tpu-restore
```

Get each seal hash before the next phase.
Use new output directories.
Start each process with `-B`.
This option prevents bytecode writes, not reads.
The separate source admission rejects existing bytecode files.
Do not import JAX or Torchax.

The owner must bind the probe source, source admission, installation proof, command, and process records.
The owner must impose deadlines, retrieve full or partial records, and close all resources.
The probe does not allocate a TPU or implement transport.

## Qualification rules

Each CPU case saves raw JSON.
Each TPU case saves raw JSON, HLO text, XLA metrics, and a tensor-only `state_dict` checkpoint.
The CPU phase has eight raw files.
Each TPU phase has 32 raw files.
Each phase also has its own seal.

The restore phase loads the checkpoint with `weights_only=True` and CPU placement.
It keeps a byte-identical copy of the input checkpoint.
The verifier compares its hash and size with the save seal.
It compares all restored state tensors and decoded metadata with the saved values.
It rejects missing keys, changed bias, incorrect shapes, incorrect metadata, and changed base weights.

The save and restore seals must have different PIDs and different process tokens.
The restore seal must identify the save process and save seal.
These fields alone do not prove external process creation.
The owner must supply the actual child records for review.

The probe requires a TPU hardware result, XLA placement, all four operator registrations, and successful synchronized execution.
It applies the Task 2 execution metric and CPU fallback criteria without exceptions.
It keeps unknown counters and failure logs.
It compares source before execution and after execution.

The final comparison reports CPU reference gates, TPU reference gates, and state restoration gates independently.
State values and repeated TPU outputs must match without a tolerance.
Numerical comparison with the CPU reference uses the fixed Task 2 tolerances.
Byte integrity can pass when a numerical test fails.

Return values are 0 for a pass, 2 for a numerical failure, and 1 for an unqualified result.
The save phase returns 0 for a full seal; it does not independently claim M3 completion.
A full M3 inspection still needs actual TPU results and external resource closure.

## Local validation

```sh
python3 -B -m unittest discover -s tests -p test_state_probe.py -v
```

All 21 isolated tests passed.
The actual CPU helper test stays disabled unless `BNB_STATE_CPU_SMOKE=1` is set.
The arrays, checkpoints, device records, and PIDs in these fixtures are synthetic.
The public restoration test uses a labeled interface stub and performs no tensor arithmetic.
These synthetic tests do not show actual CPU arithmetic, TPU execution, or state restoration in separate processes.

The controls include incorrect source, device, checkpoint, metadata, bias, process identity, gradients, and terminal status.
They also include a full case and the permitted absent module state before save.
A numerical mismatch stays `FAIL`.
Local test output stays in ignored `.work/task3-state/` files.

## Actual Mac helper smoke test

A separate local run used the existing Mac environment with PyTorch `2.14.1`.
It read the pinned bitsandbytes source without installation or source changes.
This test called helpers directly and did not bypass the qualified `prepare` runtime gate.
Its label is `UNQUALIFIED_MAC_CPU_HELPER_SMOKE`.

All eight cases passed public reconstruction, computation, `state_dict`, tensor-only save, load, and a second module reconstruction.
The reconstructed `QuantState` objects differed within the process.
Both modules kept the same state values and packed weight bytes.
Each CPU result matched its dense reference without a difference.
The four FP32 cases included dX, and both bias cases included db.
BF16 gradients did not execute.

| Result | Observed value |
| --- | --- |
| Cases | 8 of 8 passed |
| Largest CPU reference difference | 0 |
| State and output differences after reconstruction | 0 |
| Native library | Missing; bitsandbytes used its mock library |
| CPU packing flag | False for each case |
| Module mode | Training |
| Combined test suite | 22 tests passed in 8.402 seconds |
| Child duration | 9.224 seconds |
| Timeout | False |
| Child process group after exit | Absent |

The test reconstructed modules in one process.
It does not prove restoration in a new PID, the Linux native path, or TPU behavior.
Qualified Linux CPU reference data, actual TPU results, and actual separate-process restoration stay `NOT_RUN`.

The reviewer repeated the tests: 22 tests and eight subtests passed in 6.28 seconds.
The first reviewer command used an incorrect source path and stopped before computation.
The corrected command used the `bitsandbytes` package directory.

Use this template only for the explicit Mac helper smoke test:

```sh
BNB_STATE_CPU_SMOKE=1 BNB_SOURCE_ROOT="$UPSTREAM_SOURCE" \
  BNB_STATE_SMOKE_OUTPUT="$NEW_SMOKE_OUTPUT" \
  TORCHINDUCTOR_CACHE_DIR="$LOCAL_CACHE" TORCHINDUCTOR_COMPILE_THREADS=1 HF_HUB_OFFLINE=1 \
  "$MAC_TORCH_PYTHON" -B -m unittest discover -s tests -p test_state_probe.py -v
```

`UPSTREAM_SOURCE` must identify the `bitsandbytes` package directory, which contains `__init__.py`.

The worker used a 240-second local process limit.
No package installation, remote command, or TPU allocation occurred.
The ignored local output has 16 raw JSON files, eight checkpoints, and a helper report.
It also has the command log and process closure record.
The initial 21-test logs stay available.

## Sources

- [bitsandbytes module source](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/nn/modules.py)
- [bitsandbytes quantization state source](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/functional.py)
- [bitsandbytes CPU dispatch source](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/backends/cpu/ops.py)
- [bitsandbytes autograd source](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/autograd/_functions.py)

Source inspection date: 2026-10-04.
