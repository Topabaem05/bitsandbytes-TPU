# Kaggle V5E8 request admission

The new generation requests `TpuV5E8` with a 3,600-second session limit.
The builder, owner, and submission worker require that exact identifier.
The exact saved version must report the same machine shape before status polling starts.
A missing value, V6E8 value, or UI display label fails this requirement.

The pinned [Kaggle CLI documentation](https://raw.githubusercontent.com/Kaggle/kaggle-cli/v2.2.4/docs/kernels.md) supplies the API identifier.
The reviewer separately observed `TPU v5e-8` in the account's draft menu on 2026-10-09.
That menu did not list V6E8, and the draft session was off.
No accelerator was selected during that inspection.
The menu observation does not prove capacity or the cause of the previous HTTP 400.
The original failed request and saved draft remain unchanged.

The scientific ZIP and wrapper match the previous generation for identical inputs.
The source maps, fixed runtime, 46 cases, and 196 numerical comparisons remain unchanged.
This correction changes the requested hardware and its readback requirement.
The single-submission policy, absolute deadline, and independent resource closure requirements still apply.
Use a new reference and nonce for any subsequent experiment.

## Validation

The reviewer repeated nine request controls, 39 existing integration controls, and 14 error-record controls.
All 62 passed.
The nine portable request controls also passed before and after adoption.
The tests use the original pinned CLI and SDK request construction with an isolated HTTP response.
They verify the exact request, source bytes, session limit, and second-submission rejection.
They make no service calls and read no credentials.

The portable entry is `experiments/2026-10-04-bitsandbytes-tpu/m8/tests_hardware.py`.
Supply explicit baseline source, scientific source, upstream archive, and inspected Kaggle interpreter paths.
Its argument help lists these required inputs.

Refer to the [selected result](../../experiments/2026-10-04-bitsandbytes-tpu/results/kaggle-v5-request.json).
A request readback does not independently identify the physical TPU generation.
Actual CPU, TPU, output, and closure records remain required.
M8 remains unqualified, and accepted milestones remain three of eight.
