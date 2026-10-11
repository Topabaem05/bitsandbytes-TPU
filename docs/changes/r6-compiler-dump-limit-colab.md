# Compiler dump count failure

The corrected compiler request executed on Colab V5E1 on 2026-10-11.
The fixed Linux runtime passed source controls and produced five fresh CPU reference rows.
The independent host CPU gate passed before device execution.
The separate flag preflight initialized the TPU backend with all 14 requested tokens.

The `direct-fp32` and `functional-fp32` native cases passed.
The dump monitor then observed 1,025 entries against the original limit of 1,024.
It stopped the child and retained `DUMP_ENTRY_LIMIT`.
The two BF16 cases and the tail reference case did not execute.
The experiment remains incomplete.

The monitor recorded 15,559,534 bytes and a largest visited file of 530,344 bytes.
These values describe the directory prefix visited before the monitor returned.
They do not establish the complete directory size.
The original 128 MiB aggregate limit and 16 MiB file limit remain unchanged.
The failed capture retained three metadata files and omitted raw compiler dumps.
Actual compiler allocation analysis remains `NOT_RUN`.

The reviewer independently inspected all 139 scientific archive members, 11 CPU members, and three compiler metadata members.
Both completed native cases passed boundary, graph, numerical, and CPU conversion inspection.
Six incorrect native records and eight isolated archive, process, and monitor controls were rejected or accepted as specified.
The host group, 59 CLI groups, and 16 remote steps closed without cleanup errors.
Fresh server inspection found no session and zero active usage.
The temporary browser tab closed after the disconnected state was observed.
The original allocation deadline remained unchanged.

The [readback record](../../experiments/2026-10-04-bitsandbytes-tpu/results/compiler-default-dumps-previous-failure.json) retains the incomplete result and its hashes.
The [private archive](https://drive.google.com/file/d/10YuXuAw7Kc47BNEHf7hekrnU6OIbz1v3/view) contains the original records and review sources.
Its permission inspection found one owner and no sharing.
A separate Drive download matched its SHA-256 before local removal.
The [storage record](../../experiments/2026-10-04-bitsandbytes-tpu/results/resume-storage-20261011.json) records temporary file removal.

GPT-6 Astra Max inspected the failure and the pinned public compiler sources.
The proposed next generation removes the additional internal pipeline boundary request.
The [pipeline source](https://github.com/openxla/xla/blob/31eb6029e8973337ef6c99810c27e1d807791c11/xla/hlo/pass/hlo_pass_pipeline.cc) and [dump source](https://github.com/openxla/xla/blob/31eb6029e8973337ef6c99810c27e1d807791c11/xla/service/dump.cc) separate these additional dumps from default compiler dumps.
This is a proposed reduction in collection scope, not a confirmed file distribution diagnosis.
The five scientific cases and all original resource limits must remain unchanged.
A new source generation, exact request, and actual preflight remain required.
The public source revision does not identify the internal libtpu revision.

M6 remains unqualified.
Accepted milestones remain three of eight.
