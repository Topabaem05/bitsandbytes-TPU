# Upstream source patch

`params4bit-xla-v1.patch` changes only `Params4bit._quantize` in the fixed bitsandbytes revision.
The base revision is `833649043474794b8fe7a4136e0c40faf077b2e0`.
[The manifest](params4bit-xla-v1.json) identifies the original archive, patch, runtime, and resulting files.
It records all 48 package files, including 47 Python files and `py.typed`.

The patch keeps the original assignment when tensor types are compatible.
For incompatible types, it constructs an upstream `Params4bit` and exchanges its contents with the original object.
It uses the public `torch.utils.swap_tensors` function.
It preserves custom attributes and assigns the resulting `QuantState` to the parameter and module.

The incompatible path rejects an existing gradient and either enabled module conversion flag.
It also rejects parameter subclasses that the examined path does not support.
The patch does not change global conversion flags.
Its recovery for synchronous errors depends on the fixed PyTorch 2.9.0 source.
Concurrent use, asynchronous interruption, and later PyTorch versions are outside this recovery claim.
The complete module conversion is not a transaction.

The plugin permits this exact source variant only with PyTorch `2.9.0+cpu`.
It rejects other source changes.
The original source remains a separate permitted variant.
The experiment builder applies the patch to a fresh copy of the fixed Git archive.
It keeps the original archive and records the modified archive separately.
No class or method is replaced during execution.

Local CPU and Meta controls do not establish TPU operation.
The transfer experiment must repeat the source controls in the fixed Linux runtime before the device test.
It must then execute the public transfers and all 42 fixed API cases.
Refer to [the research plan](../docs/research-plan.md).

The upstream source uses the MIT License.
Refer to [the preserved notice](../packages/bitsandbytes-tpu/THIRD_PARTY_NOTICES.md).
