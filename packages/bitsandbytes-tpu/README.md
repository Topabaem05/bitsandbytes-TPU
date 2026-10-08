# bitsandbytes-tpu

This package adds an experimental XLA backend to the bitsandbytes API.
XLA is the compiler backend that PyTorch/XLA uses for TPU execution.
NF4 is a four-bit index into a nonuniform floating-point code table.
A kernel is an operator implementation for a specified device.
The operator ABI defines the argument order and the result types.

## Current scope

The initial kernels execute non-nested NF4 with block size 64 and uint8 packed weights.
The compute types are float32 and bfloat16.
The caller must supply finite input, weight, and bias values.
The caller must supply finite nonnegative scales.
These are value preconditions for each device.
The kernels do not promise value validation or an error for data outside these preconditions.
The kernels still reject incorrect structural options, shapes, layouts, types, devices, and scale counts.
The packed weights must use the pinned contiguous column layout.
GEMM accepts input tensors of rank two or three.
GEMM is matrix multiplication followed by an optional bias addition.

The kernels use Torch operations on the input device.
They do not call CPU transfer, NumPy conversion, or scalar extraction methods.
The reference GEMM makes a dense weight tensor.
It does not show fused execution or a memory benefit.

Nested quantization compresses the scales of the packed weights.
This initial GEMM rejects all nested arguments.
The default NF4 module uses nested quantization.
Thus, this initial step does not complete the first usable NF4 milestone.
The initial step rejects FP4, float16 compute, other block sizes, and float-backed packed storage.

## Registration

The installed distribution supplies the `bitsandbytes.backends` entry point `tpu`.
Bitsandbytes calls this entry point during import.
The plugin registers four existing operator overloads for XLA:

- `quantize_4bit.default`
- `dequantize_4bit.default`
- `dequantize_4bit.out`
- `gemm_4bit.default`

The plugin does not replace the bitsandbytes classes or operator schemas.
It does not replace CPU or CUDA kernels.
Repeated calls to the loaded registration function have no more effect.
An existing XLA kernel causes an error before registration starts.
A registration failure removes the registrations that the failed attempt added.

Each XLA kernel uses native functional tensor wrappers for its full Python body.
The context keeps the included dispatch keys and removes only the local functionalization exclusion.
It does not add a functorch transform or a global functionalization mode.
Functionalization replaces intermediate views and mutations with tensor operations.
The context keeps input and output mutations that the caller can see.
The mutable dequantization overload also has a `Functionalize` implementation.
It calls the existing default dequantization operator, then copies the result into the supplied output.
This keeps device dispatch and the overload's None result.
An existing `Functionalize` implementation causes an error before registration starts.

The source test compares all 47 Python files and `py.typed` with bitsandbytes revision
`833649043474794b8fe7a4136e0c40faf077b2e0`.
It also permits the exact `params4bit-xla-v1` source patch under PyTorch `2.9.0+cpu`.
The patch changes only `Params4bit._quantize` during the upstream wheel build.
The source test rejects other changes and additional Python files.
It also compares the four operator schemas.
It does not examine the entire installed distribution or native libraries.
The source test uses hashes because bitsandbytes sets its version after backend loading.
Package import does not import Torch, JAX, or PyTorch/XLA.
Backend registration uses Torch but does not initialize a device client or allocate a tensor.

## Parameter transfer patch

The original CPU-to-XLA transfer failed during parameter data assignment.
The [examined patch](../../patches/README.md) permits a transfer between incompatible tensor types through `torch.utils.swap_tensors`.
It keeps the original `Params4bit` object, custom attributes, and module state alias.
Compatible conversions keep the original assignment path.

The incompatible path rejects existing gradients, nondefault module conversion flags, and unsupported parameter subclasses.
The error recovery depends on the examined PyTorch 2.9.0 implementation.
It does not promise a transaction for the complete module.
A later bias conversion error can occur after weight conversion.

The backend never applies the patch during import or replaces class methods during execution.
The experiment builder applies the exact patch before it builds the upstream wheel.
Actual Linux and TPU qualification of this patch remains required.

## Tests and limits

Use an environment with the pinned bitsandbytes source, Torch, pytest, setuptools, and `uv`.
Run the tests from the package directory.

```sh
cd packages/bitsandbytes-tpu
python -B -m pytest
```

The quantizer counts boundaries strictly less than each normalized input value.
Values equal to a midpoint enter the lower bucket, as in the pinned `right=False` operation.
The kernels do not use `torch.bucketize` or tensor-value assertions.
These changes avoid operators with no identified native XLA implementation in the pinned source.
Each device test must show zero CPU fallback.

The tests compare CPU results with the pinned bitsandbytes operators.
They include invalid input, source changes, registration conflicts, and a real wheel installation into a temporary directory.
The temporary installation does not change the test environment.

CPU tests and XLA registry entries do not show TPU execution.
The retained Colab records include 34 passing nonlinear cases and alternative module construction paths.
The precision comparison passed its fixed FP32 gates with native precision `high` and `highest`.
The `default` mode retained the earlier FP32 failures.
These records used pristine upstream source.
Refer to [the project status](../../README.md) for scope and result links.
Full API42, public parameter transfer, new-process restoration, nested state, and training still require accepted device records.
The `xla` device declaration is an integration signal, not a device qualification result.
