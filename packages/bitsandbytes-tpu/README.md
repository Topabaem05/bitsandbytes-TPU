# bitsandbytes-tpu

This package adds an experimental XLA backend to the original bitsandbytes API.
XLA is the compiler backend that PyTorch/XLA uses for TPU execution.
NF4 is a four-bit index into a nonuniform floating-point code table.
A kernel is an operator implementation for a specified device.
The operator ABI defines the argument order and the result types.

## Current scope

The initial kernels execute non-nested NF4 with block size 64 and uint8 packed weights.
The compute types are float32 and bfloat16.
The input and scale values must be finite.
The packed weights must use the original contiguous column layout.
GEMM accepts input tensors of rank two or three.
GEMM is matrix multiplication followed by an optional bias addition.

The kernels use Torch operations on the input device.
They do not call CPU transfer, NumPy conversion, or scalar extraction methods.
The reference GEMM makes a dense weight tensor.
It does not establish fused execution or a memory benefit.

Nested quantization compresses the scales of the packed weights.
This initial GEMM rejects all nested arguments.
The default NF4 module uses nested quantization.
Thus, this initial stage does not complete the first usable NF4 milestone.
The initial stage rejects FP4, float16 compute, other block sizes, and float-backed packed storage.

## Registration

The installed distribution supplies the `bitsandbytes.backends` entry point `tpu`.
Original bitsandbytes calls this entry point during import.
The plugin registers four existing operator overloads for XLA:

- `quantize_4bit.default`
- `dequantize_4bit.default`
- `dequantize_4bit.out`
- `gemm_4bit.default`

The plugin does not replace the original classes or operator schemas.
It does not replace CPU or CUDA kernels.
Repeated calls to the loaded registration function have no additional effect.
An existing XLA kernel causes an error before registration starts.
A registration failure removes partial registrations.

The source test compares seven critical files with bitsandbytes revision
`833649043474794b8fe7a4136e0c40faf077b2e0`.
It also compares the four operator schemas.
It does not examine the entire installed distribution or native libraries.
The source test uses hashes because bitsandbytes sets its version after backend loading.
Package import does not import Torch, JAX, or PyTorch/XLA.
Backend registration uses Torch but does not create a device client or tensor.

## Tests and limits

Use an environment with the pinned bitsandbytes source, Torch, pytest, setuptools, and `uv`.
Run the tests from the package directory.

```sh
cd packages/bitsandbytes-tpu
python -B -m pytest
```

The tests compare CPU results with the original bitsandbytes operators.
They include invalid input, source changes, registration conflicts, and a real wheel installation into a temporary directory.
The temporary installation does not change the test environment.

CPU tests and XLA registry entries do not establish TPU execution.
Actual PyTorch/XLA 2.9.0 execution, nested state, module state, and training remain untested.
The `xla` device declaration is an integration signal, not a device qualification result.
XLA execution of `torch._assert_async` and `torch.bucketize` remains untested.
A later device run must show the actual execution path and fallback counters.
