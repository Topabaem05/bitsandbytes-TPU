# Task 1: NF4 reference backend

Add an installable `bitsandbytes-tpu` plugin with the original backend entry point.
Keep the original bitsandbytes classes and operator schemas.
Register four XLA operator overloads after source and ABI validation.
Preserve existing CPU, CUDA, and XLA implementations.
Add Torch reference kernels for non-nested NF4 with uint8 storage and block size 64.
Use float32 and bfloat16 compute.
Reject unsupported types, layouts, scales, and nested GEMM arguments.

NF4 stores a nonuniform code-table index in each four-bit field.
XLA is the compiler backend used by PyTorch/XLA.
An ABI defines operator arguments and result types.
An overload is a named form of an operator.
GEMM is matrix multiplication followed by an optional bias addition.
A contiguous layout has no gaps between adjacent tensor elements.

Add CPU differential tests and a real wheel installation control.
Include the upstream MIT notice and project Apache license in the wheel.

CPU results and registration entries do not establish actual TPU execution.
The default nested NF4 module is outside this initial stage.
No fused-kernel, memory, performance, or training result is claimed.
