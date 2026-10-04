# Technical terms

These terms have the specific meanings below in project documents.
They are technical terms for computer processes and mathematical operations.
They are not additions to the official ASD-STE100 dictionary.

## Technical nouns

| Term | Meaning |
| --- | --- |
| ABI | The argument and result contract of a binary or operator interface. |
| API | The public interface that application code uses. |
| backend | The software that executes operations for a selected device. |
| BF16 | The bfloat16 numerical data type. |
| contiguous layout | A tensor layout without gaps between consecutive stored elements. |
| CPU | A central processing unit. |
| CUDA | The NVIDIA interface for GPU computation. |
| dispatch key | The PyTorch identifier that selects an operator implementation. |
| entry point | Package metadata that identifies a function for automatic discovery. |
| FP16 | The IEEE binary16 numerical data type. |
| FP32 | The IEEE binary32 numerical data type. |
| GEMM | Matrix multiplication with an optional bias addition. |
| gradient | A derivative that the training algorithm uses. |
| hash | A digest that identifies file content. |
| JAX | The JAX numerical software package. |
| kernel | A function that executes a device computation. |
| libtpu | The TPU runtime library. |
| matrix multiplication | The mathematical operation that multiplies two matrices. |
| nested quantization | Quantization of the scales from a first quantization operation. |
| NF4 | The bitsandbytes four-bit NormalFloat codebook and its format. |
| nibble | Four bits within one byte. |
| overload | A named form of an operator with a specific schema. |
| operator schema | The declared arguments and results of a PyTorch operator. |
| Pallas | The JAX interface for device kernel code. |
| QLoRA | Adapter training with a quantized, frozen base model. |
| quantization | Conversion from numerical values to codes and scales. |
| reference implementation | The implementation that defines the expected result for a comparison. |
| runtime | The installed software and execution environment for a program. |
| SGD | Stochastic gradient descent. |
| SHA-256 | The hash algorithm that produces a 256-bit digest. |
| state_dict | The original PyTorch mapping of model state. |
| tensor | A numerical array with a shape and data type. |
| tile | A limited part of a matrix that a kernel processes. |
| TPU | A Google Tensor Processing Unit. |
| uint8 | An unsigned eight-bit integer data type. |
| wheel | A Python package distribution in wheel format. |
| XLA | The compiler system that PyTorch/XLA uses. |

## Technical verbs

| Verb | Meaning in this project |
| --- | --- |
| compile | Convert source operations into an executable program. |
| dequantize | Convert quantized codes and scales into numerical values. |
| dispatch | Select and call the implementation for a PyTorch operator. |
| import | Load a Python module with the Python import system. |
| initialize | Create the initial state of a software component. |
| quantize | Convert numerical values into codes and scales. |
| register | Add an implementation to the PyTorch operator registry. |

Use these verbs only for the specified computer or mathematical process.
