# Technical terms

These terms have the specific meanings below in project documents.
They are technical terms for computer processes and mathematical operations.
They are not additions to the official ASD-STE100 dictionary.

## Technical nouns

| Term | Meaning |
| --- | --- |
| ABI | The argument and result contract of a binary or operator interface. |
| accumulator | Storage for intermediate sums in matrix multiplication. |
| alias | A tensor reference that shares storage or mutation behavior with another reference. |
| allocator | Runtime software that assigns memory to computation data. |
| application | A computer program that uses the backend. |
| API | The public interface that application code uses. |
| API42 | The fixed project matrix of 42 public API test cases. |
| autograd | The PyTorch system that calculates gradients. |
| backend | The software that executes operations for a selected device. |
| BF16 | The bfloat16 numerical data type. |
| bias | The additive parameter of a linear layer. |
| bitcast | Reinterpretation of the same bits as another numerical data type. |
| checkpoint | Saved model and optimizer state used to restore execution. |
| centered scale | An NF4 scale after subtraction of the scalar offset, before nested quantization. |
| CLI | A command-line interface. |
| codebook | The numerical values that quantized codes identify. |
| collector | Software that records raw measurements and their execution context. |
| compiler | Software that converts source operations into executable operations. |
| contiguous layout | A tensor layout without gaps between consecutive stored elements. |
| CPU | A central processing unit. |
| CUDA | The NVIDIA interface for GPU computation. |
| dispatch key | The PyTorch identifier that selects an operator implementation. |
| entry point | Package metadata that identifies a function for automatic discovery. |
| endpoint | The service identifier for one remote runtime. |
| fallback | An alternative implementation selected when the requested implementation cannot execute. |
| FP16 | The IEEE binary16 numerical data type. |
| FP32 | The IEEE binary32 numerical data type. |
| GEMM | Matrix multiplication with an optional bias addition. |
| gate | A fixed acceptance condition for a test record. |
| gradient | A derivative that the training algorithm uses. |
| hash | A digest that identifies file content. |
| HBM | High-bandwidth memory attached to an accelerator. |
| HLO | The high-level operation representation used by XLA. |
| high-water value | The largest memory usage reported by an allocator within its own measurement scope. |
| JAX | The JAX numerical software package. |
| implementation | The code that performs a specified software operation. |
| fixture | Fixed data that a software test uses. |
| kernel | A function that executes a device computation. |
| libtpu | The TPU runtime library. |
| manifest | A file that records file names, byte counts, and hashes. |
| matrix multiplication | The mathematical operation that multiplies two matrices. |
| marker | A temporary file whose exact bytes identify a previously observed runtime. |
| metadata | Structured information about data, source, or execution. |
| metric | A named runtime measurement with a count, accumulated value, and samples. |
| Meta | The PyTorch device type that represents tensor structure without numerical data. |
| Mosaic | The compiler infrastructure used for the Pallas TPU kernel body. |
| NaN | A floating-point value that represents an undefined numerical result. |
| nested quantization | Quantization of the scales from a first quantization operation. |
| NF4 | The bitsandbytes four-bit NormalFloat codebook and its format. |
| nibble | Four bits within one byte. |
| overload | A named form of an operator with a specific schema. |
| offset | The scalar value subtracted before nested quantization and added after nested dequantization. |
| operator schema | The declared arguments and results of a PyTorch operator. |
| oracle | The identified reference calculation that supplies expected test values. |
| profile | The fixed configuration and criteria for an experiment. |
| profiler | A tool that records execution time, activity, or memory use. |
| probe | A small program that measures a specified software behavior. |
| precondition | A required input condition before an operation starts. |
| partial block | A quantization block with fewer values than the specified block size. |
| partial tile | A matrix tile with fewer values than the specified tile dimensions. |
| patch | A recorded source change that applies to an identified original file. |
| Pallas | The JAX interface for device kernel code. |
| plugin | A separately installed package that registers backend implementations through an entry point. |
| QLoRA | Adapter training with a quantized, frozen base model. |
| quantization | Conversion from numerical values to codes and scales. |
| receipt | A structured record that identifies an execution and its returned artifacts. |
| reference implementation | The implementation that defines the expected result for a comparison. |
| revision | A specific state of a source repository, identified by its commit. |
| reviewer | The person or agent that examines code and test records before acceptance. |
| rounding | Selection of a representable numerical value from a more precise value. |
| scale | A multiplier used to reconstruct numerical values from quantized codes. |
| runtime | The installed software and execution environment for a program. |
| SGD | Stochastic gradient descent. |
| SHA-256 | The hash algorithm that produces a 256-bit digest. |
| snapshot | A recorded source or metadata state at a specific time. |
| state_dict | The upstream PyTorch mapping of model state. |
| subnormal number | A floating-point value with a magnitude below the smallest normal value for its data type. |
| subnormal flushing | Replacement of subnormal floating-point values with zero during computation. |
| synchronization | Submission and completion of specified device operations before further host activity. |
| functionalization | The PyTorch transformation that replaces tensor mutations and views with functional operations. |
| tensor | A numerical array with a shape and data type. |
| TensorImpl | The PyTorch C++ object that holds tensor implementation state. |
| tile | A limited part of a matrix that a kernel processes. |
| timestamp | A recorded clock value for an event or interval boundary. |
| TPU | A Google Tensor Processing Unit. |
| transpose | A change in the order of array axes. |
| transport | The software component that sends network requests and returns their responses. |
| upstream package | The external bitsandbytes package at the pinned source revision. |
| uint8 | An unsigned eight-bit integer data type. |
| VMEM | The vector memory used by TPU kernel computations. |
| wheel | A Python package distribution in wheel format. |
| XLA | The compiler system that PyTorch/XLA uses. |

## Technical verbs

| Verb | Meaning in this project |
| --- | --- |
| call | Transfer program control to a function or operator. |
| execute | Perform the instructions of a computer program or device operation. |
| compile | Convert source operations into an executable program. |
| dequantize | Convert quantized codes and scales into numerical values. |
| dispatch | Select and call the implementation for a PyTorch operator. |
| import | Load a Python module with the Python import system. |
| initialize | Set the initial state of a software component. |
| quantize | Convert numerical values into codes and scales. |
| register | Add an implementation to the PyTorch operator registry. |

Use these verbs only for the specified computer or mathematical process.
