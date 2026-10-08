# R2: Colab transfer and API42 result

The reviewer accepted M2 on 2026-10-08 after the Colab V6E1 experiment.
All 42 fixed API cases and four public transfer cases passed.
The reviewer independently compared 196 numerical gates and all 273 archive members.
No numerical gate failed.
Accepted milestones are now **2 of 8**.

The runtime used Linux, Python 3.12.14, PyTorch `2.9.0+cpu`, PyTorch/XLA `2.9.0`, and libtpu `0.0.21`.
JAX and jaxlib were `0.7.1`.
The runtime probe identified a TPU and executed a separate gradient test.
The fixed Linux source controls passed all 18 cases.
The built wheel and installed Python source matched the admitted source inventory.

The experiment selected native precision `highest` once before graph construction.
The public transfer tests used FP32 and BF16 with the original `Params4bit` and `Linear4bit` classes.
They retained parameter identity, attributes, module aliases, packed weights, and frozen base weights.
The FP32 module transfer also passed input and bias gradient gates.
The records contain device placement, HLO, execution counters, and output arrays.
The verifier found no positive `aten::` fallback counter.

| Output | FP32 maximum absolute error | BF16 maximum absolute error |
| --- | --- | --- |
| Packed bytes | 0 | 0 |
| Decoded weights | 0 | 0 |
| Forward output | `1.1920928955078125e-7` | `0.00390625` |
| Input gradient | `5.960464477539063e-8` | Not tested |
| Bias gradient | 0 | Not tested |

All comparisons used the existing tolerances.
The maximum scale difference was approximately `1.0e-38`, within the fixed FP32 decode gate.
The rank-two FP32 HLO observer confirmed `highest` precision in the relevant output graphs.
Rank-three HLO files remain available without that dot observer result.

One allocation served the complete experiment.
The owner recovered the complete archive and stopped the exact session.
The server list was empty, active usage was zero, and all local and remote process groups closed.
The lifecycle took approximately 310.43 seconds.
The allocation succeeded with the reviewed transport; this result does not establish the cause of the earlier timeouts.

M3 restoration in a new TPU process remains required.
BF16 gradients, nested quantization, CUDA comparison, Pallas execution, and performance measurements remain outside this result.
Refer to [the result record](../../experiments/2026-10-04-bitsandbytes-tpu/results/transfer-colab.json).
