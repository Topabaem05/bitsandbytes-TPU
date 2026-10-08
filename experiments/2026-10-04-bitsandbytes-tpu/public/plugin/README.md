# Public Pallas Mosaic 7 candidate

This is the separate `pallas-forward-mosaic7-gather-bf16-fp32-v1` source generation. It is unqualified. The accepted gather decode kernel and adapter retain their exact reviewed bytes. A lazy admitted factory traces the original three operand objects, applies the pinned JAX 0.7.1 Mosaic 8-to-7 serde pass, and sends the converted config to the actual native binding. It retains both configs and the conversion audit. Package import does not import JAX or Torch/XLA. Conversion or native failure has no CPU or reference recovery.

Original upstream public classes, methods, autograd, parameter and bias state remain owned by bitsandbytes. Explicit unsupported tile, bias and noncontiguous routes use the existing same-device reference. NF4 block64 and non-nested state only. M4 nested, TPU gradients, loaded executable, memory and performance remain unqualified. An accepted exact Mosaic native generation is required before any public device dispatch.
