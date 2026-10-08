# R6: Native kernel and diagnostic correction

The [previous Colab experiment](r6-native-colab-errors.md) ended with five case errors.
The corrected kernel selects each packed half with fixed slices and `where`.
This change removes the two unsupported `dynamic_slice` operations.

The diagnostic synchronizes functional tensors and removes their wrappers before the original backend wrapper receives them.
It restores the required output wrapper through the existing adapter boundary.
The recorder converts the complete metric structure to JSON before validation.
The original metric validator and numerical criteria remain unchanged.

The new manifest identifies all 23 native source files.
The kernel, diagnostic, recorder, and derived source hashes have new values.
The previous actual records retain their original source identities and failed results.
The new candidate requires a fresh qualified Linux CPU oracle.
It rejects the old oracle source seal.

## Local inspection

| Inspection | Result | Scope |
| --- | --- | --- |
| Combined diagnostic and admission controls | 31 passed | Local CPU; includes incorrect source seals and metric records |
| Runtime and adapter source controls | 8 passed | JAX 0.7.1; no Torch/XLA device import |
| Previous kernel interpretation controls | 18 passed | Same corrected kernel bytes; CPU execution |
| Previous Mosaic conversion controls | 10 passed | Same kernel; real TPU target conversion on a CPU client |
| Incorrect half selection | Rejected | Kernel conversion succeeded, but the numerical comparison failed |
| Genuine Git packet | 69 members passed inspection | Upstream revision, patch, source bytes, and archive hashes |

The reviewer repeated all 68 controls within these stated scopes.
After file adoption, three test containers passed in 48.68 seconds.
They contain 26 diagnostic controls, seven Pallas controls, and 33 cloud controls.
The original actual CPU source seal remains a separate rejection fixture.
The first optional regression used identical half values and failed to detect an incorrect kernel.
The worker retained that failure and changed the fixture to use distinct half values.
The production code and numerical criteria did not change for that test correction.
The conversion output still requires the libtpu layout and backend passes.
These local results do not establish TPU compilation or execution.
They also do not establish compiler memory use or performance.

The packet retains the historical `base_cloud_revision` field from the native builder.
That field identifies the earlier cloud foundation, not this correction's complete source state.
The full file manifest and reviewed commit identify the actual candidate.

The [inspection record](../../experiments/2026-10-04-bitsandbytes-tpu/results/native-repair-preparation.json) gives the source identities and test hashes.
Actual TPU execution, compiler memory records, and public API integration remain required.
M6 remains unqualified.

## Colab allocation attempt

The reviewer requested one V6E1 runtime after this correction was committed and pushed.
The browser still showed an allocation wait after 642.58 seconds; the planned wait was 600 seconds.
The reviewer closed the temporary tab and verified an empty server list and zero active usage.
All four CLI process groups closed.
The marker, installation, CPU oracle, and TPU cases did not execute.
The browser showed 3.18 available compute units; the cause of the allocation wait remains unknown.
The [allocation record](../../experiments/2026-10-04-bitsandbytes-tpu/results/native-repair-allocation-wait.json) retains the original times and source identity.
