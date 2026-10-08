# Native diagnostic archive failure

Record date: 2026-10-08 UTC.

The first native Colab attempt stopped before the native device phase.
The runtime probe and all five qualified CPU reference cases completed.
The remote archive step then failed with `ModuleNotFoundError: No module named 'transport'`.
No native numerical comparison executed.
M6 remains unqualified.

## Cause and limits

The notebook started the remote module with `runpy.run_path`.
This entry did not add the cloud directory to the Python module search path.
The new CPU archive step used a plain `import transport` statement.
Earlier local tests did not reproduce this entry with an empty module cache.

The owner recovered all records and closed the session.
Its subsequent readback also failed because `recovered_oracle_sha256` was absent.
The records retain this secondary `KeyError` and the original remote error separately.
The correction must preserve the original failure and skip native verification when required records are absent.

The archive failure does not show a kernel or numerical defect.
It also supplies no evidence that the native kernel works on a TPU.

## Inspection

The reviewer inspected all 92 members of the complete archive.
The archive contains 313,509 bytes.
Its SHA-256 is `a77a94b3f0cb72a4304b620769b8eb63314626b8ef9efc447d14b52ef3897c9c`.
The separate CPU archive contains 11 members; their hashes and bytes also matched.

The reviewer applied the CPU oracle gate after record recovery.
The gate passed source, runtime, process identity, process closure, and deterministic input requirements.
All 164,096 reference values had the required finite type and shape.
This inspection did not authorize or start a device phase.

The TPU runtime probe passed with Python 3.12.14, PyTorch/XLA 2.9.0, JAX 0.7.1, and libtpu 0.0.21.
This probe executed basic arithmetic and a gradient.
It did not execute NF4 or Pallas.

All 38 CLI process groups, 13 remote steps, and the host process closed.
The driver ran for 283.476 seconds.
The original browser allocation lifecycle lasted 809.396 seconds, within its 3,600-second limit.
The official server list was empty, and the active usage rate was zero.
The reviewer also observed the disconnected browser and closed the temporary tab.

Eight isolated archive, array, and closure controls gave the expected results.
The [result record](../../experiments/2026-10-04-bitsandbytes-tpu/results/native-colab-archive-failure.json) retains the inspection hashes.
Accepted milestones remain **3 of 8**.

## Next work

Load the admitted transport file from its exact sibling path.
Test the CPU archive step in a new process with the actual notebook entry method.
Test complete and incomplete native records before another Colab attempt.
Keep all 23 native files and the numerical criteria unchanged.
