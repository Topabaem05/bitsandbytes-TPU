# Public reference observer version 3

The previous public training case stopped before gradient synchronization.
Its reference graph contained scalar S64 and U8 parameters.
The original shape reader rejected these parameters.
The historical result remains four passes, one error, and ten unexecuted cases.

The new reference observer binds each scalar to its fixed value, type, and consumer role.
It verifies the signed negative-index correction and the unchanged positive-index branch.
The corrected index must reach its gather through operations that preserve its values.
Different backend handles can contain the same scalar when each consumer satisfies the required role.

The native graph reader, numerical functions, protocol, tolerances, and package admission remain unchanged.
The source has a new generation and a complete manifest.
The reviewer verified all 51 source archive members and all ten adopted file changes.
Independent replay passed 77 observer controls and 12 source controls.
The controls include incorrect predicates, changed scalar roles, doubled indices, and valid equivalent representations.

These results qualify the source preparation only.
A fresh packet, qualified Linux CPU reference, and complete TPU execution remain required.
M6 remains unqualified.
Selected executable identity, memory, and performance remain unknown.

Refer to the [review records](../../experiments/2026-10-04-bitsandbytes-tpu/results/public-observer-v3-preparation/root-source-review.json).
The original recovered records remain in the verified private archive.
The temporary replay inputs are deleted after review.

The prose received sentence-length and technical-term inspection.
The complete ASD-STE100 dictionary inspection remains incomplete.
