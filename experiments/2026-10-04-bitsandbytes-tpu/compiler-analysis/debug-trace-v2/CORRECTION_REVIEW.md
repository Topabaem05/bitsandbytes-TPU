# Correction review

Both reported defects are corrected in a new isolated generation. The frozen parent candidate remains unchanged.

The offline verifier accepts completion at 110, deadline 120, and review time 1000. It rejects future completion and completion after the deadline.

The parser retains compilation and execution arities. It compares compilation against submitted module bytes and execution against optimized module bytes.

This design permits valid arity changes during compilation. Missing submitted shape evidence produces `UNSUPPORTED_SUBMITTED_MODULE_REQUIRED`.

The bounded reader uses only `host_program_shape`. It counts top-level parameters and top-level result tuple children.

The tests preserve complete module bytes and their hashes. They include nested tuples, empty tuples, helper parameters, and corrupt cold and warm counts.

The first control run exposed an obsolete deadline fixture. Its compact repair record explains the correction. No scientific criterion changed.

## Pinned primary evidence

All fifteen complete bodies passed SHA-256 and exact expression checks. Their selected excerpts retain original line numbers.

- `debug_util.cpp:387–393`: input count uses `parameters().size()`; tuple output count uses `tuple_shapes_size()`.
- `xla_graph_executor.cpp:1113–1116`: execution analysis uses the cached computation's program shape.
- `computation_client.h:114–121`: computation construction obtains the program shape with `GetProgramShape()`.
- `xla_computation.cc:28–30`: `GetProgramShape()` reads `host_program_shape`.
- `xla_graph_executor.cpp:1431–1454`: compilation obtains the submitted shape and can wrap parameters.
- `xla_graph_executor.cpp:1488–1490`: compilation analysis uses that submitted shape.
- `hlo.proto:553–566`: `host_program_shape` is module field 4.
- `xla_data.proto:326–377`: shape tuples use field 4; program parameters and result use fields 1 and 2.

Torch/XLA revision: `5fab7053df86c8d503b98d9e7202ca8b8d4978c7`.

OpenXLA revision: `92f7b5952dd585c5be17c9a5caad27407005b513`.

## Remaining limits

The fixtures contain synthetic protobuf messages. They are not executable HLO modules or actual device records.

The reader validates the fields needed for arity. It does not prove complete protobuf semantics or runtime module provenance.

The direct PJRT recorder remains unimplemented. Actual TPU verification did not run. M6 remains unqualified.

Loaded handle identity, binary identity, allocator peak, and internal Mosaic storage remain unknown.

Full English dictionary review remains incomplete. This review makes no certified STE claim.
