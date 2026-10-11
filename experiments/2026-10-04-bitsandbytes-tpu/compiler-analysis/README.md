# Offline compiler allocation bounds

This tool reads a retained binary `HloProto` dump. It does not execute a TPU experiment.
The frozen compiler capture, its source map, and its numerical result remain separate records.

## Fixed inputs

The tool imports the parser from `compiler-native/analyzer.py` only after its SHA-256 matches this value:

```text
c44ecb15696031f326a7abe64ea3d8407b9bc21f527cb94f8b5f149c456fcd8d
```

That source belongs to baseline `5dbfd05128fe741ba19c513d43a87c09b7cca017`.
The tool does not call the baseline allocation validator or its native graph validator.
It reads instruction shapes across all computations in the supplied module.
The report records the dump hash, analysis source hash, baseline hash, and caller-supplied capture packet hash.
An independent archive review must establish the relation between that packet and the dump.

## Bounds

For each assigned logical buffer, the tool requires these bounds:

```text
array_storage_lower_bound <= logical_size <= assigned_size <= allocation_size - offset
```

The first bound applies to supported array storage.
The tool permits larger logical sizes and assigned slices.
The tool also permits overlapping slices.
It does not infer valid liveness or backend alias semantics from those overlaps.
For each alias location, its supported array lower bound must fit the assigned slice.
Alias shapes can differ when the shared slice can hold them.

Supported arrays use static, dense U8, BF16, or F32 storage with the natural element width.
Their lower bound is the product of dimensions multiplied by one, two, or four bytes.
Scalars and empty arrays are supported.
Positive dense tiles, tail padding, and metadata prefixes can increase physical storage.
`Tile::kCombineDimension` uses the valid `INT64_MIN` marker.
The tool recognizes that marker only in tile dimensions and reports `UNSUPPORTED_REPRESENTATION`.
Other negative tile dimensions remain invalid.
The tool does not calculate that increase.
Explicit element widths must equal the natural width.
Sparse storage, dynamic shapes, split memory, other element widths, and other data types are unsupported.
Unknown shape or layout fields are also unsupported.

The tool rejects negative signed integers in relevant identifiers, sizes, offsets, dimensions, colors, and shape indexes.
It validates each tuple index before selecting the corresponding shape.
A tuple descriptor has an unknown backend storage size.
Its storage is not the sum of the child array sizes.

## Results

| Status | Exit code | Meaning |
| --- | --- | --- |
| `ALLOCATION_PLAN_BOUNDS_PASS` | 0 | All supported bounds pass. Tuple descriptor sizes and backend semantics remain unknown. |
| `INVALID_ALLOCATION_PLAN` | 1 | A supported bound, integer, location, or wire requirement failed. |
| `UNSUPPORTED_REPRESENTATION` | 2 | The dump or storage representation is outside the declared scope. |

Source or dump hash failures stop before the report is written.
The tool creates a new report and refuses to replace an existing file.
An unsupported representation is not a numerical failure.
Every report keeps numerical results unevaluated and M6 unqualified.

Allocation colors and layout memory spaces are backend identifiers.
They do not identify HBM or a measured peak.
The pinned source permits different colors for a preallocated temporary buffer with a nonzero allocation color.
That exception requires all four flags to be false: parameter, live-out, thread-local, and constant.
All other allocations require matching logical-buffer and allocation colors.
The report records both the temporary-buffer classification and the color-match requirement.
The allocation size sum is a compiler plan total.
Runtime allocator peak, selected executable identity, and internal Mosaic memory remain unknown.

## Replay

Use the retained dump and its independently reviewed SHA-256.
Set `CAPTURE_PACKET_SHA` to the hash of its original compiler packet.
Do not substitute a new experiment for the retained capture.

```sh
/Users/topamini/code/Post2TPU/.venv/bin/python -B \
  experiments/2026-10-04-bitsandbytes-tpu/compiler-analysis/allocation_audit.py \
  --baseline-analyzer experiments/2026-10-04-bitsandbytes-tpu/compiler-native/analyzer.py \
  --hlo-proto "$RETAINED_HLO_PROTO" \
  --hlo-proto-sha256 "$RETAINED_HLO_PROTO_SHA" \
  --capture-packet-sha256 "$CAPTURE_PACKET_SHA" \
  --output "$NEW_AUDIT_REPORT"
```

The current corrected compiler packet hash is:

```text
5992724099a519a82a265ff823b09a8d2e1524789f17b3ee69b10be977b32884
```

This hash alone does not establish actual dump ownership or compiler stage selection.
The current capture stopped at `DUMP_ENTRY_LIMIT` before a raw dump inventory was retained.
Actual allocation-plan audit status for that capture is `NOT_RUN`.
The offline controls do not change its capture or numerical result.

Run the offline controls from the repository root:

```sh
/Users/topamini/code/Post2TPU/.venv/bin/python -B \
  experiments/2026-10-04-bitsandbytes-tpu/compiler-analysis/allocation_controls.py \
  --baseline-analyzer experiments/2026-10-04-bitsandbytes-tpu/compiler-native/analyzer.py \
  --frozen-fixture experiments/2026-10-04-bitsandbytes-tpu/cloud/tests/compiler-accepted/fixtures/compiler/after.hlo.pb \
  --output "$NEW_CONTROL_REPORT"
```

The controls create small temporary binary files for subprocess tests.
They delete those files before they write the summary.
The existing frozen fixture remains unchanged.
It passes the old allocation validator with 100 bytes for `float32[128,256]`.
This auditor rejects that fixture because the array lower bound is 131072 bytes.

## Source evidence

The schema reference revision is `31eb6029e8973337ef6c99810c27e1d807791c11`.
It is a public source reference; it does not identify the internal libtpu compiler revision.

- [Buffer allocation definitions](https://github.com/openxla/xla/blob/31eb6029e8973337ef6c99810c27e1d807791c11/xla/service/buffer_assignment.h): overlapping slices and maximum size across aliased values.
- [Tile definitions](https://github.com/openxla/xla/blob/31eb6029e8973337ef6c99810c27e1d807791c11/xla/layout.h): the valid combine-dimension marker.
- [Allocation implementation](https://github.com/openxla/xla/blob/31eb6029e8973337ef6c99810c27e1d807791c11/xla/service/buffer_assignment.cc): the color exception in `BufferAllocation::AddAssignment`.
- [HLO serialization](https://github.com/openxla/xla/blob/31eb6029e8973337ef6c99810c27e1d807791c11/xla/service/hlo.proto): logical sizes, assigned sizes, locations, and colors.
- [Shape and layout serialization](https://github.com/openxla/xla/blob/31eb6029e8973337ef6c99810c27e1d807791c11/xla/xla_data.proto): array dimensions, natural widths, tuples, and backend storage options.

Source inspection date: 2026-10-11.
The report does not claim exact physical storage or measured memory use.
