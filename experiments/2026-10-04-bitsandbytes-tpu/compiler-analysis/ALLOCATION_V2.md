# Version 2 allocation bounds audit

This extension adds four primitive types to the offline allocation bounds audit.
It uses a separate source file and output schema version 2.
The original `allocation_audit.py` and its schema version 1 remain unchanged.

The contract identifier is `STATIC_DENSE_NATURAL_WIDTH_ALLOCATION_BOUNDS_V2`.
Both the Python result and CLI report identify this contract.
The CLI report also identifies the frozen version 1 source hash.

| Primitive enum | Type | Natural bytes per element |
| --- | --- | --- |
| 1 | PRED | 1 |
| 4 | S32 | 4 |
| 5 | S64 | 8 |
| 8 | U32 | 4 |

The existing U8, BF16, and F32 rules remain unchanged.
No other primitive type enters the supported scope.
The audit uses the dimension product and natural bytes per element as an array storage lower bound.
A scalar has one element.
A zero dimension gives a zero array storage lower bound.
Tuple descriptor storage remains unknown because the backend pointer size is unknown.

An absent `element_size_in_bits`, zero, or the natural byte width multiplied by eight is supported.
A valid positive nonnatural packing value returns `UNSUPPORTED_REPRESENTATION`.
For PRED, one-bit packing therefore remains unsupported.
Dynamic shapes, sparse layouts, and combine-dimension tiles remain unsupported.
All original slice, alias, signed integer, allocation end, and color rules remain unchanged.

## Primary source proof

The reference revision is `31eb6029e8973337ef6c99810c27e1d807791c11`.
It identifies a public schema reference, not the internal libtpu revision.
The source inspection date is 2026-10-11.

- [xla_data.proto](https://github.com/openxla/xla/blob/31eb6029e8973337ef6c99810c27e1d807791c11/xla/xla_data.proto#L27-L50) assigns the four enum numbers.
- [primitive_util.h](https://github.com/openxla/xla/blob/31eb6029e8973337ef6c99810c27e1d807791c11/xla/primitive_util.h#L240-L312) maps PRED to bool and the integer types to their fixed-width native types.
- [primitive_util.h](https://github.com/openxla/xla/blob/31eb6029e8973337ef6c99810c27e1d807791c11/xla/primitive_util.h#L686-L771) derives bit widths, then rounds each width up to whole bytes.
- [shape_util.cc](https://github.com/openxla/xla/blob/31eb6029e8973337ef6c99810c27e1d807791c11/xla/shape_util.cc#L934-L983) uses `ByteWidth` for natural storage and handles explicit packing separately.
- [xla_data.proto](https://github.com/openxla/xla/blob/31eb6029e8973337ef6c99810c27e1d807791c11/xla/xla_data.proto#L267-L270) defines the default when `element_size_in_bits` is absent or zero.

PRED has one value bit, which requires one natural storage byte under this rule.
S32 and S64 include their sign bit in the fixed width.
U32 has 32 value bits.
These rules give the four byte widths in the table.
The retained source seal records exact downloaded bytes, hashes, URLs, and proof ranges.

## Replay after reviewer adoption

Run these commands from the repository root.
Each output path must be new.
The controls delete their temporary binary fixtures before they write the summary.

```sh
PYTHON=/Users/topamini/code/Post2TPU/.venv/bin/python
ANALYSIS=experiments/2026-10-04-bitsandbytes-tpu/compiler-analysis
BASELINE=experiments/2026-10-04-bitsandbytes-tpu/compiler-native/analyzer.py
FROZEN=experiments/2026-10-04-bitsandbytes-tpu/cloud/tests/compiler-accepted/fixtures/compiler/after.hlo.pb
"$PYTHON" -B "$ANALYSIS/allocation_controls.py" \
  --baseline-analyzer "$BASELINE" --frozen-fixture "$FROZEN" \
  --output "$NEW_ORIGINAL_CONTROL_REPORT"
"$PYTHON" -B "$ANALYSIS/allocation_v2_controls.py" \
  --baseline-analyzer "$BASELINE" --original-controls "$ANALYSIS/allocation_controls.py" \
  --frozen-fixture "$FROZEN" --output "$NEW_V2_CONTROL_REPORT"
"$PYTHON" -B "$ANALYSIS/allocation_directory_v2.py" \
  --baseline-analyzer "$BASELINE" --frozen-auditor "$ANALYSIS/allocation_audit.py" \
  --input-directory "$RETAINED_RAW_PRIVATE_DIRECTORY" \
  --capture-packet-sha256 "$REVIEWED_CAPTURE_PACKET_SHA256" \
  --output "$NEW_STATIC_AUDIT_REPORT"
```

The directory reader parses every `.pb` file and selects assignments through outer field 3.
It does not use a filename to infer a compiler stage.
An invalid selected record remains a failure.
The reader records all protobuf inventory hashes before and after the audit.
It rejects output paths inside the input directory.
Capture packet binding remains caller supplied and requires independent archive review.

## Verification and limits

The original 65 controls passed with the frozen auditor and with version 2.
All 93 new controls passed.
These controls include valid and invalid cases for each added type.
They also test field-based selection with misleading filenames and the read-only output boundary.

The actual read-only audit parsed 55 protobufs and found 11 assignments.
Version 1 returned four bounds passes and seven unsupported results.
Version 2 returned 11 bounds passes.
All 55 input protobuf sizes and hashes remained unchanged.

The actual packet SHA-256 is `78797717b385dae4151f866f5918600e1c1023c7433e0fc4f4828fcf0e361540`.
The compact audit retains each selected module ID, input hash, status, and allocation size sum.
An allocation size sum is a compiler plan total.
It is not a runtime memory measurement.
Runtime allocator peak, internal Mosaic storage, and the actual selected executable link remain unknown.
Numerical results were not evaluated.
This audit does not qualify M6.

The author examined sentence lengths and the project terms.
Dictionary conformity remains subject to reviewer inspection.
