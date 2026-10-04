# Task 1: Zero-block normalization

## Change

The third TPU probe returned incorrect packed bytes for eight zero-weight cases.
It then failed during CPU-to-XLA parameter conversion.
Both failures and the nine passing case records stay in the experiment report.

The corrected calculation uses a denominator of one when a block magnitude is zero.
It sets the normalized values of that block to zero.
It applies the same rule to a partial final block.
The saved scale rule, NF4 boundaries, and packed layout stay unchanged.
The unused odd tail nibble stays at code `7`.
The fixed numerical limits and packed-byte criteria stay unchanged.

## Tests

The reviewer repeated 81 package tests; all passed in 22.05 seconds.
The local environment used macOS, Python 3.12.14, and PyTorch 2.14.1.
An isolated CPU process enabled subnormal flushing with `torch.set_flush_denormal(True)`.
The previous calculation produced NaN values and selected code `0` for zero elements.
The corrected calculation kept code `7` and the required packed bytes.
The 16 zero controls cover FP32, BF16, full blocks, and partial blocks.
Forty normal-value, midpoint, and neighbor controls matched the pinned default Python reference.
The previous calculation stays in a failure fixture with its license notice.
Test output files stay in temporary directories.

The package manifest SHA-256 is `be2ba05deb82b4bd59635dfe4c4b1518cd60b00ec92459b1a69d66f584250cd0`.
The reviewer compared the adopted source with the tested source.
All bytes matched.

## Limits

CPU subnormal flushing does not show the TPU intermediate values.
Tiny returned scales can still become zero on devices that flush subnormal numbers.
Nonzero subnormal inputs and reciprocal underflow at extreme magnitudes need more device tests.
Actual TPU execution of this change is untested.
This numerical change does not correct CPU-to-XLA `module.to`.
M2 is not complete.
