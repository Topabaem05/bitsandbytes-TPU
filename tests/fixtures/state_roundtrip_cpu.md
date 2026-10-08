The `state_roundtrip_cpu.json` file contains selected tensor records from the eight fixed CPU linear cases.
The local run used Python 3.12.14 and PyTorch 2.14.1 on macOS.
This runtime does not qualify the Linux TPU stack.

The `state_device_fixture.py` helper compares the literal fixture hash and fixed profile and input hashes.
It copies the fixture into each test's temporary directory and creates plain CPU tensor checkpoints with `torch.save`.
It copies each saved checkpoint to the restore path without changing its bytes.
It writes the dependent checkpoint hashes and source, receipt, and oracle hashes for that temporary fixture.

Device identity, launch identity, HLO, and metric records are synthetic test data.
They test the production verifier and owner.
They do not show TPU execution.
Mutation controls change only the temporary copies.
This fixture contains no provider logs, credentials, or binary checkpoint files.
