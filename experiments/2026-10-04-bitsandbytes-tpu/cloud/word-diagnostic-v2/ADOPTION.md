# Diagnostic input boundary source adoption

The reviewer adopted 67 selected source and recipe files.
All 88 members of the complete preparation archive passed readback.
The [adoption inventory](../../results/m4-word-diagnostic-v2-preparation/source-adoption.json) records their identities.

The packet SHA-256 is `35a40419449c274d782c75a38fdcdb1669653b97d921451065614ecc51b348fc`.
The packet has 58 members and 123,483 bytes.
Five authored payload files changed from diagnostic v1.
The manifest changed to bind those files.
All other payload files remain byte-identical.

The diagnostic now retains synchronized raw inputs before calling the frozen backend wrapper.
It preserves repeated tensor references, pending updates, dtype, shape, stride, and device.
The four bitcast cases still construct their requested raw or functional source states.
The mathematical plugin, variants, fixed inputs, runtime, limits, and 600-second clock remain unchanged.

The reviewer replayed 24 CPU probes and rejected nine incorrect controls.
Five additional controls passed with native functional inputs.
The adviser independently passed 14 correct and incorrect controls.
These tests used local Torch 2.14.1 and do not qualify Linux or TPU execution.
M4 remains unqualified.

Use this adopted directory instead of the deleted temporary path in the historical README.
Pass a fresh `/tmp` output to `root_review.py` and `functional_input_controls.py`.
Copy the source to a fresh temporary directory before replaying `review_controls.py`, which writes local reports.
Delete disposable output after selected proof preservation.

The reviewer deleted six completed preparation and review roots with 134 files.
Selected source and compact review records remain in Git.
One inherited extra final blank line remains in `packet/control/cli_display.py` for sealed source identity.
