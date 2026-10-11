# Diagnostic source adoption

The reviewer adopted 65 source and recipe files after independent checks.
The original complete preparation archive had 75 members.
Every archive member passed full hash and ZIP integrity verification.
The selected adoption inventory is in [the source record](../../results/m4-word-diagnostic-v1-preparation/source-adoption.json).
Disposable local CPU output and temporary preparation files are not part of this source directory.

The packet SHA-256 is `a73c5b044ae981752e04a0a8b67706d1d21b8d9e94ac6044610bc1dc0ef984fc`.
Its 58 members remain unchanged.
The original mathematical plugin remains unchanged in the main implementation.
The packet has isolated diagnostic variants.

The reviewer replayed all 24 CPU probes and rejected nine incorrect records or inputs.
These local CPU checks used Torch 2.14.1 and have no Linux or TPU qualification.
The worker passed 14 preparation controls.
The independent adviser checked six incorrect CPU results and three simulated observation failures.
Preparation review does not qualify M4.

Run the root checks with a fresh temporary output directory:

```sh
PY=/Users/topamini/code/Post2TPU/.venv/bin/python
ROOT=/Users/topamini/code/bitsandbytes-TPU/experiments/2026-10-04-bitsandbytes-tpu/cloud/word-diagnostic-v1
$PY -B "$ROOT/root_review.py" --candidate "$ROOT" \
  --packet-sha256 a73c5b044ae981752e04a0a8b67706d1d21b8d9e94ac6044610bc1dc0ef984fc \
  --output /tmp/NEW-ROOT-DIAGNOSTIC-REVIEW
```

The historical README identifies the original temporary preparation path.
Use this adopted directory for the packet and browser recipes.
If you replay `review_controls.py`, first copy this preparation into a fresh temporary directory.
That replay writes its two local evidence reports.
Remove disposable replay files after required records are preserved.

The browser intent starts one 600-second clock.
Setup, CPU probes, TPU probes, recovery, and exact runtime closure use that clock.
The packet permits no allocation retry.
Actual Linux and TPU results require separate records.
