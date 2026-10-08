# R4: Colab bootstrap failure

The first nested experiment stopped before package installation or numerical work.
It allocated one Colab V6E1 runtime and uploaded the complete packet.
No CPU reference cases or TPU numerical cases executed.
M4 remains unqualified.

The notebook used `runpy.run_path` to execute the uploaded `remote.py` file.
That entry point did not add the script directory to `sys.path`.
The new `nested_contract` import then raised `ModuleNotFoundError`.
Local tests had added the cloud directory to their import path.
They did not expose this notebook entry failure.

Both notebook commands returned CLI exit code zero despite their tracebacks.
The required phase receipt was absent, so the owner rejected the run.
The failed receipt download and original errors remain in the local records.
No result archive existed at this point.

The lifecycle took approximately 77.22 seconds.
The owner stopped the exact runtime and observed an empty server list and zero active usage.
The reviewer verified closure of the host and all CLI process groups.

The correction must load the admitted sibling source through the actual notebook entry point.
Its controls must include a valid unpack operation, incorrect source bytes, and a notebook failure without a receipt.
No numerical tolerance or scientific source changed after this failed attempt.
Refer to [the failure record](../../experiments/2026-10-04-bitsandbytes-tpu/results/nested-bootstrap-failure.json).
