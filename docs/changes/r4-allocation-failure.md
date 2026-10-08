# R4: Colab allocation server failure

The second nested experiment used the corrected notebook entry packet.
The Colab allocation request returned `ColabRequestError` with `Bad Gateway`.
No packet upload, bootstrap command, CPU reference, or TPU numerical case executed.
This attempt therefore gives no device result for the import correction.

The allocation request was issued once.
The complete lifecycle took approximately 242.99 seconds.
The owner observed an empty server list and zero active usage after closure.
The reviewer verified closure of the host and all eight CLI process groups.

The [failure record](../../experiments/2026-10-04-bitsandbytes-tpu/results/nested-allocation-failure.json) retains artifact hashes and the original errors.
The next attempt can use the same packet with a fresh output directory and reviewed execution authorization.
No scientific source or numerical tolerance needs a change for this server response.
M4 remains unqualified.

## Bounded retry

A fresh allocation request used the same corrected packet after observed closure.
It also returned `Bad Gateway` before payload execution.
Its lifecycle took approximately 244.02 seconds.
The reviewer verified closure of the host and all eight CLI process groups again.
The server list was empty and active usage was zero.

Refer to [the retry record](../../experiments/2026-10-04-bitsandbytes-tpu/results/nested-allocation-retry-failure.json).
Two consecutive server failures require a connection diagnosis before another CLI allocation.
Neither failed allocation changes the numerical status of the unexecuted experiment.
