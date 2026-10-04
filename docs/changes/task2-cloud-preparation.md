# Task 2: Colab owner preparation

## Change

Add one owner for the first NF4 API test on Colab.
The owner needs a reviewer admission before allocation.
It permits one new V6E1 session only.
It does not retry an allocation or a test.

The packet binds the reviewed package source, upstream commit, runtime lock, probe, and input files.
It includes the package README, license, third-party notices, and source manifest.
The packet uses Unix regular-file ZIP information.
The owner executes the remote archive guard locally before it creates a CLI instance.

The installer uses a new directory under `/content`.
It downloads the fixed CPython 3.12.14 archive and does a hash check before extraction.
It then creates an independent environment.
It downloads and checks the fixed runtime wheels.
It installs the wheels without dependency selection or bytecode compilation.

A separate lock contains six build wheels, with a total size of 1,346,502 bytes.
The shared `setuptools` and `packaging` files have the same hashes as the runtime lock.
The installer builds the upstream wheel on Linux with `BNB_SKIP_CMAKE=1`.
It builds the plugin wheel on the same system.
Both installations use `--no-deps --no-compile`.
The installer records the wheel hashes and installed package information.

## Process order

1. Start a separate runtime probe process on the TPU.
2. Close that process and its owned process group.
3. Start a separate CPU reference process without `PJRT_DEVICE`.
4. Get the CPU seal and its hash on the host.
5. Supply that hash to a separate TPU probe process.
6. Get the critical receipt and full result archive.
7. Stop the owned session.
8. Get the server and usage records.
9. Execute the probe verifier on the saved arrays.

The owner removes inherited Python path and TPU library path variables from each child environment.
It leaves `HOME` and `CODEX_HOME` unchanged.
It sets `PYTHONNOUSERSITE=1` and starts Python with `-B`.
Only the runtime and TPU probe processes receive `PJRT_DEVICE=TPU`.

The total time limit is 3,600 seconds.
The installation limit is 1,200 seconds.
The science limit is 1,800 seconds.
Each stage also uses the remaining total time.
The owner reserves 600 seconds for retrieval and 60 seconds for cleanup.
An expired work limit does not prevent an attempt to stop the owned session.
An overrun or failed cleanup prevents a pass.

Each child record keeps its first result and cleanup result independently.
The archive includes nested probe receipts and partial test files.
The archive excludes only its three top-level export files before it adds the receipt once.
The local verifier compares archive bytes and the common receipt fields.
It excludes only the three export fields that cannot refer to their own bytes.

## Local checks

The reviewer repeated all 12 focused tests; all passed in 0.05 seconds.
They include fixed archive information, changed archive information, nested receipts, symbolic links, special files, and wrong admission values.
They also include a failed allocation, an expired deadline, a failed stop, restored signal handlers, and unchanged timer state.
These tests use isolated files and a simulated service.
They do not show actual TPU execution.

The first test run exposed a fixture clock problem.
The local logs keep that failure.
The corrected test uses the active clock for the owner deadline.

The fixed packet also passed the same archive guard that the remote installer uses.
The installed CLI interpreter and 27 file identities passed read-only checks.
This worker did not call the service or allocate hardware.
The reviewer independently observed an empty server through the existing login.
No new login was necessary for that observation.

## Draft command

Use the reviewer-approved packet hash and admission file.
Keep the output directory absent before the command starts.
Keep all local logs and wheel files outside Git.

```sh
python -B experiments/2026-10-04-bitsandbytes-tpu/cloud/owner.py \
  --packet .work/task2-cloud/packet-local-v2 \
  --packet-sha256 12ea3de2db421351e6073514cd9aed55dc84e0797d1e8f2c2587eb1223621645 \
  --output .work/cloud-20261004-first \
  --root-acceptance .work/task2-cloud/root-acceptance.json \
  --root-acceptance-sha256 "$BNB_TPU_ACCEPTANCE_SHA" \
  --cli-python "$BNB_TPU_CLI_PYTHON" \
  --cli-identity .work/task2-cloud/cli-identity.json
```

The actual admission file does not exist at this preparation step.
The draft admission cannot permit allocation.
The installer, CPU reference, TPU probe, and cloud cleanup stay `NOT_RUN`.
This preparation does not complete milestone M2.
