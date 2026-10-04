# Task 2: Runtime files

## Change

The runtime files select Linux x86_64 wheels for CPython 3.12.14 and glibc 2.31 or later.
The fixed versions are `torch==2.9.0+cpu`, `torch-xla==2.9.0`, `libtpu==0.0.21`, `jax==0.7.1`, and `jaxlib==0.7.1`.
The lock includes 35 wheels with a total size of 572,327,598 bytes.

The resolver evaluates dependency conditions for Linux, not the host system.
It includes the XLA `tpu` and `pallas` options.
It excludes the JAX `tpu` option and CUDA packages.
The resolver uses only stable versions that meet the declared requirements.
It stops if the requirements conflict.
It does not search alternative dependency combinations.

Each wheel has a fixed URL, size, and SHA-256 hash.
The resolver does a check of the complete wheel, ZIP entries, package information, Python requirement, and wheel tags.
It also does a check of all active dependency requirements.
The XLA Pallas bridge source in the wheel equals the fixed release source.
The release `setup.py` has the expected hash.

## Local results

All 35 complete wheels passed the file checks.
The 16 fixture tests passed.
These tests include wrong hashes, versions, wheel tags, archive paths, and Linux dependency conditions.
They also include rejection of CPU fallback.

The first resolve attempt stopped on an old version string that did not conform to PEP 440.
The next attempt stopped because setuptools contains package information for included third-party packages.
The resolver now selects package information from the top-level `.dist-info` directory only.
The local logs retain both failures.

The Mac probe stopped before XLA import with `LINUX_X86_64_CP312_REQUIRED`.
No package installation or TPU execution occurred.
No NF4 or Pallas kernel test occurred.
These results do not complete milestone M2.

## Use

Use an existing Python environment with `packaging` for the resolver.
Keep the wheel files and test logs outside Git.
The local files are in the ignored `.work` directory.
Use `--from-lock` to download the exact files without a new version selection.
Use `--verify-only` to do file checks on an existing wheel directory.
Use `--requirements-out` to produce input for `pip --require-hashes`.

```sh
python -B experiments/2026-10-04-bitsandbytes-tpu/runtime/resolve.py \
  --from-lock \
  --lock experiments/2026-10-04-bitsandbytes-tpu/runtime/requirements.lock.json \
  --cache .work/cache/bitsandbytes-tpu-wheelhouse \
  --requirements-out .work/runtime-requirements.txt
```

Install the fixed files only in a new Linux CPython 3.12 environment.
Do not install these Linux wheels in the Mac environment.

```sh
python -m pip install --no-deps --no-compile --require-hashes \
  -r .work/runtime-requirements.txt
PJRT_DEVICE=TPU python -B \
  experiments/2026-10-04-bitsandbytes-tpu/runtime/probe_runtime.py \
  --out .work/runtime-probe.json
```

The probe requires an actual TPU backend.
It records device names, source hashes, versions, metrics, and a small gradient result.
A probe pass applies only to that runtime test.
It does not establish NF4 correctness or performance.
