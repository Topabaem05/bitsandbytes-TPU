The nested_cpu_seed.json.gz file contains selected plain JSON tensor records for the 79 fixed cases. Its uncompressed SHA256 is 7ae0b334645b4d8595c5445dd6c413a1dde89ccb3b113b3cd6861276a570ff86. The source run used Torch 2.14.1 on Darwin. These CPU values do not qualify the fixed Linux TPU runtime.

nested_saved_fixture.py verifies the compressed and uncompressed byte hashes. It reads this immutable seed and makes fresh test records in temporary directories. It sets explicit synthetic CPU and device identities, HLO, metrics, and source bindings. The real recovered-record verifier then reads those files. Mutation controls change only the temporary records. No provider logs or credentials are included.

These fixtures test source admission, archive recovery, numerical failure retention, and resource ownership. They do not show actual TPU execution or native CPU/CUDA golden results. The actual experiment must generate a new CPU oracle under the admitted Linux runtime before TPU execution.
