# R3: Colab state restoration result

The reviewer accepted M3 on 2026-10-08.
All eight fixed linear cases passed after restoration in a new TPU process.
The reviewer independently compared 136 numerical gates and verified all 204 archive members.
All eight checkpoint pairs, tensor states, and outputs matched exactly between save and restore.
Accepted milestones are now **3 of 8**.

The cases cover FP32, BF16, rank two, rank three, and both bias options.
FP32 input gradients and applicable bias gradients passed the fixed gates.
The largest scalar-reference differences were approximately `5.40e-8` for output and `2.79e-8` for input gradients.
Bias gradients matched their scalar references exactly.
Packed base weights remained fixed and had no gradients.

The save and restore processes had distinct PIDs, tokens, and lifetimes.
The coordinator confirmed save-process exit and absence before it started restoration.
Both children remained within the externally owned process group.
The restore process used fresh original public objects, `Params4bit.from_prequantized`, and strict `load_state_dict(assign=False)`.
Checkpoint inspection used `torch.load(weights_only=True)` and required plain CPU tensor values.

The experiment retained the M2 runtime, package source, input hashes, and tolerances.
Each child selected native precision `highest` once before graph construction.
The records passed source, public method, device placement, HLO, execution, and fallback checks.
The fixed Linux source controls also passed.

The original CLI returned exit code 2 despite its valid `PASS_TPU_STATE_RECORDS` result.
Its success list omitted this new status.
The scientific children returned zero, and independent inspection confirmed their passing results and complete closure.
The correction adds only this exact status to the success list.
The reviewer repeated nine controls through the actual CLI entry point; all passed.
Numerical failures, cleanup failures, and unknown states still return exit code 2.
The original driver and host records retain their unsuccessful exit codes.
This correction required no repeat of the scientific experiment.

One allocation served the experiment, and the lifecycle took approximately 333.98 seconds.
The owner recovered the complete archive and stopped the exact runtime.
The server list was empty, active usage was zero, and all local and remote process groups closed.

BF16 gradients, nested quantization, QLoRA, Pallas, CUDA comparison, and performance measurements remain outside this result.
The next device milestone is nested quantization with the upstream default statistics option.
Refer to [the result record](../../experiments/2026-10-04-bitsandbytes-tpu/results/state-colab.json).
