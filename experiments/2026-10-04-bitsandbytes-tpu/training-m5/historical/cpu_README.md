# Private M5 QLoRA preparation

Date: 2026-10-08. Status: **UNADOPTED_PENDING_M3_AND_M4**.
This is CPU preparation. It does not qualify Linux PyTorch/XLA 2.9.0 or TPU execution. This preparation does not change tracked files or R2, R3 and R4 source files.

## Fixed candidate protocol

Use two original public `Linear4bit` layers with width 256, bias and FP32 data. Both layers use NF4, block size 64, uint8 storage and public `compress_statistics=True`. Each layer has adapter A with shape [4, 256] and B with shape [256, 4]. The adapter scale is one, with alpha 4 and rank 4. Compose the two layers directly. There is no activation, dropout or normalization. These choices make the previously unspecified composition explicit.

Only the four adapter tensors train. Base weights, biases, packed bytes and all nested state remain frozen. The batch size is eight. Use mean squared error and SGD with learning rate 0.01, no momentum and no weight decay. Execute updates 1 through 20. Save after update 10. Restore in a new process and execute updates 11 through 20. Compare with an uninterrupted run.

`PROTOCOL` and `GENERATOR` in `prototype.py` define the exact choices. The generator uses integer modular arithmetic and binary fractions. It sets all base and adapter inputs explicitly. A starts with deterministic nonzero values. B starts at zero. Step inputs and targets have separate fixed formulas. Tensor shape, dtype and byte hashes cover every base input, adapter input and step input. Unused constructor initialization does not affect these values. This candidate needs no RNG state to resume.

`dependency-pins.json` pins the read-only R4 prototype, R2 source helpers and existing V5 owner code. The source bundle also binds the candidate probe and protocol. The reviewed R2 patch manifest fixes the isolated upstream Python source inventory. The fixed profile supplies the existing FP32 forward and gradient gates. No tolerance changed.

## Public source and byte reference

The isolated CPU fixture disables backend discovery only during upstream import. It restores discovery immediately. It registers R4 CPU kernels only where no native CPU kernel exists. It does not register XLA kernels, import XLA/JAX or change installed source. This is not automatic backend discovery evidence.

The original public classes and method code objects remain upstream definitions. The fixture binds the five R2 methods and four public state methods to compiled admitted source. Each child checks these identities and the Python source inventory before and after its run. Default future conversion flags must stay false.

The byte reference is the pinned default Python body from upstream revision `833649043474794b8fe7a4136e0c40faf077b2e0`. Its source SHA256 is `e39afd16dca6e6f34a14305ade0ba4acbb49dc2c7a9baf937341d7e33b682819`. The fixture extracts this body into a separate reference namespace. It removes registration and compilation decorators there. It does not replace a loaded public method.

Each child checks both base layers against the default body. Packed NF4 bytes, first scale codes, offset and second scale values must match exactly. A separate dense FP32 model uses the decoded reference weights. It computes loss, adapter gradients and SGD updates. The parent verifier independently replays this dense model for all 20 steps. It does not trust reported dense gate status as its sole numerical reference. Native CPU and CUDA byte comparisons remain `NOT_RUN`.

## Records and restore

Every step has a JSON record. It stores step identity, loss, output, all four adapter gradients, adapters after update, full frozen-state hashes and absence of base gradients. The child also stores its source, input and optimizer bindings. The verifier checks tensor shapes, finite values, loss/output consistency, step sequence, frozen state and the original FP32 gates.

The checkpoint contains plain CPU model tensors, exact tensor hashes, completed step, source/input/protocol hashes and optimizer state/configuration. It binds the adapter order by names. SGD has no momentum state in this protocol. The optimizer must contain one group with only adapter indices [0, 1, 2, 3].

Restore uses fresh original public modules and `Params4bit.from_prequantized`. It gives each layer a fresh statistics dictionary. It then uses public `load_state_dict(strict=True, assign=False)` for weights, biases and adapters. Quantization metadata is reconstructed through the public state path. The loaded full state must match the checkpoint exactly. Both nested `QuantState` classes and all module/parameter aliases must be original. Packed weights and biases remain frozen.

Save records must match the uninterrupted first ten steps. Resume records must match uninterrupted steps 11 through 20. The verifier also compares every phase with the independent dense replay. Final full model state must match exactly in this CPU candidate. TPU acceptance will need a separate reviewed policy; this CPU result does not prove exact device arithmetic.

## Process ownership

The outer V5 owner starts a fresh coordinator as process group leader. Baseline, save and resume use three fresh child PIDs and tokens. They inherit that group. Each child checks its actual parent and group. The coordinator records actual launch identities before waits, reaps each child, and checks child absence before the next phase. Finite phase deadlines fit a finite parent deadline. Parent and child inventories bind record bytes.

The outer guard has a 360-second limit. Each training child has at most 100 seconds. On timeout, the outer owner can close all descendants even if the parent cannot run `finally`. The successful smoke shows all three children reaped and absent, and the parent group absent. A separate actual timeout control killed the parent with SIGKILL. The inherited child and group were absent after cleanup.

## Checks run

The current probe SHA256 is `ebb4d60840ec2a29c33561f42f325ffbef888a47646f203702175215b468891b`.

The local runtime is macOS, Python 3.12.14 and PyTorch 2.14.1. It is unqualified.

- The uninterrupted 20-step run, save through step 10 and fresh-process resume all passed.
- All 200 resume comparison gates passed. All 400 independent dense replay gates passed.
- Final full model tensors matched exactly. Both base layers and all nested state stayed fixed.
- Ten controls ran in separate owned processes. They rejected wrong checkpoint step, learning rate, nested state, frozen weight and adapter mapping. They rejected wrong record step, learning rate and frozen-state hashes.
- A matching corrupted gradient in both baseline and resume records remained a structurally valid record, but the independent dense replay returned numerical `FAIL`.
- All final control groups closed without cleanup errors. The actual outer-timeout control also passed.

The first control harness omitted an owner log directory. That setup failed before its scientific control completed. The original failure and `CLEANUP_UNVERIFIED` receipt remain under `owned-controls/`. The process leader was reaped. A separate follow-up check observed the group absent. The corrected harness creates its directory first. Its ten controls all completed with verified closure. No tolerance or model code changed for this fixture repair.

## Review commands and limits

Run a new smoke output directory:

```text
/Users/topamini/code/Post2TPU/.venv/bin/python -B .work/r5-qlora-preparation/prototype.py run --source .work/task2-transfer-preparation/revised-source --output .work/r5-qlora-preparation/root-review-smoke
```

Repeat the controls. They create new isolated owner directories:

```text
/Users/topamini/code/Post2TPU/.venv/bin/python -B .work/r5-qlora-preparation/controls.py
```

Read `review.json`, `smoke-dense-final/review.json`, `control-results.json`, `outer-timeout-results.json` and the phase records. The earlier `smoke-01` is historical evidence from the candidate before independent verifier replay was added.

This preparation has no TPU, CUDA, performance, convergence or full feature claim. It does not complete M5. Adoption depends on M3 and M4 acceptance, source review and a separate qualified device run. Installed discovery, qualified optimizer behavior, device HLO and fallback counters, device placement, nested scale edge cases and device checkpoint synchronization remain untested. The cloud packet and portable device verifier are not prepared by this task.
