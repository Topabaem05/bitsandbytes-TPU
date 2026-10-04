"""Prepare CPU reference data and test original state restoration in a new process."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback
import uuid

HERE = Path(__file__).resolve().parent
BACKEND_SHA = "e8743e33e6a0454d5d05b77075d47a5c9f983cf32f3956c38f2b544f86f5ef6d"
PROTOCOL = {
    "format": "bnb-tpu.state-protocol.v1",
    "backend_probe_sha256": BACKEND_SHA,
    "workflow": "original Params4bit.from_prequantized then original load_state_dict for weight and bias",
    "state": "all original non-nested state_dict tensor keys; exact restored values",
    "gradient": "FP32 public autograd and independent CPU linear reference; BF16 gradients NOT_RUN",
    "case_selection": "all eight linear cases from the frozen Task2 inputs",
    "process": "save and restore require different PID and process token; external owner proves child origin",
    "cuda_golden": "NOT_RUN",
}
PROTOCOL_SHA = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def backend():
    path = HERE / "probe_backend.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != BACKEND_SHA:
        raise ValueError("BACKEND_SOURCE_IDENTITY")
    spec = importlib.util.spec_from_file_location("state_probe_backend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = backend()


def cases():
    _, inputs = B.load_spec()
    selected = [c for c in inputs["cases"] if c["op"] == "linear"]
    B.require(len(selected) == 8, "STATE_MATRIX")
    return selected


def state_keys(case):
    keys = {"weight", "weight.absmax", "weight.quant_map", "weight.quant_state.bitsandbytes__nf4"}
    if case["bias"] is not None:
        keys.add("bias")
    return keys


def validate_array(value, shape, dtype):
    B.require(value.get("shape") == shape and value.get("dtype") == dtype, "ARRAY_SHAPE_DTYPE")
    values = value.get("values")
    B.require(isinstance(values, list) and len(values) == math.prod(shape), "ARRAY_LENGTH")
    B.require(all(type(v) in (int, float) and math.isfinite(v) for v in values), "ARRAY_NONFINITE")
    if dtype == "uint8":
        B.require(all(type(v) is int and 0 <= v <= 255 for v in values), "ARRAY_UINT8")


def validate_raw(raw, case, tpu):
    base = {key: raw[key] for key in ("case_id", "input_sha256", "metadata", "placements", "counters", "outputs")}
    if tpu:
        base["execution_metrics"] = raw.get("execution_metrics")
    base["outputs"] = {k: raw["outputs"][k] for k in B.expected_outputs(case)}
    base["placements"] = {k: raw["placements"][k] for k in B.expected_outputs(case)}
    B.validate_record(base, case, tpu)
    expected = set(B.expected_outputs(case))
    if case["dtype"] == "float32":
        expected.add("dx")
        validate_array(raw["outputs"].get("dx", {}), case["x_shape"], "float32")
        if case["bias"] is not None:
            expected.add("db")
            validate_array(raw["outputs"].get("db", {}), [case["weight_shape"][0]], "float32")
    B.require(set(raw["outputs"]) == expected and set(raw["placements"]) == expected, "GRADIENT_OUTPUT_SET")
    B.require(all(v.startswith("xla:") if tpu else v == "cpu" for v in raw["placements"].values()), "GRADIENT_PLACEMENT")
    B.require(raw.get("weight_requires_grad") is False and raw.get("weight_grad_is_none") is True,
              "BASE_WEIGHT_GRADIENT")
    B.require(raw.get("packed_before") == raw["outputs"]["packed"] == raw.get("packed_after"), "BASE_WEIGHT_CHANGED")
    B.require(raw.get("original_quant_state_class") is True and
              (raw.get("module_state_alias") is True or raw.get("module_state_absent") is True),
              "QUANT_STATE_IDENTITY")
    state = raw.get("state_dict", {})
    B.require(set(state) == state_keys(case), "STATE_KEYS")
    for name, value in state.items():
        shape = value.get("shape")
        B.require(isinstance(shape, list) and all(type(n) is int and n > 0 for n in shape), "STATE_SHAPE")
        dtype = case["dtype"] if name == "bias" else "float32" if name in ("weight.absmax", "weight.quant_map") else "uint8"
        validate_array(value, shape, dtype)
    B.require(state["weight"] == raw["outputs"]["packed"] and state["weight.absmax"] == raw["outputs"]["absmax"] and
              state["weight.quant_map"] == raw["outputs"]["code"], "STATE_ARRAY_BINDING")
    blob = bytes(state["weight.quant_state.bitsandbytes__nf4"]["values"])
    B.require(state["weight.quant_state.bitsandbytes__nf4"]["shape"] == [len(blob)], "STATE_METADATA_SHAPE")
    B.require(json.loads(blob.decode()) == {"quant_type": "nf4", "blocksize": 64,
              "dtype": case["dtype"], "shape": case["weight_shape"]}, "STATE_METADATA")
    if case["bias"] is not None:
        B.require(state["bias"]["shape"] == [case["weight_shape"][0]] and state["bias"]["values"] == case["bias"],
                  "STATE_BIAS_BINDING")
    return expected


def verify_phase(root, phase, admission_sha, seal_sha=None):
    root = Path(root)
    if seal_sha is not None:
        B.require(B.sha(root / "seal.json") == seal_sha, "PHASE_SEAL_HASH")
    seal = B.read(root / "seal.json")
    profile, _ = B.load_spec()
    B.require(seal.get("status") == "COMPLETE" and seal.get("phase") == phase, "PHASE_TERMINAL")
    B.require(seal.get("protocol_sha256") == PROTOCOL_SHA and seal.get("profile_sha256") == B.PROFILE_SHA and
              seal.get("inputs_sha256") == B.INPUTS_SHA, "PHASE_PROTOCOL")
    B.require(seal.get("source_admission_sha256") == admission_sha and
              seal.get("source_pre") == admission_sha and seal.get("source_post") == admission_sha, "PHASE_SOURCE")
    B.require(seal.get("runtime_lock_sha256") == profile["runtime_lock_sha256"], "PHASE_RUNTIME_LOCK")
    B.require(seal.get("case_ids") == [c["id"] for c in cases()], "PHASE_MATRIX")
    B.require(seal.get("runtime") == profile["runtime"], "PHASE_RUNTIME")
    B.require(seal.get("cuda_golden") == "NOT_RUN", "CUDA_SCOPE")
    B.require(type(seal.get("pid")) is int and seal["pid"] > 0 and isinstance(seal.get("process_token"), str)
              and len(seal["process_token"]) == 32, "PROCESS_IDENTITY")
    B.require(seal.get("public_api") == {"Linear4bit": "bitsandbytes.nn.modules.Linear4bit",
              "Params4bit": "bitsandbytes.nn.modules.Params4bit", "original_class_identity": True}, "PUBLIC_API")
    if phase != "prepare":
        B.require(seal.get("device") == {"type": "xla", "hardware": "TPU", "pjrt": "TPU"}, "TPU_REQUIRED")
        B.require(seal.get("dispatch") == {op: True for op in B.DISPATCH_OPS}, "DISPATCH")
    else:
        B.require(seal.get("oracle_method") == "original CPU public module autograd and independent CPU dense linear", "ORACLE_METHOD")
    B.check_seal(root, "seal.json", seal)
    suffixes = (".json",) if phase == "prepare" else (".json", ".hlo.txt", ".metrics.txt", ".state.pt")
    B.require(set(seal["artifacts"]) == {"raw/" + c["id"] + s for c in cases() for s in suffixes}, "PHASE_ARTIFACTS")
    for case in cases():
        raw = B.read(root / "raw" / (case["id"] + ".json"))
        outputs = validate_raw(raw, case, phase != "prepare")
        if phase == "prepare":
            B.require(set(raw.get("dense_reference", {})) == outputs.intersection({"y", "dx", "db"}), "DENSE_REFERENCE")
            for name, array in raw["dense_reference"].items():
                validate_array(array, raw["outputs"][name]["shape"], case["dtype"])
        else:
            for suffix in (".hlo.txt", ".metrics.txt", ".state.pt"):
                B.require((root / "raw" / (case["id"] + suffix)).stat().st_size > 0, "PHASE_EMPTY_ARTIFACT")
    return seal


def compare_records(actual, reference, case, exact=False):
    profile, _ = B.load_spec()
    gates = {}
    for name, array in actual["outputs"].items():
        tolerance = {"exact": True} if exact or name == "packed" else profile["tolerances"]["fp32_decode"]
        if not exact and name in ("y", "dx", "db"):
            tolerance = profile["tolerances"]["fp32_forward_gradient" if case["dtype"] == "float32" else "bf16_forward_unit_scale"]
        gates[name] = B.numeric(array["values"], reference["outputs"][name]["values"], tolerance)
    return gates


def cpu_reference_result(root):
    rows = []
    for case in cases():
        raw = B.read(Path(root) / "raw" / (case["id"] + ".json"))
        dense = {"outputs": dict(raw["outputs"], **raw["dense_reference"])}
        rows.append({"case_id": case["id"], "gates": compare_records(raw, dense, case)})
    passed = all(g["status"] == "PASS" for row in rows for g in row["gates"].values())
    return {"numerical_status": "PASS" if passed else "FAIL", "cases": rows}


def verify_all(oracle, oracle_sha, saved, saved_sha, restored, admission_sha):
    cpu = verify_phase(oracle, "prepare", admission_sha, oracle_sha)
    before = verify_phase(saved, "save", admission_sha, saved_sha)
    after = verify_phase(restored, "restore", admission_sha)
    B.require(before["pid"] != after["pid"] and before["process_token"] != after["process_token"], "NEW_PROCESS_REQUIRED")
    B.require(before.get("oracle_sha256") == oracle_sha == after.get("oracle_sha256"), "ORACLE_BINDING")
    B.require(after.get("saved_seal_sha256") == saved_sha and after.get("saved_pid") == before["pid"] and
              after.get("saved_process_token") == before["process_token"], "SAVED_BINDING")
    rows = []
    for case in cases():
        relative = "raw/" + case["id"]
        c, s, r = (B.read(Path(root) / (relative + ".json")) for root in (oracle, saved, restored))
        B.require(r.get("fresh_reconstruction") is True and r.get("module_state_alias") is True, "FRESH_STATE_REQUIRED")
        B.require(s["state_dict"] == r["state_dict"], "RESTORED_STATE_CHANGED")
        checkpoint = before["artifacts"][relative + ".state.pt"]
        B.require(r.get("loaded_checkpoint") == checkpoint and after["artifacts"][relative + ".state.pt"] == checkpoint,
                  "CHECKPOINT_BINDING")
        dense = {"outputs": dict(c["outputs"], **c["dense_reference"])}
        rows.append({"case_id": case["id"], "save_vs_cpu": compare_records(s, c, case),
                     "restore_vs_cpu": compare_records(r, c, case), "roundtrip": compare_records(r, s, case, True),
                     "cpu_vs_dense": compare_records(c, dense, case)})
    passed = all(g["status"] == "PASS" for row in rows for group in ("save_vs_cpu", "restore_vs_cpu", "roundtrip", "cpu_vs_dense")
                 for g in row[group].values())
    return {"byte_integrity": "PASS", "numerical_status": "PASS" if passed else "FAIL", "cases": rows,
            "scope": "non-nested state; FP32 gradients; BF16 state and forward; external lifecycle and process-origin proof required",
            "bf16_gradients": "NOT_RUN", "cuda_golden": "NOT_RUN"}


def restore_module(case, state_dict, bnb, torch, device):
    B.require(set(state_dict) == state_keys(case), "CHECKPOINT_STATE_KEYS")
    dtype = getattr(torch, case["dtype"])
    n, k = case["weight_shape"]
    module = bnb.nn.Linear4bit(k, n, bias=case["bias"] is not None, compute_dtype=dtype,
                               compress_statistics=False, quant_type="nf4", quant_storage=torch.uint8).to(dtype=dtype)
    stats = {key.removeprefix("weight."): value for key, value in state_dict.items() if key.startswith("weight.")}
    module.weight = bnb.nn.Params4bit.from_prequantized(state_dict["weight"], dict(stats), requires_grad=False,
                                                        device=device, module=module)
    if module.bias is not None:
        module.bias.data = module.bias.data.to(device)
    params = {key: state_dict[key] for key in ("weight", "bias") if key in state_dict}
    module.load_state_dict(params, strict=True)
    B.require(isinstance(module.weight, bnb.nn.Params4bit) and isinstance(module.weight.quant_state, bnb.functional.QuantState),
              "ORIGINAL_STATE_CLASSES")
    B.require(module.weight.quant_state is module.quant_state, "MODULE_STATE_ALIAS")
    B.require(module.weight.quant_state.dtype == dtype and not module.weight.quant_state.nested,
              "RESTORED_STATE_PROFILE")
    return module.train()


def compute(case, module, torch, bnb, device):
    dtype = getattr(torch, case["dtype"])
    gradients = case["dtype"] == "float32"
    x = torch.tensor(case["x"], dtype=dtype, device=device).reshape(case["x_shape"]).requires_grad_(gradients)
    before = module.weight.data.detach().clone()
    y = module(x)
    if gradients:
        # Fixed dyadic upstream gradient. No corpus or random generator is used.
        dy = torch.tensor([((i % 5) - 2) / 8 for i in range(y.numel())], dtype=dtype, device=device).reshape(y.shape)
        y.backward(dy)
    decoded = bnb.functional.dequantize_4bit(module.weight.data, quant_state=module.weight.quant_state)
    outputs = {"packed": module.weight.data, "absmax": module.weight.quant_state.absmax,
               "code": module.weight.quant_state.code, "decoded": decoded, "y": y}
    if gradients:
        outputs["dx"] = x.grad
        if module.bias is not None:
            outputs["db"] = module.bias.grad
    return outputs, before


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    profile, _ = B.load_spec()
    seal = {"phase": args.command, "status": "PARTIAL", "protocol_sha256": PROTOCOL_SHA,
            "profile_sha256": B.PROFILE_SHA, "inputs_sha256": B.INPUTS_SHA,
            "source_admission_sha256": args.admission_sha256, "runtime_lock_sha256": profile["runtime_lock_sha256"],
            "case_ids": [], "cuda_golden": "NOT_RUN", "pid": os.getpid(), "process_token": uuid.uuid4().hex}
    B.write(output / "seal.json", seal)
    try:
        if args.command != "prepare":
            verify_phase(args.oracle, "prepare", args.admission_sha256, args.oracle_sha256)
            B.require(cpu_reference_result(args.oracle)["numerical_status"] == "PASS", "CPU_REFERENCE_GATES_FAILED")
            seal["oracle_sha256"] = args.oracle_sha256
        if args.command == "restore":
            saved = verify_phase(args.saved, "save", args.admission_sha256, args.saved_sha256)
            B.require(saved["pid"] != os.getpid(), "NEW_PROCESS_REQUIRED")
            seal.update(saved_seal_sha256=args.saved_sha256, saved_pid=saved["pid"], saved_process_token=saved["process_token"])
        admission, roots = B.admit(args.admission, args.admission_sha256)
        seal["source_pre"] = args.admission_sha256
        seal["runtime"] = B.runtime_check(args.command != "prepare")
        import torch
        B.require(torch.__version__ == "2.9.0+cpu", "RUNTIME_LOADED_TORCH")
        if args.command != "prepare":
            import torch_xla
            import torch_xla.core.xla_model as xm
            import torch_xla.debug.metrics as metrics
            B.require(torch_xla.__version__.split("+")[0] == "2.9.0", "RUNTIME_LOADED_XLA")
            device = xm.xla_device()
            B.require(device.type == "xla" and xm.xla_device_hw(device) == "TPU", "ACTUAL_TPU_REQUIRED")
            seal["device"] = {"type": "xla", "hardware": "TPU", "pjrt": "TPU"}
        else:
            device = "cpu"
        import bitsandbytes as bnb
        seal["public_api"] = B.public_identity(bnb, roots)
        B.require(not any(n == "jax" or n.startswith("jax.") or n.startswith("torchax") for n in sys.modules), "UNEXPECTED_JAX_TORCHAX_IMPORT")
        if args.command != "prepare":
            seal["dispatch"] = {op: torch._C._dispatch_has_kernel_for_dispatch_key(op, "XLA") for op in B.DISPATCH_OPS}
            B.require(all(seal["dispatch"].values()), "DISPATCH")
        else:
            seal["oracle_method"] = "original CPU public module autograd and independent CPU dense linear"
            cpu_functions = B.original_cpu_functions(roots["bitsandbytes"])
        for case in cases():
            prefix = output / "raw" / case["id"]
            prefix.parent.mkdir(exist_ok=True)
            dtype = getattr(torch, case["dtype"])
            if args.command == "restore":
                source = Path(args.saved) / "raw" / (case["id"] + ".state.pt")
                state_dict = torch.load(source, map_location="cpu", weights_only=True)
                module = restore_module(case, state_dict, bnb, torch, device)
                prefix.with_suffix(".state.pt").write_bytes(source.read_bytes())
            elif args.command == "prepare":
                weight = torch.tensor(case["weight"], dtype=dtype).reshape(case["weight_shape"])
                packed, absmax = cpu_functions[0](weight, 64, "nf4", torch.uint8)
                state = bnb.functional.QuantState(absmax=absmax, shape=torch.Size(case["weight_shape"]), dtype=dtype,
                          blocksize=64, code=bnb.functional.get_4bit_type("nf4", device="cpu"), quant_type="nf4")
                state_dict = {"weight": packed, **{"weight." + k: v for k, v in state.as_dict(packed=True).items()}}
                if case["bias"] is not None:
                    state_dict["bias"] = torch.tensor(case["bias"], dtype=dtype)
                module = restore_module(case, state_dict, bnb, torch, "cpu")
            else:
                weight = torch.tensor(case["weight"], dtype=dtype).reshape(case["weight_shape"])
                bias = None if case["bias"] is None else torch.tensor(case["bias"], dtype=dtype)
                module = B.cpu_linear_module(case["weight_shape"], dtype, weight, bias, bnb, torch).to(device).train()
            if args.command != "prepare":
                metrics.clear_all()
            outputs, packed_before = compute(case, module, torch, bnb, device)
            raw = {"case_id": case["id"], "input_sha256": B.case_sha(case),
                   "placements": {k: str(v.device) for k, v in outputs.items()},
                   "original_quant_state_class": type(module.weight.quant_state) is bnb.functional.QuantState,
                   "module_state_alias": module.weight.quant_state is module.quant_state,
                   "module_state_absent": module.quant_state is None,
                   "weight_requires_grad": module.weight.requires_grad, "weight_grad_is_none": module.weight.grad is None,
                   "metadata": {"shape": list(module.weight.quant_state.shape),
                                "dtype": str(module.weight.quant_state.dtype).removeprefix("torch."),
                                "blocksize": module.weight.quant_state.blocksize,
                                "quant_type": module.weight.quant_state.quant_type, "nested": module.weight.quant_state.nested,
                                "packing_format_for_cpu": bool(getattr(module.weight.quant_state, "packing_format_for_cpu", False))}}
            if args.command != "prepare":
                prefix.with_suffix(".hlo.txt").write_text(torch_xla._XLAC._get_xla_tensors_hlo(list(outputs.values())))
                xm.mark_step(wait=True)
                xm.wait_device_ops()
                raw["counters"] = {k: metrics.counter_value(k) for k in metrics.counter_names()}
                raw["execution_metrics"] = {k: list(data) for k in profile["execution_metrics"]
                                             if (data := metrics.metric_data(k)) is not None}
                prefix.with_suffix(".metrics.txt").write_text(metrics.metrics_report())
            else:
                raw["counters"] = {}
            raw["outputs"] = {k: B.tensor_record(v) for k, v in outputs.items()}
            raw["packed_before"] = B.tensor_record(packed_before)
            raw["packed_after"] = B.tensor_record(module.weight.data)
            original_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            raw["state_dict"] = {k: B.tensor_record(v) for k, v in original_state.items()}
            if args.command == "save":
                torch.save(original_state, prefix.with_suffix(".state.pt"))
            elif args.command == "restore":
                raw["fresh_reconstruction"] = True
                raw["loaded_checkpoint"] = saved["artifacts"]["raw/" + case["id"] + ".state.pt"]
            else:
                x = torch.tensor(case["x"], dtype=dtype).reshape(case["x_shape"]).requires_grad_(case["dtype"] == "float32")
                bias = None if case["bias"] is None else torch.tensor(case["bias"], dtype=dtype, requires_grad=case["dtype"] == "float32")
                dense_y = torch.nn.functional.linear(x, outputs["decoded"].detach(), bias)
                dense = {"y": dense_y}
                if case["dtype"] == "float32":
                    dy = torch.tensor([((i % 5) - 2) / 8 for i in range(dense_y.numel())], dtype=dtype).reshape(dense_y.shape)
                    dense_y.backward(dy)
                    dense["dx"] = x.grad
                    if bias is not None:
                        dense["db"] = bias.grad
                raw["dense_reference"] = {k: B.tensor_record(v) for k, v in dense.items()}
            B.write(prefix.with_suffix(".json"), raw)
            seal["case_ids"].append(case["id"])
            validate_raw(raw, case, args.command != "prepare")
        for package, root in roots.items():
            B.verify_source_tree(root, admission[package]["files"])
        seal["source_post"] = args.admission_sha256
        seal["status"] = "COMPLETE"
        seal["artifacts"] = B.inventory(output, "seal.json")
        B.write(output / "seal.json", seal)
        verify_phase(output, args.command, args.admission_sha256)
        if args.command == "restore":
            result = verify_all(args.oracle, args.oracle_sha256, args.saved, args.saved_sha256, output, args.admission_sha256)
            print(json.dumps(result, allow_nan=False))
            return 0 if result["numerical_status"] == "PASS" else 2
        result = {"status": "SEALED", "phase": args.command, "seal_sha256": B.sha(output / "seal.json")}
        if args.command == "prepare":
            result["cpu_reference"] = cpu_reference_result(output)
            print(json.dumps(result, allow_nan=False))
            return 0 if result["cpu_reference"]["numerical_status"] == "PASS" else 2
        print(json.dumps(result))
        return 0
    except Exception:
        (output / "error.log").write_text(traceback.format_exc())
        seal["status"] = "FAILED"
        seal["artifacts"] = B.inventory(output, "seal.json")
        B.write(output / "seal.json", seal)
        print(json.dumps({"status": "NOT_QUALIFIED", "reason": "See retained error.log"}))
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for phase in ("prepare", "save", "restore", "verify"):
        child = sub.add_parser(phase)
        child.add_argument("--admission-sha256", required=True)
        if phase != "verify":
            child.add_argument("--admission", required=True)
            child.add_argument("--output", required=True)
        else:
            child.add_argument("--restored", required=True)
        if phase != "prepare":
            child.add_argument("--oracle", required=True)
            child.add_argument("--oracle-sha256", required=True)
        if phase in ("restore", "verify"):
            child.add_argument("--saved", required=True)
            child.add_argument("--saved-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "verify":
        try:
            result = verify_all(args.oracle, args.oracle_sha256, args.saved, args.saved_sha256, args.restored, args.admission_sha256)
            print(json.dumps(result, allow_nan=False))
            return 0 if result["numerical_status"] == "PASS" else 2
        except Exception as error:
            print(json.dumps({"status": "NOT_QUALIFIED", "reason": str(error)}))
            return 1
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
