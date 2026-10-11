"""Run the first NF4 API slice, or verify its retained artifacts.

The owner must set the process time limit. This worker does not allocate a TPU.
"""

import argparse
import ast
import copy
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import traceback

HERE = Path(__file__).resolve().parent
PROFILE_SHA = "ad53f6416bdabf6440f08b313963c947a468cd39afd413bfe41bb381c4a7a166"
INPUTS_SHA = "d0c9cdf6a82449a56923499698f19bfd3cc8d13c042f1f5ad2bdf587701f5e54"
ORACLE_METHOD = "pinned default quantize/dequantize bodies and CPU torch.nn.functional.linear"
DISPATCH_OPS = (
    "bitsandbytes::quantize_4bit", "bitsandbytes::dequantize_4bit",
    "bitsandbytes::dequantize_4bit.out", "bitsandbytes::gemm_4bit",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def case_sha(case):
    return hashlib.sha256(json.dumps(case, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_spec():
    require(sha(HERE / "probe-profile.json") == PROFILE_SHA, "PROFILE_IDENTITY")
    require(sha(HERE / "probe-inputs.json") == INPUTS_SHA, "INPUT_IDENTITY")
    profile, inputs = read(HERE / "probe-profile.json"), read(HERE / "probe-inputs.json")
    require(profile["case_ids"] == [c["id"] for c in inputs["cases"]] and len(profile["case_ids"]) == 42,
            "MATRIX_IDENTITY")
    return profile, inputs


def inventory(root, exclude):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "INVENTORY_ROOT")
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "SYMLINK")
        if path.is_file() and path.relative_to(root).as_posix() != exclude:
            result[path.relative_to(root).as_posix()] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return result


def check_seal(root, name, seal):
    observed = inventory(root, name)
    expected = seal.get("artifacts", {})
    require(set(observed) == set(expected), "INVENTORY_INCOMPLETE_OR_EXTRA")
    require(observed == expected, "INTEGRITY_BYTES")


def verify_source_tree(root, files):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "SOURCE_ROOT")
    require(isinstance(files, dict) and files, "SOURCE_INVENTORY_EMPTY")
    for relative, expected in files.items():
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts and path.suffix == ".py", "SOURCE_PATH")
        require(isinstance(expected, str) and len(expected) == 64, "SOURCE_HASH")
    observed = {}
    for path in root.rglob("*"):
        require(not path.is_symlink(), "SOURCE_SYMLINK")
        require(path.suffix != ".pyc", "SOURCE_CACHE_PRESENT")
        if path.is_file() and path.suffix == ".py":
            observed[path.relative_to(root).as_posix()] = sha(path)
    require(observed == files, "SOURCE_BYTES_OR_INVENTORY")


def expected_outputs(case):
    n, k = case["weight_shape"]
    length = n * k
    outputs = {"packed": ([(length + 1) // 2, 1], "uint8"),
               "absmax": ([(length + 63) // 64], "float32"),
               "code": ([16], "float32"),
               "decoded": ([k, n] if length <= 2 else [n, k], case["dtype"])}
    if case["op"] == "linear":
        outputs["y"] = (case["x_shape"][:-1] + [n], case["dtype"])
    return outputs


def validate_record(record, case, tpu):
    require(record.get("case_id") == case["id"] and record.get("input_sha256") == case_sha(case), "INPUT_BINDING")
    n, k = case["weight_shape"]
    expected_meta = {"shape": [n, k], "dtype": case["dtype"], "blocksize": 64,
                     "quant_type": "nf4", "nested": False, "packing_format_for_cpu": False}
    require(record.get("metadata") == expected_meta, "OUTPUT_METADATA")
    expected = expected_outputs(case)
    require(set(record.get("outputs", {})) == set(expected), "OUTPUT_SET")
    require(set(record.get("placements", {})) == set(expected), "PLACEMENT_SET")
    for name, (shape, dtype) in expected.items():
        array = record["outputs"][name]
        require(array.get("shape") == shape and array.get("dtype") == dtype, "OUTPUT_SHAPE_DTYPE")
        values = array.get("values")
        require(isinstance(values, list) and len(values) == math.prod(shape), "OUTPUT_SIZE")
        require(all(type(v) in (int, float) and math.isfinite(v) for v in values), "NONFINITE_OUTPUT")
        if dtype == "uint8":
            require(all(type(v) is int and 0 <= v <= 255 for v in values), "OUTPUT_UINT8")
        placement = record["placements"][name]
        require(placement.startswith("xla:") if tpu else placement == "cpu", "PLACEMENT")
    if tpu:
        counters = record.get("counters")
        require(isinstance(counters, dict) and all(type(v) is int and v >= 0 for v in counters.values()), "COUNTERS_UNKNOWN")
        require(not any(k.startswith("aten::") and v for k, v in counters.items()), "CPU_FALLBACK")
        execution = record.get("execution_metrics")
        allowed = {"ExecuteTime", "ExecuteReplicatedTime"}
        require(isinstance(execution, dict) and set(execution).issubset(allowed), "EXECUTION_METRICS")
        for data in execution.values():
            require(isinstance(data, list) and len(data) == 3 and type(data[0]) is int and data[0] >= 0,
                    "EXECUTION_METRICS")
            require(type(data[1]) in (int, float) and math.isfinite(data[1]) and data[1] >= 0,
                    "EXECUTION_METRICS")
        require(any(data[0] > 0 for data in execution.values()), "EXECUTION_NOT_OBSERVED")


def numeric(actual, reference, tolerance):
    errors = [abs(a - b) for a, b in zip(actual, reference)]
    exact = tolerance.get("exact", False)
    passed = all(a == b for a, b in zip(actual, reference)) if exact else all(
        e <= tolerance["atol"] + tolerance["rtol"] * abs(b) for e, b in zip(errors, reference))
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    nrmse = rmse / max(math.sqrt(sum(b * b for b in reference) / len(reference)), 1e-12)
    if "nrmse_max" in tolerance:
        passed = passed and nrmse <= tolerance["nrmse_max"]
    return {"status": "PASS" if passed else "FAIL", "max_abs": max(errors), "rmse": rmse, "nrmse": nrmse}


def verify_oracle(root, expected_sha, admission_sha):
    profile, inputs = load_spec()
    require(Path(root).is_dir() and (Path(root) / "oracle-seal.json").is_file(), "ORACLE_MISSING")
    require(sha(Path(root) / "oracle-seal.json") == expected_sha, "ORACLE_SEAL_HASH")
    seal = read(Path(root) / "oracle-seal.json")
    require(seal.get("kind") == "CPU_ORACLE" and seal.get("status") == "COMPLETE", "ORACLE_TERMINAL")
    require(seal.get("oracle_method") == ORACLE_METHOD, "ORACLE_METHOD")
    require(seal.get("runtime_lock_sha256") == profile["runtime_lock_sha256"], "RUNTIME_LOCK")
    require(seal.get("profile_sha256") == PROFILE_SHA and seal.get("inputs_sha256") == INPUTS_SHA, "ORACLE_PROFILE")
    require(seal.get("source_admission_sha256") == admission_sha, "ORACLE_SOURCE")
    require(seal.get("runtime") == {"python": "3.12", "torch": "2.9.0+cpu"}, "ORACLE_RUNTIME")
    require(seal.get("case_ids") == profile["case_ids"], "MATRIX_INCOMPLETE")
    check_seal(root, "oracle-seal.json", seal)
    required = {"raw/" + c["id"] + ".json" for c in inputs["cases"]}
    require(set(seal["artifacts"]) == required, "ORACLE_OUTPUT_INVENTORY")
    for case in inputs["cases"]:
        validate_record(read(Path(root) / "raw" / (case["id"] + ".json")), case, False)
    return seal


def verify_artifacts(actual, oracle, oracle_sha, admission_sha):
    profile, inputs = load_spec()
    verify_oracle(oracle, oracle_sha, admission_sha)
    receipt = read(Path(actual) / "receipt.json")
    require(receipt.get("kind") == "TPU_PROBE" and receipt.get("status") == "COMPLETE", "TERMINAL_INCOMPLETE")
    require(receipt.get("profile_sha256") == PROFILE_SHA and receipt.get("inputs_sha256") == INPUTS_SHA, "PROFILE_BINDING")
    require(receipt.get("source_admission_sha256") == admission_sha and
            receipt.get("source_pre") == admission_sha and receipt.get("source_post") == admission_sha, "SOURCE_BINDING")
    require(receipt.get("oracle_sha256") == oracle_sha, "ORACLE_BINDING")
    require(receipt.get("case_ids") == profile["case_ids"], "MATRIX_INCOMPLETE")
    require(receipt.get("runtime") == profile["runtime"], "RUNTIME_PAIR")
    require(receipt.get("runtime_lock_sha256") == profile["runtime_lock_sha256"], "RUNTIME_LOCK")
    require(receipt.get("device") == {"type": "xla", "hardware": "TPU", "pjrt": "TPU"}, "TPU_DEVICE_REQUIRED")
    require(receipt.get("dispatch") == {name: True for name in DISPATCH_OPS}, "DISPATCH_INCOMPLETE")
    require(receipt.get("public_api") == {
        "Linear4bit": "bitsandbytes.nn.modules.Linear4bit", "Params4bit": "bitsandbytes.nn.modules.Params4bit",
        "original_class_identity": True}, "PUBLIC_API_SUBSTITUTE")
    require(receipt.get("cuda_golden") == "NOT_RUN", "CUDA_SCOPE")
    check_seal(actual, "receipt.json", receipt)
    required = {"raw/" + c["id"] + suffix for c in inputs["cases"] for suffix in (".json", ".hlo.txt", ".metrics.txt")}
    require(set(receipt["artifacts"]) == required, "OUTPUT_INVENTORY")
    results = []
    for case in inputs["cases"]:
        observed = read(Path(actual) / "raw" / (case["id"] + ".json"))
        reference = read(Path(oracle) / "raw" / (case["id"] + ".json"))
        validate_record(observed, case, True)
        for suffix in (".hlo.txt", ".metrics.txt"):
            require((Path(actual) / "raw" / (case["id"] + suffix)).stat().st_size > 0, "OUTPUT_EMPTY_DIAGNOSTIC")
        gates = {}
        for key in observed["outputs"]:
            tolerance = {"exact": True} if key == "packed" else profile["tolerances"]["fp32_decode"]
            if key == "y":
                tolerance = profile["tolerances"]["fp32_forward_gradient" if case["dtype"] == "float32" else "bf16_forward_unit_scale"]
            gates[key] = numeric(observed["outputs"][key]["values"], reference["outputs"][key]["values"], tolerance)
        results.append({"case_id": case["id"], "gates": gates})
    passed = all(g["status"] == "PASS" for row in results for g in row["gates"].values())
    return {"byte_integrity": "PASS", "numerical_status": "PASS" if passed else "FAIL", "cases": results,
            "scope": "first non-nested vertical slice; owner lifecycle and physical-device proof require separate review",
            "cuda_golden": "NOT_RUN"}


def admit(path, expected_sha):
    profile, _ = load_spec()
    require(sha(path) == expected_sha, "SOURCE_ADMISSION_HASH")
    admission = read(path)
    require(admission.get("format") == "bnb-tpu.probe-source-admission.v1", "SOURCE_ADMISSION_FORMAT")
    require(admission.get("runtime_lock_sha256") == profile["runtime_lock_sha256"], "RUNTIME_LOCK")
    require(admission["bitsandbytes"].get("commit") == profile["source_commit"], "SOURCE_COMMIT")
    require(all(admission["bitsandbytes"]["files"].get(p) == h for p, h in profile["bnb_files"].items()), "SOURCE_CANONICAL_PIN")
    for name in ("bitsandbytes", "bitsandbytes_tpu"):
        require(name not in sys.modules, "SOURCE_ALREADY_IMPORTED")
    roots = {}
    for name in ("bitsandbytes", "bitsandbytes_tpu"):
        spec = importlib.util.find_spec(name)
        require(spec is not None and spec.origin is not None, "SOURCE_PACKAGE_MISSING_" + name)
        roots[name] = Path(spec.origin).parent
        verify_source_tree(roots[name], admission[name]["files"])
    return admission, roots


def runtime_check(tpu):
    profile, _ = load_spec()
    require(sys.platform == "linux" and platform.machine() == "x86_64", "RUNTIME_LINUX_X86_64_REQUIRED")
    require(platform.python_implementation() == "CPython" and sys.version_info[:2] == (3, 12), "RUNTIME_PYTHON")
    require(sys.flags.dont_write_bytecode, "RUNTIME_START_WITH_B")
    require(not any(name == "jax" or name.startswith("jax.") or name.startswith("torchax") for name in sys.modules), "RUNTIME_JAX_ALREADY_IMPORTED")
    versions = {"python": "3.12"}
    for name in ("torch", "torch_xla", "libtpu"):
        versions[name] = metadata.version(name)
    require(versions == profile["runtime"], "RUNTIME_PAIR")
    if tpu:
        import os
        require(os.environ.get("PJRT_DEVICE") == "TPU", "RUNTIME_PJRT_TPU_REQUIRED")
    return versions


def public_identity(bnb, roots):
    module = importlib.import_module("bitsandbytes.nn.modules")
    require(Path(module.__file__).resolve() == (roots["bitsandbytes"] / "nn/modules.py").resolve(), "PUBLIC_API_SOURCE")
    require(bnb.nn.Linear4bit is module.Linear4bit and bnb.nn.Params4bit is module.Params4bit and
            module.Linear4bit.__module__ == "bitsandbytes.nn.modules" and
            module.Params4bit.__module__ == "bitsandbytes.nn.modules", "PUBLIC_API_SUBSTITUTE")
    return {"Linear4bit": "bitsandbytes.nn.modules.Linear4bit", "Params4bit": "bitsandbytes.nn.modules.Params4bit",
            "original_class_identity": True}


def original_cpu_functions(root):
    """Remove registration only. Run the pinned default math on CPU."""
    default = importlib.import_module("bitsandbytes.backends.default.ops")
    tree = ast.parse((root / "backends/default/ops.py").read_text())
    candidates = [n for n in tree.body if isinstance(n, ast.FunctionDef) and any(
        isinstance(d, ast.Call) and len(d.args) >= 2 and isinstance(d.args[0], ast.Constant)
        and d.args[0].value == "bitsandbytes::quantize_4bit" and isinstance(d.args[1], ast.Constant)
        and d.args[1].value == "default" for d in n.decorator_list)]
    require(len(candidates) == 1, "ORACLE_DEFAULT_FUNCTION_NOT_FOUND")
    function = copy.deepcopy(candidates[0])
    function.decorator_list = []
    namespace = dict(vars(default))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "pinned-default-quantize", "exec"), namespace)
    decode = default._dequantize_4bit_compute.__wrapped__
    return namespace[function.name], decode


def tensor_record(tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch."),
            "values": tensor.detach().cpu().reshape(-1).tolist()}


def cpu_linear_module(shape, dtype, weight, bias, bnb, torch):
    module = bnb.nn.Linear4bit(shape[1], shape[0], bias=bias is not None, compute_dtype=dtype,
                               compress_statistics=False, quant_type="nf4", quant_storage=torch.uint8)
    module = module.to(dtype=dtype)
    require(isinstance(module.weight, bnb.nn.Params4bit) and module.weight.dtype == dtype,
            "PUBLIC_API_FLOAT_WEIGHT_DTYPE")
    module.weight.data.copy_(weight)
    if bias is not None:
        module.bias.data.copy_(bias)
    return module


def cpu_public_decode(decode, packed, state, shape, dtype):
    restored = decode(packed.reshape(-1), state.absmax, state.code, 64, shape, dtype)
    return restored.t() if packed.shape[0] == 1 else restored


def run_case(case, torch, bnb, device, cpu_functions=None):
    dtype = getattr(torch, case["dtype"])
    shape = case["weight_shape"]
    bias = None if case.get("bias") is None else torch.tensor(case["bias"], dtype=dtype)
    y = None
    if case["op"] == "decode":
        packed = torch.tensor(case["packed"], dtype=torch.uint8, device=device).reshape(-1, 1)
        absmax = torch.tensor(case["absmax"], dtype=torch.float32, device=device)
        state = bnb.functional.QuantState(absmax=absmax, shape=torch.Size(shape), dtype=dtype, blocksize=64,
                                          code=bnb.functional.get_4bit_type("nf4", device=device), quant_type="nf4")
    else:
        weight = torch.tensor(case["weight"], dtype=dtype).reshape(shape)
        if cpu_functions:
            packed, absmax = cpu_functions[0](weight, 64, "nf4", torch.uint8)
            state = bnb.functional.QuantState(absmax=absmax, shape=torch.Size(shape), dtype=dtype, blocksize=64,
                                              code=bnb.functional.get_4bit_type("nf4", device="cpu"), quant_type="nf4")
        elif case["op"] == "linear":
            module = cpu_linear_module(shape, dtype, weight, bias, bnb, torch)
            module = module.to(device)
            require(isinstance(module.weight, bnb.nn.Params4bit) and module.weight.dtype == torch.uint8,
                    "PUBLIC_API_PARAMS4BIT_LOST")
            require(module.weight.quant_state.dtype == dtype and not module.weight.quant_state.nested,
                    "PUBLIC_API_QUANT_STATE_DTYPE")
            module.eval()
            x = torch.tensor(case["x"], dtype=dtype, device=device).reshape(case["x_shape"])
            with torch.no_grad():
                y = module(x)
            packed, state = module.weight.data, module.weight.quant_state
        else:
            packed, state = bnb.functional.quantize_4bit(weight.to(device), blocksize=64,
                                                        compress_statistics=False, quant_type="nf4", quant_storage=torch.uint8)
    if cpu_functions:
        decoded = cpu_public_decode(cpu_functions[1], packed, state, shape, dtype)
        if case["op"] == "linear":
            x = torch.tensor(case["x"], dtype=dtype).reshape(case["x_shape"])
            y = torch.nn.functional.linear(x, decoded, bias)
    else:
        decoded = bnb.functional.dequantize_4bit(packed, quant_state=state)
    outputs = {"packed": packed, "absmax": state.absmax, "code": state.code, "decoded": decoded}
    if y is not None:
        outputs["y"] = y
    record = {"case_id": case["id"], "input_sha256": case_sha(case),
              "metadata": {"shape": list(state.shape), "dtype": str(state.dtype).removeprefix("torch."),
                           "blocksize": state.blocksize, "quant_type": state.quant_type, "nested": state.nested,
                           "packing_format_for_cpu": bool(getattr(state, "packing_format_for_cpu", False))},
              "placements": {name: str(tensor.device) for name, tensor in outputs.items()}}
    return outputs, record


def run(args):
    profile, inputs = load_spec()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    name = "oracle-seal.json" if args.command == "prepare" else "receipt.json"
    record = {"kind": "CPU_ORACLE" if args.command == "prepare" else "TPU_PROBE", "status": "PARTIAL",
              "profile_sha256": PROFILE_SHA, "inputs_sha256": INPUTS_SHA,
              "source_admission_sha256": args.admission_sha256,
              "runtime_lock_sha256": profile["runtime_lock_sha256"], "case_ids": [], "cuda_golden": "NOT_RUN"}
    try:
        if args.command == "execute":
            verify_oracle(args.oracle, args.oracle_sha256, args.admission_sha256)
            record["oracle_sha256"] = args.oracle_sha256
        admission, roots = admit(args.admission, args.admission_sha256)
        versions = runtime_check(args.command == "execute")
        import torch
        require(torch.__version__ == "2.9.0+cpu", "RUNTIME_LOADED_TORCH")
        if args.command == "execute":
            import torch_xla
            import torch_xla.core.xla_model as xm
            import torch_xla.debug.metrics as metrics
            device = xm.xla_device()
            require(device.type == "xla" and xm.xla_device_hw(device) == "TPU", "ACTUAL_TPU_REQUIRED")
            require(torch_xla.__version__.split("+")[0] == "2.9.0", "RUNTIME_LOADED_XLA")
        else:
            device = "cpu"
        import bitsandbytes as bnb
        record["public_api"] = public_identity(bnb, roots)
        require(not any(n == "jax" or n.startswith("jax.") or n.startswith("torchax") for n in sys.modules), "UNEXPECTED_JAX_TORCHAX_IMPORT")
        if args.command == "execute":
            record["runtime"] = versions
            record["device"] = {"type": device.type, "hardware": xm.xla_device_hw(device), "pjrt": "TPU"}
            record["dispatch"] = {op: torch._C._dispatch_has_kernel_for_dispatch_key(op, "XLA") for op in DISPATCH_OPS}
            require(all(record["dispatch"].values()), "DISPATCH_INCOMPLETE")
            record["source_pre"] = args.admission_sha256
            cpu_functions = None
        else:
            record["runtime"] = {"python": versions["python"], "torch": versions["torch"]}
            record["oracle_method"] = ORACLE_METHOD
            cpu_functions = original_cpu_functions(roots["bitsandbytes"])
        for case in inputs["cases"]:
            if args.command == "execute":
                metrics.clear_all()
            outputs, raw = run_case(case, torch, bnb, device, cpu_functions)
            if args.command == "execute":
                tensors = list(outputs.values())
                hlo = torch_xla._XLAC._get_xla_tensors_hlo(tensors)
                (output / "raw").mkdir(exist_ok=True)
                (output / "raw" / (case["id"] + ".hlo.txt")).write_text(hlo)
                xm.mark_step(wait=True)
                xm.wait_device_ops()
                raw["counters"] = {key: metrics.counter_value(key) for key in metrics.counter_names()}
                raw["execution_metrics"] = {key: list(data) for key in profile["execution_metrics"]
                                             if (data := metrics.metric_data(key)) is not None}
                (output / "raw" / (case["id"] + ".metrics.txt")).write_text(metrics.metrics_report())
            else:
                raw["counters"] = {}
            raw["outputs"] = {key: tensor_record(tensor) for key, tensor in outputs.items()}
            write(output / "raw" / (case["id"] + ".json"), raw)
            record["case_ids"].append(case["id"])
            validate_record(raw, case, args.command == "execute")
        for package, root in roots.items():
            verify_source_tree(root, admission[package]["files"])
        if args.command == "execute":
            record["source_post"] = args.admission_sha256
        require(not any(n == "jax" or n.startswith("jax.") for n in sys.modules), "UNEXPECTED_JAX_IMPORT")
        record["status"] = "COMPLETE"
        record["artifacts"] = inventory(output, name)
        write(output / name, record)
        if args.command == "execute":
            audit = verify_artifacts(output, args.oracle, args.oracle_sha256, args.admission_sha256)
            # Print the derivative. Keep the original raw inventory immutable.
            print(json.dumps(audit, allow_nan=False))
            return 0 if audit["numerical_status"] == "PASS" else 2
        print(json.dumps({"oracle_status": "SEALED", "oracle_sha256": sha(output / name), "cases": len(inputs["cases"])}))
        return 0
    except Exception:
        (output / "error.log").write_text(traceback.format_exc())
        record["status"] = "FAILED"
        record["artifacts"] = inventory(output, name)
        write(output / name, record)
        print(json.dumps({"status": "NOT_QUALIFIED", "reason": "See retained error.log"}))
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "execute", "verify"):
        command = sub.add_parser(name)
        command.add_argument("--admission-sha256", required=True)
        if name != "verify":
            command.add_argument("--admission", required=True)
            command.add_argument("--output", required=True)
        else:
            command.add_argument("--actual", required=True)
        if name != "prepare":
            command.add_argument("--oracle", required=True)
            command.add_argument("--oracle-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "verify":
        try:
            audit = verify_artifacts(args.actual, args.oracle, args.oracle_sha256, args.admission_sha256)
            print(json.dumps(audit, allow_nan=False))
            return 0 if audit["numerical_status"] == "PASS" else 2
        except Exception as error:
            print(json.dumps({"status": "NOT_QUALIFIED", "reason": str(error)}))
            return 1
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
