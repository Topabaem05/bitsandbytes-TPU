"""Synthetic state reader tests. These records do not show actual execution."""

import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import traceback
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "experiments/2026-10-04-bitsandbytes-tpu/probe_state.py"
spec = importlib.util.spec_from_file_location("state_probe_controls", ENTRY)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def seal(root, record):
    record["artifacts"] = P.B.inventory(root, "seal.json")
    P.B.write(root / "seal.json", record)
    return P.B.sha(root / "seal.json")


class StateControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.oracle, self.saved, self.restored = [self.base / n for n in ("oracle", "saved", "restored")]
        self.admission = "a" * 64
        profile, _ = P.B.load_spec()
        self.records = {}
        for phase, root, pid in (("prepare", self.oracle, 1001), ("save", self.saved, 1002), ("restore", self.restored, 1003)):
            root.mkdir()
            record = {"phase": phase, "status": "COMPLETE", "protocol_sha256": P.PROTOCOL_SHA,
                      "profile_sha256": P.B.PROFILE_SHA, "inputs_sha256": P.B.INPUTS_SHA,
                      "source_admission_sha256": self.admission, "source_pre": self.admission, "source_post": self.admission,
                      "runtime_lock_sha256": profile["runtime_lock_sha256"], "runtime": profile["runtime"],
                      "case_ids": [c["id"] for c in P.cases()], "cuda_golden": "NOT_RUN", "pid": pid,
                      "process_token": str(pid).zfill(32),
                      "public_api": {"Linear4bit": "bitsandbytes.nn.modules.Linear4bit",
                                     "Params4bit": "bitsandbytes.nn.modules.Params4bit", "original_class_identity": True}}
            if phase == "prepare":
                record["oracle_method"] = "original CPU public module autograd and independent CPU dense linear"
            else:
                record["device"] = {"type": "xla", "hardware": "TPU", "pjrt": "TPU"}
                record["dispatch"] = {op: True for op in P.B.DISPATCH_OPS}
            for case in P.cases():
                expected = dict(P.B.expected_outputs(case))
                if case["dtype"] == "float32":
                    expected["dx"] = (case["x_shape"], "float32")
                    if case["bias"] is not None:
                        expected["db"] = ([case["weight_shape"][0]], "float32")
                outputs = {n: {"shape": shape, "dtype": dtype, "values": [0 if dtype == "uint8" else 0.0] * math.prod(shape)}
                           for n, (shape, dtype) in expected.items()}
                blob = list(json.dumps({"quant_type": "nf4", "blocksize": 64, "dtype": case["dtype"],
                                        "shape": case["weight_shape"]}).encode())
                state = {"weight": outputs["packed"], "weight.absmax": outputs["absmax"], "weight.quant_map": outputs["code"],
                         "weight.quant_state.bitsandbytes__nf4": {"shape": [len(blob)], "dtype": "uint8", "values": blob}}
                if case["bias"] is not None:
                    state["bias"] = {"shape": [case["weight_shape"][0]], "dtype": case["dtype"],
                                     "values": case["bias"]}
                raw = {"case_id": case["id"], "input_sha256": P.B.case_sha(case), "outputs": outputs,
                       "placements": {n: "cpu" if phase == "prepare" else "xla:0" for n in outputs},
                       "metadata": {"shape": case["weight_shape"], "dtype": case["dtype"], "blocksize": 64,
                                    "quant_type": "nf4", "nested": False, "packing_format_for_cpu": False},
                       "counters": {}, "execution_metrics": {"ExecuteTime": [1, 0.01, [[0, 0.01]]]},
                       "original_quant_state_class": True, "module_state_alias": True,
                       "weight_requires_grad": False, "weight_grad_is_none": True,
                       "packed_before": outputs["packed"], "packed_after": outputs["packed"], "state_dict": state}
                if phase == "prepare":
                    raw["dense_reference"] = {k: v for k, v in outputs.items() if k in ("y", "dx", "db")}
                elif phase == "restore":
                    raw["fresh_reconstruction"] = True
                    raw["loaded_checkpoint"] = self.records["save"]["artifacts"]["raw/" + case["id"] + ".state.pt"]
                P.B.write(root / "raw" / (case["id"] + ".json"), raw)
                if phase != "prepare":
                    for extension in (".hlo.txt", ".metrics.txt", ".state.pt"):
                        (root / "raw" / (case["id"] + extension)).write_text("SYNTHETIC ONLY\n")
            if phase != "prepare":
                record["oracle_sha256"] = self.oracle_sha
            if phase == "restore":
                record.update(saved_seal_sha256=self.saved_sha, saved_pid=1002, saved_process_token=str(1002).zfill(32))
            identity = seal(root, record)
            self.records[phase] = record
            if phase == "prepare":
                self.oracle_sha = identity
            elif phase == "save":
                self.saved_sha = identity

    def verify(self):
        return P.verify_all(self.oracle, self.oracle_sha, self.saved, self.saved_sha, self.restored, self.admission)

    def mutate_seal(self, key, value):
        self.records["restore"][key] = value
        seal(self.restored, self.records["restore"])

    def mutate_raw(self, function):
        path = self.restored / "raw" / (P.cases()[0]["id"] + ".json")
        value = P.B.read(path)
        function(value)
        P.B.write(path, value)
        seal(self.restored, self.records["restore"])

    def test_complete_fixture(self):
        result = self.verify()
        self.assertEqual(result["byte_integrity"], "PASS")
        self.assertEqual(result["numerical_status"], "PASS")
        self.assertEqual(len(result["cases"]), 8)
        self.assertEqual(result["bf16_gradients"], "NOT_RUN")

    def test_cpu_dense_mismatch_remains_fail(self):
        path = self.oracle / "raw" / (P.cases()[0]["id"] + ".json")
        raw = P.B.read(path)
        raw["dense_reference"]["dx"]["values"][0] = 10.0
        P.B.write(path, raw)
        self.assertEqual(P.cpu_reference_result(self.oracle)["numerical_status"], "FAIL")

    def test_same_pid_rejected(self):
        self.mutate_seal("pid", 1002)
        with self.assertRaisesRegex(ValueError, "NEW_PROCESS"):
            self.verify()

    def test_reused_process_token_rejected(self):
        self.mutate_seal("process_token", str(1002).zfill(32))
        with self.assertRaisesRegex(ValueError, "NEW_PROCESS"):
            self.verify()

    def test_absent_module_alias_before_save_is_valid(self):
        # The pinned forward uses weight.quant_state when module.quant_state is None.
        path = self.saved / "raw" / (P.cases()[0]["id"] + ".json")
        raw = P.B.read(path)
        raw.update(module_state_alias=False, module_state_absent=True)
        P.B.write(path, raw)
        self.saved_sha = seal(self.saved, self.records["save"])
        self.mutate_seal("saved_seal_sha256", self.saved_sha)
        self.assertEqual(self.verify()["numerical_status"], "PASS")

    def test_wrong_bias_rejected(self):
        path = self.restored / "raw/linear-float32-rank2-bias1.json"
        raw = P.B.read(path)
        raw["state_dict"]["bias"]["values"][0] += 1
        P.B.write(path, raw)
        seal(self.restored, self.records["restore"])
        with self.assertRaisesRegex(ValueError, "BIAS_BINDING"):
            self.verify()

    def test_wrong_source_rejected(self):
        self.mutate_seal("source_post", "b" * 64)
        with self.assertRaisesRegex(ValueError, "SOURCE"):
            self.verify()

    def test_cpu_fallback_rejected(self):
        self.mutate_raw(lambda r: r["counters"].update({"aten::bucketize": 1}))
        with self.assertRaisesRegex(ValueError, "FALLBACK"):
            self.verify()

    def test_cpu_device_rejected(self):
        self.mutate_seal("device", {"type": "xla", "hardware": "CPU", "pjrt": "TPU"})
        with self.assertRaisesRegex(ValueError, "TPU"):
            self.verify()

    def test_missing_gradient_rejected(self):
        self.mutate_raw(lambda r: r["outputs"].pop("dx"))
        with self.assertRaisesRegex(ValueError, "ARRAY"):
            self.verify()

    def test_state_metadata_rejected(self):
        def mutate(raw):
            key = "weight.quant_state.bitsandbytes__nf4"
            blob = json.loads(bytes(raw["state_dict"][key]["values"]))
            blob["blocksize"] = 128
            values = list(json.dumps(blob).encode())
            raw["state_dict"][key] = {"shape": [len(values)], "dtype": "uint8", "values": values}
        self.mutate_raw(mutate)
        with self.assertRaisesRegex(ValueError, "STATE_METADATA"):
            self.verify()

    def test_missing_state_key_rejected(self):
        self.mutate_raw(lambda r: r["state_dict"].pop("weight.absmax"))
        with self.assertRaisesRegex(ValueError, "STATE_KEYS"):
            self.verify()

    def test_changed_base_weight_rejected(self):
        self.mutate_raw(lambda r: r.update(packed_after=dict(r["packed_after"], values=[1] * len(r["packed_after"]["values"]))))
        with self.assertRaisesRegex(ValueError, "BASE_WEIGHT_CHANGED"):
            self.verify()

    def test_base_gradient_rejected(self):
        self.mutate_raw(lambda r: r.update(weight_grad_is_none=False))
        with self.assertRaisesRegex(ValueError, "BASE_WEIGHT_GRADIENT"):
            self.verify()

    def test_stale_state_alias_rejected(self):
        self.mutate_raw(lambda r: r.update(module_state_alias=False))
        with self.assertRaisesRegex(ValueError, "STATE_IDENTITY"):
            self.verify()

    def test_plain_load_claim_rejected(self):
        self.mutate_raw(lambda r: r.update(fresh_reconstruction=False))
        with self.assertRaisesRegex(ValueError, "FRESH_STATE"):
            self.verify()

    def test_wrong_checkpoint_rejected(self):
        self.mutate_raw(lambda r: r.update(loaded_checkpoint={"sha256": "b" * 64, "bytes": 1}))
        with self.assertRaisesRegex(ValueError, "CHECKPOINT"):
            self.verify()

    def test_incomplete_matrix_rejected(self):
        self.mutate_seal("case_ids", self.records["restore"]["case_ids"][:-1])
        with self.assertRaisesRegex(ValueError, "MATRIX"):
            self.verify()

    def test_numeric_difference_remains_fail(self):
        self.mutate_raw(lambda r: r["outputs"]["dx"]["values"].__setitem__(0, 10.0))
        result = self.verify()
        self.assertEqual(result["byte_integrity"], "PASS")
        self.assertEqual(result["numerical_status"], "FAIL")

    def test_wrong_terminal_rejected(self):
        self.mutate_seal("status", "PARTIAL")
        with self.assertRaisesRegex(ValueError, "TERMINAL"):
            self.verify()

    def test_public_restore_calls_and_dict_copy(self):
        # Faithful interface stub; it executes no tensor arithmetic.
        events = []
        class State:
            dtype = "float32"
            nested = False
        class Params:
            @classmethod
            def from_prequantized(cls, data, stats, **kw):
                events.append(("from_prequantized", data, kw["device"], kw["requires_grad"]))
                stats.pop("quant_state.bitsandbytes__nf4")
                result = cls()
                result.quant_state = State()
                kw["module"].quant_state = result.quant_state
                return result
        class Module:
            def __init__(self, *a, **kw):
                self.bias = None
            def to(self, **kw):
                events.append(("dtype", kw["dtype"]))
                return self
            def load_state_dict(self, data, strict):
                events.append(("load", sorted(data), strict))
            def train(self):
                return self
        case = next(c for c in P.cases() if c["dtype"] == "float32" and c["bias"] is None)
        state = {k: k for k in P.state_keys(case)}
        before = dict(state)
        bnb = SimpleNamespace(nn=SimpleNamespace(Linear4bit=Module, Params4bit=Params),
                              functional=SimpleNamespace(QuantState=State))
        module = P.restore_module(case, state, bnb, SimpleNamespace(float32="float32", uint8="uint8"), "xla:0")
        self.assertIs(module.weight.quant_state, module.quant_state)
        self.assertEqual(state, before)
        self.assertEqual(events, [("dtype", "float32"), ("from_prequantized", "weight", "xla:0", False),
                                  ("load", ["weight"], True)])


@unittest.skipUnless(os.environ.get("BNB_STATE_CPU_SMOKE") == "1", "Explicit CPU helper smoke only")
class CpuHelperSmoke(unittest.TestCase):
    def test_all_eight_public_state_cases(self):
        # This test deliberately calls helpers, not the qualified prepare mode.
        upstream = Path(os.environ["BNB_SOURCE_ROOT"])
        output = Path(os.environ["BNB_STATE_SMOKE_OUTPUT"])
        output.mkdir(parents=True, exist_ok=False)
        profile, _ = P.B.load_spec()
        for relative, identity in profile["bnb_files"].items():
            self.assertEqual(P.B.sha(upstream / relative), identity)
        sys.path.insert(0, str(upstream.parent))
        import torch
        import bitsandbytes as bnb
        from bitsandbytes.cextension import lib, ErrorHandlerMockBNBNativeLibrary
        self.assertEqual(Path(bnb.__file__).resolve(), (upstream / "__init__.py").resolve())
        self.assertTrue(sys.flags.dont_write_bytecode)
        self.assertEqual(sys.platform, "darwin")
        self.assertEqual(torch.__version__, "2.14.1")
        functions = P.B.original_cpu_functions(upstream)
        results = []
        report = {"scope": "UNQUALIFIED_MAC_CPU_HELPER_SMOKE", "qualified_linux_oracle": "NOT_RUN",
                  "actual_tpu": "NOT_RUN", "platform": platform.platform(), "torch": torch.__version__,
                  "native_library_mock": isinstance(lib, ErrorHandlerMockBNBNativeLibrary),
                  "source_subset": profile["bnb_files"], "cases": results}
        for case in P.cases():
            with self.subTest(case=case["id"]):
                try:
                    dtype = getattr(torch, case["dtype"])
                    weight = torch.tensor(case["weight"], dtype=dtype).reshape(case["weight_shape"])
                    packed, absmax = functions[0](weight, 64, "nf4", torch.uint8)
                    state = bnb.functional.QuantState(absmax=absmax, shape=torch.Size(case["weight_shape"]), dtype=dtype,
                              blocksize=64, code=bnb.functional.get_4bit_type("nf4", device="cpu"), quant_type="nf4")
                    state_dict = {"weight": packed, **{"weight." + k: v for k, v in state.as_dict(packed=True).items()}}
                    if case["bias"] is not None:
                        state_dict["bias"] = torch.tensor(case["bias"], dtype=dtype)
                    raw_records = []
                    module_states = []
                    for iteration in range(2):
                        module = P.restore_module(case, state_dict, bnb, torch, "cpu")
                        module_states.append(module.weight.quant_state)
                        outputs, before = P.compute(case, module, torch, bnb, "cpu")
                        qs = module.weight.quant_state
                        raw = {"case_id": case["id"], "input_sha256": P.B.case_sha(case),
                               "metadata": {"shape": list(qs.shape), "dtype": str(qs.dtype).removeprefix("torch."),
                                            "blocksize": qs.blocksize, "quant_type": qs.quant_type, "nested": qs.nested,
                                            "packing_format_for_cpu": bool(getattr(qs, "packing_format_for_cpu", False))},
                               "placements": {k: str(v.device) for k, v in outputs.items()}, "counters": {},
                               "outputs": {k: P.B.tensor_record(v) for k, v in outputs.items()},
                               "weight_requires_grad": module.weight.requires_grad, "weight_grad_is_none": module.weight.grad is None,
                               "original_quant_state_class": type(qs) is bnb.functional.QuantState,
                               "module_state_alias": module.quant_state is qs,
                               "packed_before": P.B.tensor_record(before), "packed_after": P.B.tensor_record(module.weight.data),
                               "state_dict": {k: P.B.tensor_record(v) for k, v in module.state_dict().items()}}
                        P.validate_raw(raw, case, False)
                        gradients = case["dtype"] == "float32"
                        x = torch.tensor(case["x"], dtype=dtype).reshape(case["x_shape"]).requires_grad_(gradients)
                        bias = None if case["bias"] is None else torch.tensor(case["bias"], dtype=dtype, requires_grad=gradients)
                        y = torch.nn.functional.linear(x, outputs["decoded"].detach(), bias)
                        dense = {"y": y}
                        if gradients:
                            dy = torch.tensor([((i % 5) - 2) / 8 for i in range(y.numel())], dtype=dtype).reshape(y.shape)
                            y.backward(dy)
                            dense["dx"] = x.grad
                            if bias is not None:
                                dense["db"] = bias.grad
                        reference = {"outputs": dict(raw["outputs"], **{k: P.B.tensor_record(v) for k, v in dense.items()})}
                        gates = P.compare_records(raw, reference, case)
                        self.assertTrue(all(g["status"] == "PASS" for g in gates.values()), gates)
                        P.B.write(output / (case["id"] + f".process-local-{iteration}.json"), raw)
                        raw_records.append(raw)
                        if iteration == 0:
                            checkpoint = output / (case["id"] + ".state.pt")
                            torch.save({k: v.detach().cpu().clone() for k, v in module.state_dict().items()}, checkpoint)
                            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    self.assertIsNot(module_states[0], module_states[1])
                    self.assertEqual(raw_records[0]["state_dict"], raw_records[1]["state_dict"])
                    roundtrip = P.compare_records(raw_records[1], raw_records[0], case, True)
                    self.assertTrue(all(g["status"] == "PASS" for g in roundtrip.values()), roundtrip)
                    results.append({"case_id": case["id"], "status": "PASS", "cpu_vs_dense": gates, "roundtrip": roundtrip,
                                    "fresh_quant_state_object": True})
                except Exception:
                    results.append({"case_id": case["id"], "status": "FAIL", "traceback": traceback.format_exc()})
                    raise
                finally:
                    P.B.write(output / "helper-smoke.json", report)
        self.assertEqual(len(results), 8)


if __name__ == "__main__":
    unittest.main()
