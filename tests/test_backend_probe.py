"""Local artifact controls. These fixtures do not prove TPU execution."""

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "experiments/2026-10-04-bitsandbytes-tpu/probe_backend.py"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reseal(root, name):
    seal = json.loads((root / name).read_text())
    seal["artifacts"] = {
        p.relative_to(root).as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size}
        for p in sorted(root.rglob("*")) if p.is_file() and p.name != name
    }
    write(root / name, seal)
    return digest(root / name)


class ProbeControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ENTRY.is_file():
            return
        spec = importlib.util.spec_from_file_location("probe_backend_tested", ENTRY)
        cls.probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.probe)

    def setUp(self):
        self.assertTrue(ENTRY.is_file(), "The backend probe implementation is missing")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.cpu, self.tpu = self.base / "oracle", self.base / "actual"
        self.profile, self.inputs = self.probe.load_spec()
        self.admission = "a" * 64
        for root in (self.cpu, self.tpu):
            root.mkdir()
            for case in self.inputs["cases"]:
                n, k = case["weight_shape"]
                total = n * k
                outputs = {
                    "packed": {"shape": [(total + 1) // 2, 1], "dtype": "uint8", "values": [0] * ((total + 1) // 2)},
                    "absmax": {"shape": [(total + 63) // 64], "dtype": "float32", "values": [0.0] * ((total + 63) // 64)},
                    "code": {"shape": [16], "dtype": "float32", "values": [0.0] * 16},
                    "decoded": {"shape": [k, n] if total <= 2 else [n, k], "dtype": case["dtype"], "values": [0.0] * total},
                }
                if case["op"] == "linear":
                    shape = case["x_shape"][:-1] + [n]
                    outputs["y"] = {"shape": shape, "dtype": case["dtype"], "values": [0.0] * math.prod(shape)}
                record = {
                    "case_id": case["id"], "input_sha256": self.probe.case_sha(case),
                    "outputs": outputs,
                    "metadata": {"shape": [n, k], "dtype": case["dtype"], "blocksize": 64,
                                 "quant_type": "nf4", "nested": False, "packing_format_for_cpu": False},
                    "placements": {key: "cpu" if root == self.cpu else "xla:0" for key in outputs},
                    "counters": {} if root == self.cpu else {"xla::add": 1},
                    "execution_metrics": {} if root == self.cpu else {"ExecuteTime": [1, 0.001, [[0, 0.001]]]},
                }
                write(root / "raw" / (case["id"] + ".json"), record)
                if root == self.tpu:
                    (root / "raw" / (case["id"] + ".hlo.txt")).write_text("HloModule fixture; TEST DATA ONLY\n")
                    (root / "raw" / (case["id"] + ".metrics.txt")).write_text("TEST DATA ONLY\n")
        common = {"profile_sha256": self.probe.PROFILE_SHA, "inputs_sha256": self.probe.INPUTS_SHA,
                  "runtime_lock_sha256": self.profile["runtime_lock_sha256"],
                  "source_admission_sha256": self.admission, "case_ids": self.profile["case_ids"],
                  "status": "COMPLETE", "cuda_golden": "NOT_RUN", "artifacts": {}}
        write(self.cpu / "oracle-seal.json", dict(common, kind="CPU_ORACLE", oracle_method=self.probe.ORACLE_METHOD,
                                                 runtime={"python": "3.12", "torch": "2.9.0+cpu"}))
        self.oracle_sha = reseal(self.cpu, "oracle-seal.json")
        self.receipt = dict(common, kind="TPU_PROBE", oracle_sha256=self.oracle_sha,
                            runtime={"python": "3.12", "torch": "2.9.0+cpu", "torch_xla": "2.9.0", "libtpu": "0.0.21"},
                            device={"type": "xla", "hardware": "TPU", "pjrt": "TPU"},
                            public_api={"Linear4bit": "bitsandbytes.nn.modules.Linear4bit",
                                        "Params4bit": "bitsandbytes.nn.modules.Params4bit", "original_class_identity": True},
                            dispatch={name: True for name in self.probe.DISPATCH_OPS},
                            source_pre=self.admission, source_post=self.admission)
        write(self.tpu / "receipt.json", self.receipt)
        reseal(self.tpu, "receipt.json")

    def verify(self):
        return self.probe.verify_artifacts(self.tpu, self.cpu, self.oracle_sha, self.admission)

    def mutate_receipt(self, key, value):
        receipt = json.loads((self.tpu / "receipt.json").read_text())
        receipt[key] = value
        write(self.tpu / "receipt.json", receipt)

    def mutate_record(self, function):
        path = next((self.tpu / "raw").glob("*.json"))
        record = json.loads(path.read_text())
        function(record)
        write(path, record)
        reseal(self.tpu, "receipt.json")

    def test_length_two_public_shape_shim(self):
        case = next(c for c in self.inputs["cases"] if c["id"] == "quant-float32-2-codes")
        self.assertEqual(self.probe.expected_outputs(case)["decoded"], ([2, 1], "float32"))
        class Restored:
            def t(self):
                return "transposed"
        self.assertEqual(self.probe.cpu_public_decode(lambda *a: Restored(), SimpleNamespace(shape=[1, 1], reshape=lambda *a: None),
                          SimpleNamespace(absmax=None, code=None), [1, 2], "float32"), "transposed")

    def test_bf16_module_uses_cpu_dtype_before_copy(self):
        # This stub models constructor FP32 storage and copy_ destination casting.
        actions = []
        class Data:
            def __init__(self):
                self.dtype = "float32"
            def copy_(self, value):
                actions.append(("copy", self.dtype, value))
        class Params:
            def __init__(self):
                self.data = Data()
            @property
            def dtype(self):
                return self.data.dtype
        class Module:
            def __init__(self, *a, **kw):
                self.weight = Params()
                self.bias = SimpleNamespace(data=Data())
            def to(self, *, dtype):
                actions.append(("dtype", dtype))
                self.weight.data.dtype = dtype
                self.bias.data.dtype = dtype
                return self
        bnb = SimpleNamespace(nn=SimpleNamespace(Linear4bit=Module, Params4bit=Params))
        module = self.probe.cpu_linear_module([19, 17], "bfloat16", "W", "B", bnb,
                                               SimpleNamespace(uint8="uint8"))
        self.assertIsInstance(module.weight, Params)
        self.assertEqual(actions, [("dtype", "bfloat16"), ("copy", "bfloat16", "W"),
                                   ("copy", "bfloat16", "B")])

    def test_missing_or_zero_execution_samples_are_rejected(self):
        for value in ({}, {"ExecuteTime": [0, 0.0, []]}):
            self.mutate_record(lambda r: r.update(execution_metrics=value))
            with self.assertRaisesRegex(ValueError, "EXECUTION"):
                self.verify()

    def test_bf16_state_dtype_is_rejected(self):
        path = self.tpu / "raw/linear-bfloat16-rank2-bias1.json"
        value = json.loads(path.read_text())
        value["metadata"]["dtype"] = "float32"
        write(path, value)
        reseal(self.tpu, "receipt.json")
        with self.assertRaisesRegex(ValueError, "METADATA"):
            self.verify()

    def test_complete_fixture_is_valid(self):
        result = self.verify()
        self.assertEqual(result["byte_integrity"], "PASS")
        self.assertEqual(result["numerical_status"], "PASS")
        self.assertEqual(len(result["cases"]), 42)

    def test_wrong_source_is_rejected(self):
        self.mutate_receipt("source_post", "b" * 64)
        with self.assertRaisesRegex(ValueError, "SOURCE"):
            self.verify()

    def test_cpu_device_is_rejected(self):
        self.mutate_receipt("device", {"type": "xla", "hardware": "CPU", "pjrt": "TPU"})
        with self.assertRaisesRegex(ValueError, "TPU"):
            self.verify()

    def test_wrong_runtime_lock_is_rejected(self):
        self.mutate_receipt("runtime_lock_sha256", "b" * 64)
        with self.assertRaisesRegex(ValueError, "RUNTIME_LOCK"):
            self.verify()

    def test_wrong_runtime_is_rejected(self):
        runtime = copy.deepcopy(self.receipt["runtime"])
        runtime["torch_xla"] = "2.8.0"
        self.mutate_receipt("runtime", runtime)
        with self.assertRaisesRegex(ValueError, "RUNTIME"):
            self.verify()

    def test_partial_receipt_is_rejected(self):
        self.mutate_receipt("status", "PARTIAL")
        with self.assertRaisesRegex(ValueError, "TERMINAL"):
            self.verify()

    def test_missing_case_from_both_seals_is_rejected(self):
        for root, name in ((self.cpu, "oracle-seal.json"), (self.tpu, "receipt.json")):
            seal = json.loads((root / name).read_text())
            seal["case_ids"].pop()
            write(root / name, seal)
        self.oracle_sha = digest(self.cpu / "oracle-seal.json")
        self.mutate_receipt("oracle_sha256", self.oracle_sha)
        with self.assertRaisesRegex(ValueError, "MATRIX"):
            self.verify()

    def test_wrong_oracle_seal_hash_is_rejected(self):
        self.oracle_sha = "b" * 64
        with self.assertRaisesRegex(ValueError, "ORACLE"):
            self.verify()

    def test_truncated_raw_is_rejected(self):
        path = next((self.tpu / "raw").glob("*.json"))
        path.write_bytes(path.read_bytes()[:-5])
        with self.assertRaisesRegex(ValueError, "INTEGRITY"):
            self.verify()

    def test_missing_output_is_rejected_after_reseal(self):
        self.mutate_record(lambda r: r["outputs"].pop("decoded"))
        with self.assertRaisesRegex(ValueError, "OUTPUT"):
            self.verify()

    def test_unknown_output_file_is_rejected(self):
        (self.tpu / "extra.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "INVENTORY"):
            self.verify()

    def test_wrong_dtype_is_rejected(self):
        self.mutate_record(lambda r: r["outputs"]["decoded"].update(dtype="float16"))
        with self.assertRaisesRegex(ValueError, "OUTPUT"):
            self.verify()

    def test_fallback_counter_is_rejected(self):
        self.mutate_record(lambda r: r["counters"].update({"aten::nonzero": 1}))
        with self.assertRaisesRegex(ValueError, "FALLBACK"):
            self.verify()

    def test_cpu_tensor_is_rejected(self):
        self.mutate_record(lambda r: r["placements"].update(decoded="cpu"))
        with self.assertRaisesRegex(ValueError, "PLACEMENT"):
            self.verify()

    def test_nonfinite_is_rejected(self):
        self.mutate_record(lambda r: r["outputs"]["decoded"]["values"].__setitem__(0, float("nan")))
        with self.assertRaisesRegex(ValueError, "NONFINITE"):
            self.verify()

    def test_wrong_input_binding_is_rejected(self):
        self.mutate_record(lambda r: r.update(input_sha256="b" * 64))
        with self.assertRaisesRegex(ValueError, "INPUT"):
            self.verify()

    def test_numeric_mismatch_is_fail_not_corruption(self):
        self.mutate_record(lambda r: r["outputs"]["decoded"]["values"].__setitem__(0, 10.0))
        result = self.verify()
        self.assertEqual(result["byte_integrity"], "PASS")
        self.assertEqual(result["numerical_status"], "FAIL")

    def test_missing_dispatch_is_rejected(self):
        dispatch = dict(self.receipt["dispatch"])
        dispatch.pop(next(iter(dispatch)))
        self.mutate_receipt("dispatch", dispatch)
        with self.assertRaisesRegex(ValueError, "DISPATCH"):
            self.verify()

    def test_substitute_public_module_is_rejected(self):
        api = dict(self.receipt["public_api"], Linear4bit="port2tpu.compat.NF4Linear")
        self.mutate_receipt("public_api", api)
        with self.assertRaisesRegex(ValueError, "PUBLIC_API"):
            self.verify()

    def test_source_tree_correct_tampered_extra_and_cache(self):
        root = self.base / "source"
        root.mkdir()
        path = root / "a.py"
        path.write_text("x = 1\n")
        files = {"a.py": digest(path)}
        self.probe.verify_source_tree(root, files)
        path.write_text("x = 2\n")
        with self.assertRaisesRegex(ValueError, "SOURCE"):
            self.probe.verify_source_tree(root, files)
        path.write_text("x = 1\n")
        (root / "extra.py").write_text("pass\n")
        with self.assertRaisesRegex(ValueError, "SOURCE"):
            self.probe.verify_source_tree(root, files)
        (root / "extra.py").unlink()
        (root / "stale.pyc").write_bytes(b"cache")
        with self.assertRaisesRegex(ValueError, "CACHE"):
            self.probe.verify_source_tree(root, files)

    def test_raw_symlink_is_rejected(self):
        path = next((self.tpu / "raw").glob("*.json"))
        outside = self.base / "outside.json"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "SYMLINK"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
