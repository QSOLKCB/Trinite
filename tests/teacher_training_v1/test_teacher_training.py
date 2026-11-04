"""TT-003 offline plan, admission, approval and CUDA software controls."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from trinite.contracts import ContractError, identity, json_bytes
from teacher_dataset_v1 import core as tt002
from teacher_training_v1 import spec, cuda_preflight


class TT003Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.dataset = Path(cls.temp.name) / "tt002"
        cls.dataset_anchor = tt002.prepare(cls.dataset)
        cls.request_folder = Path(cls.temp.name) / "tt003"
        cls.request_anchor = spec.freeze(cls.request_folder, cls.dataset, cls.dataset_anchor)
        cls.request = spec.verify_request(cls.request_folder, cls.request_anchor, cls.dataset)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_small_native_teacher_and_frozen_schedule(self):
        p, model = spec.settings()
        self.assertEqual(model.parameter_count, 90624)
        self.assertEqual((p["steps"], p["batch_size"], p["max_target_tokens"]), (32, 4, 640))
        self.assertEqual(self.request["model_parameters"], 90624)
        self.assertEqual(self.request["scheduled_visits"], 128)
        self.assertLessEqual(self.request["scheduled_target_tokens"], 640)
        self.assertFalse(p["training_authorized"])

    def test_no_gpu_or_heldout_authorization(self):
        self.assertEqual(self.request["role"], "teacher")
        self.assertEqual(self.request["backend"], "pinned-cpu-float32")
        self.assertFalse(self.request["training_authorized"])
        self.assertFalse(self.request["heldout_scored"])
        self.assertFalse(self.request["cuda_gpu_training_available"])
        self.assertEqual(self.request["weights_license"], "UNDECIDED")

    def test_source_closure_covers_native_runtime_and_protocol(self):
        paths = {item["path"] for item in self.request["source_closure"]}
        for required in ("src/trinite/model.py", "src/trinite/training.py",
                         "src/trinite/comparison_checkpoint.py",
                         "src/trinite/comparison_data.py",
                         "src/trinite/qec_source.py",
                         "src/trinite/qec-pin.json",
                         "scripts/teacher_training_v1/numerical.py",
                         "scripts/teacher_training_v1/cuda_preflight.py",
                         "docs/TEACHER-LAB-TT-003.md", "requirements.lock"):
            self.assertIn(required, paths)

    def test_request_binding_and_nonoverwrite(self):
        with self.assertRaises(ContractError):
            spec.verify_request(self.request_folder, "sha256:" + "0"*64, self.dataset)
        with self.assertRaises(ContractError):
            spec.freeze(self.request_folder, self.dataset, self.dataset_anchor)

    def test_normative_document_drift_rejects_previously_frozen_request(self):
        original = spec.tt001.source_bytes
        def changed(path):
            raw = original(path)
            return raw + b"\n# changed" if path == "docs/TEACHER-LAB-TT-003.md" else raw
        with mock.patch.object(spec.tt001, "source_bytes", side_effect=changed):
            with self.assertRaises(ContractError):
                spec.verify_request(self.request_folder, self.request_anchor, self.dataset)

    def test_separate_owner_approval_required_and_exact(self):
        approved = {
            "schema": spec.APPROVAL_SCHEMA,
            "request_identity": self.request_anchor,
            "dataset_manifest_identity": self.dataset_anchor,
            "owner": "QSOLKCB",
            "scope": "one-explicit-owner-local-cpu-teacher-experiment",
            "approved_updates": 32,
            "training_authorized": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "approval.json"
            path.write_bytes(json_bytes(approved))
            anchor = identity(path.read_bytes())
            self.assertEqual(spec.load_approval(
                path, anchor, self.request, self.request_anchor), approved)
            with self.assertRaises(ContractError):
                spec.load_approval(path, "sha256:"+"0"*64, self.request, self.request_anchor)
            approved["approved_updates"] = 33
            path.write_bytes(json_bytes(approved))
            with self.assertRaises(ContractError):
                spec.load_approval(path, identity(path.read_bytes()),
                                   self.request, self.request_anchor)

    def test_cuda_124_driver_and_arch_rejection(self):
        example = {"cuda_build":"12.8", "driver_api":12080,
                   "gpu_name":"NVIDIA GeForce RTX 5060 Ti",
                   "capability":[12,0], "arch_list":["sm_86","sm_120"],
                   "kernel_executed":True,"kernel_correct":True}
        valid = cuda_preflight.validate(example)
        self.assertTrue(valid["eligible_preflight_only"])
        self.assertFalse(valid["gpu_training_supported"])
        for k, val in (("cuda_build","12.4"),("cuda_build","12.7"),
                       ("driver_api",12040),("driver_api",True),
                       ("capability",[8,6]),("capability",[True,0]),
                       ("gpu_name","GeForce RTX 3050"),
                       ("arch_list",["sm_86"]),
                       ("kernel_executed",False),("kernel_correct",False)):
            with self.subTest(k=k, v=val), self.assertRaises(ContractError):
                cuda_preflight.validate(dict(example, **{k:val}))

    def test_cuda_invalid_schema_and_index(self):
        with self.assertRaises(ContractError):
            cuda_preflight.validate({})
        with self.assertRaises(ContractError):
            cuda_preflight.live(-1)


if __name__ == "__main__":
    unittest.main()
