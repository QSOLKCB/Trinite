from pathlib import Path
import json
import os
import subprocess
import sys
import unittest

import torch

from trinite.contracts import ContractError, ModelConfig, identity, json_bytes
from trinite.inspection import (capture_inventory, inventory, quantizer_snapshots,
                               tensor_identity)
from trinite.model import Decoder, CaptureSpec
from trinite.quantizer import quantize

ROOT = Path(__file__).resolve().parents[2]
SMALL = ModelConfig(context_length=8, blocks=1, width=4, heads=2, feed_forward_width=8)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


class InspectionTests(unittest.TestCase):
    def test_actual_counts_sharing_scope_and_content_identities(self):
        model = Decoder()
        report = inventory(model)
        self.assertEqual(report["counts"], {"unique_parameters": 1247232,
                         "ternary_forward_elements": 1179648, "floating_forward_elements": 67584,
                         "latent_float32_bytes": 4988928})
        aliases = [p["names"] for p in report["parameters"] if len(p["names"]) > 1]
        self.assertEqual(aliases, [["token_embedding", "output_weight"]])
        self.assertEqual(len(report["linear_layers"]), 36)
        for layer in report["linear_layers"]:
            self.assertEqual(sum(layer["codes"].values()), layer["elements"])
            self.assertGreater(float.fromhex(layer["scale_hex"]), 0)
        dense = inventory(Decoder(lane="dense"))
        self.assertEqual(dense["counts"]["ternary_forward_elements"], 0)
        self.assertTrue(all("codes" not in x for x in dense["linear_layers"]))
        self.assertEqual(tensor_identity(torch.tensor([1.], dtype=torch.float32)),
                         identity(b"\x00\x00\x80\x3f"))
        # Metadata and raw bytes both bind the inventory, excluding environment.
        core = {k: v for k, v in report.items() if k not in ("inventory_identity", "environment")}
        self.assertEqual(report["inventory_identity"], identity(json_bytes(core)))

    def test_quantizer_snapshots_are_exact_detached_and_independent(self):
        model = Decoder(SMALL)
        names = ("blocks.0.q", "blocks.0.up")
        rng = torch.get_rng_state().clone()
        before = inventory(model)
        snapshots = quantizer_snapshots(model, names, max_bytes=56)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, capture in snapshots.items():
            expected = quantize(dict(model.named_modules())[name].weight)
            self.assertTrue(torch.equal(capture.codes, expected.codes))
            self.assertTrue(torch.equal(capture.scale, expected.scale))
            self.assertFalse(capture.scale.requires_grad)
            self.assertIsNone(capture.scale.grad_fn)
            capture.codes.zero_(); capture.scale.fill_(99)
        self.assertEqual(before, inventory(model))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_inspection_rejects_unbounded_selection_and_attached_captures(self):
        model = Decoder(SMALL)
        with self.assertRaises(ContractError): inventory(model, max_bytes=1)
        for names in ((), ("missing",), ("blocks.0.q", "blocks.0.q")):
            with self.assertRaises(ContractError): quantizer_snapshots(model, names)
        with self.assertRaises(ContractError):
            quantizer_snapshots(model, ("blocks.0.q",), max_bytes=19)
        with self.assertRaises(ContractError):
            quantizer_snapshots(Decoder(SMALL, lane="dense"), ("blocks.0.q",))
        with self.assertRaises(ContractError): capture_inventory({"attached": model.token_embedding})
        ids = torch.tensor([[256, 257]])
        output = model(ids, torch.ones_like(ids, dtype=torch.bool),
                       capture=CaptureSpec(("final",), (0, 1)))
        self.assertEqual(capture_inventory(output.captures)["final"]["shape"], [1, 2, 4])

    def test_cli_inventory_forward_capture_and_failures(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT/"src"))
        command = [sys.executable, "-m", "trinite", "inspect-model"]
        result = subprocess.run(command + ["--text", "1", "--capture-layer", "final",
                                           "--capture-position", "1"],
                                env=env, cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["counts"]["unique_parameters"], 1247232)
        self.assertEqual(report["forward_result"]["logits_shape"], [1, 3, 259])
        self.assertEqual(report["forward_result"]["captures"]["final"]["shape"], [1, 1, 128])
        self.assertEqual(report["forward_result"]["capture_selection"]["positions"], [1])
        for args in (["--capture-layer", "final"], ["--text", "1", "--capture-position", "0"],
                     ["--text", "x"*255], ["--text", "1", "--capture-layer", "final",
                                              "--capture-position", "1", "--capture-bytes", "1"]):
            result = subprocess.run(command+args, env=env, cwd=ROOT,
                                    capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertIn("trinite:", result.stderr)

    def test_locked_cpu_backend_is_the_conformance_lane(self):
        self.assertEqual(torch.__version__, "2.8.0+cpu")
        self.assertIsNone(torch.version.cuda)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
