"""Artifact-boundary and numerical export conformance, never a shipping importer."""
import dataclasses
import random
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.torch import load
import torch
from torch import nn

from trinite.contracts import ContractError, ModelConfig, identity, json_bytes, parse_json
from trinite.forward_export import FORMAT_ID, MAX_MANIFEST_BYTES, MAX_PAYLOAD_BYTES
from trinite.forward_export.encoder import export_forward
from trinite.forward_export.source import SourceSnapshot, read_source_snapshot
from trinite.inspection import tensor_identity
from trinite.model import CaptureSpec, Decoder, Linear
from trinite.packed_ternary import CODEC_ID, unpack_codes
from trinite.quantizer import QuantizedWeight, quantize

TINY = ModelConfig(blocks=1, width=16, heads=2, feed_forward_width=32, context_length=32)
ODD = ModelConfig(blocks=1, width=3, heads=1, feed_forward_width=5, context_length=8)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def oracle_reconstruction(artifact):
    """TEST ONLY: safetensors' authoritative reader and the native packed decoder.

    This consumes only encoder-produced trusted bytes. It is not a strict
    untrusted-artifact importer and must never enter the production package.
    """
    manifest = parse_json(artifact.manifest, canonical=True)
    tensors = load(artifact.float_tensors)
    model = Decoder(ModelConfig.from_dict(manifest["config"]), lane="dense", seed=999)
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for entry in manifest["floating_tensors"]:
            parameters[entry["name"]].copy_(tensors[entry["tensor"]])
        for entry in manifest["ternary_tensors"]:
            raw = artifact.packed_codes[entry["offset"]:entry["offset"]+entry["bytes"]]
            codes = torch.tensor(unpack_codes(raw, entry["code_count"]), dtype=torch.float32)
            parameters[entry["name"]].copy_(codes.reshape(entry["shape"])*tensors[entry["scale_tensor"]])
    return model


class ForwardExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = read_source_snapshot()

    def encode(self, model, **kwargs):
        return export_forward(model, source=self.source,
                              expected_source_identity=self.source.content_identity, **kwargs)

    def test_deterministic_immutable_complete_inventory_and_hashes(self):
        for config in (ODD, TINY, ModelConfig()):
            model = Decoder(config, seed=2)
            artifact = self.encode(model)
            self.assertEqual(artifact, self.encode(model))
            for field in dataclasses.fields(artifact):
                self.assertIs(type(getattr(artifact, field.name)), bytes)
            with self.assertRaises(dataclasses.FrozenInstanceError):
                artifact.packed_codes = b""
            manifest = parse_json(artifact.manifest, canonical=True)
            self.assertEqual(artifact.manifest, json_bytes(manifest))
            self.assertEqual(manifest["schema"], FORMAT_ID)
            self.assertEqual(manifest["codec"], CODEC_ID)
            self.assertEqual(manifest["config_identity"], config.content_identity)
            self.assertEqual(manifest["source_identity"], self.source.content_identity)
            self.assertEqual(manifest["state_identity"], identity(json_bytes(manifest["state"])))
            self.assertEqual(manifest["tokenizer_identity"], identity(json_bytes(manifest["tokenizer"])))
            self.assertEqual(manifest["tokenizer"]["byte_tokens"], [0, 255])
            self.assertEqual([manifest["tokenizer"][k] for k in ("bos", "eos", "pad")], [256, 257, 258])
            parameters = dict(model.named_parameters())
            latent = manifest["state"]["tensors"]
            self.assertEqual({e["name"] for e in latent}, set(parameters))
            self.assertEqual(sum(e["elements"] for e in latent), config.parameter_count)
            self.assertEqual([e["aliases"] for e in latent if e["aliases"]], [["output_weight"]])
            for e in latent:
                self.assertEqual(e["identity"], tensor_identity(parameters[e["name"]]))
            floats = load(artifact.float_tensors)
            self.assertEqual(set(floats), set(manifest["float_tensor_inventory"]))
            for n, e in manifest["float_tensor_inventory"].items():
                self.assertEqual(list(floats[n].shape), e["shape"])
                self.assertEqual(floats[n].dtype, torch.float32)
                self.assertEqual(tensor_identity(floats[n]), e["identity"])
                self.assertEqual(floats[n].numel()*4, e["bytes"])
            seen = set()
            end = 0
            for e in manifest["ternary_tensors"]:
                self.assertNotIn(e["name"], seen); seen.add(e["name"])
                self.assertEqual(e["offset"], end)
                end += e["bytes"]
                payload = artifact.packed_codes[e["offset"]:end]
                self.assertEqual(identity(payload), e["identity"])
                actual = quantize(parameters[e["name"]])
                codes = unpack_codes(payload, e["code_count"])
                self.assertEqual(codes, tuple(actual.codes.reshape(-1).tolist()))
                self.assertEqual(e["code_identity"], tensor_identity(actual.codes))
                self.assertEqual(e["occupancy"], {str(i): codes.count(i) for i in (-1, 0, 1)})
                self.assertEqual(sum(e["occupancy"].values()), e["code_count"])
                self.assertEqual(e["scale_identity"], tensor_identity(actual.scale))
                self.assertTrue(torch.equal(floats[e["scale_tensor"]], actual.scale))
                self.assertEqual(float.fromhex(e["scale_hex"]), float(actual.scale))
                self.assertEqual(float.fromhex(e["scale_hex"]).hex(), e["scale_hex"])
            self.assertEqual(end, len(artifact.packed_codes))
            for e in manifest["floating_tensors"]:
                self.assertNotIn(e["name"], seen); seen.add(e["name"])
                self.assertTrue(torch.equal(floats[e["tensor"]], parameters[e["name"]]))
            self.assertEqual(seen, set(parameters))
            for key in ("packed_codes", "float_tensors"):
                payload = getattr(artifact, key)
                self.assertEqual(manifest["members"][key], {
                    **manifest["members"][key], "identity": identity(payload), "bytes": len(payload)})
            counts = manifest["counts"]
            self.assertEqual(counts["unique_parameters"], config.parameter_count)
            self.assertEqual(counts["ternary_codes"]+counts["floating_forward_elements"], config.parameter_count)
            self.assertEqual(counts["float_raw_bytes"], sum(t.numel()*4 for t in floats.values()))
            self.assertEqual(counts["safetensors_framing_bytes"]+counts["float_raw_bytes"], len(artifact.float_tensors))
            self.assertEqual(counts["serialized_payload_bytes"], len(artifact.float_tensors)+len(artifact.packed_codes))

    def test_exact_logits_captures_and_bounded_greedy_trajectories_two_shapes_two_seeds(self):
        for config in (TINY, ModelConfig()):
            for seed in (0, 1):
                with self.subTest(config=config.width, seed=seed):
                    ternary = Decoder(config, seed=seed)
                    dense = oracle_reconstruction(self.encode(ternary))
                    ids = torch.tensor([[256, 49, 43, 50, 258, 258], [256, 65, 66, 67, 68, 257]])
                    mask = torch.tensor([[True]*4+[False]*2, [True]*6])
                    layers = ("embedding", *(f"block.{i}" for i in range(config.blocks)), "final")
                    capture = CaptureSpec(layers, (0, 2, 5))
                    with torch.no_grad():
                        original = ternary(ids, mask, capture=capture)
                        reconstructed = dense(ids, mask, capture=capture)
                        self.assertTrue(torch.equal(original.logits, reconstructed.logits))
                        for name in layers:
                            self.assertTrue(torch.equal(original.captures[name], reconstructed.captures[name]))
                        # Conformance-only greedy oracle: native full-prefix recompute,
                        # smallest-ID argmax tie handling, EOS or six-token hard bound.
                        for prompt in ([256, 49, 43], [256, 65]):
                            a, b = list(prompt), list(prompt)
                            for _ in range(6):
                                outputs = []
                                for model, tokens in ((ternary, a), (dense, b)):
                                    tokens_tensor = torch.tensor([tokens])
                                    out = model(tokens_tensor, torch.ones_like(tokens_tensor, dtype=torch.bool))
                                    outputs.append(out.logits[0, -1])
                                self.assertTrue(torch.equal(outputs[0], outputs[1]))
                                ta, tb = (int(o.argmax()) for o in outputs)
                                self.assertEqual(ta, tb)
                                a.append(ta); b.append(tb)
                                if ta == 257:
                                    break
                            self.assertEqual(a, b)

    def test_current_forward_after_latent_edits_not_initialization_or_checkpoint_claim(self):
        model = Decoder(TINY, seed=17)
        before = self.encode(model)
        with torch.no_grad():
            model.blocks[0].q.weight.mul_(3).add_(0.02)
            model.token_embedding[65].mul_(2)
        after = self.encode(model)
        a, b = parse_json(before.manifest), parse_json(after.manifest)
        self.assertEqual(a["initialization_seed_label"], b["initialization_seed_label"])
        self.assertNotEqual(a["state_identity"], b["state_identity"])
        self.assertNotEqual(before.float_tensors, after.float_tensors)
        self.assertNotEqual(before.packed_codes, after.packed_codes)
        self.assertIn("no checkpoint", b["lineage"])
        dense = oracle_reconstruction(after)
        ids = torch.tensor([[256, 65, 257]])
        mask = torch.ones_like(ids, dtype=torch.bool)
        self.assertTrue(torch.equal(model(ids, mask).logits, dense(ids, mask).logits))

    def test_zero_subnormal_extreme_codes_and_positive_float32_scales(self):
        for fill in (0.0, float.fromhex("0x1p-149"), torch.finfo(torch.float32).tiny,
                     torch.finfo(torch.float32).max):
            model = Decoder(ODD)
            with torch.no_grad():
                for module in model.modules():
                    if type(module) is Linear:
                        module.weight.fill_(fill)
            artifact = self.encode(model)
            manifest = parse_json(artifact.manifest)
            floats = load(artifact.float_tensors)
            for e in manifest["ternary_tensors"]:
                scale = floats[e["scale_tensor"]]
                self.assertEqual(scale.shape, ())
                self.assertTrue(bool(torch.isfinite(scale)))
                self.assertGreater(float(scale), 0)
                codes = unpack_codes(artifact.packed_codes[e["offset"]:e["offset"]+e["bytes"]], e["code_count"])
                self.assertEqual(set(codes), {1 if fill == torch.finfo(torch.float32).max else 0})

    def test_no_state_rng_gradient_mode_or_io_mutation(self):
        model = Decoder(TINY)
        ids = torch.tensor([[256, 49, 257]])
        model(ids, torch.ones_like(ids, dtype=torch.bool)).logits.sum().backward()
        before = {n: (p.clone(), p.grad.clone(), p._version, id(p), p.requires_grad)
                  for n, p in model.named_parameters()}
        modes = {n: m.training for n, m in model.named_modules()}
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        threads = torch.get_num_threads()
        with patch("builtins.open", side_effect=AssertionError("implicit file I/O")):
            artifact = self.encode(model)
        self.assertIsNotNone(artifact)
        self.assertTrue(torch.is_grad_enabled())
        self.assertEqual(threads, torch.get_num_threads())
        self.assertEqual(modes, {n: m.training for n, m in model.named_modules()})
        for n, p in model.named_parameters():
            value, grad, version, pointer, requires_grad = before[n]
            self.assertTrue(torch.equal(value, p)); self.assertTrue(torch.equal(grad, p.grad))
            self.assertEqual((version, pointer, requires_grad), (p._version, id(p), p.requires_grad))
        self.assertEqual(rng[0], random.getstate())
        after_np = np.random.get_state()
        self.assertEqual(rng[1][0], after_np[0]); self.assertTrue(np.array_equal(rng[1][1], after_np[1]))
        self.assertEqual(rng[1][2:], after_np[2:])
        self.assertTrue(torch.equal(rng[2], torch.get_rng_state()))
        # Export owns all bytes: later live-model mutation cannot change them.
        frozen = dataclasses.astuple(artifact)
        with torch.no_grad(): model.token_embedding.zero_()
        self.assertEqual(frozen, dataclasses.astuple(artifact))

    def test_invalid_models_inventory_aliases_layout_and_operations(self):
        mutations = [
            lambda m: setattr(m, "lane", "dense"),
            lambda m: setattr(m, "lane", "four-state"),
            lambda m: setattr(m, "config", ModelConfig()),
            lambda m: setattr(m.blocks[0].q, "lane", "dense"),
            lambda m: setattr(m.blocks[0].q, "weight", nn.Parameter(torch.ones(1, 1))),
            lambda m: setattr(m, "output_weight", nn.Parameter(m.token_embedding.detach().clone())),
            lambda m: setattr(m.blocks[0].k, "weight", m.blocks[0].q.weight),
            lambda m: setattr(m.blocks[0].k, "weight", nn.Parameter(m.blocks[0].q.weight.detach())),
            lambda m: setattr(m.blocks[0].q, "weight", nn.Parameter(m.blocks[0].q.weight.t())),
            lambda m: m.double(),
            lambda m: setattr(m.blocks[0].attention_norm, "epsilon", 1.0),
            lambda m: setattr(m.blocks[0], "heads", 1),
            lambda m: setattr(m.blocks[0], "q", nn.Linear(16, 16, bias=False)),
            lambda m: m.register_buffer("unexpected", torch.ones(1)),
            lambda m: m.register_parameter("extra", nn.Parameter(torch.ones(1))),
            lambda m: m.register_forward_hook(lambda *args: None),
            lambda m: setattr(m, "forward", lambda *args: None),
            lambda m: setattr(m, "validate_state", lambda: None),
            lambda m: setattr(m, "_compiled_call_impl", lambda: None),
            lambda m: m._parameter_shapes.update({"token_embedding": (1, 1)}),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                model = Decoder(TINY); mutate(model)
                with self.assertRaises(ContractError): self.encode(model)
        class SubDecoder(Decoder):
            pass
        with self.assertRaises(ContractError): self.encode(SubDecoder(TINY))
        with self.assertRaises(ContractError): self.encode(None)
        for value in (float("nan"), float("inf"), -float("inf")):
            for name in ("token_embedding", "blocks.0.q.weight", "final_norm.weight"):
                model = Decoder(TINY)
                with torch.no_grad(): dict(model.named_parameters())[name].fill_(value)
                with self.assertRaises(ContractError): self.encode(model)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with self.assertRaises(ContractError): self.encode(Decoder(TINY))

    def test_registered_parameter_and_child_attribute_shadows_are_rejected(self):
        # nn.Module.__setattr__ normally protects registration. These adversarial
        # __dict__ edits bypass it while leaving registered classes/names intact.
        # Cover every registered parameter (including the output alias), every
        # child, and even same-object shadows: the export profile is canonical.
        template = Decoder(TINY)
        parameter_paths = [(path, leaf) for path, module in template.named_modules()
                           for leaf in module._parameters]
        child_paths = [(path, leaf) for path, module in template.named_modules()
                       for leaf in module._modules]
        for kind, paths in (("parameter", parameter_paths), ("child", child_paths)):
            for path, leaf in paths:
                for same_object in (False, True):
                    with self.subTest(kind=kind, path=path, leaf=leaf, same_object=same_object):
                        model = Decoder(TINY)
                        module = dict(model.named_modules())[path]
                        registered = getattr(module, leaf)
                        shadow = (registered if same_object else
                                  torch.ones_like(registered) if kind == "parameter" else nn.Identity())
                        module.__dict__[leaf] = shadow
                        with self.assertRaisesRegex(ContractError, "shadowed"):
                            self.encode(model)
        # Both root embedding names can be shadowed by one identical object,
        # satisfying the native alias check but still diverging from registration.
        model = Decoder(TINY)
        shadow = torch.ones_like(model.token_embedding)
        model.__dict__["token_embedding"] = shadow
        model.__dict__["output_weight"] = shadow
        with self.assertRaisesRegex(ContractError, "shadowed"):
            self.encode(model)
        for changed_children in (False, True):
            model = Decoder(TINY)
            shadow = list(model.blocks)
            if changed_children:
                shadow[0] = Decoder(TINY, seed=42).blocks[0]
            model.__dict__["blocks"] = shadow
            with self.assertRaisesRegex(ContractError, "shadowed"):
                self.encode(model)
        for field, value in (("heads", True), ("heads", 2.0), ("head_width", 8.0)):
            model = Decoder(TINY)
            setattr(model.blocks[0], field, value)
            with self.assertRaises(ContractError): self.encode(model)
        model = Decoder(ODD)
        model._parameter_shapes["final_norm.weight"] = (3.0,)
        with self.assertRaises(ContractError): self.encode(model)

    def test_bounds_fail_before_quantization_or_finite_scan(self):
        model = Decoder(TINY)
        with patch("trinite.forward_export.encoder.quantize", side_effect=AssertionError("quantized")), \
             patch.object(Decoder, "validate_state", side_effect=AssertionError("scanned")):
            for budget in (0, True, 1.0, -1, MAX_PAYLOAD_BYTES+1, 1):
                with self.assertRaises(ContractError): self.encode(model, max_payload_bytes=budget)
            for budget in (0, True, 1.0, -1, MAX_MANIFEST_BYTES+1, 1):
                with self.assertRaises(ContractError): self.encode(model, max_manifest_bytes=budget)
            config = ModelConfig(blocks=1, width=512, heads=4, feed_forward_width=2049)
            large_matrix = Decoder(config, parameter_budget=5_000_000)
            with self.assertRaisesRegex(ContractError, "codec code ceiling"):
                self.encode(large_matrix)

    def test_source_identity_and_inventory_mismatch_rejection(self):
        model = Decoder(TINY)
        for source in (None, {}, SourceSnapshot(tuple()), SourceSnapshot(self.source.members[:-1]),
                       SourceSnapshot(self.source.members[::-1]),
                       SourceSnapshot((self.source.members[0],)*len(self.source.members))):
            with self.assertRaises(ContractError):
                export_forward(model, source=source, expected_source_identity=self.source.content_identity)
        for expected in (True, "sha256:"+"f"*64, "not a hash"):
            with self.assertRaises(ContractError):
                export_forward(model, source=self.source, expected_source_identity=expected)
        changed = SourceSnapshot(((self.source.members[0][0], b"changed"), *self.source.members[1:]))
        with self.assertRaises(ContractError):
            export_forward(model, source=changed, expected_source_identity=self.source.content_identity)
        with self.assertRaises(ContractError):
            read_source_snapshot(expected_identity="sha256:"+"f"*64)

    def test_bad_quantizer_outputs_or_interleaved_mutation_do_not_publish(self):
        for scale in (torch.tensor(0.), torch.tensor(-1.), torch.tensor(float("nan")),
                      torch.tensor(float("inf")), torch.tensor(1., dtype=torch.float64), torch.ones(1)):
            def bad(weight):
                original = quantize(weight)
                return QuantizedWeight(original.codes, scale, original.weight)
            with patch("trinite.forward_export.encoder.quantize", side_effect=bad):
                with self.assertRaises(ContractError): self.encode(Decoder(TINY))
        for kind in ("reserved", "shape", "dtype", "device"):
            def bad_codes(weight):
                original = quantize(weight)
                codes = (torch.full_like(original.codes, 3) if kind == "reserved" else
                         original.codes.reshape(-1) if kind == "shape" else
                         original.codes.float() if kind == "dtype" else original.codes.to("meta"))
                return QuantizedWeight(codes, original.scale, original.weight)
            with patch("trinite.forward_export.encoder.quantize", side_effect=bad_codes):
                with self.assertRaises(ContractError): self.encode(Decoder(TINY))
        model = Decoder(TINY)
        def mutate(weight):
            result = quantize(weight)
            with torch.no_grad(): model.token_embedding.add_(1)
            return result
        with patch("trinite.forward_export.encoder.quantize", side_effect=mutate):
            with self.assertRaisesRegex(ContractError, "model changed"):
                self.encode(model)


if __name__ == "__main__":
    unittest.main()
