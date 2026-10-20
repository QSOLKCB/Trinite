"""Independent numerical and generation adversaries using software doubles only."""
import copy
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from trinite.contracts import ContractError, identity
from batch_composition_v1 import comparison as c
from batch_composition_v1 import experiment as e
from batch_composition_v1 import spec


def vector(value=0.0):
    value = float(torch.tensor(value, dtype=torch.float32))
    return [value.hex()] * 259


def workload(prompts):
    return [{"item_id": f"item-{i}", "split": "train", "prompt": prompt,
             "prompt_identity": identity(prompt.encode())}
            for i, prompt in enumerate(prompts)]


class BranchingModel:
    """No row-index dependence; deliberately different stopping and special IDs."""
    def __init__(self):
        self.inputs = []

    def __call__(self, ids, mask):
        self.inputs.append((ids.clone(), mask.clone()))
        result = torch.zeros((*ids.shape, 259), dtype=torch.float32)
        for row in range(len(ids)):
            prefix = ids[row][mask[row]].tolist()
            initial = prefix[1]
            if initial == ord("A"):
                selected = 257
            elif initial == ord("B"):
                selected = 49 if len(prefix) == 2 else 257
            elif initial == ord("C"):
                selected = 256 if len(prefix) == 2 else 258
            else:
                selected = 50
            for position in range(len(prefix)):
                result[row, position, 54] = 2
                result[row, position, selected] = 10
        return SimpleNamespace(logits=result)


class NumericReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_last_live_not_physical_last_column(self):
        mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
        output = torch.arange(2 * 4 * 259, dtype=torch.float32).reshape(2, 4, 259)
        output[~mask] = 0
        actual = e.last_live(output, mask)
        self.assertEqual(actual[0], [float(x).hex() for x in output[0, 1]])
        self.assertEqual(actual[1], [float(x).hex() for x in output[1, 2]])
        self.assertNotEqual(actual[0], actual[1])

    def test_nonfinite_rejected_even_outside_selected_or_allowed_logits(self):
        mask = torch.tensor([[True, True, False]])
        for bad in (float("nan"), float("inf"), float("-inf")):
            for position in (0, 1, 2):
                output = torch.zeros((1, 3, 259), dtype=torch.float32)
                output[0, position, 200] = bad
                with self.subTest(bad=bad, position=position), self.assertRaises(ContractError):
                    e.last_live(output, mask)

    def test_invalid_padding_mask_and_nonzero_masked_outputs_rejected(self):
        for mask in (torch.tensor([[False, True, False]]),
                     torch.tensor([[True, False, True]])):
            with self.assertRaises(ContractError):
                e.last_live(torch.zeros((1, 3, 259)), mask)
        mask = torch.tensor([[True, False, False]])
        output = torch.zeros((1, 3, 259))
        output[0, 2, 258] = 1
        with self.assertRaises(ContractError):
            e.last_live(output, mask)
        with self.assertRaises(ContractError):
            e.last_live(torch.zeros((1, 3, 259), dtype=torch.float64), mask)

    def test_payload_and_dynamic_padding_preserve_live_prefix(self):
        prefixes = [[256, 65], [256, 66, 257, 258]]
        ids, mask = e.tensors(prefixes, "payload8_32")
        self.assertEqual(ids[0, :2].tolist(), prefixes[0])
        self.assertEqual(ids[1, :4].tolist(), prefixes[1])
        self.assertEqual(ids[0, 2:8].tolist(), [0, 257, 255, 0, 257, 255])
        self.assertEqual(ids[1, 4:10].tolist(), [0, 257, 255, 0, 257, 255])
        self.assertEqual(mask.sum(1).tolist(), [2, 4])
        short_ids, short_mask = e.tensors(prefixes, "batch8_min")
        self.assertEqual(tuple(short_ids.shape), (2, 4))
        self.assertEqual(short_mask.sum(1).tolist(), [2, 4])
        self.assertEqual(prefixes, [[256, 65], [256, 66, 257, 258]])

    def test_full_vector_tolerance_detects_nonwinner_and_near_zero_error(self):
        reference = vector()
        candidate = vector()
        candidate[258] = float(torch.tensor(0.000002, dtype=torch.float32)).hex()
        self.assertTrue(c.compare_vectors(reference, candidate)["numeric_pass"])
        above = torch.nextafter(torch.tensor(0.000002), torch.tensor(float("inf")))
        candidate[258] = float(above).hex()
        report = c.compare_vectors(reference, candidate)
        self.assertFalse(report["numeric_pass"])
        self.assertEqual(report["violating_token_ids"], [258])
        self.assertGreater(float.fromhex(report["max_normalized_error_hex"]), 1)

    def test_exact_decisions_are_separate_from_close_logits(self):
        reference = vector()
        candidate = vector()
        candidate[1] = float(torch.tensor(1e-7)).hex()
        self.assertTrue(c.compare_vectors(reference, candidate)["numeric_pass"])
        self.assertEqual(c.decision(reference, "greedy", 0)["token"], 0)
        self.assertEqual(c.decision(candidate, "greedy", 0)["token"], 1)
        self.assertTrue(c.decision(candidate, "greedy", 0)["near_tie_diagnostic"])

    def test_ties_all_digits_and_real_syntax_eos_decision(self):
        tied = vector()
        self.assertEqual(c.decision(tied, "greedy", 0)["token"], 0)
        self.assertEqual(c.decision(tied, "syntax", 0)["token"], 48)
        self.assertTrue(c.decision(tied, "syntax", 0)["exact_tie"])
        self.assertIsNone(c.decision(tied, "syntax", 1)["top_two_margin_hex"])
        for digit in range(48, 58):
            values = vector()
            values[digit] = (1.0).hex()
            self.assertEqual(c.decision(values, "syntax", 0)["token"], digit)
            self.assertEqual(c.decision(values, "syntax", 1)["token"], 257)

    def test_noncanonical_nonfinite_and_nonfloat32_vector_entries_rejected(self):
        for bad in ("nan", "inf", "-inf", "0x1p+0", "0x1p+99999", "0x1.0000000000001p+0",
                    "0x1.fffffffffffffp+1023", 0, True):
            values = vector()
            values[200] = bad
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                c.values(values)
        with self.assertRaises(ContractError):
            c.values(vector()[:-1])

    def test_independent_generations_restore_canonical_rows_and_keep_special_tokens(self):
        rows = workload(["A", "B", "C", *["D"] * 14])
        expected = [[257], [49, 257], [256, 258], *[[50, 50] for _ in range(14)]]
        for layout in spec.LAYOUTS:
            model = BranchingModel()
            result, calls = e.generate(model, rows, layout, "greedy", e.Budget())
            with self.subTest(layout=layout):
                self.assertEqual([r["item_id"] for r in result], [r["item_id"] for r in rows])
                self.assertEqual([r["tokens"] for r in result], expected)
                self.assertEqual([r["stop_reason"] for r in result],
                                 ["eos", "eos", *["token_cap"] * 15])
                self.assertEqual([len(r["steps"]) for r in result], [1, *[2] * 16])
                self.assertEqual(len(calls), len(model.inputs))

    def test_finished_carrier_does_not_append_eos_or_receive_extra_decision(self):
        rows = workload(["A", "B"])
        model, budget = BranchingModel(), e.Budget()
        result, calls = e.generate(model, rows, "batch8_32", "greedy", budget)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["active_local_rows"], [1])
        self.assertEqual(calls[1]["live_lengths"], [2, 3])
        self.assertEqual(model.inputs[1][0][0][model.inputs[1][1][0]].tolist(), [256, 65])
        self.assertEqual(result[0]["tokens"], [257])
        self.assertEqual(budget.snapshot(), {"physical_forward_calls": 2,
                                           "row_evaluations": 4, "active_decisions": 3})

    def test_context_exhaustion_without_repair_and_syntax_eos_cost(self):
        rows = workload(["D" * 31])
        for mode in ("greedy", "syntax"):
            model = BranchingModel()
            result, _ = e.generate(model, rows, "singleton_min", mode, e.Budget())
            self.assertEqual(result[0]["stop_reason"], "context_cap")
            self.assertEqual(len(result[0]["tokens"]), 1)
            self.assertEqual(len(model.inputs), 1)
        model = BranchingModel()
        result, _ = e.generate(model, workload(["A"]), "singleton_32", "syntax", e.Budget())
        self.assertEqual(result[0]["tokens"], [54, 257])
        self.assertEqual(len(model.inputs), 2)
        with self.assertRaises(ContractError):
            e.generate(BranchingModel(), workload(["D" * 32]), "singleton_min", "syntax", e.Budget())

    def test_generated_trajectories_have_no_reference_or_label_input(self):
        rows = workload(["A", "B"])
        original = copy.deepcopy(rows)
        with patch.object(e, "references", side_effect=AssertionError("generation must not read reference")):
            result, _ = e.generate(BranchingModel(), rows, "batch8_32", "syntax", e.Budget())
        self.assertEqual(rows, original)
        self.assertEqual([r["tokens"] for r in result], [[54, 257], [49, 257]])
        with self.assertRaises(TypeError):
            e.generate(BranchingModel(), rows, "batch8_32", "greedy", e.Budget(), answer="0")

    def test_budget_rejects_overrun_before_charge(self):
        budget = e.Budget()
        budget.physical_calls = 3000
        before = budget.snapshot()
        with self.assertRaises(ContractError):
            budget.charge(1, 1)
        self.assertEqual(budget.snapshot(), before)
        budget = e.Budget()
        budget.row_evaluations = 8064
        with self.assertRaises(ContractError):
            budget.charge(1, 1)
        budget = e.Budget()
        budget.started -= 301
        with self.assertRaises(ContractError):
            budget.charge(1, 1)

    def test_rng_state_detects_python_numpy_and_torch_changes(self):
        states = e.rng_state()
        try:
            for draw in (random.random, np.random.random, lambda: torch.rand(1)):
                before = e.rng_state()
                self.assertTrue(e.unchanged_rng(before))
                draw()
                self.assertFalse(e.unchanged_rng(before))
        finally:
            random.setstate(states[0])
            np.random.set_state(states[1])
            torch.set_rng_state(states[2])

    def test_immutability_guard_rejects_gradients_training_alias_and_identity_changes(self):
        for mutation in ("gradient", "training", "alias", "identity", "rng"):
            model = torch.nn.Module()
            model.token_embedding = torch.nn.Parameter(torch.zeros((2, 2)))
            model.output_weight = model.token_embedding
            model.eval()
            state = e.rng_state()
            if mutation == "gradient":
                model.token_embedding.grad = torch.zeros_like(model.token_embedding)
            elif mutation == "training":
                model.train()
            elif mutation == "alias":
                model.output_weight = torch.nn.Parameter(torch.zeros((2, 2)))
            observed_identity = "different" if mutation == "identity" else "unchanged"
            with patch.object(e, "model_identity", return_value=observed_identity), \
                    patch.object(e, "unchanged_rng", return_value=mutation != "rng"):
                with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                    e.assert_immutable(model, "unchanged", state)

    def test_forced_prefix_uses_prior_model_token_and_not_independent_output(self):
        rows = workload(["B"])
        records, _ = e.forced(BranchingModel(), rows, [[57, 257]], "singleton_min", e.Budget())
        self.assertEqual(records[0]["steps"][0]["prefix_ids"], [256, 66])
        self.assertEqual(records[0]["steps"][1]["prefix_ids"], [256, 66, 57])
        generated, _ = e.generate(BranchingModel(), rows, "singleton_min", "greedy", e.Budget())
        self.assertEqual(generated[0]["tokens"], [49, 257])
        self.assertEqual(generated[0]["steps"][1]["prefix_ids"], [256, 66, 49])

    def test_comparison_rejects_missing_extra_duplicate_or_reordered_rows_and_layouts(self):
        rows = workload(["A", "B"])
        layouts = {}
        for layout in spec.LAYOUTS:
            fixed, fixed_calls = e.forced(BranchingModel(), rows, [[54, 257], [49, 257]],
                                         layout, e.Budget())
            generations, calls = {}, {"forced": fixed_calls}
            for mode in ("greedy", "syntax"):
                generations[mode], calls[mode] = e.generate(BranchingModel(), rows, layout,
                                                            mode, e.Budget())
            layouts[layout] = {"forced": fixed, "generated": generations, "calls": calls}
        item_ids = [r["item_id"] for r in rows]
        self.assertEqual(len(c.comparisons(layouts, item_ids)), 13)
        for mutation in ("missing_layout", "extra_layout", "missing_row", "duplicate_row",
                         "reordered_rows", "extra_mode"):
            changed = copy.deepcopy(layouts)
            if mutation == "missing_layout":
                del changed["payload8_32"]
            elif mutation == "extra_layout":
                changed["invented"] = copy.deepcopy(changed["batch8_32"])
            elif mutation == "extra_mode":
                changed["batch8_32"]["generated"]["invented"] = []
            else:
                target = changed["batch8_32"]["forced"]
                if mutation == "missing_row":
                    target.pop()
                elif mutation == "duplicate_row":
                    target[1] = copy.deepcopy(target[0])
                else:
                    target.reverse()
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                c.comparisons(changed, item_ids)


if __name__ == "__main__":
    unittest.main()
