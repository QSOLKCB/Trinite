"""Independent strict-report tampering tests with synthetic models and bindings."""
import copy
from contextlib import contextmanager
import unittest
from unittest.mock import patch

import torch

from trinite.contracts import ContractError, identity, json_bytes
from batch_composition_v1 import experiment as e, spec, validation as v


class ReportModel(torch.nn.Module):
    def __init__(self, contaminated=False):
        super().__init__()
        self.token_embedding = torch.nn.Parameter(torch.zeros(1))
        self.output_weight = self.token_embedding
        self.contaminated = contaminated
        self.eval()

    def forward(self, ids, mask):
        result = torch.zeros((*ids.shape, 259), dtype=torch.float32)
        for row in range(len(ids)):
            prefix = ids[row][mask[row]].tolist()
            token = 257 if prefix[1] == 65 or len(prefix) > 2 else 49
            result[row, :, 54] = 2
            result[row, :, token] = 10
        if self.contaminated and len(ids) > 1:
            result[:, :, 3] = 0.1
        result *= mask[:, :, None]
        return type("Output", (), {"logits": result})()


@contextmanager
def synthetic_context(contaminated=False):
    rows = [{"item_id": f"item-{i}", "split": "train", "prompt": prompt,
             "prompt_identity": identity(prompt.encode())}
            for i, prompt in enumerate(("A", "B"))]
    reference_tokens = [[54, 257], [49, 257]]
    refs = {key: reference_tokens for key in spec.CELLS}
    weight_identity = e.model_identity(ReportModel())
    old = {"cells": [{"seed": seed, "arm": arm, "final_checkpoint": {}}
                     for seed, arm in spec.CELLS]}
    historical = {"cells": []}
    for seed, arm in spec.CELLS:
        modes = {}
        for mode, tokens in (("greedy", [[257], [49, 257]]), ("syntax", reference_tokens)):
            modes[mode] = {"predictions": [
                {"example_identity": row["item_id"], "generated_tokens": token, "stop_reason": "eos"}
                for row, token in zip(rows, tokens)]}
        historical["cells"].append({"seed": seed, "arm": arm, "modes": modes,
                                    "model_identity": weight_identity})
    binding = {"workload": rows,
               "schedules": {name: spec.schedule(rows, name) for name in spec.LAYOUTS},
               "checkpoint_identities": [identity(json_bytes({}))] * 4}
    with patch.object(e, "request", return_value=binding), \
            patch.object(e, "workload", return_value=rows), \
            patch.object(e, "references", return_value=refs), \
            patch.object(e, "reference_report", return_value=historical), \
            patch.object(e, "original", return_value=old), \
            patch.object(e, "load_model", side_effect=lambda _: ReportModel(contaminated)), \
            patch.object(v, "workload", return_value=rows), \
            patch.object(v, "references", return_value=refs), \
            patch.object(v, "reference_report", return_value=historical):
        yield e.report(), binding


class ValidationReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_positive_and_negative_reports_validate_without_erasing_negative_gates(self):
        for contaminated in (False, True):
            with synthetic_context(contaminated) as (value, binding):
                before = json_bytes(value)
                result = v.validate_report(value, binding)
                self.assertIs(result["conforms"], not contaminated)
                self.assertEqual(json_bytes(value), before)
                self.assertEqual(len(value["cells"]), 4)
                self.assertTrue(all(cell["decision_pass"] for cell in value["cells"]))

    def test_strict_schema_decisions_prefixes_stops_and_accounting_are_not_trusted(self):
        def mutate(value, target):
            cell = value["cells"][0]
            layout = cell["layouts"]["batch8_32"]
            generated = layout["generated"]["syntax"][0]
            forced = layout["forced"][0]["steps"][0]
            if target == "root_extra":
                value["ignored"] = True
            elif target == "training_bool":
                value["training_updates"] = False
            elif target == "cell_missing":
                value["cells"].pop()
            elif target == "cell_reorder":
                value["cells"][0], value["cells"][1] = value["cells"][1], value["cells"][0]
            elif target == "model_identity":
                cell["model_identity"] = identity(b"different weights")
            elif target == "checkpoint_identity":
                cell["checkpoint_identity"] = identity(b"different endpoint")
            elif target == "step_extra":
                forced["ignored"] = "extra"
            elif target == "row_extra":
                generated["answer"] = "0"
            elif target == "generated_prefix":
                generated["steps"][1]["prefix_ids"][-1] = 55
            elif target == "reference_prefix":
                layout["forced"][0]["steps"][1]["prefix_ids"][-1] = 55
            elif target == "reference_next":
                forced["reference_next_token"] = 55
            elif target == "stored_decision":
                generated["steps"][0]["decision"]["token"] = 55
            elif target == "stored_margin":
                generated["steps"][0]["decision"]["top_two_margin_hex"] = (999.0).hex()
            elif target == "emitted_token":
                generated["tokens"][0] = 55
            elif target == "stop_reason":
                generated["stop_reason"] = "token_cap"
            elif target == "missing_step":
                generated["steps"].pop()
            elif target == "step_after_eos":
                raw = layout["generated"]["greedy"][0]
                raw["steps"].append(copy.deepcopy(raw["steps"][0]))
            elif target == "call_identity":
                layout["calls"]["greedy"][0]["input_identity"] = identity(b"wrong input")
            elif target == "finished_row_active":
                layout["calls"]["greedy"][1]["active_local_rows"] = [0, 1]
            elif target == "call_live_length":
                layout["calls"]["greedy"][1]["live_lengths"][0] += 1
            elif target == "budget":
                value["budget"]["row_evaluations"] += 1
            elif target == "boolean_gate_as_integer":
                cell["numeric_pass"] = 1
            elif target == "comparison_summary":
                cell["comparisons"][0]["numeric_pass"] = False
            else:
                raise AssertionError(target)

        mutations = ("root_extra", "training_bool", "cell_missing", "cell_reorder",
                     "model_identity", "checkpoint_identity", "step_extra", "row_extra",
                     "generated_prefix", "reference_prefix", "reference_next", "stored_decision",
                     "stored_margin", "emitted_token", "stop_reason", "missing_step",
                     "step_after_eos", "call_identity", "finished_row_active", "call_live_length",
                     "budget", "boolean_gate_as_integer", "comparison_summary")
        with synthetic_context() as (value, binding):
            for target in mutations:
                changed = copy.deepcopy(value)
                mutate(changed, target)
                with self.subTest(target=target), self.assertRaises(ContractError):
                    v.validate_report(changed, binding)


if __name__ == "__main__":
    unittest.main()
