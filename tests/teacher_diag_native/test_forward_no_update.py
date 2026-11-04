# SPDX-License-Identifier: MPL-2.0
"""Small pinned-CPU forward-only canary; not the real teacher checkpoint."""
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from teacher_diag_v1.native import raw_greedy
from teacher_dataset_v1 import core as tt002
from teacher_training_v1 import numerical


class TT003NativeForwardCanary(unittest.TestCase):
    def test_two_exposed_train_rows_no_optimizer_update(self):
        numerical.configure()
        from trinite.contracts import ModelConfig, parse_json
        from trinite.model import Decoder
        from trinite.training import model_identity
        import torch
        # Reference architecture, not an actual trainer or checkpoint restore.
        config = ModelConfig.from_dict(numerical.spec.tt001.source_json(
            "configs/teacher/gf3-teacher-90k-v1.json"))
        model = Decoder(config, lane="dense", seed=17, parameter_budget=2_000_000)
        before = model_identity(model)
        packet, _ = tt002.make_packet()
        rows = [parse_json(x + b"\n", canonical=True)
                for x in packet["train.jsonl"].splitlines()[:2]]
        with mock.patch.object(torch.optim.AdamW, "step",
                               side_effect=AssertionError("optimizer update forbidden")):
            for row in rows:
                got = raw_greedy(model, row["prompt"])
                self.assertLessEqual(len(got), 5)
                self.assertTrue(all(type(t) is int and 0 <= t <= 258 for t in got))
        self.assertEqual(model_identity(model), before)


if __name__ == "__main__":
    unittest.main()
