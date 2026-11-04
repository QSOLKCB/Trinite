"""One actual native CPU optimizer update, only engineering conformance."""
from pathlib import Path
import tempfile
import unittest

from teacher_dataset_v1 import core as tt002
from teacher_training_v1 import spec, numerical


class TT003NativeSmoke(unittest.TestCase):
    def test_one_native_update_and_safe_tensor_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "tt002"
            dataset_anchor = tt002.prepare(dataset)
            request_dir = Path(tmp) / "request"
            request_anchor = spec.freeze(request_dir, dataset, dataset_anchor)
            request = spec.verify_request(request_dir, request_anchor, dataset)
            numerical.configure()
            from trinite.training import run_steps, model_identity
            from trinite.comparison_checkpoint import restore_tensors
            state = numerical.fresh_state(dataset, request, request_anchor)
            initial = state.initialization_identity
            run_steps(state, stop_after=1)
            self.assertEqual(state.step, 1)
            self.assertGreater(state.target_tokens, 0)
            self.assertNotEqual(model_identity(state.model), initial)
            payload, _ = numerical.checkpoint(state, request, request_anchor)
            cold = numerical.fresh_state(dataset, request, request_anchor)
            cold.step, cold.cursor, cold.target_tokens = state.step, state.cursor, state.target_tokens
            cold.history, cold.validation = state.history, state.validation
            restored = restore_tensors(cold, payload, model_identity(state.model))
            self.assertEqual(model_identity(restored.model), model_identity(state.model))
            self.assertEqual(restored.target_tokens, state.target_tokens)


if __name__ == "__main__":
    unittest.main()
