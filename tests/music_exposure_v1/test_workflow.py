"""Synthetic receipt/score doubles test auditing mechanics, not native custody."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from trinite.contracts import ContractError, identity, json_bytes, parse_json
from music_exposure_v1 import numerical, workflow, reporting
from test_spec import predictions

RAW = b'synthetic request'
RID = identity(RAW)
AID = identity(b'synthetic admission')
MID = identity(b'synthetic model')


def bundle(root):
    """Deliberately fake tensors/checkpoint structure; native reader is mocked."""
    artifacts = {}
    for step in (0, 64, 640):
        prefix = f'step-{step:03d}'
        records = predictions()
        artifacts[prefix+'.safetensors'] = b'NOT A REAL CHECKPOINT'
        artifacts[prefix+'-checkpoint.json'] = json_bytes({'step': step,
            'cursor': step*8, 'target_tokens': step*16, 'model_identity': MID,
            'initial_train_loss_hex': (1.0).hex(),
            'history': [{'synthetic_step': i} for i in range(step)],
            'validation': [{'synthetic_step': i} for i in range(64, step+1, 64)]})
        artifacts[prefix+'-predictions.json'] = json_bytes(records)
        artifacts[prefix+'-journal.jsonl'] = b''.join(json_bytes(p) for p in records['predictions'])
    report = {'schema': 'trinite.music-exposure-report.v1', 'seed': 17,
        'request_identity': RID, 'admission_identity': AID, 'model_identity': MID,
        'verification_scope': workflow.SCOPE, 'step': 640, 'target_tokens': 10240,
        'observations': {str(step): {'correct': 40, 'examples': 40} for step in (0,64,640)},
        'observations_contrasts': {str(step): reporting.summarize(artifacts[f'step-{step:03d}-predictions.json'])
                                  for step in (0,64,640)},
        'training_fit': True, 'update_seconds_hex': (100.0).hex(), **workflow.CLAIMS}
    artifacts['report.json'] = json_bytes(report)
    (root/'request.json').write_bytes(RAW)
    for name, data in artifacts.items():
        (root/name).write_bytes(data)
    return reseal(root)


def reseal(root):
    completion = {'schema': 'trinite.music-exposure-completion.v1', 'request_identity': RID,
        'cell': 'train-17', 'artifacts': {name: identity((root/name).read_bytes())
                                       for name in workflow.names() if (root/name).exists()}}
    raw = json_bytes(completion); (root/'completion.json').write_bytes(raw)
    return identity(raw)


def mutate_json(root, name, operation):
    value = parse_json((root/name).read_bytes(), canonical=True)
    operation(value)
    (root/name).write_bytes(json_bytes(value))
    return reseal(root)


class WorkflowAuditTests(unittest.TestCase):
    def audit(self, root, expected):
        # Native tests separately exercise actual safe-tensor parsing/progress.
        with patch.object(workflow, 'validate', return_value={'admission_identity': AID}), \
                patch.object(numerical, 'check_score', return_value={'correct': 40, 'examples': 40}), \
                patch.object(numerical, 'verify_checkpoint',
                    side_effect=lambda payload, meta, *args: SimpleNamespace(cursor=meta['cursor'], target_tokens=meta['target_tokens'])):
            return workflow.audit(root, RAW, RID, 17, expected)

    def test_synthetic_completed_receipt_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); expected = bundle(root)
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            result = self.audit(root, expected)
            self.assertTrue(result['verified'])
            self.assertFalse(result['predictions_replayed'])
            self.assertFalse(result['optimizer_replayed'])
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_missing_intermediate_checkpoint_artifact_rejected_even_resealed(self):
        for suffix in workflow.SUFFIXES:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); bundle(root)
                (root/('step-064'+suffix)).unlink()
                with self.assertRaises(ContractError):
                    self.audit(root, reseal(root))

    def test_fabricated_final_progress_score_fit_and_budget_rejected(self):
        mutations = [lambda r: r.update(step=64), lambda r: r.update(target_tokens=1024),
            lambda r: r.update(training_fit=False), lambda r: r.update(training_fit=1),
            lambda r: r.update(update_seconds_hex=(120.001).hex()),
            lambda r: r['observations']['640'].update(correct=39),
            lambda r: r.update(model_identity=identity(b'other model')),
            lambda r: r['observations_contrasts']['640']['all'].update(complete_groups=0)]
        for mutation in mutations:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); bundle(root)
                expected = mutate_json(root, 'report.json', mutation)
                with self.assertRaises(ContractError):
                    self.audit(root, expected)

    def test_history_and_validation_prefix_lineage_rejected(self):
        for field in ('history', 'validation'):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); bundle(root)
                expected = mutate_json(root, 'step-640-checkpoint.json',
                    lambda value: value[field][0].update(synthetic_step=-1))
                with self.assertRaisesRegex(ContractError, 'lineage'):
                    self.audit(root, expected)

    def test_duplicate_journal_mismatch_rejected_even_resealed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bundle(root)
            path = root/'step-064-journal.jsonl'
            path.write_bytes(path.read_bytes()+json_bytes(predictions()['predictions'][0]))
            with self.assertRaisesRegex(ContractError, 'journal differs'):
                self.audit(root, reseal(root))

    def test_extra_artifact_and_unbound_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); expected = bundle(root)
            (root/'extra.json').write_bytes(b'{}\n')
            with self.assertRaisesRegex(ContractError, 'physical inventory'):
                self.audit(root, expected)
            (root/'extra.json').unlink()
            (root/'step-064.safetensors').write_bytes(b'OTHER FAKE TENSOR BYTES')
            with self.assertRaisesRegex(ContractError, 'bound artifact bytes'):
                self.audit(root, expected)


if __name__ == '__main__':
    unittest.main()
