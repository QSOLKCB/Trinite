from pathlib import Path
import tempfile
import unittest

from trinite.contracts import ContractError, json_bytes, parse_json
from trinite.generalization import (answer_body, component_counts, corpus_coverage, diagnose,
                                   protocol, run_inventory)
from trinite.learning_data import dataset
from trinite.tokenizer import EOS


class GeneralizationTests(unittest.TestCase):
    def test_eos_and_special_token_partition(self):
        for tokens, body, error in (([49,EOS], '1', None), ([49], None, 'missing-eos'),
            ([49,EOS,EOS], None, 'special-token'), ([256,EOS], None, 'special-token'),
            ([49,258], None, 'special-token'), ([255,EOS], None, 'non-ascii')):
            self.assertEqual(answer_body(tokens), (body,error))
        for tokens in ([], [True,EOS], [-1,EOS], [259]):
            with self.assertRaises(ContractError): answer_body(tokens)

    def test_components_require_complete_canonical_output(self):
        item = {'task':'fold', 'answer':'3 1 9'}
        self.assertEqual(component_counts('fold', item, [*b'3 2 9',EOS]), {'count':1,'sum':0,'squares':1})
        for text in (b'3 1 9 ', b'03 1 9', b'3 -0 9', b'3 1', b'3 1 9\xff'):
            self.assertEqual(component_counts('fold', item, [*text,EOS]), dict.fromkeys(('count','sum','squares'),0))
        self.assertEqual(component_counts('fold',item,[*b'3 1 9']),dict.fromkeys(('count','sum','squares'),0))
        item = {'task':'traversal', 'answer':'L[1,2,0]'}
        self.assertEqual(component_counts('lattice',item,[*b'L[0,2,0]',EOS]),{'x':0,'y':1,'z':1})
        self.assertEqual(component_counts('lattice',item,[*b'L[3,2,0]',EOS]),{'x':0,'y':0,'z':0})

    def test_counts_keep_all_examples_and_do_not_unlock_gates(self):
        examples = parse_json(dataset('arithmetic')[0])['examples']
        predictions = []
        for i, item in enumerate(r for r in examples if r['split'] == 'test'):
            tokens = [*item['answer'].encode(),EOS] if i == 0 else [256,EOS] if i == 1 else [255,EOS] if i == 2 else [49]
            predictions.append({'example_identity':item['example_identity'], 'generated_tokens':tokens, 'correct':i==0})
        groups = diagnose('arithmetic', examples, predictions, 'test')
        tasks = [v for k,v in groups.items() if k.startswith('task:')]
        self.assertEqual(sum(v['examples'] for v in tasks),64)
        self.assertEqual(sum(v['correct'] for v in tasks),1)
        for v in groups.values(): self.assertEqual(sum(v['errors'].values()),v['examples'])
        self.assertTrue(all(value is False for value in protocol()['gates'].values()))
        with self.assertRaises(ContractError): diagnose('arithmetic',examples,predictions[:-1],'test')
        predictions[0]['correct'] = False
        with self.assertRaises(ContractError): diagnose('arithmetic',examples,predictions,'test')

    def test_coverage_detects_semantic_and_family_leakage(self):
        for workload in ('arithmetic','fold','lattice'):
            examples = parse_json(dataset(workload)[0])['examples']; result = corpus_coverage(examples)
            self.assertEqual(set(result), {r['task'] for r in examples})
            extra = {**examples[0], 'split':'test' if examples[0]['split']!='test' else 'train'}
            with self.assertRaises(ContractError): corpus_coverage([*examples,extra])
            extra['family_id'] = 'another-family'
            with self.assertRaises(ContractError): corpus_coverage([*examples,extra])

    def test_inventory_rejects_symlinks_and_preserves_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/'file').write_bytes(b'original')
            self.assertEqual(run_inventory(root),run_inventory(root))
            (root/'alias').symlink_to(root/'file')
            with self.assertRaises(ContractError): run_inventory(root)
            self.assertEqual((root/'file').read_bytes(),b'original')

class DiagnosticEvidenceTests(unittest.TestCase):
    def test_fresh_closed_evidence_rejects_tampered_root_receipt_and_payload(self):
        from unittest.mock import patch
        from trinite.generalization import retain_evidence, verify_evidence
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); output = root/'output'; output.mkdir(); run = root/'run'; run.mkdir()
            (output/'request.json').write_bytes(b'{}\n'); (output/'report.json').write_bytes(b'{}\n')
            parent = {}
            from trinite.contracts import identity
            for name,key in (('request.json','request_identity'),('summary.json','summary_identity'),('test-decision.json','decision_identity')):
                raw = ('{"name":"'+name+'"}\n').encode(); (run/name).write_bytes(raw); parent[key] = identity(raw)
            for workload in ('arithmetic','fold','lattice'):
                directory=output/'curriculum'/workload;directory.mkdir(parents=True)
                for name in ('dataset.json','manifest.json'): (directory/name).write_bytes(b'{}\n')
            with patch('trinite.generalization.protocol',return_value={'parent':parent}):
                retain_evidence(output,run); self.assertTrue(verify_evidence(output,run)['integrity_verified'])
                receipt = output/'verification.json'; original = receipt.read_bytes()
                changed = {**parse_json(original), 'manifest_identity': 'sha256:'+'0'*64}
                for label, raw in (('missing', None), ('modified', json_bytes(changed)),
                                   ('noncanonical', original+b'\n')):
                    with self.subTest(receipt=label):
                        if raw is None: receipt.unlink()
                        else: receipt.write_bytes(raw)
                        with self.assertRaises(ContractError): verify_evidence(output,run)
                    receipt.write_bytes(original)
                    self.assertEqual(json_bytes(verify_evidence(output,run)), original)
                    self.assertEqual(receipt.read_bytes(), original)
                (output/'report.json').write_bytes(b'{"changed":true}\n')
                with self.assertRaises(ContractError): verify_evidence(output,run)
                (output/'report.json').write_bytes(b'{}\n')
                payload=next((output/'provenance/artifacts/sha256').iterdir());payload.write_bytes(b'corrupt')
                with self.assertRaises(ContractError): verify_evidence(output,run)


if __name__ == '__main__': unittest.main()
