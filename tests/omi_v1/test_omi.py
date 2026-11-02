"""Public engineering conformance, never fresh scientific evaluation."""
import hashlib
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

import omi_module_v1 as omi
from trinite.tokenizer import BOS, EOS


class OmiTests(unittest.TestCase):
    def invoke(self, operation, raw):
        # communicate handles an early stdin close; no unconditional write race.
        return subprocess.run([str(omi.BINARY), operation], input=raw,
                              capture_output=True, timeout=5)

    def test_native_tasks_and_oracle(self):
        for word in (0, 1, 42, 0x5A3C, 0xABCD, 65535):
            for task in omi.TASKS:
                for horizon in (range(1, 8) if task in ('forward', 'backward') else (1,)):
                    observation = omi.public(word, task, horizon)
                    self.assertEqual(omi.native('observe', observation), observation)
                    self.assertEqual(omi.native('evaluate-privileged', observation),
                                     dict(schema='trinite.omi-outcome.v1', answer=omi.expected(observation)))

    def test_exhaustive_arithmetic_inventory(self):
        table = [omi.arithmetic_delta(word) for word in range(65536)]
        self.assertEqual(len(set(table)), 65536)
        # Byte-order explicit table binding; no cached receipt accepted as oracle.
        digest = hashlib.sha256(b''.join(x.to_bytes(2, 'little') for x in table)).hexdigest()
        self.assertEqual(digest, 'a0bb32b98090680604ff561e02db2d576194b767bd75bef7284a3d3877fa7c1a')
        seen, counts, families = set(), {1:0, 2:0, 4:0, 8:0}, 0
        for word in range(65536):
            if word in seen:
                continue
            states, current = [], word
            for _ in range(8):
                states.append(current)
                current = table[current]
                if current == word:
                    break
            self.assertEqual(current, word)
            self.assertFalse(seen.intersection(states))
            seen.update(states)
            counts[len(states)] += len(states)
            families += 1
            # Actual family function is exercised at every canonical orbit root.
            self.assertEqual(omi.family(word), word)
            self.assertEqual(omi.family(states[-1]), word)
            self.assertEqual(omi.split(word), omi.split(states[-1]))
        self.assertEqual(counts, {1:8, 2:56, 4:4032, 8:61440})
        self.assertEqual(families, 8724)
        self.assertEqual(counts[8]/65536, .9375)

    def test_public_carrier_and_real_token_counts(self):
        for task in omi.TASKS:
            obs = omi.public(65535, task, 7 if task in ('forward', 'backward') else 1)
            text = omi.prompt(obs)
            self.assertEqual(omi.decode_prompt(text), obs)
            tokens = omi.public_only(obs)
            self.assertEqual(tokens, [BOS, *text.encode('ascii')])
            self.assertNotIn(EOS, tokens)
            ledger = omi.ledger(obs)
            answer_size = 1 if task == 'period' else 4
            self.assertEqual(ledger['complete_context_tokens'], len(text) + answer_size + 2)
            self.assertEqual(ledger['supervised_tokens'], answer_size + 1)
            self.assertEqual(ledger['emitted_tokens_including_eos'], answer_size+1)
            self.assertLessEqual(ledger['complete_context_tokens'], 126)
            self.assertEqual(ledger['optimizer_updates'], 0)
        self.assertEqual(sum(121+k for k in range(5))*8, 4920)
        with patch.object(omi, 'PROMPT_BYTE_CAP', 1):
            with self.assertRaises(ValueError): omi.prompt(omi.public(0))

    def test_no_hidden_oracle_or_provenance_public(self):
        for field in ('answer', 'seed', 'family', 'split', 'source', 'oracle', 'period'):
            with self.assertRaises(ValueError):
                omi.public_only(dict(omi.public(0), **{field:0}))
        with patch.object(omi, 'expected', side_effect=AssertionError('oracle leak')), \
             patch.object(omi, 'orbit', side_effect=AssertionError('orbit leak')), \
             patch.object(omi, 'native', side_effect=AssertionError('native leak')):
            omi.public_only(omi.public(0))
            omi.ledger(omi.public(0))
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('trinite.training', sys.modules)

    def test_train_filter_precedes_oracle_and_mask(self):
        held = next(word for word in range(100) if omi.split(word) != 'train')
        with patch.object(omi, 'native', side_effect=AssertionError('heldout oracle called')):
            with self.assertRaises(ValueError): omi.admit_train(omi.public(held))
        word = next(word for word in range(100) if omi.split(word) == 'train')
        row = omi.admit_train(omi.public(word))
        prefix = len(omi.public_only(row['observation']))
        self.assertEqual(row['encoding']['loss_mask'], [0]*prefix + [1]*5)
        self.assertFalse(row['provenance']['training_authorized'])
        self.assertEqual(row['encoding']['input_ids'][-1], EOS)

    def test_malformed_cli_and_public_schema(self):
        valid = dict(schema='trinite.omi-request.v1', word=0, task='next', horizon=1)
        bad = [b'', b'null', b'{}', b' '*16385, b'{"word":0,"word":1}',
               json.dumps(dict(valid, word=65536)).encode(), json.dumps(dict(valid, word=-1)).encode(),
               json.dumps(dict(valid, word=True)).encode(), json.dumps(dict(valid, word=1.0)).encode(),
               json.dumps(dict(valid, word=2**64)).encode(), json.dumps(dict(valid, horizon=8)).encode(),
               json.dumps(dict(valid, horizon=0)).encode(), json.dumps(dict(valid, task='hidden')).encode(),
               json.dumps(dict(valid, answer='0000')).encode(), json.dumps(dict(valid, horizon=2)).encode(),
               json.dumps(valid).encode()+b' null']
        for raw in bad:
            result = self.invoke('observe', raw)
            self.assertEqual(result.returncode, 2, raw)
            self.assertEqual(result.stdout, b'', raw)
        for operation in ('', 'score', 'default'):
            result = self.invoke(operation, b'x'*16385)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, b'')
        for word in (-1, 65536, True, 1.0):
            with self.assertRaises(ValueError): omi.public(word)
        for text in ('["O1","abcd","next",1]\n=', '["O1","0000","next",8]\n=',
                     '["O1", "0000","next",1]\n=', '["O1","0000","next",1,0]\n='):
            with self.assertRaises(ValueError): omi.decode_prompt(text)

    def test_stale_source_binding_rejected(self):
        self.assertIn('src/trinite/__init__.py', omi.sources())
        actual = omi.identity()
        with patch.object(omi, 'identity', return_value=dict(actual, binary_sha256='0'*64)):
            with self.assertRaises(ValueError): omi.native('observe', omi.public(0))


if __name__ == '__main__':
    unittest.main()
