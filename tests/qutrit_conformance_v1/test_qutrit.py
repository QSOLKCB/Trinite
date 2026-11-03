"""Public engineering cases; these tests do not create scientific held-out data."""
from itertools import product
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import qutrit_conformance_v1 as q
from trinite.tokenizer import BOS, EOS

WORDS = [''.join(map(str,x)) for x in product(range(3), repeat=4)]


class QutritTests(unittest.TestCase):
    def invoke(self, operation, raw):
        return subprocess.run([str(q.BINARY), operation], input=raw, capture_output=True, timeout=5)

    def test_exhaustive_native_vs_ket_permutation(self):
        for circuit in q.CIRCUITS:
            for word in WORDS:
                for task in ('apply', 'inverse'):
                    obs = q.public(word, circuit, task)
                    actual = q.native('evaluate-privileged', obs)
                    self.assertEqual(actual, dict(schema='trinite.qutrit-conformance-outcome.v1', answer=q.expected(obs)))
                answer = q.expected(q.public(word, circuit))
                self.assertEqual(q.expected(q.public(answer, circuit, 'inverse')), word)
                for candidate in (answer, str((int(answer[0])+1)%3)+answer[1:]):
                    obs = q.public(word, circuit, 'check', candidate)
                    self.assertEqual(q.native('evaluate-privileged',obs)['answer'], q.expected(obs))
        self.assertEqual(q.expected(q.public('1001','S1')), '1121')
        self.assertEqual(q.expected(q.public('1001','S2')), '1211')

    def test_sign_order_weight_negative_controls_not_blanket_negatives(self):
        correct = q.expected(q.public('1001','S1'))
        for wrong in ('1111', q.expected(q.public('1001','S2'))):
            self.assertEqual(q.expected(q.public('1001','S1','check',wrong)), '0')
        left = q.expected(q.public('1001','S1W'))
        right = q.expected(q.public('1001','WS1'))
        self.assertNotEqual(left, right)
        self.assertEqual(q.expected(q.public('1001','S1W','check',right)), '0')
        # Mutation labels alone don't make a negative: zero is fixed by every gate.
        self.assertEqual(q.expected(q.public('0000','S1','check','0000')), '1')

    def test_public_schema_and_token_masks(self):
        for circuit in q.CIRCUITS:
            for task in q.TASKS:
                obs = q.public('2222', circuit, task, '2222' if task=='check' else '')
                self.assertEqual(q.native('observe',obs),obs)
                text = q.prompt(obs)
                self.assertEqual(q.decode_prompt(text),obs)
                self.assertEqual(q.public_only(obs),[BOS,*text.encode('ascii')])
                ledger=q.ledger(obs)
                size=1 if task=='check' else 4
                self.assertEqual(ledger['complete_context_tokens'],len(text)+size+2)
                self.assertEqual(ledger['supervised_tokens'],size+1)
                self.assertLessEqual(ledger['complete_context_tokens'],126)
                self.assertEqual(ledger['full_prefix_forward_positions'],sum(len(text)+1+k for k in range(size+1)))
                self.assertEqual(ledger['optimizer_updates'],0)
        with patch.object(q,'PROMPT_BYTE_CAP',1):
            with self.assertRaises(ValueError):q.prompt(q.public('0000'))

    def test_split_before_privilege_and_no_public_oracle(self):
        self.assertEqual(q.split('S1'),q.split('S2'))
        for circuit in ('W','S1W','WS1','S2W','WS2'):
            with patch.object(q,'native',side_effect=AssertionError('privilege')):
                with self.assertRaises(ValueError):q.admit_train(q.public('0000',circuit))
        with patch.object(q,'expected',side_effect=AssertionError('oracle')), patch.object(q,'native',side_effect=AssertionError('native')):
            q.public_only(q.public('0000')); q.ledger(q.public('0000'))
        for task in q.TASKS:
            obs=q.public('1001','S1',task,'1121' if task=='check' else '')
            row=q.admit_train(obs)
            self.assertEqual(row['encoding']['loss_mask'],[0]*len(q.public_only(obs))+[1]*(2 if task=='check' else 5))
            self.assertEqual(row['encoding']['input_ids'][-1],EOS)
            self.assertFalse(row['provenance']['training_authorized'])
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('trinite.training',sys.modules)

    def test_cli_fail_closed_and_strict_carriers(self):
        good=dict(q.public('1001'),schema='trinite.qutrit-conformance-request.v1')
        bad=[b'',b'null',b'{}',b' '*16385,b'{"word":"0000","word":"1111"}',json.dumps(good).encode()+b' null']
        for key,values in {'word':[0,True,'0123','012','-100','１２１１'], 'circuit':['S0','S3','WSS'], 'task':['decode',''], 'candidate':['0000',None,False], 'answer':['1121']}.items():
            bad += [json.dumps(dict(good,**{key:v})).encode() for v in values]
        for raw in bad:
            result=self.invoke('observe',raw)
            self.assertEqual((result.returncode,result.stdout),(2,b''),raw)
        for op in ('','default','decode'):
            result=self.invoke(op,b'x'*16385)
            self.assertEqual((result.returncode,result.stdout),(2,b''))
        for extra in ('answer','seed','family','split','source','hidden_error'):
            with self.assertRaises(ValueError):q.public_only(dict(q.public('0000'),**{extra:0}))
        for text in ('["Q1", "0000","S1","apply",""]\n=', '["Q1","0000","S0","apply",""]\n='):
            with self.assertRaises(ValueError):q.decode_prompt(text)

    def test_build_probes_pinned_crate_directory(self):
        def command(argv, **kwargs):
            self.assertEqual(kwargs.get('cwd'), q.CRATE)
            return subprocess.CompletedProcess(argv, 0, b'rustc 1.90.0 (fixture)\n' if argv[0]=='rustc' else b'', b'')
        original = q.RECORD.read_bytes()
        with tempfile.TemporaryDirectory() as temp, patch.object(q, 'RECORD', Path(temp)/'build.json'), patch.object(q.subprocess, 'run', side_effect=command) as run:
            q.build()
        self.assertEqual(q.RECORD.read_bytes(), original)
        self.assertEqual([call.args[0][0] for call in run.call_args_list], ['rustc', 'cargo'])

    def test_binding_and_infrastructure_are_not_negative_labels(self):
        self.assertIn('src/trinite/__init__.py',q.sources())
        actual=q.identity()
        with patch.object(q,'identity',return_value=dict(actual,binary_sha256='0'*64)):
            with self.assertRaises(ValueError):q.native('observe',q.public('0000'))
        with patch.object(q.gate,'bounded_process',side_effect=q.gate.GateError('infrastructure failure')):
            with self.assertRaises(ValueError):q.admit_train(q.public('1001','S1','check','0000'))


if __name__ == '__main__':unittest.main()
