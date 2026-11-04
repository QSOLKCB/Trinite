# SPDX-License-Identifier: MPL-2.0
"""Synthetic, standard-library-only negative and numerical evidence controls."""
import hashlib
import json
import math
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

from teacher_fit_review_v1 import core


def panel():
    out = []
    for family in ('S1', 'S2', 'W'):
        for word in ('0000', '1000', '2000', '0100', '0200', '0010'):
            for task, answer in (('apply', '0120'), ('inverse', '2010'),
                                 ('check', '0'), ('check', '1')):
                payload = f'{family}|{word}|{task}|{answer}'
                out.append({'id': core.sha(payload.encode()),
                            'family': family, 'answer': answer,
                            'observation': {'task': task, 'word': word, 'circuit': family}})
    return out


class NumericReview(unittest.TestCase):
    def test_full_vocabulary_and_correct_binary_eos_odds(self):
        for eos_logit in (-735., -730., -700., 0., 700., 730., 735.):
            with self.subTest(eos_logit=eos_logit):
                z = [0.] * core.VOCAB
                z[core.EOS] = eos_logit
                raw = core.VECTOR.pack(*z)
                metric = core.probs(core.VECTOR.unpack(raw), 48)
                self.assertAlmostEqual(float.fromhex(metric['eos_rest_log_odds_hex']),
                                       eos_logit - math.log(258), places=11)
                self.assertTrue(core.metric_equal(metric, metric))
                self.assertEqual(metric['eos_rest_probability_check_hex'] is None,
                                 abs(eos_logit) >= 730.)

    def test_ties_and_full_distribution_rank(self):
        metric = core.probs((0.,) * core.VOCAB, 49)
        self.assertEqual(metric['argmax'], 0)
        self.assertEqual(metric['target_rank'], 50)
        self.assertEqual(metric['best_residue_token'], 48)
        self.assertEqual(metric['target_residue_rank'], 2)
        self.assertAlmostEqual(float.fromhex(metric['full_eos_probability_hex']), 1/259)
        tampered = dict(metric, argmax=49)
        self.assertFalse(core.metric_equal(tampered, metric))
        tampered = dict(metric, eos_rest_log_odds_hex='0x1.0p+0')
        self.assertFalse(core.metric_equal(tampered, metric))
        tampered = dict(metric, target_nll_hex=None)
        self.assertFalse(core.metric_equal(tampered, metric))

    def test_classifier_no_repairs_and_premature_eos(self):
        row = next(r for r in panel() if r['observation']['task'] == 'apply')
        self.assertEqual(core.classify([48, core.EOS], row)['classification'], 'premature_eos')
        self.assertEqual(core.classify([49, 48, 50, 48, core.EOS], row)['classification'],
                         'wrong_answer')
        self.assertEqual(core.classify([61] * 5, row)['classification'], 'invalid_symbol')
        with self.assertRaises(core.EvidenceError):
            core.classify([core.EOS, 48], row)

    def test_complete_fixed_control_scoring(self):
        rows = panel()
        self.assertEqual(len(rows), 72)
        for constant in (48, 49):
            predictions = [{'id': row['id'],
                            'tokens': [constant] * len(row['answer']) + [core.EOS]}
                           for row in rows]
            scored = core.score(rows, predictions)
            self.assertEqual(scored['groups']['all']['exact'], 18)
            self.assertEqual(scored['groups']['apply']['exact'], 0)
            self.assertEqual(scored['groups']['inverse']['exact'], 0)
            self.assertEqual(scored['groups']['all']['premature_eos'], 0)
            self.assertEqual(sum(x['nonzero_targets'] for x in scored['vector_positions']), 72)
            with self.assertRaises(core.EvidenceError):
                core.score(rows, predictions[:-1])
            corrupt = list(predictions)
            corrupt[0] = dict(corrupt[0], id='unexpected')
            with self.assertRaises(core.EvidenceError):
                core.score(rows, corrupt)


class CustodyReview(unittest.TestCase):
    def _zip(self, entries):
        temp = tempfile.TemporaryDirectory()
        path = Path(temp.name) / 'packet.zip'
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as z:
            for name, data, mode in entries:
                if mode is None:
                    z.writestr(name, data)
                else:
                    info = zipfile.ZipInfo(name)
                    info.create_system = 3
                    info.external_attr = mode << 16
                    z.writestr(info, data)
        return temp, path

    def test_anchored_archive_bytes_and_crc(self):
        temp, path = self._zip([('a.txt', b'a', None)])
        with temp:
            self.assertNotEqual(core.anchored_hash(path), 'sha256:' + '0'*64)
            z, members = core.safe_archive(path, core.anchored_hash(path))
            with z:
                self.assertEqual(members['a.txt']['identity'], core.sha(b'a'))
            with self.assertRaisesRegex(core.EvidenceError, 'archive SHA-256'):
                core.safe_archive(path, 'sha256:'+'0'*64)

    def test_zip_traversal_symlink_duplicate_and_extra_rejected(self):
        for entries in (
            [('../escape', b'x', None)],
            [('/absolute', b'x', None)],
            [('dir\\file', b'x', None)],
            [('link', b'x', stat.S_IFLNK | 0o777)],
            [('same', b'a', None), ('same', b'b', None)],
        ):
            with self.subTest(entries=[e[0] for e in entries]):
                temp, path = self._zip(entries)
                with temp, self.assertRaises(core.EvidenceError):
                    core.safe_archive(path, core.anchored_hash(path))
        hashes = {'s/a': {'identity': core.sha(b'a'), 'bytes': 1},
                  's/unlisted': {'identity': core.sha(b'b'), 'bytes': 1}}
        with self.assertRaisesRegex(core.EvidenceError, 'inventory/hash mismatch'):
            core.members_check(hashes, 's/', [
                {'name':'a','identity':core.sha(b'a'),'bytes':1}])

    def test_external_sidecar_canonical_and_conflicts(self):
        digest = hashlib.sha256(b'test').hexdigest()
        self.assertEqual(core.parse_anchor((digest + '  /private/packet.zip\n').encode()),
                         'sha256:'+digest)
        for item in (b'', b'invalid\n', (digest+'\n'+digest).encode(), b'UPPERCASE'):
            with self.assertRaises(core.EvidenceError):
                core.parse_anchor(item)

    def test_json_requires_canonical_sorted_lf_and_no_nan(self):
        for raw, valid in ((b'{"a":1,"b":2}\n', True),
                           (b'{"b":2,"a":1}\n', False),
                           (b'{"a":1}\n', True),
                           (b'{"a":NaN}\n', False)):
            with self.subTest(raw=raw):
                temp, path = self._zip([('a.json', raw, None)])
                with temp, zipfile.ZipFile(path) as z:
                    if valid:
                        self.assertIsInstance(core.jdoc(z,'a.json'),dict)
                    else:
                        with self.assertRaises((core.EvidenceError, ValueError)):
                            core.jdoc(z,'a.json')

    def test_no_training_or_archive_source_execution_imports(self):
        path = Path(__file__).resolve().parents[2] / 'scripts/teacher_fit_review_v1/core.py'
        data = path.read_text()
        self.assertNotIn('import torch', data)
        self.assertNotIn('exec(', data)
        self.assertNotIn('subprocess', data)
        self.assertNotIn('pickle.load', data)


if __name__ == '__main__':
    unittest.main()
