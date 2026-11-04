# SPDX-License-Identifier: MPL-2.0
"""Synthetic, standard-library-only negative and numerical evidence controls."""
import hashlib
import json
import math
import os
import sys
from unittest import mock
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


class HostileEvidenceRegressions(unittest.TestCase):
    """Executed minimal, synthetic reproductions of seven reviewed defects."""

    def test_matching_gold_and_greedy_prefixes_require_exact_fp32_bytes(self):
        z0 = core.VECTOR.pack(*([0.0] * core.VOCAB))
        altered = [0.0] * core.VOCAB
        altered[core.EOS] = 3.0
        z1 = core.VECTOR.pack(*altered)
        index = [{'prefix_tokens': [256, 49], 'probe': 'gold'},
                 {'prefix_tokens': [256, 49], 'probe': 'greedy'},
                 {'prefix_tokens': [256, 49, 48], 'probe': 'gold'},
                 {'prefix_tokens': [256, 49, 48], 'probe': 'greedy'}]
        core.check_prefix_vectors(index, z0 * 4)
        for tampered in (z0 + z1 + z0 + z0, z0 + z0 + z0 + z1):
            with self.assertRaisesRegex(core.EvidenceError, 'identical checkpoint prefix'):
                core.check_prefix_vectors(index, tampered)
        different = [dict(rec) for rec in index]
        different[1]['prefix_tokens'] = [256, 50]
        core.check_prefix_vectors(different, z0 + z1 + z0 + z0)

    def test_required_source_dataset_and_rights_inventories_are_not_self_chosen(self):
        source_names = set(core.REQUIRED_SOURCE_FILES | core.TT002_SOURCE_FILES)
        source_names.update('docs/synthetic-%04d.txt' % i for i in
                            range(core.FROZEN_SOURCE_COUNT - len(source_names)))
        source = {'commit': core.FROZEN_SOURCE_COMMIT,
                  'members': [{'path': n, 'identity': core.sha(n.encode())}
                              for n in sorted(source_names)]}
        src_lookup = {m['path']: m['identity'] for m in source['members']}
        ds = {'schema': 'trinite.teacher-gf3-dataset.v1',
              'files': [{'name': n, 'identity': core.sha(n.encode()), 'bytes': 3}
                        for n in sorted(core.DATASET_FILES)],
              'source_closure': [{'path': n, 'identity': src_lookup[n]}
                                 for n in sorted(core.TT002_SOURCE_FILES)]}
        hashes = {'dataset/manifest.json': {'identity': core.FROZEN_DATASET_ID},
                  'dataset/dataset-rights.json': {'identity': core.FROZEN_RIGHTS_ID}}
        with mock.patch.object(core, 'FROZEN_SOURCE_ID', core.sha(core.canonical(source))):
            core.fixed_inventories(source, ds, hashes)
            for invalid in (
                dict(ds, files=[m for m in ds['files'] if m['name'] != 'dataset-rights.json']),
                dict(ds, files=[m for m in ds['files'] if m['name'] != 'privileged-targets.jsonl']),
                dict(ds, source_closure=[]),
            ):
                with self.assertRaises(core.EvidenceError):
                    core.fixed_inventories(source, invalid, hashes)
            truncated = dict(source, members=source['members'][:-1])
            with mock.patch.object(core, 'FROZEN_SOURCE_ID', core.sha(core.canonical(truncated))):
                with self.assertRaisesRegex(core.EvidenceError, 'source closure'):
                    core.fixed_inventories(truncated, ds, hashes)
            with self.assertRaisesRegex(core.EvidenceError, 'dataset/rights'):
                core.fixed_inventories(source, ds,
                   {**hashes, 'dataset/dataset-rights.json': {'identity': core.sha(b'wrong')}})

    def test_approval_requires_all_fields_exactly_bound_to_request(self):
        request = {'source': {'commit': core.FROZEN_SOURCE_COMMIT}, 'seed': 17,
                   'dataset_manifest_identity': core.FROZEN_DATASET_ID,
                   'schedule_identity': core.sha(b'schedule'),
                   'environment': {'runtime': {'cpu': True}},
                   'output_directory': '/private/seed-17', 'limits': {'updates': 256}}
        anchor, identity = core.sha(b'fixed-request'), core.sha(b'approval')
        correct = core.approval_expected(request, anchor)
        core.validate_approval(correct, request, anchor, identity, identity)
        for field, value in (('owner','NOT THE OWNER'),('scope','unrelated scope'),
                             ('dataset_manifest_identity','sha256:'+'0'*64),
                             ('schedule_identity','sha256:'+'0'*64),
                             ('environment_identity','sha256:'+'0'*64),
                             ('output_directory','/tmp/elsewhere'),
                             ('training_authorized',False)):
            with self.subTest(field=field):
                with self.assertRaisesRegex(core.EvidenceError, 'complete owner approval'):
                    core.validate_approval(dict(correct, **{field:value}),
                                           request, anchor, identity, identity)
        with self.assertRaises(core.EvidenceError):
            core.validate_approval({k:v for k,v in correct.items() if k != 'owner'},
                                   request, anchor, identity, identity)
        with self.assertRaises(core.EvidenceError):
            core.validate_approval(dict(correct, unexpected=True),
                                   request, anchor, identity, identity)
        with self.assertRaises(core.EvidenceError):
            core.validate_approval(correct, request, anchor, 'sha256:'+'0'*64, identity)

    def test_checkpoint_loss_is_finite_nonnegative_and_matches_native_history(self):
        initial = (5.5).hex()
        valid = {'initial_train_loss_hex': initial,
                 'validation': [{'step': 64, 'loss_hex': (1.25).hex()}]}
        self.assertEqual(core.verified_loss({'train_loss_hex': initial},
                                            dict(valid,validation=[]),0),5.5)
        self.assertEqual(core.verified_loss({'train_loss_hex': (1.25).hex()},valid,64),1.25)
        for invalid in ('-0x1.0p+0','nan','inf',(1.5).hex(),'0x1p+0'):
            with self.subTest(loss=invalid), self.assertRaises(core.EvidenceError):
                core.verified_loss({'train_loss_hex': invalid}, valid, 64)
        with self.assertRaises(core.EvidenceError):
            core.verified_loss({'train_loss_hex': (1.25).hex()},
                               dict(valid, validation=[]),64)
        with self.assertRaises(core.EvidenceError):
            core.verified_loss({'train_loss_hex': initial},
                               dict(valid,initial_train_loss_hex=(-5.).hex(),validation=[]),0)

    def test_cli_rejects_symlinked_output_parent_into_checkout(self):
        import run_teacher_fit_review as cli
        root = Path(cli.__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            archive = tmp / 'source.zip'
            archive.write_bytes(b'synthetic-non-archive-never-parsed')
            alias = tmp / 'alias'
            alias.symlink_to(root, target_is_directory=True)
            target = root / '.fit-review-synthetic-output-alias.json'
            self.assertFalse(target.exists())
            args = ['review', '--archive', str(archive), '--archive-sha256',
                    core.sha(b'placeholder'), '--out', str(alias / target.name)]
            with mock.patch.object(sys, 'argv', args), \
                 mock.patch.object(cli.core, 'review', return_value={'tested': True}), \
                 mock.patch('sys.stderr'):
                self.assertEqual(cli.main(),2)
            self.assertFalse(target.exists())
            destination = tmp / 'private.json'
            args[-1] = str(destination)
            with mock.patch.object(sys, 'argv', args), \
                 mock.patch.object(cli.core, 'review', return_value={'tested': True}):
                self.assertEqual(cli.main(),0)
            self.assertEqual(destination.read_bytes(),core.canonical({'tested':True}))

    def test_machine_readable_conclusion_uses_verified_endpoints(self):
        for scores, expected, seeds in (
            ((23,18,23),'NONE_OF_THREE_SEEDS_FIT_72_OF_72',[]),
            ((72,18,23),'SOME_OF_THREE_SEEDS_FIT_72_OF_72',[17]),
            ((72,72,72),'ALL_THREE_SEEDS_FIT_72_OF_72',[17,29,43]),
        ):
            with self.subTest(scores=scores):
                final = [{'seed': s, 'exact': n} for s,n in zip(core.SEEDS,scores)]
                observed_seeds, conclusion = core.fit_conclusion(final)
                self.assertEqual(observed_seeds,seeds)
                self.assertEqual(conclusion,expected)

    def test_zip_replacement_after_open_does_not_redirect_hashed_snapshot(self):
        original_open = Path.open
        with tempfile.TemporaryDirectory() as tmp:
            path, replacement = Path(tmp)/'original.zip', Path(tmp)/'replacement.zip'
            with zipfile.ZipFile(path, 'w') as z:
                z.writestr('original.txt',b'original')
            with zipfile.ZipFile(replacement, 'w') as z:
                z.writestr('replacement.txt',b'changed')
            anchor=core.anchored_hash(path)
            switched=[False]
            def swap_after_open(obj,*args,**kwargs):
                stream=original_open(obj,*args,**kwargs)
                if Path(obj)==path and not switched[0]:
                    os.replace(replacement,path)
                    switched[0]=True
                return stream
            with mock.patch.object(Path,'open',swap_after_open):
                z, hashes=core.safe_archive(path,anchor)
                with z:
                    self.assertEqual(z.read('original.txt'),b'original')
            self.assertTrue(switched[0])
            self.assertIn('original.txt',hashes)
            self.assertNotIn('replacement.txt',hashes)
            self.assertNotEqual(core.anchored_hash(path),anchor)


if __name__ == '__main__':
    unittest.main()
