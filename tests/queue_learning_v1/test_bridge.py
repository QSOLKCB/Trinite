"""Exposed engineering conformance; never train or score proposed pilot outcomes."""
from copy import deepcopy
import random
import subprocess
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import queue_evaluation_v1 as gate
from queue_learning_v1 import admission, codec, controller, evaluation, producer, spec
from queue_learning_v1.artifacts import inventory, read_json, write_json


def request(future=0):
    # Engineering fixture, deliberately outside the prospective 16-tick producer.
    return {'schema': 'trinite.queue-request.v1', 'capacity': 2, 'initial_queue': 0,
            'arrivals': [1, 4 + future, 3 + future], 'service': [1, 0, 1],
            'outages': [False] * 3, 'observe_ticks': 1, 'observed': [True] * 6}


def observation(n=8):
    return {'schema': 'trinite.queue-observation.v1', 'capacity': 128,
            'observe_ticks': n, 'horizon': 8, 'rows': [
                {'tick': t, 'outage': bool(t % 2), 'values': [0] * 6, 'observed': [True] * 6}
                for t in range(n)]}


class CodecTests(unittest.TestCase):
    def test_independent_roundtrip_and_manual_layout(self):
        obs = observation(1)
        obs['rows'][0] = {'tick': 0, 'outage': True, 'values': [128, 32, 0, 2, 1, 1],
                          'observed': [True, True, False, True, True, True]}
        raw = gate.encode(obs)
        text = codec.compact(raw)
        # Independent hand encoding: mask=59 ('x'), values high*64+low.
        self.assertEqual(text, 'Q1;128;1;8;1x200W00020101')
        self.assertEqual(codec.expand(text), obs)
        rng = random.Random(718)
        for n in range(1, 17):
            value = observation(n)
            for t, row in enumerate(value['rows']):
                row['observed'] = [bool(rng.randrange(2)) for _ in range(6)]
                row['values'] = [rng.randrange(m + 1) if known else 0
                                 for m, known in zip((128, 32, 32, 32, 32, t+1), row['observed'])]
            self.assertEqual(codec.expand(codec.compact(gate.encode(value))), value)

    def test_masks_and_answer_only_loss(self):
        obs = observation()
        known = codec.prompt(gate.encode(obs))
        obs['rows'][0]['observed'][0] = False
        missing = codec.prompt(gate.encode(obs))
        self.assertNotEqual(known, missing)
        self.assertFalse(codec.expand(missing.split('?')[0])['rows'][0]['observed'][0])
        raw = gate.encode(obs)
        enc = codec.supervised(raw, 0)
        prefix = codec.prompt(raw)
        self.assertEqual(enc.input_ids, (256, *prefix.encode('ascii'), ord('0'), 257))
        self.assertEqual(enc.loss_mask, (0,) * (1 + len(prefix)) + (1, 1))
        self.assertTrue(all(enc.attention_mask))
        obs['rows'][0]['values'][0] = 1
        with self.assertRaises(ValueError):
            codec.prompt(gate.encode(obs))

    def test_context_boundary_and_malformed_carriers(self):
        accepted, rejected = [], []
        for n in range(1, 19):
            try:
                prefix = codec.prompt(gate.encode(observation(n)))
                accepted.append(n)
                self.assertLessEqual(1 + len(prefix) + 5, 256)
                self.assertLessEqual(len(codec.supervised(gate.encode(observation(n)), 256).input_ids), 256)
            except ValueError:
                rejected.append(n)
        self.assertEqual(max(accepted) + 1, min(rejected))
        good = codec.compact(gate.encode(observation(1)))
        for bad in (good + '0', good[:-1], good.replace('Q1', 'Q2'), good.replace(';128;', ';0128;'),
                    good.replace(';1;', ';0;', 1), good + '\u00e9', good[:-1] + '!'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                codec.expand(bad)

    def test_leakage_rejection_and_future_noninterference(self):
        a, _ = gate.run('observe', gate.encode(request(0)))
        b, _ = gate.run('observe', gate.encode(request(3)))
        self.assertEqual(a, b)
        self.assertEqual(codec.prompt(a), codec.prompt(b))
        for name in ('answer', 'seed', 'family_id', 'filename', 'provenance', 'private_hash', 'future', 'labels'):
            value = gate.decode(a)
            value[name] = 1
            with self.subTest(name=name), self.assertRaises(ValueError):
                codec.prompt(gate.encode(value))
        with self.assertRaises(ValueError):
            codec.prompt(gate.encode(request()))
        with self.assertRaises(ValueError):
            codec.prompt(a, 'future_service')
        for n in (0, 9, 32, 64):
            self.assertEqual(codec.prompt(a), codec.prompt(a))
            self.assertEqual(codec.supervised(a, n).input_ids[1:1+len(codec.prompt(a))], tuple(codec.prompt(a).encode()))

    def test_outputs_never_repair_or_make_invalid_numeric(self):
        raw = gate.encode(observation(1))
        for tokens in ([], [257], [48], [48, 48, 257], [45, 49, 257], [32, 48, 257],
                       [256, 257], [258, 257], [255, 257], [50, 53, 55, 257], [True, 257]):
            with self.subTest(tokens=tokens):
                self.assertEqual(codec.output(tokens, raw), {'status': 'malformed'})
        self.assertEqual(codec.output([48, 257], raw), {'status': 'prediction', 'value': 0})
        self.assertEqual(codec.output([63, 257], raw), {'status': 'abstain'})


class ProtocolTests(unittest.TestCase):
    def test_split_alias_budget_and_no_outcomes(self):
        rows = producer.episodes()
        self.assertEqual(len(rows), 384)
        self.assertEqual(len({gate.digest(gate.encode(r['request'])) for r in rows}), 384)
        self.assertEqual({s: sum(r['split'] == s for r in rows) for s in ('train', 'validation', 'test')},
                         {'train': 256, 'validation': 64, 'test': 64})
        families = producer.families()
        self.assertEqual(len(families), 48)
        for family, split in families.items():
            selected = [r for r in rows if r['family'] == family]
            self.assertEqual(len(selected), 8)
            self.assertEqual({r['split'] for r in selected}, {split})
        real_run = gate.run
        def only_observe(op, data):
            self.assertEqual(op, 'observe', 'pilot outcome must not be computed by engineering preflight')
            return real_run(op, data)
        with patch.object(gate, 'run', side_effect=only_observe):
            observations = controller.observed_inventory(rows)
            receipt = admission.audit_observations(rows, observations)
        self.assertLessEqual(receipt['max_prompt_tokens'] + 5, 256)
        self.assertEqual(receipt['max_scheduled_target_tokens'], 2048)
        self.assertLess(receipt['max_scheduled_target_tokens'], 32768)
        # Every counterfactual differs only in the future (same initial and mask).
        indexed = {r['id']: r for r in rows}
        for row in rows:
            if row['id'][-2] == '0':
                alternate = row['id'][:-2] + '1' + row['id'][-1]
                self.assertEqual(observations[row['id']], observations[alternate])
                self.assertEqual(row['split'], indexed[alternate]['split'])
        for seed in (17, 29):
            config = spec.plan(seed)
            self.assertEqual((config.steps, config.batch_size, config.max_seconds, config.max_target_tokens), (64, 8, 120, 32768))
            self.assertEqual((config.model.blocks, config.model.width, config.model.heads, config.model.feed_forward_width), (1, 32, 4, 64))
        with self.assertRaises(ValueError):
            spec.plan(0)
        altered = deepcopy(rows)
        altered[0]['split'] = 'test'
        with self.assertRaises(ValueError):
            admission.audit_observations(altered, observations)
        aliases = dict(observations)
        foreign = next(r for r in rows if r['family'] != rows[0]['family'])
        aliases[foreign['id']] = observations[rows[0]['id']]
        with self.assertRaisesRegex(ValueError, 'crosses queue family'):
            admission.audit_observations(rows, aliases)

    def test_actual_queue_admission_rejects_foreign_and_contradictions(self):
        rows, source = producer.episodes(), {'producer': 'bound-test-source'}
        admitted = admission.record(rows, source)
        self.assertEqual(admission.validate_admission(admitted, rows, source), admitted)
        for key in ('schema', 'rights_basis', 'scope', 'outcome', 'verification', 'source_id'):
            value = deepcopy(admitted)
            value[key] = 'music-or-foreign-data'
            with self.subTest(key=key), self.assertRaises(ValueError):
                admission.validate_admission(value, rows, source)
        changed = deepcopy(rows)
        changed[0]['request']['arrivals'][-1] += 1
        with self.assertRaises(ValueError):
            admission.record(changed, source)
        with self.assertRaises(ValueError):
            admission.check_training_rows([{'split': 'test'}])

    def test_train_filter_precedes_any_private_target(self):
        with patch.object(admission, 'prepared', side_effect=AssertionError('private read')):
            with self.assertRaisesRegex(ValueError, 'train-only inventory'):
                admission.training_rows(Path('/unread'), {'episodes': [{'split': 'test'}]}, {})

    def test_authorization_and_feasibility_do_not_default_to_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            approval = root / 'approval.json'
            write_json(approval, {})
            with patch.object(controller, 'git_identity', side_effect=AssertionError('late git')):
                with self.assertRaises(ValueError):
                    controller.authorize({'git': {'head': 'a' * 40}}, 'b' * 64, approval)
            proof = root / 'proof.json'
            sha = write_json(proof, {'schema': 'wrong'})
            with self.assertRaises((ValueError, KeyError)):
                controller.freeze(proof, sha, root / 'frozen', root / 'anchors.json')
            self.assertFalse((root / 'frozen').exists())

    def test_missing_main_is_metadata_only_and_cannot_authorize_training(self):
        def git(argv, **kwargs):
            if 'refs/remotes/origin/main' in argv:
                raise subprocess.CalledProcessError(1, argv)
            return b'a' * 40 if 'HEAD' in argv else b''
        with patch.object(spec.subprocess, 'check_output', side_effect=git):
            value = spec.git_identity()
        self.assertEqual(value, {'head': 'a' * 40, 'clean': True, 'main': None})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'declaration.json'
            write_json(path, {'schema': 'trinite.queue-learning-owner-approval.v1',
                'freeze_sha256': 'b' * 64, 'implementation_commit': 'a' * 40,
                'approved_by': 'Trent Slade', 'reviewed_merged_green': True,
                'authorization': 'execute queue-learning-v1 seeds 17 and 29 once'})
            with patch.object(controller, 'git_identity', return_value=value):
                with self.assertRaisesRegex(ValueError, 'clean reviewed main'):
                    controller.authorize({'git': {'head': 'a' * 40}}, 'b' * 64, path)

    def test_child_environment_mismatch_rejects_before_admission_or_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(frozen=str(root / 'frozen'), freeze_sha='b' * 64,
                anchors=str(root / 'anchors.json'), anchors_sha='c' * 64, out=str(root / 'train'))
            with (patch('queue_learning_v1.numerical.configure', return_value={'foreign': True}),
                  patch.object(controller, 'frozen_context', return_value=(root / 'frozen', {'environment': {'expected': True}}, {})),
                  patch.object(admission, 'training_rows', side_effect=AssertionError('no labels'))):
                with self.assertRaisesRegex(ValueError, 'worker frozen environment changed'):
                    controller.train_worker(args)
            self.assertFalse((root / 'train').exists())

    def test_bounded_failed_worker_retains_raw_process(self):
        import sys
        timeout = controller.launch([sys.executable, '-I', '-c', 'import time; print("raw", flush=True); time.sleep(5)'], .1)
        self.assertEqual(timeout['process']['failure'], 'timeout')
        self.assertIn('child_user_seconds_hex', timeout)
        self.assertNotEqual(timeout['process']['stdout_b64'], '')
        failed = controller.launch([sys.executable, '-I', '-c', 'raise SystemExit(3)'], 1)
        self.assertEqual(failed['process']['exit_code'], 3)

    def test_full_freeze_observe_only_and_anchored_context_tampering(self):
        # Fake timing/environment only: this is not an executable scientific
        # freeze or native feasibility evidence. All 384 real preparations and
        # both journal/manifest anchor bindings are exercised without targets.
        rows = producer.episodes()
        source = spec.sources()
        observations = controller.observed_inventory(rows)
        structural = admission.audit_observations(rows, observations)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proof = {'schema': 'trinite.queue-learning-feasibility.v1', 'sources': source,
                'binary_sha256': gate.binary_identity(), 'protocol': spec.PROTOCOL,
                'structural': structural, 'kernel': {'optimizer_updates': 0,
                    'parameters': spec.plan(17).model.parameter_count,
                    'environment': {'engineering-double': True},
                    'conservative_64_update_estimate_seconds_hex': (1.).hex()}}
            sha = write_json(root / 'proof.json', proof)
            run = gate.run
            def observe_only(op, data):
                self.assertEqual(op, 'observe')
                return run(op, data)
            with (patch('queue_learning_v1.numerical.configure', return_value={'engineering-double': True}),
                  patch.object(controller, 'git_identity', return_value={'head': 'a'*40, 'main': 'a'*40, 'clean': True}),
                  patch.object(gate, 'run', side_effect=observe_only)):
                result = controller.freeze(root / 'proof.json', sha, root / 'frozen', root / 'anchors.json')
            args = (root / 'frozen', result['freeze_sha256'], root / 'anchors.json', result['preparation_anchors_sha256'])
            _, manifest, prepares = controller.frozen_context(*args)
            self.assertEqual(len(prepares), 384)
            self.assertEqual(len(manifest['episodes']), 384)
            journal = root / 'anchors.jsonl'
            data = journal.read_bytes()
            journal.write_bytes(data + b'{}\n')
            with self.assertRaisesRegex(ValueError, 'preparation journal changed'):
                controller.frozen_context(*args)
            journal.write_bytes(data)
            public = root / 'frozen' / rows[0]['id'] / 'public/observation.json'
            public.write_bytes(gate.encode(observation(1)))
            with self.assertRaisesRegex(ValueError, 'public bytes changed'):
                controller.frozen_context(*args)


class GateIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.frozen = gate.fresh(self.root / 'frozen')
        self.anchors = gate.fresh(self.root / 'independent-anchors')
        self.entries, self.prepares = [], {}
        for i, split in enumerate(('validation', 'test')):
            name = 'engineering-' + str(i)
            slot = gate.fresh(self.frozen / name)
            gate.write(slot / 'request.json', gate.encode(request(i)))
            self.prepares[name] = gate.prepare(slot / 'request.json', slot / 'private', slot / 'public')
            self.entries.append({'id': name, 'family': 'engineering-only', 'split': split})
        write_json(self.anchors / 'prepares.json', self.prepares)
        self.predictions = {}
        for method in evaluation.METHODS:
            for row in self.entries:
                self.predictions[method, row['id']] = {'status': 'prediction', 'value': 0}
        self.predictions['seed17', 'engineering-0'] = {'status': 'malformed', 'tokens': [48]}
        self.predictions['seed17', 'engineering-1'] = {'status': 'abstain', 'tokens': [63, 257]}
        self.predictions['seed29', 'engineering-0'] = {'status': 'timeout'}
        self.predictions['seed29', 'engineering-1'] = {'status': 'not_attempted'}

    def commit(self):
        real_run = gate.run
        # Public commitment cannot execute any runner, even the observer.
        with patch.object(gate, 'run', side_effect=AssertionError('no runner during commit')):
            sha = evaluation.commit_all(self.frozen, self.entries, self.predictions,
                self.root / 'commits', self.anchors / 'barrier.json')
        return sha

    def test_actual_prepare_commit_score_verify_and_all_denominators(self):
        barrier = self.commit()
        before = {p: p.read_bytes() for p in self.anchors.iterdir()}
        report = evaluation.score_all(self.frozen, self.prepares, self.root / 'commits',
            self.anchors / 'barrier.json', barrier, self.entries, self.root / 'scores')
        for method in evaluation.METHODS:
            for split in ('validation', 'test'):
                r = report['methods'][method][split]
                self.assertEqual(r['planned'], 1)
                self.assertEqual(r['exact_match']['denominator'], 1)
        a = report['methods']['seed17']['validation']
        self.assertIsNone(a['mae'])
        self.assertIsNone(a['conditional_numeric_mae'])
        self.assertEqual(a['coverage'], {'numerator': 0, 'denominator': 1})
        self.assertEqual(a['valid_output_rate']['numerator'], 0)
        self.assertEqual(report['methods']['seed17']['test']['valid_output_rate']['numerator'], 1)
        self.assertEqual(report['methods']['seed17']['test']['exact_match']['numerator'], 0)
        self.assertEqual(before, {p: p.read_bytes() for p in self.anchors.iterdir()})
        for row in self.entries:
            for method in evaluation.METHODS:
                name = method + '-' + row['id']
                evaluated = read_json(self.root / 'scores' / name / 'evaluation.json')
                # Independent simple FIFO counts for these exposed requests.
                actual = 5 if row['split'] == 'validation' else 7
                self.assertEqual(evaluated['actual'], actual)
                if method.startswith('seed'):
                    self.assertFalse({'predicted', 'exact_match', 'absolute_error'} & set(evaluated))
                else:
                    self.assertEqual(evaluated['absolute_error'], actual)
                    self.assertFalse(evaluated['exact_match'])

    def test_public_barrier_tampering_rejects_before_private_reads(self):
        barrier = self.commit()
        path = self.root / 'commits/mode-engineering-1-raw.json'
        path.write_bytes(gate.encode({'status': 'prediction', 'value': 1}))
        with patch.object(gate, 'evaluate', side_effect=AssertionError('private execution too early')):
            with self.assertRaisesRegex(ValueError, 'raw output changed'):
                evaluation.score_all(self.frozen, self.prepares, self.root / 'commits',
                    self.anchors / 'barrier.json', barrier, self.entries, self.root / 'scores')
        self.assertFalse((self.root / 'scores').exists())

    def test_independent_anchors_and_missing_commit_rejected(self):
        barrier = self.commit()
        path = self.anchors / 'barrier.json'
        raw = path.read_bytes()
        path.write_bytes(raw.replace(b'engineering-0', b'engineering-X', 1))
        with self.assertRaisesRegex(ValueError, 'anchor mismatch'):
            evaluation.public_barrier(self.root / 'commits', path, barrier, self.entries)
        path.write_bytes(raw)
        (self.root / 'commits/causal-engineering-1/prediction.json').unlink()
        with self.assertRaises(ValueError):
            evaluation.public_barrier(self.root / 'commits', path, barrier, self.entries)

    def test_private_tamper_and_exclusive_failure_retention(self):
        barrier = self.commit()
        path = self.frozen / 'engineering-0/private/request.json'
        path.write_bytes(gate.encode(request(8)))
        with self.assertRaises(ValueError):
            evaluation.score_all(self.frozen, self.prepares, self.root / 'commits',
                self.anchors / 'barrier.json', barrier, self.entries, self.root / 'scores')
        self.assertTrue((self.root / 'scores/failure.json').is_file())
        old = (self.root / 'scores/failure.json').read_bytes()
        with self.assertRaises(FileExistsError):
            evaluation.score_all(self.frozen, self.prepares, self.root / 'commits',
                self.anchors / 'barrier.json', barrier, self.entries, self.root / 'scores')
        self.assertEqual((self.root / 'scores/failure.json').read_bytes(), old)

    def test_baselines_are_train_only_and_causal_public_only(self):
        pool = [{'split': 'train', 'answer': v} for v in (0, 1, 1, 3, 3, 6)]
        self.assertEqual(evaluation.constants(pool), {'median': 1, 'mode': 1})
        with self.assertRaises(ValueError):
            evaluation.constants(pool + [{'split': 'test', 'answer': 0}])
        observed = gate.read(self.frozen / 'engineering-0/public/observation.json')
        self.assertEqual(evaluation.causal(observed), {'status': 'prediction', 'value': 0})
        value = gate.decode(observed)
        value['rows'][-1]['observed'][0] = False
        self.assertEqual(evaluation.causal(gate.encode(value)), {'status': 'abstain'})
        with self.assertRaises(ValueError):
            evaluation.causal(gate.encode(request()))

    def test_controller_workers_doubled_gate_real_barrier_order_and_readonly_verify(self):
        # This tests controller lifecycle, not training or checkpoint numerics.
        # Actual native safe-reader behavior is tested separately at zero steps.
        destination = self.root / 'run'
        anchor_dir = self.root / 'run-anchors'
        pool = [{'split': 'train', 'answer': 0}]
        manifest = {'episodes': [{**r, 'observation_sha256': gate.digest(gate.read(
            self.frozen / r['id'] / 'public/observation.json'))} for r in self.entries],
            'environment': {'engineering': True}}
        args = SimpleNamespace(frozen=str(self.frozen), freeze_sha='b'*64,
            anchors=str(self.anchors / 'prepares.json'), anchors_sha='c'*64,
            approval=str(self.root / 'unused-owner-declaration.json'),
            out=str(destination), run_anchors=str(anchor_dir))
        calls = []
        def launch(argv, seconds):
            stage = '_train' if '_train' in argv else '_infer'
            seed = int(argv[argv.index('--seed') + 1])
            slot = gate.fresh(argv[argv.index('--out') + 1])
            calls.append((stage, seed, seconds))
            if stage == '_train':
                gate.write(slot / 'tensors.safetensors', b'controller-double-no-tensors')
                gate.write(slot / 'metadata.json', b'{}\n')
                write_json(slot / 'train-data.json', pool)
                write_json(slot / 'training.json', {'checkpoint_sha256': [
                    gate.digest(b'controller-double-no-tensors'), gate.digest(b'{}\n')],
                    'train_data_sha256': gate.digest(gate.encode(pool))})
            else:
                checkpoint_file = anchor_dir / 'checkpoints.json'
                self.assertTrue(checkpoint_file.is_file())
                self.assertEqual(set(read_json(checkpoint_file)['checkpoints']), {'17', '29'})
                gate.write(slot / 'raw.jsonl', b''.join(gate.encode({'id': r['id'],
                    'raw': {'status': 'prediction', 'tokens': [48, 257]}}) for r in self.entries))
                gate.write(slot / 'attempts.jsonl', b''.join(gate.encode({'id': r['id']}) for r in self.entries))
            return {'process': {'exit_code': 0, 'failure': None}, 'wall_seconds_hex': (.1).hex(),
                    'elapsed_limit_exceeded': False}
        real_gate = gate.run
        private_requests = {gate.read(self.frozen / r['id'] / 'private/request.json') for r in self.entries}
        def bounded_gate(op, data):
            if op == 'target' and data in private_requests:
                barrier = read_json(anchor_dir / 'barrier.json')
                self.assertEqual(len(barrier['entries']), 2 * len(evaluation.METHODS))
            return real_gate(op, data)
        context = (self.frozen, manifest, self.prepares)
        with (patch.object(controller, 'frozen_context', return_value=context),
              patch.object(controller, 'authorize'),
              patch('queue_learning_v1.numerical.configure', return_value={'engineering': True}),
              patch.object(controller, 'launch', side_effect=launch),
              patch.object(admission, 'training_rows', return_value=pool),
              patch.object(gate, 'run', side_effect=bounded_gate)):
            result = controller.run(args)
        self.assertFalse(result['failed_workers'])
        self.assertEqual([(a, b) for a, b, _ in calls], [('_train',17), ('_train',29), ('_infer',17), ('_infer',29)])
        self.assertEqual([s for a, _, s in calls if a == '_train'], [120, 120])
        self.assertTrue(all(s < 900 for a, _, s in calls if a == '_infer'))
        run_sha = write_json(anchor_dir / 'run-inventory.json', inventory(destination))
        verify_args = SimpleNamespace(**vars(args), run=str(destination),
            checkpoint_sha=result['checkpoint_sha256'], barrier_sha=result['barrier_sha256'], run_sha=run_sha)
        before = inventory(destination)
        rejected = SimpleNamespace(**{**vars(verify_args), 'barrier_sha': '0' * 64})
        with (patch.object(controller, 'frozen_context', return_value=context),
              patch.object(controller, 'inventory', side_effect=AssertionError('no scored outcome reads'))):
            with self.assertRaisesRegex(ValueError, 'anchor mismatch'):
                controller.verify(rejected)
        with (patch.object(controller, 'frozen_context', return_value=context),
              patch('queue_learning_v1.numerical.configure', return_value={'engineering': True}),
              patch('queue_learning_v1.numerical.restore', return_value=SimpleNamespace(step=64))):
            self.assertEqual(controller.verify(verify_args)['optimizer_replay'], False)
            self.assertEqual(inventory(destination), before)
            (destination / 'resources.json').write_bytes(b'{}\n')
            with self.assertRaisesRegex(ValueError, 'inventory changed'):
                controller.verify(verify_args)


if __name__ == '__main__':
    unittest.main()
