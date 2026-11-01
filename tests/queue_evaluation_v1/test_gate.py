"""Public exposed engineering cases, not model or learning experiments."""
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('gate', ROOT / 'scripts/queue_evaluation_v1.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def request():
    return {'schema': 'trinite.queue-request.v1', 'capacity': 2, 'initial_queue': 0,
            'arrivals': [1, 4, 3], 'service': [1, 0, 1], 'outages': [False] * 3,
            'observe_ticks': 1, 'observed': [True] * 6}


def oracle(req):
    waiting = [0] * req['initial_queue']
    dropped = 0
    for t, arrivals in enumerate(req['arrivals']):
        for _ in range(arrivals):
            if len(waiting) == req['capacity']:
                dropped += int(t >= req['observe_ticks'])
            else:
                waiting.append(t)
        for _ in range(0 if req['outages'][t] else req['service'][t]):
            if waiting:
                waiting.pop(0)
    return dropped


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.req = self.base / 'input.json'
        self.req.write_bytes(g.encode(request()))
        self.private = self.base / 'private'
        self.public = self.base / 'public'
        self.committed = self.base / 'commit'
        self.result = self.base / 'result'

    def prepare(self):
        return g.prepare(self.req, self.private, self.public)

    def commit(self, value=5, abstain=False):
        pred = {'schema': 'trinite.queue-prediction.v1',
                'observation_sha256': g.digest((self.public / 'observation.json').read_bytes()),
                'status': 'abstain' if abstain else 'prediction'}
        if not abstain:
            pred['value'] = value
        path = self.base / 'prediction.json'
        path.write_bytes(g.encode(pred))
        return g.commit(self.public, path, self.committed)

    def test_exact_wrong_and_abstain(self):
        for value, abstain in ((5, False), (4, False), (None, True)):
            with self.subTest(value=value):
                self.setUp()
                p = self.prepare()
                c = self.commit(value, abstain)
                g.score(self.private, self.committed, p, c, self.result)
                result = g.decode((self.result / 'evaluation.json').read_bytes())
                self.assertEqual(result['actual'], 5)
                if abstain:
                    self.assertNotIn('absolute_error', result)
                    self.assertNotIn('exact_match', result)
                else:
                    self.assertEqual(result['absolute_error'], abs(value - 5))
                    self.assertEqual(result['exact_match'], value == 5)
                before = {f: f.read_bytes() for f in self.base.rglob('*') if f.is_file()}
                g.verify(self.private, self.committed, p, c, self.result)
                self.assertEqual(before, {f: f.read_bytes() for f in self.base.rglob('*') if f.is_file()})

    def test_independent_accounting_and_observation_parity(self):
        rng = random.Random(192)
        queue_binary = ROOT / 'external/queue-sim-v1/target/release/trinite-queue-sim-v1'
        for _ in range(48):
            r = request()
            n = rng.randrange(2, 15)
            r.update(capacity=rng.randrange(1, 129), initial_queue=0,
                     arrivals=[rng.randrange(33) for _ in range(n)],
                     service=[rng.randrange(33) for _ in range(n)],
                     outages=[bool(rng.randrange(2)) for _ in range(n)], observe_ticks=rng.randrange(1, n))
            data = g.encode(r)
            target, _ = g.run('target', data)
            self.assertEqual(g.decode(target)['dropped'], oracle(r))
            observed, _ = g.run('observe', data)
            reference = subprocess.run([str(queue_binary), 'observe'], input=data, capture_output=True, check=True)
            self.assertEqual(observed, reference.stdout)

    def test_same_observation_different_suffix_requires_private_anchor(self):
        p = self.prepare()
        c = self.commit()
        r = request()
        r['arrivals'][1:] = [0, 0]
        other = self.base / 'other'
        other_public = self.base / 'other-public'
        self.req.write_bytes(g.encode(r))
        other_anchor = g.prepare(self.req, other, other_public)
        self.assertEqual((self.public/'observation.json').read_bytes(), (other_public/'observation.json').read_bytes())
        self.assertNotEqual(p, other_anchor)
        with self.assertRaises(g.GateError):
            g.evaluate(other, self.committed, p, c)
        actual = g.decode(g.evaluate(other, self.committed, other_anchor, c)['evaluation.json'])
        self.assertEqual(actual['actual'], 0)  # A new anchor is a different episode, not a repaired history.

    def test_commit_does_not_access_private_state_or_execute(self):
        self.prepare()
        original = g.read
        def guarded(path, *args):
            self.assertNotIn(self.private, Path(path).parents)
            return original(path, *args)
        with patch.object(g, 'read', side_effect=guarded), patch.object(g, 'run', side_effect=AssertionError('runner in commit')):
            self.commit()

    def test_bad_commit_rejected_before_private_access(self):
        self.prepare()
        c = self.commit()
        original = g.read
        def guarded(path, *args):
            self.assertNotIn(self.private, Path(path).parents)
            return original(path, *args)
        with patch.object(g, 'read', side_effect=guarded):
            with self.assertRaises(g.GateError):
                g.score(self.private, self.committed, '0'*64, '0'*64, self.result)
        self.assertFalse(self.result.exists())
        self.assertNotEqual(c, '0'*64)

    def test_artifact_tampering(self):
        p, c = self.prepare(), self.commit()
        for directory, names in ((self.private, g.PREPARED), (self.committed, g.COMMITTED)):
            for name in names:
                path = directory / name
                before = path.read_bytes()
                path.write_bytes(before + b' ')
                with self.subTest(name=name), self.assertRaises(g.GateError):
                    g.evaluate(self.private, self.committed, p, c)
                path.write_bytes(before)
        g.score(self.private, self.committed, p, c, self.result)
        for name in g.RESULT:
            path = self.result / name
            before = path.read_bytes()
            path.write_bytes(before + b' ')
            with self.assertRaises(g.GateError):
                g.verify(self.private, self.committed, p, c, self.result)
            path.write_bytes(before)
        with patch.object(g, 'source_identity', return_value={}):
            with self.assertRaises(g.GateError):
                g.evaluate(self.private, self.committed, p, c)
        with patch.object(g, 'binary_identity', return_value='0'*64):
            with self.assertRaises(g.GateError):
                g.evaluate(self.private, self.committed, p, c)

    def test_missing_extra_and_failed_artifacts_rejected(self):
        p, c = self.prepare(), self.commit()
        path = self.committed / 'prediction.json'
        before = path.read_bytes()
        path.unlink()
        with self.assertRaises(g.GateError):
            g.evaluate(self.private, self.committed, p, c)
        path.write_bytes(before)
        (self.private/'failure.json').write_bytes(b'{}')
        with self.assertRaises(g.GateError):
            g.evaluate(self.private, self.committed, p, c)

    def test_invalid_predictions(self):
        self.prepare()
        obs = (self.public/'observation.json').read_bytes()
        valid = {'schema':'trinite.queue-prediction.v1','status':'prediction','value':0,'observation_sha256':g.digest(obs)}
        variants = [dict(valid, value=v) for v in (True, None, -1, 65, 1.1)]
        variants += [dict(valid, extra=1), dict(valid, status='abstain'), dict(valid, observation_sha256='0'*64),
                     {'schema':'trinite.queue-prediction.v1','status':'prediction','observation_sha256':g.digest(obs)}]
        for item in variants:
            with self.subTest(item=item), self.assertRaises(g.GateError):
                g.prediction(g.encode(item), obs)
        for data in (b'{"a":1,"a":2}', b'{"a":1e999}', b'{"a":NaN}', b'{"a":9223372036854775808}', b'['*14+b'0'+b']'*14, b' '* (g.LIMIT+1)):
            with self.assertRaises(g.GateError):
                g.decode(data)

    def test_fresh_exclusive_paths_and_symlinks(self):
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()
        with self.assertRaises(g.GateError):
            g.commit(self.public, self.req, self.public/'nested')
        link = self.base/'link'
        link.symlink_to(self.public, target_is_directory=True)
        with self.assertRaises(g.GateError):
            g.bundle(link, g.PUBLIC)
        link.unlink()
        link.symlink_to(self.req)
        with self.assertRaises(g.GateError):
            g.read(link)
        hard = self.base/'hard'
        os.link(self.req, hard)
        with self.assertRaises(g.GateError):
            g.read(hard)

    def test_concurrent_destination(self):
        self.prepare()
        pred = self.base/'prediction.json'
        pred.write_bytes(g.encode({'schema':'trinite.queue-prediction.v1','status':'abstain','observation_sha256':g.digest((self.public/'observation.json').read_bytes())}))
        def attempt(_):
            try:
                return g.commit(self.public, pred, self.committed)
            except FileExistsError:
                return None
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(attempt, range(2)))
        self.assertEqual(sum(item is not None for item in results), 1)
        g.committed(self.committed, next(item for item in results if item))

    def test_process_failures_are_bounded(self):
        for code, timeout, expected in (("import time;time.sleep(2)", .05, 'timeout'),
                                        ("import sys;sys.stdout.write('x'*1000000)", 2, 'output_limit'),
                                        ("import sys;sys.stderr.write('x'*1000000)", 2, 'output_limit'),
                                        ("raise SystemExit(7)", 2, None)):
            with self.assertRaises(g.ProcessError) as caught:
                g.bounded_process([sys.executable, '-I', '-B', '-c', code], b'x'*16000, timeout, 1024)
            self.assertEqual(caught.exception.evidence['failure'], expected)
            self.assertLessEqual(len(caught.exception.evidence['stdout_b64']), 1368)
            self.assertLessEqual(len(caught.exception.evidence['stderr_b64']), 1368)

    def test_failure_retention(self):
        fake = {'exit_code':7,'failure':None,'stdout_b64':'','stderr_b64':''}
        with patch.object(g, 'run', side_effect=g.ProcessError(fake)):
            with self.assertRaises(g.ProcessError):
                self.prepare()
        self.assertEqual(g.decode((self.private/'failure.json').read_bytes())['process'], fake)
        self.assertFalse((self.private/'manifest.json').exists())

    def test_malformed_runner_output_retained(self):
        p, c = self.prepare(), self.commit()
        original = g.run
        def bad(operation, data):
            if operation == 'target':
                return b'{', {'exit_code':0,'failure':None,'stdout_b64':'ew==','stderr_b64':''}
            return original(operation, data)
        with patch.object(g, 'run', side_effect=bad):
            with self.assertRaises(g.ProcessError):
                g.score(self.private, self.committed, p, c, self.result)
        record = g.decode((self.result/'failure.json').read_bytes())
        self.assertEqual(record['process']['target']['stdout_b64'], 'ew==')
        self.assertFalse((self.result/'evaluation.json').exists())

    def test_public_packet_has_no_private_identity(self):
        self.prepare()
        self.assertEqual(set(p.name for p in self.public.iterdir()), set(g.PUBLIC))
        raw = (self.public/'observation.json').read_bytes()
        self.assertNotIn(g.digest(self.req.read_bytes()).encode(), raw)
        self.assertNotIn(b'initial_queue', raw)
        self.assertNotIn(b'sources', raw)

    def test_cli_rejection_and_no_bytecode(self):
        result = subprocess.run([sys.executable, '-I', '-B', str(ROOT/'scripts/queue_evaluation_v1.py'), 'score', '--prepared', str(self.private), '--commit', str(self.committed), '--prepare-sha','0'*64,'--commit-sha','0'*64,'--out',str(self.result)], capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b'')
        self.assertFalse(self.result.exists())

if __name__ == '__main__':
    unittest.main()
