"""Independent adversarial checks, using synthetic CPU-only evidence doubles.

The locally fabricated records below certify no CUDA execution or model output.
No fabricated record is persisted or offered as a retained inference fixture.
"""
import copy
from contextlib import redirect_stderr
import importlib.util
import io
import json
import marshal
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
from ncm_reference_v1 import core, evidence, runner, runtime


def synthetic_records(req):
    """Schema doubles only, deliberately distinguished by environment/log text."""
    result = {
        'schema': 'ncm-reference-result-v1',
        'evidence_class': 'actual-local-cuda-inference',
        'environment': {
            'versions': runtime.VERSIONS.copy(),
            'python': '3.11.99 (SYNTHETIC CPU TEST DOUBLE)',
            'platform': 'SYNTHETIC CPU TEST DOUBLE',
            'torch_cuda': '12.8',
            'cudnn': 91002,
            'device': 'NVIDIA GeForce RTX 5060 Ti',
            'capability': [12, 0],
            'total_memory': 16 * 1024**3,
            'torch_build': 'SYNTHETIC CPU TEST DOUBLE',
            'driver': 'SYNTHETIC CPU TEST DOUBLE',
            'installed_packages': runtime.VERSIONS.copy(),
        },
        'outcomes': [],
        'reconstructions': [],
    }
    timings = {
        'load_seconds': 0.1,
        'measurements': [],
        'max_allocated_bytes': 1024,
        'max_reserved_bytes': 2048,
        'memory_scope': 'PyTorch allocator, not whole-device VRAM',
    }
    schedule = [('cold', req['inputs'][0]), ('warmup', req['inputs'][0])]
    schedule += [('warm', x) for x in req['inputs']]
    for phase, case in schedule:
        raw_ids = core.encode_board(case['target'])
        result['outcomes'].append({
            'phase': phase, 'id': case['id'], 'raw_ids': raw_ids,
            'latent_ids': [0] * sum(core.LEVELS),
            'metrics': core.score(raw_ids, case['clues'], case['target']),
        })
        timings['measurements'].append({
            'phase': phase, 'id': case['id'], 'prepare_seconds': 0.01,
            'generator_seconds': 0.02, 'codec_and_transfer_seconds': 0.03,
            'total_seconds': 0.06,
        })
    for case in req['inputs']:
        raw_ids = core.encode_board(case['target'])
        result['reconstructions'].append({
            'id': case['id'], 'raw_ids': raw_ids,
            'metrics': core.score(raw_ids, case['clues'], case['target']),
        })
        timings['measurements'].append({
            'phase': 'gold-codec-diagnostic', 'id': case['id'],
            'total_seconds': 0.01,
        })
    return result, timings


class EvidenceReview(unittest.TestCase):
    def setUp(self):
        self.req = core.request()
        self.result, self.timings = synthetic_records(self.req)
        self.members = {
            'request.json': core.json_bytes(self.req),
            'status.json': core.json_bytes({'state': 'completed', 'returncode': 0}),
            'worker.log': b'SYNTHETIC CPU TEST DOUBLE; NO GPU OR MODEL EXECUTED\n',
            'result.json': core.json_bytes(self.result),
            'timings.json': core.json_bytes(self.timings),
        }
        self.members.update({'source/' + p: b for p, b in core.source_bytes().items()})

    def verify_members(self, members):
        data = evidence.pack(members)
        return evidence.verify(data, core.digest(data))

    def test_structural_verification_never_claims_inference_attestation(self):
        verdict = self.verify_members(self.members)
        self.assertTrue(verdict['metrics_recomputed'])
        self.assertFalse(verdict['inference_verified'])
        self.assertIn('not hardware attestation', verdict['limitation'])

    def test_preparation_clock_starts_before_python_prefix_encoding(self):
        events = []
        class Cuda:
            def synchronize(self):
                events.append('synchronize')
        class Torch:
            cuda = Cuda()
            long = 'CPU TEST DOUBLE'
            def tensor(self, *args, **kwargs):
                events.append('tensor')
                raise RuntimeError('stop before any model execution')
        def clock():
            events.append('clock')
            return 0.0
        def encode(clues):
            events.append('encode')
            return core.encode_board(clues)
        with patch.object(runtime.time, 'perf_counter', side_effect=clock):
            with patch.object(runtime, 'encode_board', side_effect=encode):
                with self.assertRaisesRegex(RuntimeError, 'stop before any model execution'):
                    runtime.infer(Torch(), None, None, self.req['inputs'][0]['clues'])
        self.assertEqual(events, ['synchronize', 'clock', 'encode', 'tensor'])

    def test_metrics_and_complete_order_are_independently_checked(self):
        for mutation in ('metric', 'duplicate', 'omission', 'reorder'):
            with self.subTest(mutation=mutation):
                result = copy.deepcopy(self.result)
                if mutation == 'metric':
                    result['outcomes'][0]['metrics']['solved'] = False
                elif mutation == 'duplicate':
                    result['outcomes'].append(result['outcomes'][-1])
                elif mutation == 'omission':
                    result['outcomes'].pop()
                else:
                    result['outcomes'][-2:] = reversed(result['outcomes'][-2:])
                with self.assertRaises(ValueError):
                    evidence.verify_result(result, self.req)

    def test_latent_ids_reject_boolean_bounds_and_wrong_length(self):
        for value in (-1, 16, True, 0.0):
            result = copy.deepcopy(self.result)
            result['outcomes'][0]['latent_ids'][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                evidence.verify_result(result, self.req)
        result = copy.deepcopy(self.result)
        result['outcomes'][0]['latent_ids'].pop()
        with self.assertRaises(ValueError):
            evidence.verify_result(result, self.req)

    def test_required_environment_identity_cannot_be_omitted(self):
        for key in ('python', 'platform', 'torch_cuda', 'cudnn', 'torch_build',
                    'driver', 'installed_packages'):
            result = copy.deepcopy(self.result)
            del result['environment'][key]
            with self.subTest(key=key), self.assertRaises((ValueError, KeyError)):
                evidence.verify_result(result, self.req)

    def test_wrong_python_or_cuda_environment_is_rejected(self):
        for key, value in [('python', '3.12.0'), ('torch_cuda', '11.8'),
                           ('total_memory', float('inf'))]:
            result = copy.deepcopy(self.result)
            result['environment'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                evidence.verify_result(result, self.req)

    def test_installed_inventory_cannot_contradict_direct_dependencies(self):
        for name in runtime.VERSIONS:
            result = copy.deepcopy(self.result)
            result['environment']['installed_packages'][name] = 'wrong-version'
            with self.subTest(name=name), self.assertRaises(ValueError):
                evidence.verify_result(result, self.req)

    def test_diagnostic_duration_is_required(self):
        timings = copy.deepcopy(self.timings)
        del timings['measurements'][-1]['total_seconds']
        with self.assertRaises((ValueError, KeyError)):
            evidence.verify_timings(timings, self.req)

    def test_timing_numbers_are_finite_nonnegative_and_not_boolean(self):
        for index in (0, -1):
            for value in (float('nan'), float('inf'), -1, True):
                timings = copy.deepcopy(self.timings)
                timings['measurements'][index]['total_seconds'] = value
                with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                    evidence.verify_timings(timings, self.req)

    def test_timing_scope_and_memory_accounting_are_consistent(self):
        for mutation in ('scope', 'memory_order', 'budget'):
            timings = copy.deepcopy(self.timings)
            if mutation == 'scope':
                timings['memory_scope'] = 'whole-device VRAM'
            elif mutation == 'memory_order':
                timings['max_allocated_bytes'] = timings['max_reserved_bytes'] + 1
            else:
                timings['max_reserved_bytes'] = self.req['protocol']['max_allocator_bytes'] + 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                evidence.verify_timings(timings, self.req)

    def test_request_comparison_preserves_json_numeric_types(self):
        members = self.members.copy()
        req = copy.deepcopy(self.req)
        req['protocol']['seed'] = False  # False == 0 must not bypass frozen identity.
        members['request.json'] = core.json_bytes(req)
        with self.assertRaises(ValueError):
            self.verify_members(members)

    def test_self_consistent_rehashed_source_change_is_not_current_source(self):
        members = self.members.copy()
        name = next(n for n in members if n.startswith('source/'))
        members[name] += b'\n# changed\n'
        with self.assertRaises(ValueError):
            self.verify_members(members)

    def test_external_archive_digest_is_mandatory(self):
        data = evidence.pack(self.members)
        with self.assertRaises(ValueError):
            evidence.verify(data, '0' * 64)

    def test_zip_duplicate_and_unsafe_names_are_rejected(self):
        for names in [('x', 'x'), ('../x',), ('/x',), ('a\\x',)]:
            stream = io.BytesIO()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                with zipfile.ZipFile(stream, 'w') as archive:
                    for name in names:
                        archive.writestr(name, b'{}')
            data = stream.getvalue()
            with self.subTest(names=names), self.assertRaises(ValueError):
                evidence.unpack(data, core.digest(data))

    def test_failure_evidence_cannot_be_presented_as_inference_success(self):
        for state in ('failed', 'timeout', 'interrupted'):
            members = self.members.copy()
            members.pop('result.json')
            members.pop('timings.json')
            members['status.json'] = core.json_bytes({'state': state, 'returncode': -9})
            verdict = self.verify_members(members)
            self.assertEqual(verdict['state'], state)
            self.assertFalse(verdict['inference_verified'])
            self.assertNotIn('warm_solved', verdict)

    def test_concurrent_publication_has_exactly_one_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'evidence.json'
            values = [str(i).encode() for i in range(12)]
            def attempt(value):
                try:
                    core.publish(path, value)
                    return value
                except FileExistsError:
                    return None
            with ThreadPoolExecutor(max_workers=12) as pool:
                winners = [v for v in pool.map(attempt, values) if v is not None]
            self.assertEqual(len(winners), 1)
            self.assertEqual(path.read_bytes(), winners[0])
            self.assertEqual(list(Path(directory).iterdir()), [path])

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'POSIX FIFO boundary')
    def test_nonregular_fifo_is_rejected_without_blocking(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fifo'
            os.mkfifo(path)
            probe = (
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "from ncm_reference_v1.core import read_bytes; "
                "read_bytes(sys.argv[2], 100)"
            )
            completed = subprocess.run(
                [sys.executable, '-I', '-B', '-c', probe, str(REPO / 'scripts'), str(path)],
                capture_output=True, text=True, timeout=5,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn('nonregular', completed.stderr)

    def test_launcher_ignores_own_package_unverified_bytecode(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            scripts = repo / 'scripts'
            scripts.mkdir()
            shutil.copytree(core.ROOT, scripts / 'ncm_reference_v1',
                            ignore=shutil.ignore_patterns('__pycache__'))
            launcher = scripts / 'run_ncm_reference.py'
            shutil.copyfile(REPO / 'scripts/run_ncm_reference.py', launcher)
            source = scripts / 'ncm_reference_v1/runner.py'
            cache = Path(importlib.util.cache_from_source(str(source)))
            cache.parent.mkdir(parents=True)
            info = source.stat()
            malicious_code = compile("raise RuntimeError('UNVERIFIED_BYTECODE_EXECUTED')",
                                     str(source), 'exec')
            cache.write_bytes(
                importlib.util.MAGIC_NUMBER
                + struct.pack('<III', 0, int(info.st_mtime) & 0xffffffff, info.st_size)
                + marshal.dumps(malicious_code)
            )
            output = repo / 'frozen-request.json'
            completed = subprocess.run(
                [sys.executable, '-I', '-B', str(launcher), 'freeze', '--output', str(output)],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertNotIn('UNVERIFIED_BYTECODE_EXECUTED', completed.stderr)
            self.assertEqual(json.loads(output.read_bytes())['schema'], 'ncm-reference-request-v1')

    def replay_arguments(self, root):
        expected = root / 'expected.zip'
        packet = evidence.pack(self.members)
        expected.write_bytes(packet)
        frozen = root / 'request.json'
        frozen.write_bytes(self.members['request.json'])
        output = root / 'replay.zip'
        return output, [
            'replay', '--expected', str(expected),
            '--expected-sha256', core.digest(packet),
            '--request', str(frozen), '--request-sha256', core.digest(frozen.read_bytes()),
            '--checkpoints', str(root / 'NO_REAL_WEIGHTS'), '--output', str(output),
        ]

    def test_replay_existing_receipt_prevents_expensive_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            output, args = self.replay_arguments(Path(directory))
            receipt = Path(str(output) + '.replay.json')
            receipt.write_bytes(b'preserve previous receipt')
            with patch.object(runner, 'run') as run, redirect_stderr(io.StringIO()):
                self.assertEqual(runner.main(args), 1)
            run.assert_not_called()
            self.assertEqual(receipt.read_bytes(), b'preserve previous receipt')
            self.assertFalse(output.exists())

    def test_failed_replay_retains_unknown_equality_receipt(self):
        for state in ('failed', 'timeout', 'interrupted'):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as directory:
                output, args = self.replay_arguments(Path(directory))
                members = self.members.copy()
                members.pop('result.json')
                members.pop('timings.json')
                members['status.json'] = core.json_bytes({'state': state, 'returncode': -9})
                packet = evidence.pack(members)
                def failed_controller(*unused_args):
                    output.write_bytes(packet)
                    return packet, state
                with patch.object(runner, 'run', side_effect=failed_controller), redirect_stderr(io.StringIO()):
                    self.assertEqual(runner.main(args), 1)
                receipt = json.loads(Path(str(output) + '.replay.json').read_bytes())
                self.assertIsNone(receipt['exact_result_match'])
                self.assertEqual(receipt['state'], state)
                self.assertEqual(receipt['replay_archive'], core.digest(output.read_bytes()))

    def test_mismatched_replay_retains_both_archives_and_false_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            output, args = self.replay_arguments(Path(directory))
            members = self.members.copy()
            result = copy.deepcopy(self.result)
            result['outcomes'][0]['raw_ids'] = [0] * 89
            case = self.req['inputs'][0]
            result['outcomes'][0]['metrics'] = core.score([0] * 89, case['clues'], case['target'])
            members['result.json'] = core.json_bytes(result)
            packet = evidence.pack(members)
            def mismatched_controller(*unused_args):
                output.write_bytes(packet)
                return packet, 'completed'
            with patch.object(runner, 'run', side_effect=mismatched_controller), redirect_stderr(io.StringIO()):
                self.assertEqual(runner.main(args), 1)
            receipt = json.loads(Path(str(output) + '.replay.json').read_bytes())
            self.assertFalse(receipt['exact_result_match'])
            self.assertTrue((Path(directory) / 'expected.zip').exists())
            self.assertTrue(output.exists())


if __name__ == '__main__':
    unittest.main()
