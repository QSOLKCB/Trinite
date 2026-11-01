"""Real supervised fresh-process validation of truthful zero-update test inputs.

This is engineering rejection coverage, not a completed fit or fit certificate.
No retained study is read or changed, and no optimizer update is permitted.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest


@contextmanager
def observe_native_calls():
    """Observe real functions without replacing configuration/guard/safe restore."""
    evidence = dict(optimizer_calls=0, update_loop_calls=0, configured=0,
                    guarded=0, restored=[], deterministic_before_configuration=[])
    previous = sys.getprofile()

    def observe(frame, event, result):
        module = frame.f_globals.get('__name__', '')
        name = frame.f_code.co_name
        if event == 'call':
            if module.startswith('torch.optim.') and name == 'step':
                evidence['optimizer_calls'] += 1
                raise RuntimeError('engineering regression forbids optimizer updates')
            if module == 'trinite.training' and name == 'run_steps':
                evidence['update_loop_calls'] += 1
                raise RuntimeError('engineering regression forbids update loops')
            if module == 'context_symbol_learning_v1.numerical' and name == 'configure':
                import torch
                evidence['deterministic_before_configuration'].append(
                    torch.are_deterministic_algorithms_enabled())
                evidence['configured'] += 1
        elif event == 'return':
            if module == 'trinite.training' and name == 'require_environment' and result is not None:
                evidence['guarded'] += 1
            if module == 'trinite.comparison_checkpoint' and name == 'restore_tensors' and result is not None:
                from trinite.training import model_identity
                evidence['restored'].append(dict(step=result.step,
                    target_tokens=result.target_tokens, optimizer_slots=len(result.optimizer.state),
                    model_identity=model_identity(result.model)))
        return observe

    sys.setprofile(observe)
    try:
        yield evidence
    finally:
        sys.setprofile(previous)


def probe_launcher():
    # -I excludes PYTHONPATH/sitecustomize. The actual fixed launcher still owns
    # path setup, resource limits, thread environment and supervisor-parent check.
    if 'torch' in sys.modules:
        raise RuntimeError('fresh launcher probe must not pre-import Torch')
    launcher, *arguments = sys.argv[2:]
    sys.argv = [launcher, *arguments]
    with observe_native_calls() as evidence:
        try:
            runpy.run_path(launcher, run_name='__main__')
        finally:
            print('NATIVE_OBSERVATION=' + json.dumps(evidence, sort_keys=True))


class ValidationProcessTests(unittest.TestCase):
    def setUp(self):
        import queue_evaluation_v1 as gate
        from context_symbol_learning_v1 import data, numerical
        from procedural_dataset_v1.cases import expected_target
        from queue_learning_v1.artifacts import write_json, inventory
        from trinite.training import model_identity
        from trinite.contracts import json_bytes

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inputs = self.root / 'engineering-zero-step-inputs'
        self.inputs.mkdir()
        self.checkpoint = self.inputs / 'checkpoint'
        self.checkpoint.mkdir()
        self.packet = self.inputs / 'train-only.json'
        self.binding = 'engineering-only-zero-step-not-a-scientific-freeze'
        with observe_native_calls() as evidence:
            numerical.configure()
            rows = data.training_rows(data.inventory(), lambda r: expected_target('context', r))
            state = numerical.state_for(rows, 17, self.binding, True)
            self.initial_identity = model_identity(state.model)
            payload, metadata = numerical.snapshot(state, self.binding)
            self.assertEqual(model_identity(state.model), self.initial_identity)
            self.assertEqual(state.step, 0)
            self.assertEqual(state.target_tokens, 0)
            self.assertEqual(dict(state.optimizer.state), {})
        self.assertEqual(evidence['optimizer_calls'], 0)
        self.assertEqual(evidence['update_loop_calls'], 0)
        gate.write(self.checkpoint / 'tensors.safetensors', payload)
        gate.write(self.checkpoint / 'metadata.json', metadata)
        self.info = dict(seed=17, fit=True, step=0, target_tokens=0,
            checkpoint_sha256=[gate.digest(payload), gate.digest(metadata)],
            schedule=numerical.schedule(rows, 17, True),
            train_sha256=gate.digest(json_bytes(rows)),
            update_seconds_hex=float(0).hex())
        self.training_sha = write_json(self.checkpoint / 'training.json', self.info)
        write_json(self.packet, dict(freeze_sha256=self.binding, train=rows, public=[]))
        self.before = inventory(self.inputs)

    def command(self, output, training_sha=None):
        return ['_validate-fit', '--checkpoint', self.checkpoint,
            '--training-sha', training_sha or self.training_sha, '--packet', self.packet,
            '--freeze-sha', self.binding, '--out', output]

    def assert_rejected(self, receipt, output, message):
        from queue_learning_v1.artifacts import inventory
        self.assertIsNone(receipt['failure'], receipt)
        self.assertEqual(receipt['exit_code'], 2, receipt)
        self.assertIn(message, receipt['stderr'])
        self.assertNotIn('training requires the pinned CPU backend', receipt['stderr'])
        self.assertFalse((output / 'certificate.json').exists())
        self.assertEqual(inventory(self.inputs), self.before)

    def test_production_launch_restores_then_rejects_zero_step_fit(self):
        from context_symbol_learning_v1 import controller
        output = self.root / 'production-validation'
        receipt = controller.launch(self.command(output), 60, max_updates=0, max_targets=0)
        self.assert_rejected(receipt, output, 'incomplete checkpoint/attempt budget')

    def test_observed_fresh_launcher_has_real_guard_restore_and_zero_optimizer_calls(self):
        from context_symbol_learning_v1 import controller
        from queue_learning_v2.resources import supervise
        output = self.root / 'observed-validation'
        command = [sys.executable, '-I', '-B']
        if sys.flags.optimize:
            command.append('-O')
        command += [str(Path(__file__).resolve()), '--probe-launcher', str(controller.LAUNCHER),
                    *map(str, self.command(output))]
        receipt = supervise(command, 60, max_updates=0, max_targets=0)
        self.assert_rejected(receipt, output, 'incomplete checkpoint/attempt budget')
        lines = [line.removeprefix('NATIVE_OBSERVATION=') for line in receipt['stdout'].splitlines()
                 if line.startswith('NATIVE_OBSERVATION=')]
        self.assertEqual(len(lines), 1, receipt)
        evidence = json.loads(lines[0])
        self.assertEqual(evidence['optimizer_calls'], 0)
        self.assertEqual(evidence['update_loop_calls'], 0)
        self.assertEqual(evidence['deterministic_before_configuration'][0], False)
        self.assertGreaterEqual(evidence['configured'], 1)
        self.assertGreaterEqual(evidence['guarded'], 1)
        self.assertEqual(evidence['restored'], [dict(step=0, target_tokens=0,
            optimizer_slots=0, model_identity=self.initial_identity)])

    def test_fresh_validation_preserves_anchor_context_and_safe_tensor_rejection(self):
        import queue_evaluation_v1 as gate
        from context_symbol_learning_v1 import controller
        from queue_learning_v1.artifacts import inventory
        from trinite.contracts import json_bytes, parse_json
        original_metadata = (self.checkpoint / 'metadata.json').read_bytes()
        original_payload = (self.checkpoint / 'tensors.safetensors').read_bytes()
        for kind, message in [('anchor', 'checkpoint anchor mismatch'),
                              ('binding', 'checkpoint frozen context mismatch'),
                              ('source', 'checkpoint frozen context mismatch'),
                              ('environment', 'checkpoint frozen context mismatch'),
                              ('tensor', 'invalid comparison safetensors')]:
            with self.subTest(kind=kind):
                info = deepcopy(self.info)
                metadata, payload = original_metadata, original_payload
                if kind in ('binding', 'source', 'environment'):
                    value = parse_json(metadata, canonical=True)
                    if kind == 'binding':
                        value['binding'] = 'wrong-engineering-binding'
                    elif kind == 'environment':
                        value['environment']['deterministic'] = False
                    else:
                        value['sources']['scripts/context_symbol_learning_v1/workers.py'] = '0' * 64
                    metadata = json_bytes(value)
                else:
                    payload = b'not-safe-tensors'
                if kind != 'anchor':
                    info['checkpoint_sha256'] = [gate.digest(payload), gate.digest(metadata)]
                (self.checkpoint / 'metadata.json').write_bytes(metadata)
                (self.checkpoint / 'tensors.safetensors').write_bytes(payload)
                training = json_bytes(info)
                (self.checkpoint / 'training.json').write_bytes(training)
                self.before = inventory(self.inputs)
                output = self.root / ('rejected-' + kind)
                receipt = controller.launch(self.command(output, gate.digest(training)), 60,
                                            max_updates=0, max_targets=0)
                self.assert_rejected(receipt, output, message)

    def test_internal_launcher_still_requires_actual_supervisor(self):
        from context_symbol_learning_v1 import controller
        result = subprocess.run([sys.executable, '-I', '-B', str(controller.LAUNCHER),
            *map(str, self.command(self.root / 'unsupervised'))], capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('internal worker requires fixed resource supervisor', result.stderr)

    def test_unconfigured_fresh_native_guard_still_rejects(self):
        from context_symbol_learning_v1.spec import ROOT
        code = ('import sys; sys.path.insert(0, ' + repr(str(ROOT / 'src')) + '); '
                'from trinite.training import require_environment; require_environment()')
        result = subprocess.run([sys.executable, '-I', '-B', '-c', code],
                                capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('training requires the pinned CPU backend', result.stderr)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--probe-launcher']:
        probe_launcher()
    else:
        unittest.main()
