"""Synthetic controller mechanics only, not target musical performance evidence."""
from contextlib import ExitStack
from types import SimpleNamespace
import random
import hashlib
import os
import subprocess
import shutil
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from frozen_music_eval_v1.worker import evaluate_models, replay_gate, original_source
from music_contrast_v1.spec import rows


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.register_buffer('marker', torch.tensor([2.0]))
        self.child = torch.nn.Identity()
        self.config = SimpleNamespace(context_length=256)

    def forward(self, value):
        return value * self.weight


class FrozenInference(unittest.TestCase):
    def setUp(self):
        raw = rows() + rows('heldout', True)
        self.inputs = [{k:r[k] for k in ('example_identity','prompt','split')} for r in raw]
        self.gold = {r['prompt']:[*r['answer'].encode(),257] for r in raw}
        self.retained = {s:{r['example_identity']:[*r['answer'].encode(),257] for r in raw[:40]} for s in (17,29)}
        self.models = {s:FakeModel() for s in (17,29)}
        self.models[17].child.training = False
        self.events = []

    def greedy(self, model, prompt, cap):
        self.assertIs(type(prompt), str)
        self.assertEqual(cap, 64)
        generated = self.gold[prompt]
        for _ in generated:
            model(torch.tensor([1.0]))
        return generated[:]

    def run_fake(self, decode=None):
        with ExitStack() as stack:
            for target in ('trinite.training.create_optimizer','trinite.training.run_steps',
                           'trinite.training.evaluate','trinite.training.TrainingState'):
                stack.enter_context(patch(target, side_effect=AssertionError('training forbidden')))
            stack.enter_context(patch('trinite.comparison.greedy', side_effect=decode or self.greedy))
            return evaluate_models(self.models,self.inputs,self.retained,
                lambda record:self.events.append(record),lambda gate:self.events.append(gate))

    def test_both_replays_barrier_complete_coverage_modes_rng(self):
        modes = {s:[m.training for m in model.modules()] for s,model in self.models.items()}
        rng = torch.get_rng_state().clone(); python_rng = random.getstate()
        result = self.run_fake()
        self.assertEqual(result['decodes'],320)
        self.assertEqual(result['forward_calls'],640)
        self.assertEqual(self.events[80], replay_gate())
        self.assertTrue(all(r['stage']=='train_replay' for r in self.events[:80]))
        self.assertTrue(all(r['stage']=='heldout' for r in self.events[81:]))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(python_rng,random.getstate())
        self.assertEqual(modes,{s:[m.training for m in model.modules()] for s,model in self.models.items()})

    def test_seed29_replay_failure_no_heldout_or_gate(self):
        self.retained[29][self.inputs[39]['example_identity']] = [56,257]
        with self.assertRaisesRegex(ValueError,'train replay differs'):
            self.run_fake()
        self.assertEqual(len(self.events),80)
        self.assertTrue(all(r['stage']=='train_replay' for r in self.events))

    def test_state_mutation_stops_before_first_journal(self):
        for what in ('weight','buffer','rng','mode','grad'):
            with self.subTest(what=what):
                self.setUp()
                def broken(model,prompt,cap):
                    result=self.greedy(model,prompt,cap)
                    if what=='weight':
                        with torch.no_grad():model.weight.add_(1)
                    elif what=='buffer':model.marker.add_(1)
                    elif what=='rng':random.random()
                    elif what=='mode':model.child.training=True
                    else:model.weight.grad=torch.ones_like(model.weight)
                    return result
                with self.assertRaisesRegex(ValueError,'mutated'):
                    self.run_fake(broken)
                self.assertEqual(self.events,[])

    def test_last_train_emit_mutation_stops_both_heldout_stages(self):
        events = []
        def emit(record):
            events.append(record)
            if len(events) == 80:
                with torch.no_grad(): self.models[29].weight.add_(1)
        with patch('trinite.comparison.greedy',side_effect=self.greedy):
            with self.assertRaisesRegex(ValueError,'mutated'):
                evaluate_models(self.models,self.inputs,self.retained,emit,
                    lambda gate:events.append(gate))
        self.assertEqual(len(events),80)
        self.assertTrue(all(r['stage']=='train_replay' for r in events))

    def test_gate_callback_mutation_stops_both_heldout_stages(self):
        events = []
        def gate(record):
            events.append(record)
            with torch.no_grad(): self.models[29].weight.add_(1)
        with patch('trinite.comparison.greedy',side_effect=self.greedy):
            with self.assertRaisesRegex(ValueError,'mutated'):
                evaluate_models(self.models,self.inputs,self.retained,events.append,gate)
        self.assertEqual(len(events),81)
        self.assertEqual(events[-1],replay_gate())
        self.assertTrue(all(r['stage']=='train_replay' for r in events[:-1]))

    def test_forward_budget_accounting(self):
        def broken(model,prompt,cap):
            result=self.greedy(model,prompt,cap)
            model(torch.tensor([1.0]))
            return result
        with self.assertRaisesRegex(ValueError,'accounting'):
            self.run_fake(broken)
        self.assertEqual(self.events,[])

    def test_gold_fields_cannot_enter_inference(self):
        self.inputs[0]['answer']='8'
        with self.assertRaisesRegex(ValueError,'gold'):
            self.run_fake()
        self.assertEqual(self.events,[])

    def test_missing_model_rejected(self):
        del self.models[29]
        with self.assertRaisesRegex(ValueError,'both fixed'):
            self.run_fake()

    def test_source_blob_tamper_rejected_despite_clean_pinned_git(self):
        from frozen_music_eval_v1.worker import ORIGINAL_COMMIT, ORIGINAL_TREE
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'src').mkdir(); (root/'scripts').mkdir()
            (root/'src/marker.py').write_bytes(b'tampered\n')
            expected = b'original\n'
            blob = hashlib.sha1(b'blob '+str(len(expected)).encode()+b'\0'+expected).hexdigest()
            def git(argv, **kwargs):
                command = argv[7:]
                if command == ['rev-parse','--show-toplevel']: return str(root).encode()
                if command == ['rev-parse','HEAD']: return ORIGINAL_COMMIT.encode()
                if command == ['rev-parse','HEAD^{tree}']: return ORIGINAL_TREE.encode()
                if command[0] in ('status','ls-files'): return b''
                if command[0] == 'ls-tree': return f'100644 blob {blob}\tsrc/marker.py\n'.encode()
                self.fail('unexpected git command: '+repr(command))
            with patch('subprocess.check_output',side_effect=git), \
                    patch('trinite.model.Decoder',side_effect=AssertionError('model construction forbidden')):
                with self.assertRaisesRegex(ValueError,'bytes differ'):
                    original_source(root)

    def test_original_source_ignored_cache_rejected(self):
        from frozen_music_eval_v1.worker import ORIGINAL_COMMIT, ORIGINAL_TREE
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'src').mkdir(); (root/'scripts').mkdir()
            (root/'src/hidden.pyc').write_bytes(b'not executable')
            def git(argv, **kwargs):
                command = argv[7:]
                if command == ['rev-parse','--show-toplevel']: return str(root).encode()
                if command == ['rev-parse','HEAD']: return ORIGINAL_COMMIT.encode()
                if command == ['rev-parse','HEAD^{tree}']: return ORIGINAL_TREE.encode()
                if command[0] == 'status': return b''
                self.fail('cache should fail before further Git checks')
            with patch('subprocess.check_output',side_effect=git), \
                    patch('trinite.model.Decoder',side_effect=AssertionError('model construction forbidden')):
                with self.assertRaisesRegex(ValueError,'bytecode/extension'):
                    original_source(root)

    def test_real_git_replacement_tree_cannot_bypass_source_pin(self):
        import frozen_music_eval_v1.worker as worker
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = {k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
            env.update(GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL=os.devnull)
            def git(*args):
                return subprocess.check_output(['git','-C',str(root),*args],
                    env=env,stderr=subprocess.PIPE).decode().strip()
            git('init')
            git('config','user.name','Conformance')
            git('config','user.email','conformance@example.invalid')
            (root/'src').mkdir(); (root/'scripts').mkdir()
            (root/'src/marker.py').write_text('original = True\n')
            git('add','src/marker.py'); git('commit','-m','original')
            commit = git('rev-parse','HEAD'); tree = git('rev-parse','HEAD^{tree}')
            with patch.object(worker,'ORIGINAL_COMMIT',commit),patch.object(worker,'ORIGINAL_TREE',tree):
                # Ambient Git selectors cannot redirect even the positive check.
                with patch.dict(os.environ,{'GIT_DIR':'/nonexistent/redirect',
                                           'GIT_WORK_TREE':'/nonexistent/redirect'}):
                    self.assertEqual(original_source(root),root)
                (root/'src/marker.py').write_text('malicious = True\n')
                git('add','src/marker.py')
                replacement = git('write-tree')
                git('replace',tree,replacement)
                git('reset','--hard','HEAD')
                self.assertEqual(git('status','--porcelain'),'')
                self.assertEqual(git('rev-parse','HEAD'),commit)
                self.assertEqual(git('rev-parse','HEAD^{tree}'),tree)
                self.assertEqual((root/'src/marker.py').read_text(),'malicious = True\n')
                with patch('trinite.model.Decoder',side_effect=AssertionError('model construction forbidden')):
                    with self.assertRaisesRegex(ValueError,'exact original|bytes differ'):
                        original_source(root)

    def test_core_worktree_redirect_cannot_hide_native_import_shadows(self):
        import frozen_music_eval_v1.worker as worker
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); root = base/'original'; root.mkdir()
            mirror = base/'clean-mirror'
            env = {k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
            env.update(GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL=os.devnull,
                       PYTHONDONTWRITEBYTECODE='1')
            def git(*args):
                return subprocess.check_output(['git','-C',str(root),*args],
                    env=env,stderr=subprocess.PIPE).decode().strip()
            git('init'); git('config','user.name','Conformance')
            git('config','user.email','conformance@example.invalid')
            (root/'src').mkdir(); (root/'scripts').mkdir()
            (root/'src/marker.py').write_text('original = True\n')
            (root/'scripts/marker.py').write_text('original = True\n')
            git('add','.'); git('commit','-m','original')
            commit=git('rev-parse','HEAD'); tree=git('rev-parse','HEAD^{tree}')
            shutil.copytree(root,mirror,ignore=shutil.ignore_patterns('.git'))
            git('config','core.worktree',str(mirror))
            self.assertEqual(git('rev-parse','--show-toplevel'),str(mirror))
            with patch.object(worker,'ORIGINAL_COMMIT',commit),patch.object(worker,'ORIGINAL_TREE',tree):
                # A clean actual root remains admissible despite irrelevant config.
                self.assertEqual(original_source(root),root)
                for ignored in (False,True):
                    with self.subTest(ignored=ignored):
                        sentinel=base/'SENTINEL_EXECUTED'
                        shadow=root/'src/torch.py'
                        shadow.write_text('from pathlib import Path\nPath('+repr(str(sentinel))+').touch()\n'
                                          'raise RuntimeError("shadow imported")\n')
                        (root/'.git/info/exclude').write_text('src/torch.py\n' if ignored else '')
                        # Unpinned Git sees a clean mirror in both cases.
                        self.assertEqual(git('status','--porcelain','--untracked-files=all'),'')
                        code = """import sys
sys.path.insert(0,sys.argv[1])
from frozen_music_eval_v1 import worker
worker.ORIGINAL_COMMIT=sys.argv[3]; worker.ORIGINAL_TREE=sys.argv[4]
try:
    worker.inspect_original(sys.argv[2],sys.argv[5], 'sha256:'+'0'*64,'sha256:'+'1'*64)
except ValueError as error:
    assert 'exact original' in str(error) or 'ignored original' in str(error), str(error)
    assert 'torch' not in sys.modules
    print('rejected-before-native-import')
else:
    raise AssertionError('redirected worktree admitted shadow')
"""
                        result=subprocess.run([sys.executable,'-I','-B','-c',code,
                            str(Path(worker.__file__).resolve().parents[1]),str(root),commit,tree,
                            str(base/'no-run-input')],env=env,capture_output=True,text=True,timeout=30)
                        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                        self.assertEqual(result.stdout.strip(),'rejected-before-native-import')
                        self.assertFalse(sentinel.exists())
                        shadow.unlink()

    def test_original_source_leaf_symlink_rejected_before_git(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'real').mkdir()
            (root/'linked').symlink_to(root/'real',target_is_directory=True)
            with patch('subprocess.check_output',side_effect=AssertionError('Git forbidden')):
                with self.assertRaisesRegex(ValueError,'original producer checkout'):
                    original_source(root/'linked')

    def test_wrong_original_source_rejected_before_import(self):
        from pathlib import Path
        with self.assertRaisesRegex(ValueError,'exact original'):
            original_source(Path(__file__).resolve().parents[2])


if __name__ == '__main__':
    unittest.main()
