"""Ordinary source-only audit correctness; no adversarial cache execution."""
import ast
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_batch_composition_v4.py"
loader = importlib.util.spec_from_file_location("ordinary_auditor_v4", SCRIPT)
auditor = importlib.util.module_from_spec(loader)
loader.loader.exec_module(auditor)
helper_spec = importlib.util.spec_from_file_location(
    "ordinary_audit_fixture", ROOT / "tests/test_batch_composition_audit_review.py")
helper = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(helper)


class SourceOnlyAuditTests(unittest.TestCase):
    def test_clean_verified_package_executes_bound_bytes_and_cleans_import_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / 'src/ordinary_audit_fixture_v4'
            package.mkdir(parents=True)
            files = {'src/ordinary_audit_fixture_v4/__init__.py': b'VALUE = 7\n',
                     'src/ordinary_audit_fixture_v4/child.py': b'from . import VALUE\nRESULT = VALUE + 5\n'}
            for name, raw in files.items():
                (root / name).write_bytes(raw)
            approved = helper.canonical({'source': {name: helper.identity(raw) for name, raw in files.items()}})
            old_path, old_meta = list(sys.path), list(sys.meta_path)
            for _ in range(2):
                with auditor.VerifiedSourceImports(root, approved) as boundary:
                    module = importlib.import_module('ordinary_audit_fixture_v4.child')
                    self.assertEqual(module.RESULT, 12)
                    self.assertEqual(boundary.checked_executed_sources(), {
                        'ordinary_audit_fixture_v4': helper.identity(files['src/ordinary_audit_fixture_v4/__init__.py']),
                        'ordinary_audit_fixture_v4.child': helper.identity(files['src/ordinary_audit_fixture_v4/child.py'])})
                    self.assertIs(module.__loader__, boundary)
                self.assertFalse(any(name.startswith('ordinary_audit_fixture_v4') for name in sys.modules))
                self.assertEqual(sys.path, old_path)
                self.assertEqual(sys.meta_path, old_meta)
            self.assertFalse(list(root.rglob('*.pyc')))

    def test_clean_positive_and_nonconformant_cli_records_source_execution(self):
        fixture = helper.AuditReviewTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        for conforms in (True, False):
            fixture.report['conforms'] = conforms
            fixture.write_packet()
            output = fixture.root / ('ordinary-%s.json' % conforms)
            result = fixture.invoke(output, script=SCRIPT)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads(output.read_bytes())
            self.assertEqual(receipt['schema'], 'trinite.independent-batch-ci-audit.v4')
            self.assertIs(receipt['strict_recorded_report_validation']['conforms'], conforms)
            self.assertEqual(receipt['auditor_identity'], helper.identity(SCRIPT.read_bytes()))
            expected = json.loads(fixture.approved.read_bytes())['source']
            self.assertEqual(receipt['executed_project_sources'], {
                'trinite': expected['src/trinite/__init__.py'],
                'trinite.contracts': expected['src/trinite/contracts.py'],
                'batch_composition_v1': expected['scripts/batch_composition_v1/__init__.py'],
                'batch_composition_v1.spec': expected['scripts/batch_composition_v1/spec.py'],
                'batch_composition_v1.validation': expected['scripts/batch_composition_v1/validation.py']})

    def test_retained_source_only_receipts_bind_original_artifacts(self):
        folder = ROOT / 'fixtures/batch-composition-audit-v4'
        raw = (folder / 'commitment.json').read_bytes()
        self.assertEqual(helper.identity(raw), 'sha256:dc68a24c2e85833f365e700fe7814e1bbc9689c705e933be4f492c6e3f0a6ef1')
        commitment = json.loads(raw)
        self.assertIs(commitment['independent_adversarial_review_completed'], False)
        self.assertIs(commitment['model_inference_executed'], False)
        self.assertEqual(helper.identity(SCRIPT.read_bytes()), commitment['auditor_identity'])
        for label in ('approved_review', 'historical_auditor'):
            self.assertEqual(helper.identity((ROOT / commitment[label + '_path']).read_bytes()),
                             commitment[label + '_identity'])
        approved = json.loads((ROOT / commitment['approved_review_path']).read_bytes())
        originals = json.loads((ROOT / 'fixtures/batch-composition-v2/commitment.json').read_bytes())
        self.assertEqual(set(p.name for p in folder.iterdir()), set(commitment['files']) | {'commitment.json'})
        for version, info in commitment['matrix'].items():
            raw = (folder / info['receipt_path']).read_bytes()
            self.assertEqual(helper.identity(raw), info['receipt_identity'])
            self.assertEqual(helper.identity(raw), commitment['files'][info['receipt_path']])
            receipt = json.loads(raw)
            self.assertEqual(receipt['schema'], 'trinite.independent-batch-ci-audit.v4')
            self.assertEqual(receipt['auditor_identity'], commitment['auditor_identity'])
            self.assertEqual(receipt['approved_review_identity'], commitment['approved_review_identity'])
            self.assertEqual(receipt['report_identity'], originals['matrix'][version]['report_identity'])
            self.assertEqual(receipt['request_identity'], originals['matrix'][version]['request_identity'])
            self.assertEqual(receipt['all_artifact_file_hashes'], {
                name: value['identity'] for name, value in originals['matrix'][version]['members'].items()})
            self.assertEqual(len(receipt['executed_project_sources']), 8)
            for name, source_id in receipt['executed_project_sources'].items():
                relative = name.replace('.', '/')
                candidates = [base + relative + suffix for base in ('src/', 'scripts/')
                              for suffix in ('.py', '/__init__.py')]
                matches = [path for path in candidates if path in approved['source']]
                self.assertEqual(len(matches), 1)
                self.assertEqual(source_id, approved['source'][matches[0]])
            self.assertEqual(receipt['same_prefix_vector_comparisons'], 10920)
            self.assertEqual(receipt['violating_coordinates'], 0)
            self.assertIs(receipt['local_inference_executed'], False)

    def test_loader_compiles_captured_source_without_cached_loader_calls(self):
        tree = ast.parse(SCRIPT.read_text())
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == 'VerifiedSourceImports']
        self.assertEqual(len(classes), 1)
        calls = [node for node in ast.walk(classes[0]) if isinstance(node, ast.Call)]
        names = [node.func.id for node in calls if isinstance(node.func, ast.Name)]
        self.assertIn('compile', names)
        self.assertIn('exec', names)
        attributes = [node.func.attr for node in calls if isinstance(node.func, ast.Attribute)]
        self.assertFalse(set(attributes) & {'SourceFileLoader', 'SourcelessFileLoader', 'get_code', 'cache_from_source'})
        self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)))


if __name__ == '__main__':
    unittest.main()
