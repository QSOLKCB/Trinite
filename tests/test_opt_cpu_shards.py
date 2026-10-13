"""Stdlib-only adversarial checks for the OPT model CI partition, not a model test."""
from pathlib import Path
import hashlib
import io
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from check_research_cpu_shards import (
    BASELINE_V020_CASES, SHARDS, CoverageError, RecordingResult, audit,
    case_ids, main, required_suites, successful, validate_partition,
    validate_shard_index,
)


def case(name, *, skip=False):
    # Dynamic local fixture classes are not collected as root-suite tests.
    def body(self):
        self.assertEqual(2+2,4)
    impl = unittest.skip('mandatory skip test')(body) if skip else body
    cls = type('Toy_'+name, (unittest.TestCase,), {'test_case': impl})
    return cls('test_case')


def fake_parts():
    return unittest.TestSuite((
        unittest.TestSuite([case('alpha')]),
        unittest.TestSuite([case('beta')]),
        unittest.TestSuite([case('gamma')]),
    ))


def fake_discovered():
    return {
        'alpha': unittest.TestSuite([case('alpha')]),
        'beta': unittest.TestSuite([case('beta')]),
        'gamma': unittest.TestSuite([case('gamma')]),
    }


class OptCiShardingContract(unittest.TestCase):
    def test_exact_source_suites_one_per_shard_in_legacy_order(self):
        names = required_suites()  # Does not discover/import model tests.
        self.assertTrue(validate_partition(names))
        self.assertEqual(len(SHARDS), 4)
        self.assertEqual(set().union(*map(set, SHARDS)), set(names))
        self.assertEqual(len({name for group in SHARDS for name in group}), len(names))
        self.assertEqual(BASELINE_V020_CASES, 261)
        self.assertEqual(SHARDS[0][0], 'task_balance')
        self.assertIn('retention_budget', SHARDS[1])
        self.assertIn('retention_v2', SHARDS[2])
        self.assertIn('fold_order', SHARDS[3])

    def test_duplicate_missing_unknown_and_reordered_assignments_fail(self):
        names = ('alpha','beta','gamma')
        bad = (
            (('alpha',), ('beta',)),
            (('alpha',), ('beta','beta','gamma')),
            (('alpha',), ('beta','omega'), ('gamma',)),
            (('beta','alpha'), ('gamma',)),
            (('alpha',), (), ('beta','gamma')),
            (('alpha',), ['beta','gamma']),
        )
        for candidate in bad:
            with self.subTest(candidate=candidate), self.assertRaises(CoverageError):
                validate_partition(names, candidate)
        for broken in (('alpha','alpha'), (), ('alpha',False), ['alpha','beta']):
            with self.subTest(names=broken), self.assertRaises(CoverageError):
                validate_partition(broken, (('alpha',),))

    def test_dynamic_audit_uses_actual_case_identities_and_rejects_drift(self):
        names = ('alpha','beta','gamma')
        assignments = (('gamma','alpha'), ('beta',))
        # Same content without correct order is not an acceptable shard.
        with self.assertRaisesRegex(CoverageError,'order'):
            audit(fake_parts(),names,shards=assignments,expected_cases=3,
                      discovered=fake_discovered())
        assignments = (('alpha','beta'), ('gamma',))
        groups,ids,hashes = audit(fake_parts(),names,shards=assignments,expected_cases=3,
                      discovered=fake_discovered())
        # Genuine combined discovery need not preserve a top-level slot per suite.
        flattened = unittest.TestSuite([unittest.TestSuite(
            [case('alpha'),case('beta'),case('gamma')])])
        self.assertEqual(len(list(flattened)),1)
        audit(flattened,names,shards=assignments,expected_cases=3,
              discovered=fake_discovered())
        self.assertEqual(ids, [case('alpha').id(),case('beta').id(),case('gamma').id()])
        self.assertEqual(set(groups),set(names))
        self.assertEqual(len(hashes['full_test_ids_sha256']),64)
        for expected in (2, 4):
            with self.assertRaisesRegex(CoverageError,'case count'):
                audit(fake_parts(),names,shards=assignments,expected_cases=expected,
                      discovered=fake_discovered())
        duplicate = unittest.TestSuite([
            unittest.TestSuite([case('alpha')]),
            unittest.TestSuite([case('alpha')]),
            unittest.TestSuite([case('gamma')]),
        ])
        with self.assertRaisesRegex(CoverageError,'duplicate/empty test IDs'):
            audit(duplicate,names,shards=assignments,expected_cases=3,
                      discovered=fake_discovered())
        empty = unittest.TestSuite([
            unittest.TestSuite([case('alpha')]), unittest.TestSuite(),
            unittest.TestSuite([case('gamma')]),
        ])
        with self.assertRaisesRegex(CoverageError,'case count'):
            audit(empty,names,shards=assignments,expected_cases=3,
                      discovered=fake_discovered())
        empty_parts=fake_discovered()
        empty_parts['beta']=unittest.TestSuite()
        with self.assertRaisesRegex(CoverageError,'empty'):
            audit(fake_parts(),names,shards=assignments,expected_cases=3,
                  discovered=empty_parts)
        with self.assertRaisesRegex(CoverageError,'case count'):
            audit(unittest.TestSuite([unittest.TestSuite([case('alpha')])]),
                  names,shards=assignments,expected_cases=3,
                      discovered=fake_discovered())

    def test_true_execution_ids_and_required_skip_fail_closed(self):
        log = io.StringIO()
        result=unittest.TextTestRunner(stream=log,verbosity=0,resultclass=RecordingResult).run(
            unittest.TestSuite([case('alpha'),case('beta')]))
        ordered = [case('alpha').id(),case('beta').id()]
        self.assertTrue(successful(result,ordered))
        self.assertFalse(successful(result,list(reversed(ordered))))
        self.assertFalse(successful(result,ordered[:1]))
        skipped=unittest.TextTestRunner(stream=io.StringIO(),verbosity=0,
            resultclass=RecordingResult).run(unittest.TestSuite([case('skip',skip=True)]))
        self.assertFalse(successful(skipped,[case('skip',skip=True).id()]))

    def test_closed_shard_selector_and_preserved_v020_full_runner(self):
        for value in (-1, 4, None, True, False, '0'):
            with self.subTest(value=value), self.assertRaises(CoverageError):
                validate_shard_index(value)
        self.assertEqual(main(['--shard','99']),1)
        original=(ROOT/'scripts/check_research_cpu.py').read_bytes()
        blob=hashlib.sha1(b'blob '+str(len(original)).encode()+b'\0'+original).hexdigest()
        # Exact v0.2.0 native full-run entrypoint remains available unchanged.
        self.assertEqual(blob,'d55f8736b43d80386ba77fda5369684beba00446')
        yaml=(ROOT/'.github/workflows/cpu.yml').read_text()
        self.assertIn("python: ['3.11', '3.12', '3.13']",yaml)
        self.assertIn('shard: [0, 1, 2, 3]',yaml)
        self.assertIn('  model-conformance-shard:',yaml)
        self.assertIn('  model-conformance:\n    needs: model-conformance-shard',yaml)
        self.assertIn('if: ${{ always() }}',yaml)
        self.assertIn('SHARDS_RESULT: ${{ needs.model-conformance-shard.result }}',yaml)
        self.assertIn("if [ \"$SHARDS_RESULT\" != 'success' ]; then",yaml)
        self.assertIn('python scripts/check_research_cpu_shards.py --shard',yaml)
        self.assertIn('python -m pip install --require-hashes --no-deps -r requirements.lock',yaml)
        self.assertIn('python -m pip check',yaml)
        self.assertIn('python -m trinite inspect-model --lane dense',yaml)
        self.assertIn('python -m trinite inspect-model --lane ternary',yaml)


if __name__=='__main__':
    unittest.main()
