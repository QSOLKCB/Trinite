import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def runner():
    spec = importlib.util.spec_from_file_location('research_cpu_runner',ROOT/'scripts/check_research_cpu.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def ids(suite):
    for case in suite:
        if isinstance(case,unittest.TestSuite): yield from ids(case)
        else: yield case.id()


class ResearchRunnerTests(unittest.TestCase):
    def test_cpu_research_coverage_001_exact_ordered_standalone_case_identity(self):
        module = runner(); expected = []
        for name in ('model','training','geometry',*module.RESEARCH_SUITES):
            expected.extend(ids(unittest.TestLoader().discover(str(ROOT/'tests'/name))))
        actual = list(ids(module.load_suites()))
        self.assertEqual(actual,expected); self.assertEqual(len(actual),len(set(actual)))

    def test_empty_import_error_failure_and_required_skip_never_succeed(self):
        module = runner()
        with patch.object(module.unittest.TestLoader,'discover',return_value=unittest.TestSuite()):
            with self.assertRaises(RuntimeError): module.load_suites()
        result = unittest.TestResult(); self.assertTrue(module.successful(result))
        result.skipped.append(('required','skip')); self.assertFalse(module.successful(result))
        result.skipped.clear(); result.errors.append(('import','error')); self.assertFalse(module.successful(result))
        result.errors.clear(); result.failures.append(('required','failed')); self.assertFalse(module.successful(result))
