"""Run every required CPU research test, preserving standalone suite order."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from check_cpu import load_suites as load_native_suites

RESEARCH_SUITES = ('comparison', 'foundations', 'convergence', 'budget', 'learning', 'generalization', 'scalar', 'fold_exposure', 'fold_order', 'maths')


def load_suites():
    suite = load_native_suites()
    for name in RESEARCH_SUITES:
        discovered = unittest.TestLoader().discover(str(ROOT/'tests'/name))
        if not discovered.countTestCases():
            raise RuntimeError('required research suite is empty: '+name)
        suite.addTests(discovered)
    return suite


def successful(result):
    return result.wasSuccessful() and not result.skipped


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(load_suites())
    raise SystemExit(0 if successful(result) else 1)
