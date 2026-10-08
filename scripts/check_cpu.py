"""Run every required CPU suite in one process; keep isolation subprocess tests."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SUITES = ('model', 'training', 'geometry')


def load_suites():
    suite = unittest.TestSuite()
    # Separate discovery roots preserve the existing standalone module IDs.
    for name in SUITES:
        discovered = unittest.TestLoader().discover(str(ROOT/'tests'/name))
        if not discovered.countTestCases():
            raise RuntimeError('required CPU suite is empty: '+name)
        suite.addTests(discovered)
    return suite


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(load_suites())
    # Required CPU coverage cannot become green by skipping a broken backend.
    raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)
