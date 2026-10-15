"""Run the unchanged historical CPU aggregate plus new scale-refit engineering tests."""
from pathlib import Path
import unittest

from check_research_cpu import load_suites as load_historical, successful

ROOT = Path(__file__).resolve().parents[1]


def load_suites():
    suite = load_historical()
    added = unittest.TestLoader().discover(str(ROOT/'tests/scale_refit'))
    if not added.countTestCases():
        raise RuntimeError('required scale-refit engineering suite is empty')
    suite.addTests(added)
    return suite


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(load_suites())
    raise SystemExit(0 if successful(result) else 1)
