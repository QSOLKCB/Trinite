"""Append frozen decoding assurance without replacing any historical CPU suite."""
from pathlib import Path
import unittest
from check_scale_refit_training_cpu import load_suites as historical, successful


def load_suites():
    suite=historical()
    added=unittest.TestLoader().discover(str(Path(__file__).resolve().parents[1]/'tests/frozen_decoding_v1'))
    if not added.countTestCases():raise RuntimeError('required frozen decoding suite empty')
    suite.addTests(added)
    return suite

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(load_suites())
    raise SystemExit(0 if successful(result) else 1)
