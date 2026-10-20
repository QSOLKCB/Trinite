"""New additive runner; historical source-bound aggregate remains unchanged."""
from pathlib import Path
import unittest
from check_frozen_decoding_cpu import load_suites as historical, successful


def load_suites():
    suite = historical()
    root = Path(__file__).resolve().parents[1]
    for folder in ('frozen_decoding_transport_v2', 'batch_composition_v1', 'batch_composition_review'):
        added = unittest.TestLoader().discover(str(root/'tests'/folder))
        if not added.countTestCases():
            raise RuntimeError('required suite empty: ' + folder)
        suite.addTests(added)
    return suite

if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(load_suites())
    raise SystemExit(0 if successful(result) else 1)
