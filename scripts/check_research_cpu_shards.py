"""OPT-guided parallel CI wrapper around the unchanged full research suite.

Each CI job checks the complete required test inventory *before* running its
assigned subset. The source of truth remains check_research_cpu.load_suites().
No tests, assertions, numeric operations or verification paths are bypassed.
The legacy complete sequential runner remains runnable without modification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

# Four whole-suite shards, assigned from the immutable v0.2.0 run timing
# profile. Keep each suite in one job to preserve setUpClass/tearDownClass,
# per-suite order, and the real subprocess and replay checks. Rebalance only
# with a new reviewed profile; do not silently remove expensive tests.
SHARDS = (
    ('task_balance', 'ternary_dynamics', 'aizawa_reference'),
    ('budget', 'maths', 'maths_retention', 'retention_budget'),
    ('learning', 'generalization', 'scalar', 'retention_v2'),
    ('model', 'training', 'geometry', 'comparison', 'foundations',
     'convergence', 'fold_exposure', 'fold_order'),
)
BASELINE_V020_CASES = 261
SCHEMA = 'trinite.opt-research-ci-shard.v1'


class CoverageError(RuntimeError):
    """Full-suite/shard coverage is incomplete or ambiguous."""


def case_ids(suite):
    """Collect actual unittest case IDs, not guessed test-file names."""
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from case_ids(item)
        else:
            yield item.id()


def required_suites():
    """Read authoritative suite names lazily; importing this file needs no Torch."""
    from check_cpu import SUITES
    from check_research_cpu import RESEARCH_SUITES
    return (*SUITES, *RESEARCH_SUITES)


def validate_partition(names, shards=SHARDS):
    """Require an exact partition, with original order inside each shard."""
    if (type(names) is not tuple or not names or
            any(type(name) is not str or not name for name in names) or
            len(set(names)) != len(names)):
        raise CoverageError('invalid authoritative suite inventory')
    if type(shards) is not tuple or not shards:
        raise CoverageError('missing complete shard assignment')
    allocated = []
    for members in shards:
        if type(members) is not tuple or not members:
            raise CoverageError('empty/malformed CI shard')
        if any(type(name) is not str or name not in names for name in members):
            raise CoverageError('CI shard references unknown suite')
        positions = [names.index(name) for name in members]
        if positions != sorted(positions):
            raise CoverageError('CI shard must preserve authoritative suite order')
        allocated.extend(members)
    if len(allocated) != len(names) or set(allocated) != set(names) or len(set(allocated)) != len(names):
        raise CoverageError('CI shards omit or repeat required suites')
    return True


def audit(full, names, *, shards=SHARDS, expected_cases=BASELINE_V020_CASES):
    """Compare each shard against the actual complete, ordered suite inventory.

    Returns the original suite objects without executing their tests. Import
    failures are not promoted: real unittest _FailedTest placeholders fail when
    run; a false match cannot turn execution success into a fixture guarantee.
    """
    validate_partition(names, shards)
    if type(expected_cases) is not int or expected_cases < 1:
        raise CoverageError('invalid expected case budget')
    if not isinstance(full, unittest.TestSuite):
        raise CoverageError('original runner must return a unittest suite')
    parts = list(full)
    if len(parts) != len(names):
        raise CoverageError('discovered suite boundaries differ from full runner')
    suites = {}
    all_ids = []
    for name, part in zip(names, parts):
        if not isinstance(part, unittest.TestSuite):
            raise CoverageError('missing discovered test suite: '+name)
        ids = list(case_ids(part))
        if not ids or any(type(i) is not str or not i for i in ids):
            raise CoverageError('empty/invalid discovered suite: '+name)
        suites[name] = {'suite': part, 'ids': ids}
        all_ids.extend(ids)
    if len(all_ids) != expected_cases:
        raise CoverageError('case count diverges from v0.2.0 baseline; review the inventory: '
                            +str(len(all_ids))+' != '+str(expected_cases))
    if len(all_ids) != len(set(all_ids)):
        raise CoverageError('duplicate test IDs in full-suite discovery')
    for members in shards:
        ids = [case for name in members for case in suites[name]['ids']]
        if not ids:
            raise CoverageError('empty shard after full-suite discovery')
    digests = {'full_test_ids_sha256': hashlib.sha256(json.dumps(
        all_ids, ensure_ascii=True, separators=(',', ':')).encode('utf-8')).hexdigest()}
    return suites, all_ids, digests


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executed_ids = []

    def startTest(self, test):
        self.executed_ids.append(test.id())
        super().startTest(test)


def successful(result, expected_ids):
    """Enforce complete execution, no skips and ordinary unittest success."""
    return (isinstance(result, RecordingResult) and result.wasSuccessful()
            and not result.skipped and result.executed_ids == expected_ids
            and result.testsRun == len(expected_ids))


def validate_shard_index(index, shards=SHARDS):
    if type(index) is not int or index < 0 or index >= len(shards):
        raise CoverageError('unknown fixed CPU shard index')
    return index


def receipt(index, names, all_ids, expected_ids, digest, elapsed_seconds):
    """Evidence disclosure, not a verifier result cache or signed certificate."""
    def source(name):
        return 'sha256:'+hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return {
        'schema': SCHEMA,
        'shard': index,
        'authoritative_suites': list(names),
        'assigned_suites': list(SHARDS[index]),
        'total_required_cases': len(all_ids),
        'shard_executed_cases': len(expected_ids),
        'shard_wall_seconds_hex': elapsed_seconds.hex(),
        'full_test_ids_sha256': digest['full_test_ids_sha256'],
        'shard_test_ids_sha256': hashlib.sha256(json.dumps(
            expected_ids, ensure_ascii=True, separators=(',', ':')).encode('utf-8')).hexdigest(),
        'source': {
            'scripts/check_cpu.py': source('scripts/check_cpu.py'),
            'scripts/check_research_cpu.py': source('scripts/check_research_cpu.py'),
            'scripts/check_research_cpu_shards.py': source('scripts/check_research_cpu_shards.py'),
            'requirements.lock': source('requirements.lock'),
        },
        'scope': 'executed whole-suite CI shard; original numerical and custody verifiers unchanged',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=int, required=True)
    args = parser.parse_args(argv)
    try:
        index = validate_shard_index(args.shard)
        # Import and discover ALL suites as the authoritative combined runner
        # does. Tests retain their original root, names and internal order.
        from check_research_cpu import load_suites
        names = required_suites()
        suites, all_ids, digest = audit(load_suites(), names)
        members = SHARDS[index]
        selected = unittest.TestSuite(suites[name]['suite'] for name in members)
        expected = [case for name in members for case in suites[name]['ids']]
        print('OPT-CI-SHARD: covered '+str(len(all_ids))+' required v0.2.0 test IDs; '
              'running shard '+str(index)+': '+', '.join(members), file=sys.stderr, flush=True)
        started = time.monotonic()
        result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(selected)
        elapsed = time.monotonic()-started
        if not successful(result, expected):
            raise CoverageError('shard failed, skipped, or did not execute the exact ordered test IDs')
        print(json.dumps(receipt(index, names, all_ids, expected, digest, elapsed),
                         sort_keys=True, ensure_ascii=True, separators=(',', ':')), flush=True)
        return 0
    except (CoverageError, OSError, ValueError) as error:
        print('OPT-CI-SHARD: '+str(error), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
