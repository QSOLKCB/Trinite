#!/usr/bin/env python3
"""Actual-module public engineering conformance, never a hidden evaluation run.

Requires explicit prior `run_procedural_dataset_v1.py build`. Every invocation
executes the three real modules and verifies their entire outcomes. Published
histograms concern exposed test fixtures, not trained-model performance.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]

import queue_evaluation_v1 as gate
from procedural_dataset_v1 import admission, cases, transport
from trinite.tokenizer import ByteTokenizer

FIXTURE = ROOT / 'fixtures/procedural-dataset-v1/conformance.json'


def generate_record():
    """Return deterministic seed-17 real-oracle evidence bound to current sources.

    Compiler/host-specific executable hashes are rechecked at runtime but omitted
    from portable record equality. This is semantic replay, not binary identity
    across hosts, independent reproduction, compute equivalence or learning.
    """
    source = admission.sources()
    relative = Path(__file__).resolve().relative_to(ROOT).as_posix()
    source[relative] = gate.digest(gate.read(ROOT / relative))
    identities = transport.identities()
    rows = admission.inventory(17)
    observations = []
    digest = hashlib.sha256()
    families = {}
    max_actual = 0
    calls = 0
    for row in rows:
        module = row['module']
        observed = admission.observed(row)
        observations.append(observed)
        calls += module != 'ternary'
        outcome = transport.run(module, 'score' if module == 'ternary' else 'evaluate-privileged', row['request'])
        calls += 1
        target = cases.verify_outcome(module, row['request'], outcome)
        text = admission.prompt(module, observed)
        encoding = admission.supervised(module, observed, target)
        actual_tokens = len(encoding.input_ids)
        # Independent actual tokenizer call verifies the measured complete string.
        gate.require(encoding == ByteTokenizer().encode(text + target, answer_start=len(text.encode('ascii'))),
                     'actual token measurement disagreement')
        gate.require(cases.decode_public(text[:-len(admission.PROTOCOL['separator'])]) == (module, observed),
                     'public carrier roundtrip disagreement')
        max_actual = max(max_actual, actual_tokens)
        record = families.setdefault(row['family'], {'module': module, 'split': row['split'], 'cases': 0,
                                      'label_histogram': {}, 'max_actual_prompt_answer_tokens': 0})
        record['cases'] += 1
        record['max_actual_prompt_answer_tokens'] = max(record['max_actual_prompt_answer_tokens'], actual_tokens)
        labels = [e['status'] for e in outcome['entries']] if module == 'ternary' else [target]
        for label in labels:
            record['label_histogram'][label] = record['label_histogram'].get(label, 0) + 1
        digest.update(gate.encode({'row': row, 'public': observed, 'outcome': outcome,
                                   'prompt': text, 'target': target, 'tokens': list(encoding.input_ids)}))
    gate.require(transport.identities() == identities, 'binary identities changed during conformance')
    after = admission.sources()
    after[relative] = gate.digest(gate.read(ROOT / relative))
    gate.require(source == after, 'sources changed during conformance')
    return {'schema': 'trinite.procedural-dataset-conformance.v1', 'seed': 17,
            'sources': source, 'protocol': admission.PROTOCOL,
            'families': families, 'cases': len(rows), 'actual_module_calls': calls,
            'case_outcome_sha256': digest.hexdigest(),
            'max_actual_prompt_answer_tokens': max_actual,
            'audit': admission.audit(rows, observations),
            'training_executed': False,
            'scope': 'public synthetic actual-module conformance; all split outcomes exposed; not unseen evaluation',
            'replay_scope': 'deterministic semantic/source equality; executable identity checked locally, not equal across hosts'}


def verify(path=FIXTURE):
    """Rerun every oracle, compare exact canonical fixture bytes read-only."""
    expected = gate.read(Path(path), admission.LIMIT)
    decoded = gate.decode(expected)
    gate.require(expected == gate.encode(decoded), 'noncanonical conformance fixture')
    actual = generate_record()
    gate.require(expected == gate.encode(actual), 'conformance fixture/source/result disagreement')
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-fixture', action='store_true')
    mode.add_argument('--write-fixture', action='store_true', help='create only; never replace historical evidence')
    args = parser.parse_args()
    result = verify() if args.verify_fixture else generate_record()
    if args.write_fixture:
        gate.safe_path(FIXTURE)
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation: source changes require a new version, not resealing.
        with FIXTURE.open('xb') as handle:
            handle.write(gate.encode(result))
    print(gate.encode({'cases': result['cases'], 'actual_module_calls': result['actual_module_calls'],
                       'case_outcome_sha256': result['case_outcome_sha256'],
                       'max_actual_prompt_answer_tokens': result['max_actual_prompt_answer_tokens'],
                       'verified_fixture': args.verify_fixture, 'training_executed': False}).decode('ascii'), end='')


if __name__ == '__main__':
    main()
