"""Reuse authoritative retained-output analysis; add frozen factorial contrasts."""
from position_study_v1 import reporting as original
from .protocol import CONTRASTS, SEEDS, STRATA, PROTOCOL
from .data import validate_matching


def describe(arm, training_record, development_records):
    result = original.describe('control', training_record, development_records)
    return {**result, 'schema': 'trinite.position-exposure-description.v1'}


def controls(arm, stratum):
    return original.controls('control', stratum)


def design_summary():
    return {'arithmetic_design': original.design_summary(),
        'address_role_exposure': validate_matching(),
        'coverage_scope': PROTOCOL['coverage_gate'],
        'limitations': PROTOCOL['coverage_limit']}


def criteria(descriptions):
    contrasts = {}
    for name, (left, right) in CONTRASTS.items():
        mapped = {}
        for seed in SEEDS:
            for src, dst in ((left, 'control'), (right, 'candidate')):
                if f'{src}-{seed}' in descriptions:
                    mapped[f'{dst}-{seed}'] = descriptions[f'{src}-{seed}']
        # Exact historical family thresholds, independently applied to each
        # prespecified directed contrast, never selected from observed outcomes.
        result = original.criteria(mapped)
        counts = {}
        for seed in SEEDS:
            a, b = descriptions.get(f'{left}-{seed}'), descriptions.get(f'{right}-{seed}')
            if a is None or b is None: continue
            counts[str(seed)] = {}
            for s in STRATA:
                x, y = a['development'][s]['total'], b['development'][s]['total']
                counts[str(seed)][s] = {'left_families': x['all_renderings_correct_families'],
                    'right_families': y['all_renderings_correct_families'],
                    'difference': y['all_renderings_correct_families'] - x['all_renderings_correct_families'],
                    'families': x['families']}
        contrasts[name] = {'left': left, 'right': right, **result, 'family_counts': counts}
    return {'contrasts': contrasts,
        'interpretation': 'separate prespecified descriptive contrasts; no omnibus gate, significance, fresh confirmation or causal mechanism proof'}
