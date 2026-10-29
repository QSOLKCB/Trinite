"""Reuse unchanged contrast metrics; train-only strata and raw output diversity."""
from collections import Counter
from trinite.contracts import parse_json
from music_contrast_v1.reporting import summarize as contrast_summary
from .spec import rows


def summarize(raw):
    result = contrast_summary(raw, 'train')
    records = parse_json(raw, canonical=True)['predictions']
    by_id = {r['example_identity']: r for r in records}
    train = rows()
    for key, field in (('by_answer', 'answer'), ('by_parent', 'parent_id')):
        result[key] = {value: {'rows': len(selected),
            'exact_plus_eos': sum(by_id[r['example_identity']]['correct'] for r in selected)}
            for value in sorted({r[field] for r in train})
            for selected in [[r for r in train if r[field] == value]]}
    counts = Counter(tuple(r['generated_tokens']) for r in records)
    result['raw_output_histogram'] = [{'tokens': list(tokens), 'rows': count}
        for tokens, count in sorted(counts.items())]
    result['unique_raw_outputs'] = len(counts)
    return result
