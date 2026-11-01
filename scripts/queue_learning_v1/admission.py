"""Queue-only exact synthetic-fact admission. No music or generic bypass policy."""
import queue_evaluation_v1 as gate
from trinite.contracts import json_bytes
from .codec import prompt, supervised
from .producer import episodes, families

POLICY = 'trinite.queue-learning-admission.v1'


def record(rows, sources):
    expected = episodes()
    gate.require(rows == expected, 'queue admission requires exact producer regeneration')
    gate.require(len(rows) == 384 and len({gate.digest(gate.encode(r['request'])) for r in rows}) == 384,
                 'queue episode inventory or duplicate request')
    return {
        'schema': POLICY, 'outcome': 'admitted', 'source_id': 'commissioned-queue-formal-v1',
        'origin': 'original finite integer schedules; no external data, teacher, music or corpus',
        'author': 'Codex generator commissioned by Trent Slade / QSOLKCB',
        'generation': 'producer.episodes; 4 workload kinds x 12 parameter groups x 2 initial states x 2 futures x 2 masks',
        'verification': 'exact request regeneration; authoritative Rust observation/target; independent codec roundtrip and test oracle',
        'rights_basis': 'new formal queue counts and minimal symbolic notation; no third-party expressive content embedded',
        'supporting_reference': {'sources': sources, 'requests_sha256': gate.digest(json_bytes(rows)),
                                 'split_sha256': gate.digest(json_bytes(families()))},
        'scope': 'only the exact 384 generated requests under queue-learning-v1; no blanket synthetic admission',
        'reviewer': 'automated source-specific formal-fact, rights-scope, grammar and family audit commissioned by owner',
        'reviewed_on': '2026-11-01',
        'execution_authorization': 'admission is not permission to train; reviewed merged green implementation and explicit frozen-run approval required',
        'limitations': 'not legal certification or operator isolation; public generator grammar and exposed engineering tests; held-out outcomes remain unscored until commit barrier',
        'counts': {'episodes': 384, 'families': 48, 'train': 256, 'validation': 64, 'test': 64},
        'exclusions': {'truncated': 0, 'external_content': 0, 'selected_for_model_success': 0},
    }


def validate_admission(value, rows, sources):
    gate.require(json_bytes(value) == json_bytes(record(rows, sources)), 'queue-specific admission mismatch')
    return value


def audit_observations(rows, observations):
    """No targets: detect prefix aliases crossing structural groups/splits."""
    gate.require(rows == episodes() and set(observations) == {r['id'] for r in rows}, 'queue observation inventory')
    seen, group_splits = {}, {}
    max_prompt = max_input = 0
    for row in rows:
        raw = observations[row['id']]
        text = prompt(raw)
        alias = gate.digest(raw)
        if alias in seen:
            gate.require(seen[alias] == (row['family'], row['split']), 'identical prefix crosses queue family/split')
        seen[alias] = row['family'], row['split']
        group_splits.setdefault(row['family'], set()).add(row['split'])
        max_prompt = max(max_prompt, 1 + len(text))
        # Upper label bound, not the episode's future outcome.
        n = len(supervised(raw, 32 * gate.observation(raw)['horizon']).input_ids)
        max_input = max(max_input, n)
    gate.require(group_splits == {f: {s} for f, s in families().items()}, 'queue grouped split mismatch')
    return {'episodes': len(rows), 'public_prefixes': len(seen), 'max_prompt_tokens': max_prompt,
            'max_supervised_tokens': max_input, 'generation_allowance': 5,
            'max_scheduled_target_tokens': 64 * 8 * 4,  # horizon 8: at most three digits + EOS
            'split_sha256': gate.digest(json_bytes(families()))}


def prepared(path, expected):
    """Validate the original preparation anchor before consuming a train label."""
    gate.hash_arg(expected)
    files = gate.bundle(path, gate.PREPARED)
    gate.require(gate.digest(files['manifest.json']) == expected, 'queue training preparation anchor mismatch')
    m = gate.decode(files['manifest.json'])
    gate.require(m['schema'] == 'trinite.queue-preparation.v1'
                 and m['files'] == {n: gate.digest(b) for n, b in files.items() if n != 'manifest.json'}
                 and m['sources'] == gate.source_identity() and m['binary_sha256'] == gate.binary_identity(),
                 'queue training preparation changed')
    obs, evidence = gate.run('observe', files['request.json'])
    gate.require(obs == files['observation.json'] and gate.encode(evidence) == files['process.json']
                 and files['protocol.json'] == gate.encode(gate.PROTOCOL), 'queue training observation changed')
    return files


def training_rows(frozen, manifest, anchors):
    result = []
    # This filter precedes the target call. Held-out requests cannot enter this loop.
    for row in manifest['episodes']:
        if row['split'] != 'train':
            continue
        files = prepared(frozen / row['id'] / 'private', anchors[row['id']])
        raw, target_process = gate.run('target', files['request.json'])
        target = gate.decode(raw)
        gate.keys(target, ('schema', 'dropped'))
        gate.require(target['schema'] == 'trinite.queue-target.v1', 'queue train target schema')
        encoding = supervised(files['observation.json'], target['dropped'])
        result.append({'identity': row['id'], 'family': row['family'], 'split': 'train',
                       'observation': gate.decode(files['observation.json']), 'answer': target['dropped'],
                       'target_process': target_process,
                       'ids': list(encoding.input_ids), 'mask': list(encoding.attention_mask),
                       'loss': list(encoding.loss_mask)})
    gate.require(len(result) == 256, 'queue train-only inventory mismatch')
    return sorted(result, key=lambda r: gate.digest(r['identity'].encode('ascii')))


def check_training_rows(rows):
    expected = {r['id']: r for r in episodes() if r['split'] == 'train'}
    gate.require(type(rows) is list and len(rows) == len(expected), 'queue train data count')
    for row in rows:
        gate.keys(row, ('identity', 'family', 'split', 'observation', 'answer', 'target_process', 'ids', 'mask', 'loss'))
        formal = expected.pop(row['identity'], None)
        gate.require(formal is not None and row['family'] == formal['family'] and row['split'] == 'train',
                     'queue training contains foreign/held-out example')
        # Do not trust a caller-supplied answer, token mask or verification flag.
        # These are train requests only; this function never targets held-outs.
        observed, _ = gate.run('observe', gate.encode(formal['request']))
        target, evidence = gate.run('target', gate.encode(formal['request']))
        gate.require(gate.decode(observed) == row['observation']
                     and gate.decode(target) == {'schema': 'trinite.queue-target.v1', 'dropped': row['answer']}
                     and evidence == row['target_process'], 'queue train admission label/observation replay mismatch')
        encoded = supervised(gate.encode(row['observation']), row['answer'])
        gate.require((row['ids'], row['mask'], row['loss']) ==
                     (list(encoded.input_ids), list(encoded.attention_mask), list(encoded.loss_mask)),
                     'queue train tokens or masks changed')
    gate.require(not expected and rows == sorted(rows, key=lambda r: gate.digest(r['identity'].encode('ascii'))),
                 'queue train schedule order changed')
