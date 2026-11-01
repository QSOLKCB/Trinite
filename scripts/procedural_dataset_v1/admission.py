"""Source-specific procedural admission and train-only target custody.

Artifacts are publicly reproducible engineering examples, not fresh benchmarks.
This module cannot train, select a model, or evaluate held-out predictions.
"""
from pathlib import Path

import queue_evaluation_v1 as gate
from trinite.tokenizer import ByteTokenizer
from . import cases, transport

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = {
    'schema': 'trinite.procedural-dataset-protocol.v1',
    'generator_version': 1, 'context_tokens': 256, 'max_examples': 2048,
    'seed_min': 0, 'seed_max': 65535, 'separator': '\n=',
    'answer_byte_caps': {'ternary': 96, 'tombstone': 12, 'context': 2},
    'execution_enabled': False,
    'exposure': 'public engineering families; not unseen evaluation evidence',
    'held_out_policy': 'no validation/test targets in admission or training export',
    'split_unit': 'whole structural generator family including every seed and variant',
}
FILES = ('manifest.json', 'attempt.json', 'public.json', 'requests.json', 'train-targets.json', 'admission.json')
LIMIT = gate.LIMIT  # Same strict bounded JSON ceiling as the shared decoder.


def same(a, b, message):
    gate.require(gate.encode(a) == gate.encode(b), message)


def sources():
    names = {'AGENTS.md', 'scripts/run_procedural_dataset_v1.py',
             'scripts/queue_evaluation_v1.py', 'scripts/tombstone_adapter_v1.py',
             'src/trinite/__init__.py', 'src/trinite/tokenizer.py', 'src/trinite/contracts.py',
             'docs/PROCEDURAL-DATASET-V1.md'}
    names.update(str(p.relative_to(ROOT)) for p in (ROOT / 'scripts/procedural_dataset_v1').glob('*.py'))
    for module in ('ternary-evidence-v1', 'tombstone-adapter-v1', 'context-symbol-relation-v1'):
        base = ROOT / 'external' / module
        names.update(str(p.relative_to(ROOT)) for p in base.rglob('*') if p.is_file()
                     and 'target' not in p.relative_to(base).parts and 'tests' not in p.relative_to(base).parts)
    return {n: gate.digest(gate.read(ROOT / n, LIMIT)) for n in sorted(names)}


def inventory(seed):
    gate.integer(seed, 0, 65535)
    rows = []
    for family, info in sorted(cases.FAMILIES.items()):
        for index, request in enumerate(cases.generate(family, seed)):
            rows.append({'id': f'case-{len(rows):04d}', 'family': family, 'split': info['split'],
                         'module': info['module'], 'seed': seed, 'variant': index, 'request': request})
    gate.require(0 < len(rows) <= PROTOCOL['max_examples'], 'inventory count exceeds bound')
    return rows


def prompt(module, observation):
    """Only public schema is accepted; provenance/targets are not arguments."""
    text = cases.encode_public(module, observation)
    decoded_module, decoded = cases.decode_public(text)
    same([decoded_module, decoded], [module, observation], 'lossless public roundtrip failed')
    text += PROTOCOL['separator']
    ByteTokenizer().encode(text)  # Includes BOS/EOS, never truncates.
    return text


def supervised(module, observation, target):
    text = prompt(module, observation)
    gate.require(type(target) is str and target.isascii()
                 and len(target) <= PROTOCOL['answer_byte_caps'][module], 'target byte bound')
    return ByteTokenizer().encode(text + target, answer_start=len(text.encode('ascii')))


def public_only(observation, module):
    """Inference surface returns only prompt tokens, never reads a target file."""
    text = prompt(module, observation)
    # EOS terminates completed text, not an unfinished inference prefix.
    return list(ByteTokenizer().encode(text).input_ids[:-1])


def observed(row):
    module, request = row['module'], row['request']
    expected = cases.public_observation(module, request)
    if module == 'ternary':
        # Its only native operation is score; all input fields are public.
        cases.validate_request(module, request)
        return expected
    result = transport.run(module, 'observe', request)
    same(result, expected, 'native public observation differs from regenerated request')
    return result


def target(row):
    operation = 'score' if row['module'] == 'ternary' else 'evaluate-privileged'
    outcome = transport.run(row['module'], operation, row['request'])
    actual = cases.verify_outcome(row['module'], row['request'], outcome)
    gate.require(actual == cases.expected_target(row['module'], row['request']),
                 'native label differs from independent formal oracle')
    return actual


def audit(rows, observations):
    gate.require(len(rows) == len(observations), 'public inventory count')
    seen, prefixes, counts = {}, {}, dict.fromkeys(('train', 'validation', 'test'), 0)
    max_prompt = max_total = 0
    for row, obs in zip(rows, observations):
        family_split = (row['family'], row['split'])
        key = gate.digest(gate.encode([row['module'], obs]))
        gate.require(key not in seen, 'duplicate canonical public input')
        seen[key] = family_split
        # Query-independent prefix aliases must never cross a family boundary.
        prefix = {k: v for k, v in obs.items() if k != 'query'}
        prefix_key = gate.digest(gate.encode([row['module'], prefix]))
        gate.require(prefix_key not in prefixes or prefixes[prefix_key] == family_split,
                     'public query prefix crosses family/split')
        prefixes[prefix_key] = family_split
        text = prompt(row['module'], obs)
        # Reserve a fixed task-specific answer allowance without computing a held-out label.
        encoding = supervised(row['module'], obs, '0' * PROTOCOL['answer_byte_caps'][row['module']])
        max_prompt = max(max_prompt, len(ByteTokenizer().encode(text).input_ids) - 1)
        max_total = max(max_total, len(encoding.input_ids))
        counts[row['split']] += 1
    return {'counts': counts, 'families': len(cases.FAMILIES), 'canonical_inputs': len(seen),
            'query_prefixes': len(prefixes), 'max_prompt_tokens': max_prompt,
            'max_reserved_prompt_answer_tokens': max_total, 'context_tokens': 256,
            'truncated': 0, 'exposure': PROTOCOL['exposure']}


def rights(rows, source, stats):
    return {'schema': 'trinite.procedural-admission.v1', 'outcome': 'admitted',
            'source_id': 'commissioned-three-module-formal-v1',
            'origin': 'original finite numeric arrays and rule tables; no external corpus or teacher',
            'author': 'Codex generator commissioned by Trent Slade / QSOLKCB',
            'rights_basis': 'formal numeric facts and minimal symbolic carriers; no expressive third-party data',
            'scope': 'only exact regenerated inventory and source closure in this manifest',
            'generation': 'cases.generate; whole structural families include all seeds and transformations',
            'verification': 'strict public roundtrip; actual native train oracle and independent formal oracle; exact regeneration',
            'reviewer': 'automated source-specific formal fact and provenance audit commissioned by owner',
            'reviewed_on': '2026-11-01', 'sources': source,
            'inventory_sha256': gate.digest(gate.encode(rows)), 'audit': stats,
            'execution_authorization': 'not permission to train; fresh reviewed frozen-run approval required',
            'limitations': 'not legal certification, secret benchmark, OS isolation or model evidence'}


def materialize(seed):
    rows = inventory(seed)
    observations = []
    for row in rows:
        try:
            observations.append(observed(row))
        except Exception as error:
            error.procedural_context = {'row': row, 'operation': 'observe'}
            raise
    stats = audit(rows, observations)
    targets = []
    for row, obs in zip(rows, observations):
        if row['split'] != 'train':
            continue  # Filter before privileged native evaluation, never afterwards.
        try:
            answer = target(row)
        except Exception as error:
            error.procedural_context = {'row': row, 'operation': 'score' if row['module'] == 'ternary' else 'evaluate-privileged'}
            raise
        enc = supervised(row['module'], obs, answer)
        targets.append({'id': row['id'], 'answer': answer, 'tokens': list(enc.input_ids),
                        'attention_mask': list(enc.attention_mask), 'loss_mask': list(enc.loss_mask)})
    return rows, observations, stats, targets


def prepare(out, seed=17):
    destination = gate.fresh(out)
    try:
        gate.integer(seed, 0, 65535)
        source, binaries = sources(), transport.identities()
        gate.write(destination / 'attempt.json', gate.encode(
            {'seed': seed, 'protocol': PROTOCOL, 'sources': source, 'binaries': binaries}))
        rows, observations, stats, targets = materialize(seed)
        content = {'public.json': observations, 'requests.json': rows,
                   'train-targets.json': targets, 'admission.json': rights(rows, source, stats)}
        for name, value in content.items():
            gate.write(destination / name, gate.encode(value))
        same(source, sources(), 'source changed during preparation')
        same(binaries, transport.identities(), 'binary changed during preparation')
        manifest = {'schema': 'trinite.procedural-manifest.v1', 'seed': seed,
                    'protocol': PROTOCOL, 'families': cases.FAMILIES, 'sources': source,
                    'binaries': binaries, 'audit': stats,
                    'files': {name: gate.digest(gate.read(destination / name, LIMIT)) for name in (*content, 'attempt.json')}}
        raw = gate.encode(manifest)
        gate.write(destination / 'manifest.json', raw)
        return gate.digest(raw)
    except Exception as error:
        failure = {'schema': 'trinite.procedural-failure.v1', 'success': False,
                   'reason': type(error).__name__, 'seed': seed, 'context': getattr(error, 'procedural_context', None)}
        if isinstance(error, gate.ProcessError):
            failure['process'] = error.evidence
        gate.write(destination / 'failure.json', gate.encode(failure))
        raise


def verify(path, expected_sha256):
    """Read-only full regeneration; supplied hashes are not sufficient evidence."""
    gate.hash_arg(expected_sha256)
    path = gate.safe_path(path)
    gate.require(path.is_dir() and set(p.name for p in path.iterdir()) == set(FILES),
                 'missing/extra artifact or retained failure')
    raw = {name: gate.read(path / name, LIMIT) for name in FILES}
    gate.require(gate.digest(raw['manifest.json']) == expected_sha256, 'external manifest anchor mismatch')
    value = {name: gate.decode(data) for name, data in raw.items()}
    manifest = value['manifest.json']
    gate.keys(manifest, ('schema', 'seed', 'protocol', 'families', 'sources', 'binaries', 'audit', 'files'))
    gate.require(manifest['schema'] == 'trinite.procedural-manifest.v1', 'unknown manifest schema')
    same(manifest['protocol'], PROTOCOL, 'protocol mismatch')
    same(manifest['families'], cases.FAMILIES, 'family/split mismatch')
    same(manifest['sources'], sources(), 'source mismatch')
    same(manifest['binaries'], transport.identities(), 'binary mismatch')
    same(manifest['files'], {n: gate.digest(raw[n]) for n in FILES if n != 'manifest.json'}, 'file digest mismatch')
    rows, observations, stats, targets = materialize(manifest['seed'])
    for name, expected in {'attempt.json': {'seed': manifest['seed'], 'protocol': PROTOCOL,
                                            'sources': manifest['sources'], 'binaries': manifest['binaries']},
                           'requests.json': rows, 'public.json': observations,
                           'train-targets.json': targets,
                           'admission.json': rights(rows, sources(), stats)}.items():
        same(value[name], expected, 'regenerated artifact mismatch: ' + name)
        gate.require(raw[name] == gate.encode(expected), 'noncanonical artifact bytes')
    same(manifest['audit'], stats, 'regenerated count/token audit mismatch')
    gate.require(raw['manifest.json'] == gate.encode(manifest), 'noncanonical manifest')
    same(manifest['sources'], sources(), 'source changed during verification')
    same(manifest['binaries'], transport.identities(), 'binary changed during verification')
    return value


def training_rows(path, expected_sha256):
    """Admission export only, no model import. Held-out targets are never exported."""
    values = verify(path, expected_sha256)
    return values['train-targets.json']
