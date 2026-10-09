"""Complete per-item admission over unchanged, independently checked foundations."""
from pathlib import Path

from .contracts import ContractError, identity, json_bytes, parse_json
from .foundations_data import dataset as parent_dataset, audit_bytes as parent_audit, source_receipt as parent_sources

POLICY = 'trinite.learning-data.v1'
SPLIT_SEED = 31


def source_receipt():
    return {**parent_sources(), 'learning_data.py': identity(Path(__file__).read_bytes())}


def dataset(workload):
    original, original_manifest = parent_dataset(workload)
    parent = parent_audit(original, original_manifest)
    source = source_receipt(); revision = identity(json_bytes(source))
    rows = []
    for item in parse_json(original, canonical=True)['examples']:
        core = {k: v for k, v in item.items() if k != 'example_identity'}
        core.update(source_ids=[*core['source_ids'], POLICY+':'+workload],
                    generator_revision=revision, seed=SPLIT_SEED,
                    ordering_identity=item['example_identity'],
                    transformation_lineage={**core['transformation_lineage'],
                        'parent_example_identity': item['example_identity'],
                        'operation': 'attach-per-item-admission.v1'})
        rows.append({**core, 'example_identity': identity(json_bytes(core))})
    raw = json_bytes({'schema': POLICY, 'workload': workload,
                      'examples': sorted(rows, key=lambda r: r['example_identity'])})
    admission = {**parent['admission']}
    evidence = {**admission['evidence'], 'dataset_identity': identity(raw),
                'parent_dataset_identity': identity(original),
                'parent_manifest_identity': identity(original_manifest)}
    admission.update(source_id=POLICY+':'+workload,
        generation=admission['generation']+'; attach content-bound per-item generator revision, split seed and parent/carrier lineage; preserve parent ordering keys',
        supporting_reference={'implementation': source,
            'functions': [*admission['supporting_reference']['functions'], 'learning_data.dataset'],
            'evidence_identity': identity(json_bytes(evidence))}, evidence=evidence,
        scope='this exact bounded learning-data admission over unchanged foundations facts only')
    manifest = {'schema': 'trinite.learning-admission.v1', 'workload': workload,
        'dataset_identity': identity(raw), 'admission': admission,
        'parent_dataset_identity': identity(original), 'parent_manifest_identity': identity(original_manifest),
        'family_splits': parent['family_splits'], 'counts': parent['counts']}
    return raw, json_bytes(manifest)


def audit_bytes(raw, manifest):
    if (type(raw) is not bytes or type(manifest) is not bytes
            or len(raw) > 4*1024*1024 or len(manifest) > 2*1024*1024):
        raise ContractError('learning admission requires bounded bytes')
    record = parse_json(raw, canonical=True); parse_json(manifest, canonical=True)
    workload = record.get('workload') if type(record) is dict else None
    if workload not in ('arithmetic', 'fold', 'lattice') or (raw, manifest) != dataset(workload):
        raise ContractError('learning data/admission differs from verified per-item source')
    return parse_json(manifest, canonical=True)


def validate_admission(record, workload):
    expected = parse_json(dataset(workload)[1], canonical=True)['admission']
    if type(record) is not dict or json_bytes(record) != json_bytes(expected):
        raise ContractError('learning admission contradicts scoped evidence')
