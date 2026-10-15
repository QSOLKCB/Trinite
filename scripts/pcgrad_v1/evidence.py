"""Fresh bounded numerical replay receipts, not a PROVENANCE custody bundle."""
from fractions import Fraction
import torch
from trinite.contracts import ContractError, identity, json_bytes, parse_json
from trinite.training import require_environment
from .kernel import combine
from .spec import PROTOCOL_ID, SPEC_COMMIT, sources, specification
from .toy import create, run, resume, snapshot, values


def report():
    spec = specification()
    environment, source = require_environment(), sources()
    cases = []
    for case in spec['cases']:
        a, b = (torch.tensor(case[k], dtype=torch.float32) for k in ('a', 'b'))
        result = combine(a, b, arm='pcgrad')
        expected = [torch.tensor([float(Fraction(v)) for v in row], dtype=torch.float32)
                    for row in case['projected']]
        if any(not torch.equal(x, y) for x, y in zip(result.projected, expected)):
            raise ContractError('hand-derived rational projection fixture mismatch')
        cases.append({'name': case['name'], 'projected_hex': [values(g) for g in result.projected],
                      'combined_hex': values(result.gradient), 'diagnostics': result.diagnostics})
    cells = []
    for seed in spec['seeds']:
        for arm in spec['arms']:
            complete = snapshot(run(create(arm, seed)))
            interruptions = []
            for stop in (0, 3, 7, 8):
                prefix = snapshot(run(create(arm, seed), stop_after=stop))
                restored = resume(prefix, expected_identity=identity(prefix), arm=arm, seed=seed)
                replayed = snapshot(run(restored))
                if replayed != complete:
                    raise ContractError('full toy optimizer/RNG/history replay mismatch')
                interruptions.append({'stop': stop, 'prefix_identity': identity(prefix),
                                      'final_identity': identity(replayed), 'exact_match': True})
            cells.append({'seed': seed, 'arm': arm, 'final_checkpoint': parse_json(complete, canonical=True),
                          'final_checkpoint_identity': identity(complete), 'interruptions': interruptions})
    if require_environment() != environment or sources() != source:
        raise ContractError('source/environment changed during engineering conformance')
    return {'schema': 'trinite.pcgrad-engineering-report.v1', 'specification_commit': SPEC_COMMIT,
            'protocol_identity': PROTOCOL_ID, 'source': source, 'environment': environment,
            'cases': cases, 'cells': cells, 'scientific_execution': False,
            'training_gate_evaluated': False, 'learning_claim': False, 'release_ready': False,
            'closed_custody': False, 'native_checkpoint_replay': False,
            'scope': 'two-vector algebra and fixed toy full model/AdamW/RNG/history replay only'}
