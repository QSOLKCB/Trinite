"""Fresh source-bound engineering receipt; no learning/custody certification."""
import math
from fractions import Fraction
import torch
from trinite.contracts import ContractError, identity, parse_json
from trinite.training import require_environment
from .model import MODEL_ID
from .quantizer import QUANTIZER_ID, quantize
from .spec import PROTOCOL_ID, SPEC_COMMIT, sources, specification
from .toy import create, run, resume, snapshot, tensors, values


def analytic():
    fixture = specification()['analytic_fixture']
    w = torch.tensor([[float(Fraction(x)) for x in fixture['weights']]], dtype=torch.float32, requires_grad=True)
    s = torch.tensor(float(fixture['scale']), dtype=torch.float32, requires_grad=True)
    incoming = torch.tensor([[float(Fraction(x)) for x in fixture['incoming']]], dtype=torch.float32)
    result = quantize(w, s)
    result.weight.backward(incoming)
    expected_w = torch.tensor([[float(Fraction(x)) for x in fixture['weight_grad']]], dtype=torch.float32)
    expected_s = torch.tensor(9/math.sqrt(7), dtype=torch.float32)
    if (result.codes.tolist() != [fixture['codes']] or not torch.equal(w.grad, expected_w)
            or not torch.equal(s.grad, expected_s) or not torch.equal(result.weight, result.codes.float()*s.detach())):
        raise ContractError('independent hand-derived analytic fixture mismatch')
    return {'codes': result.codes.tolist(), 'actual_forward_hex': values(result.weight),
            'weight_gradient_hex': values(w.grad), 'scale_gradient_hex': values(s.grad),
            'independent_expected_scale_gradient': '9/sqrt(7) rounded float32'}


def report():
    spec = specification()
    environment, source = require_environment(), sources()
    case = analytic()
    cells, paired = [], []
    for seed in spec['seeds']:
        learned, frozen = create('learned', seed), create('frozen', seed)
        if tensors(learned.model) != tensors(frozen.model):
            raise ContractError('learned/frozen model initialization differs')
        paired.append({'seed': seed, 'initial_model_exact_match': True,
                       'learned_inventory': learned.model.inventory(), 'frozen_inventory': frozen.model.inventory()})
        for initial in (learned, frozen):
            initial_values = tensors(initial.model)
            complete = snapshot(run(initial))
            final = parse_json(complete, canonical=True)
            changes = [n for n in initial_values if n.endswith('.scale') and initial_values[n] != final['model_hex'][n]]
            if (initial.arm == 'frozen' and changes) or (initial.arm == 'learned' and not changes):
                raise ContractError('trainable/frozen scale update conformance failed')
            interruptions = []
            for stop in spec['replay_stops']:
                prefix = snapshot(run(create(initial.arm, seed), stop_after=stop))
                restored = resume(prefix, expected_identity=identity(prefix), arm=initial.arm, seed=seed)
                replayed = snapshot(run(restored))
                if replayed != complete:
                    raise ContractError('full model/optimizer/RNG/history replay mismatch')
                interruptions.append({'stop': stop, 'prefix_identity': identity(prefix),
                                      'final_identity': identity(replayed), 'exact_match': True})
            cells.append({'seed': seed, 'arm': initial.arm, 'changed_scales': changes,
                          'final_checkpoint': final, 'final_checkpoint_identity': identity(complete),
                          'interruptions': interruptions})
    if require_environment() != environment or sources() != source:
        raise ContractError('source/environment changed during engineering conformance')
    return {'schema': 'trinite.learned-scales-engineering-report.v1',
            'specification_commit': SPEC_COMMIT, 'protocol_identity': PROTOCOL_ID,
            'model_id': MODEL_ID, 'quantizer_id': QUANTIZER_ID,
            'source': source, 'environment': environment, 'analytic': case,
            'paired_initialization': paired, 'cells': cells,
            'scientific_execution': False, 'training_gate_evaluated': False,
            'learning_claim': False, 'release_ready': False, 'closed_custody': False,
            'native_checkpoint_replay': False,
            'scope': 'analytic surrogate and four-update synthetic decoder scale/model/AdamW/RNG/history conformance only'}
