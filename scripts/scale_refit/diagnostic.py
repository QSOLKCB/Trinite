"""Bounded deterministic approximation diagnostics, not a learning experiment."""
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from trinite.contracts import ContractError
from trinite.quantizer import quantize
from trinite.training import require_environment
from .quantizer import MAX_ELEMENTS, QUANTIZER_ID, exact_sse_numerator, refit

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'scripts/scale_refit/protocol.json'
PROTOCOL_SHA256 = '9783c0c590a9c561c0daa015531ba2c2294c46cce2186b10e67f534cada78a68'


def identity(raw):
    return 'sha256:' + hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def suite():
    raw = PROTOCOL.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ContractError('engineering matrix suite differs from published specification')
    specification = json.loads(raw)
    for case in specification['cases']:
        values = [float.fromhex(v) for v in case['values']] * case['repeat']
        if not 1 <= len(values) <= MAX_ELEMENTS:
            raise ContractError('matrix suite exceeds engineering bound')
        yield case['name'], torch.tensor([values], dtype=torch.float32, device='cpu')


def measure_suite():
    rows = []
    for name, weight in suite():
        baseline, selected = quantize(weight), refit(weight)
        candidate = selected.quantized
        n = weight.shape[1]
        inputs = torch.tensor([[(-1 if i % 2 else 1)/256 for i in range(n)],
                               [(i % 3 - 1)/256 for i in range(n)]], dtype=torch.float32)
        dense_output = F.linear(inputs, weight)
        baseline_output = F.linear(inputs, baseline.weight)
        selected_output = F.linear(inputs, candidate.weight)
        codes_equal = torch.equal(candidate.codes, baseline.codes)
        if not codes_equal or selected.selected_sse_numerator > selected.baseline_sse_numerator:
            raise ContractError('scale-refit conformance invariant failed')
        rows.append({'name': name, 'shape': list(weight.shape),
                     'codes_preserved': codes_equal,
                     'nonzero_codes': int((candidate.codes != 0).sum()),
                     'decision': selected.decision,
                     'baseline_scale_hex': baseline.scale.item().hex(),
                     'selected_scale_hex': candidate.scale.item().hex(),
                     'weight_sse_numerators': [str(selected.baseline_sse_numerator),
                                               str(selected.selected_sse_numerator)],
                     'linear_output_sse_numerators': [str(exact_sse_numerator(dense_output, baseline_output)),
                                                      str(exact_sse_numerator(dense_output, selected_output))]})
    return rows


def source_receipt():
    # Bind the complete native package, including the authentic quantizer and
    # environment guard's transitive runtime dependencies. Never reseal originals.
    paths = sorted((ROOT/'src/trinite').rglob('*.py'))
    paths += sorted((ROOT/'scripts/scale_refit').glob('*.py'))
    paths += sorted((ROOT/'tests/scale_refit').glob('*.py'))
    paths += [PROTOCOL, ROOT/'scripts/check_scale_refit.py', ROOT/'requirements.lock',
              ROOT/'src/trinite/cpu-dependencies.lock', ROOT/'scripts/check_scale_refit_cpu.py',
              ROOT/'scripts/check_cpu.py', ROOT/'scripts/check_research_cpu.py',
              ROOT/'scripts/check_research_cpu_shards.py', ROOT/'.github/workflows/scale-refit.yml']
    return {p.relative_to(ROOT).as_posix(): identity(p.read_bytes()) for p in paths}


def report():
    before = require_environment()
    sources = source_receipt()
    rows = measure_suite()
    if require_environment() != before or source_receipt() != sources:
        raise ContractError('environment or source changed during diagnostic')
    before = dict(before)
    before.pop('optimizer')  # No optimizer is run by this diagnostic.
    return {'schema': 'trinite.scale-refit-engineering-report.v1',
            'quantizer_id': QUANTIZER_ID, 'source': sources, 'environment': before,
            'protocol_identity': identity(PROTOCOL.read_bytes()), 'cases': rows,
            'sse_denominator_power_of_two': 298,
            'aggregate_weight_sse_numerators': [str(sum(int(r['weight_sse_numerators'][i]) for r in rows)) for i in (0, 1)],
            'improved_cases': sum(r['decision'] == 'improved' for r in rows),
            'maximum_weight_sse_increase_numerator': str(max(int(r['weight_sse_numerators'][1])-int(r['weight_sse_numerators'][0]) for r in rows)),
            'scientific_execution': False, 'learning_claim': False, 'speed_claim': False,
            'native_model_or_checkpoint_lane': False, 'release_ready': False}
