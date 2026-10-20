"""Versioned prospective protocol, full native source closure, and fixed schedule."""
from pathlib import Path
from trinite.contracts import ContractError, identity, json_bytes
from .data import admission, examples, validate, ARMS
ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = 'trinite.integral-geometry-dense-128.v1'
PROTOCOL = {
    'schema':'trinite.integral-geometry-protocol.v1', 'model_id':MODEL_ID,
    'model':{'context_length':128,'blocks':1,'width':16,'heads':2,'feed_forward_width':32},
    'lane':'dense','geometry_loss':'0','seeds':[0,1], 'arms':list(ARMS),
    'steps':32,'batch_size':4,'max_target_tokens':4096,'cell_seconds':180,
    'initialization':'paired native seed; no pretrained model',
    'optimizer':'unchanged native AdamW default RunConfig values',
    'schedule':'seeded family-row identity rank; setup/evaluation interleaved; fixed 128 visits',
    'selection':'final step only; no early stopping or tuning',
    'control':'symbolic integral operand serialization plus same exact integral evaluation rows',
    'treatment':'geometry-to-integral setup plus same exact integral evaluation rows',
    'matched':'initial weights, schedule row IDs, target strings/tokens, steps, optimizer, model',
    'unmatched':'prompt tokens/length and task difficulty; no equal-FLOP or walltime claim',
    'setup_cap':97,'answer_cap':16,
    'scores':['setup_semantic','setup_canonical','oracle_setup_evaluation','end_to_end'],
    'decoding':'unconstrained greedy; EOS required; no solver, repair, gold substitution or retry',
    'splits':'normalized coefficient families before affine transforms; train 6/validation 1/test 2',
    'exposure':'public exploratory suite; holdout excludes optimizer use but is not secret or confirmatory',
    'bounds':'four cells; CPU only; external whole-run deadline 900 seconds',
    'claims':'no learned reasoning, benefit, speedup or Phase 5 gate inferred from oracle conformance',
}


def sources():
    paths = list((ROOT/'src/trinite').rglob('*.py'))
    for folder in ('scripts/integral_geometry_v1','tests/integral_geometry_v1'):
        paths += list((ROOT/folder).glob('*.py'))
    paths += [ROOT/'scripts/run_integral_geometry.py', ROOT/'requirements.lock',
              ROOT/'src/trinite/cpu-dependencies.lock', ROOT/'.github/workflows/integral-geometry.yml',
              ROOT/'docs/INTEGRAL-GEOMETRY.md']
    return {p.relative_to(ROOT).as_posix():identity(p.read_bytes()) for p in sorted(paths)}


def schedule(seed):
    rows = [r for r in examples() if r['split']=='train']
    rows.sort(key=lambda r:identity(json_bytes({'seed':seed,'row':r['identity']})))
    return [(r,kind) for r in rows for kind in ('setup','evaluation')]


def targets(seed):
    rows = schedule(seed)
    return sum(len(rows[i%len(rows)][0]['setup' if rows[i%len(rows)][1]=='setup' else 'answer'])+1
               for i in range(PROTOCOL['steps']*PROTOCOL['batch_size']))


def request(environment):
    rows = examples()
    validate(rows)
    counts = {str(seed):targets(seed) for seed in PROTOCOL['seeds']}
    if any(n > PROTOCOL['max_target_tokens'] for n in counts.values()):
        raise ContractError('scheduled targets exceed frozen budget')
    return {'schema':'trinite.integral-geometry-request.v1','protocol':PROTOCOL,
            'protocol_identity':identity(json_bytes(PROTOCOL)), 'source':sources(),
            'environment':environment,'admission':admission(),
            'dataset_identity':identity(json_bytes(rows)),'scheduled_target_tokens':counts}
