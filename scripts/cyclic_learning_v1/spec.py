"""Frozen bounded matched encoding comparison; no full target run before publication."""
from pathlib import Path
from trinite.contracts import ContractError, identity, json_bytes
from .data import ARMS, examples, validate, admission, validate_admission
ROOT=Path(__file__).resolve().parents[2]
MODEL_ID='trinite.cyclic-encoding-dense-64.v1'
PROTOCOL={'schema':'trinite.cyclic-learning-protocol.v1','model_id':MODEL_ID,
 'model':{'context_length':64,'blocks':1,'width':16,'heads':2,'feed_forward_width':32},
 'lane':'dense','geometry_loss':'0','seeds':[0,1],'arms':list(ARMS),
 'steps':32,'batch_size':4,'max_target_tokens':512,'cell_seconds':180,
 'initialization':'same native seeded weights per arm; no pretrained weights',
 'optimizer':'unchanged native AdamW RunConfig defaults','selection':'final step only; no tuning',
 'schedule':'seeded SHA256 rank of latent row identity; first128 visits of192 unique train rows',
 'matched':'latent cases/order, initial weights, byte-level prompt lengths, canonical3digit decimal targets plus EOS, target counts, model, updates',
 'unmatched':'radix label, colon, digit distribution and positional difficulty; encoding comparison not pure-base causal effect',
 'task':'composition in fixed Z/360; no changing modulus or discretization',
 'split':'24 relation orbits under permutations and simultaneous negation: train16/validation4/test4',
 'exposure':'public same-task family holdout; not secret, unseen modulus/template or confirmatory evidence',
 'diagnostics':'unique training rows only; repeated scheduled visits reported separately',
 'baseline':'same-seed initial identity verified and raw scored per arm only after all training; descriptive final-minus-initial counts',
 'answer_cap':4,'decoding':'raw unconstrained greedy; exact3ASCII decimal digits and EOS required; no repair or solver',
 'bounds':'four CPU cells; controller deadline900s; no GPU',
 'claims':'no improved reasoning, base superiority, ternary advantage, speedup or roadmap gate from conformance'}


def sources():
    paths=list((ROOT/'src/trinite').rglob('*.py'))
    for folder in ('scripts/cyclic_learning_v1','tests/cyclic_learning_v1'):
        paths+=list((ROOT/folder).glob('*.py'))
    paths += [ROOT/p for p in ('scripts/run_cyclic_learning.py','scripts/cyclic_conformance_v1/reference.py',
      'scripts/cyclic_conformance_v1/__init__.py','requirements.lock','src/trinite/cpu-dependencies.lock',
      '.github/workflows/cyclic-learning.yml','docs/CYCLIC-LEARNING.md')]
    return {p.relative_to(ROOT).as_posix():identity(p.read_bytes()) for p in sorted(paths)}


def schedule(seed):
    if type(seed) is not int or seed not in PROTOCOL['seeds']:
        raise ContractError('unknown seed')
    return sorted((r for r in examples() if r['split']=='train'),
                  key=lambda r:identity(json_bytes({'seed':seed,'row':r['identity']})))


def targets(seed):
    schedule(seed)
    return PROTOCOL['steps']*PROTOCOL['batch_size']*4


def request(environment):
    rows=examples(); validate(rows); validate_admission(admission())
    ordered={str(seed):schedule(seed) for seed in PROTOCOL['seeds']}
    return {'schema':'trinite.cyclic-learning-request.v1','protocol':PROTOCOL,
      'protocol_identity':identity(json_bytes(PROTOCOL)),'source':sources(),'environment':environment,
      'admission':admission(),'items':rows,'dataset_identity':identity(json_bytes(rows)),
      'counts':{s:{'rows':sum(r['split']==s for r in rows),'families':len({r['family'] for r in rows if r['split']==s})}
                for s in ('train','validation','test')},
      'schedules':{str(seed):{'ordered_unique_train_ids':[r['identity'] for r in ordered[str(seed)]],
          'scheduled_visits':[ordered[str(seed)][i%len(ordered[str(seed)])]['identity'] for i in range(PROTOCOL['steps']*PROTOCOL['batch_size'])],
          'scheduled_target_tokens':targets(seed)} for seed in PROTOCOL['seeds']}}
