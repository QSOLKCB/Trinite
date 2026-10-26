"""Exact finite source-specific admission; no network, model or answer feedback."""
from collections import Counter
from pathlib import Path

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from trinite.tokenizer import ByteTokenizer
from position_coupling_v1.positions import operand_width, prefix_positions
from .protocol import ARMS, MODES, STRATA, PROTOCOL

POLICY = 'trinite.position-study-data.v1'
ROOT = Path(__file__).resolve().parents[2]
SMALL = ((0,2),(0,4),(0,6),(0,8),(1,2),(1,4),(1,6),(2,3),
         (2,5),(2,7),(3,4),(3,6),(4,5),(4,7),(5,6),(6,7))
BASE = (10,11,19,22,37,48)
LARGE = tuple((a,b) for a in BASE for b in (50,55,66,79,88,99)) + tuple(
    (0,b) for b in BASE) + tuple((b,b) for b in BASE)
UNSEEN = ((0,0),(0,1),(0,9),(1,1),(1,8),(2,2),(2,8),(3,3),
    (0,98),(0,99),(1,98),(1,99),(10,89),(11,89),(12,87),(12,88),
    (22,77),(22,78),(33,66),(33,67),(44,55),(44,56),(49,50),(49,51))
LENGTH3 = ((0,999),(1,999),(2,998),(2,999),(99,900),(99,901),
    (111,222),(111,223),(444,555),(444,556),(555,555),(555,556))
LENGTH4 = tuple((a,b) for a,b in ((0,9999),(1,9999),(2,9998),(2,9999),
    (999,9000),(999,9001),(1111,2222),(1111,2223),(4444,5555),
    (4444,5556),(5555,5555),(5555,5556)))
FAMILIES = {'train': SMALL + LARGE, 'unseen': UNSEEN,
            'length3': LENGTH3, 'length4': LENGTH4}


def widths(a, b):
    d = max(len(str(a)), len(str(b)))
    return (1,2) if d == 1 else (d,)


def family(a, b):
    return POLICY + ':' + identity(json_bytes(sorted((a,b))))


def parse_prompt(prompt):
    width = operand_width(prompt)
    a,b = prompt[:-1].split('+')
    return int(a), int(b), width


def reference(a, b, width):
    # Decimal column algorithm independent of integer addition used by generator.
    left, right = str(a).zfill(width)[::-1], str(b).zfill(width)[::-1]
    carry = 0; out = []
    for x,y in zip(left,right):
        total = int(x) + int(y) + carry
        out.append(str(total % 10)); carry = total // 10
    return ''.join(out) + str(carry)


def row(a, b, width, stratum):
    prompt = f'{a:0{width}d}+{b:0{width}d}='
    answer = str(a+b).zfill(width+1)[::-1]
    if parse_prompt(prompt) != (a,b,width) or reference(a,b,width) != answer:
        raise ContractError('position study label/parser disagreement')
    carry = 0; carries = []
    for x,y in zip(str(a).zfill(width)[::-1],str(b).zfill(width)[::-1]):
        carry = int(int(x)+int(y)+carry >= 10); carries.append(carry)
    core = {'task': 'add', 'formal': {'task': 'add', 'a': a, 'b': b},
        'prompt': prompt, 'answer': answer, 'text': prompt+answer,
        'family_id': family(a,b), 'split': 'train' if stratum == 'train' else 'test',
        'stratum': stratum, 'width': width, 'source_ids': [POLICY],
        'verifier_outcome': 'verified', 'generator_revision': POLICY,
        'split_seed': 'fixed-enumerated-inventory.v1',
        'semantic_identity': identity(json_bytes(sorted((a,b)))),
        'transformation_lineage': {'operand_order': [a,b], 'padded_width': width,
                                   'carrier': 'equal-width-digits+digits='},
        'features': {'zero_operand': min(a,b) == 0,
            'repeated_digits': any(len(set(str(v))) == 1 and len(str(v)) > 1 for v in (a,b)),
            'carry_count': sum(carries), 'longest_carry_chain': max(
                (len(s) for s in ''.join(map(str,carries)).split('0')), default=0),
            'historical_0_to_12': max(a,b) <= 12}}
    ByteTokenizer(64).encode(core['text'], answer_start=len(prompt))
    return {**core, 'example_identity': identity(json_bytes(core))}


def all_rows():
    result = []
    for stratum, families in FAMILIES.items():
        for a,b in families:
            for x,y in sorted({(a,b),(b,a)}):
                for d in widths(a,b):
                    result.append(row(x,y,d,stratum))
    return result


def dataset():
    rows = all_rows()
    raw = json_bytes({'schema': POLICY, 'examples': rows})
    names = ('scripts/position_study_v1/data.py','scripts/position_study_v1/protocol.py',
             'scripts/position_coupling_v1/positions.py','src/trinite/tokenizer.py',
             'src/trinite/contracts.py')
    manifest = {'schema': 'trinite.position-study-admission.v1',
        'dataset_identity': identity(raw),
        'sources': {p: identity((ROOT/p).read_bytes()) for p in names},
        'admission': {'source_id': POLICY,
            'origin': 'local exact bounded unsigned addition; enumerated semantic families',
            'author': 'Codex deterministic generator commissioned by Trent Slade / QSOL-IMC',
            'rights_basis': 'new mathematical facts and minimal symbols only; no imported prose, paper data, teacher output or weights',
            'generation': 'bounded integer addition independently checked by decimal column arithmetic and strict prompt parser',
            'supporting_reference': 'complete formal inventory in dataset and source-bound generator',
            'scope': 'this finite corpus only; orders and padded widths grouped by unordered operands',
            'reviewer': 'automated exact regeneration, rights, labels, grammar, family and token audit',
            'reviewed_on': '2026-10-27', 'outcome': 'admitted',
            'limitations': 'not legal certification or fresh confirmation; public historical overlaps are development'},
        'counts': {s: [len(fs),sum(r['stratum']==s for r in rows)] for s,fs in FAMILIES.items()}}
    return raw, json_bytes(manifest)


def audit_bytes(raw, manifest):
    if (type(raw) is not bytes or type(manifest) is not bytes or
            len(raw)>1024*1024 or len(manifest)>65536):
        raise ContractError('bounded position-study admission bytes required')
    parse_json(raw,canonical=True); parse_json(manifest,canonical=True)
    if (raw,manifest) != dataset():
        raise ContractError('position-study corpus/admission differs from exact source')
    rows = parse_json(raw,canonical=True)['examples']
    if len({r['prompt'] for r in rows}) != len(rows):
        raise ContractError('duplicate prompts')
    groups = [{family(a,b) for a,b in fs} for fs in FAMILIES.values()]
    if any(a & b for i,a in enumerate(groups) for b in groups[i+1:]):
        raise ContractError('semantic family leakage')
    result = parse_json(manifest,canonical=True)
    if result['counts'] != PROTOCOL['inventory']:
        raise ContractError('frozen inventory counts differ')
    return result


def examples(selector):
    if selector in ARMS: selector = 'train'
    if selector not in FAMILIES: raise ContractError('unknown study stratum')
    return [r for r in all_rows() if r['stratum'] == selector]


def schedule(arm):
    if arm not in ARMS: raise ContractError('unknown study arm')
    by = {(r['formal']['a'],r['formal']['b'],r['width']):r for r in examples('train')}
    # Both orders even on diagonals: equal family exposure, unique diagnostics.
    return [by[x,y,d] for a,b in FAMILIES['train'] for x,y in ((a,b),(b,a)) for d in widths(a,b)]


def exposure(arm):
    rows = schedule(arm); tok = ByteTokenizer(64)
    encoded = [tok.encode(r['text'],answer_start=len(r['prompt'])) for r in rows]
    positions = []; roles = {}
    # Native loss slices the padded BATCH, not individual rows. Short examples
    # retain EOS as a live unscored input alongside longer rows. Count exactly
    # those lookups, never assume every row loses its own last token.
    for start in range(0,len(rows),PROTOCOL['batch_size']):
        batch = encoded[start:start+PROTOCOL['batch_size']]
        shifted_length = max(len(e.input_ids) for e in batch)-1
        for r,e in zip(rows[start:start+len(batch)],batch):
            live = min(len(e.input_ids),shifted_length)
            ps=prefix_positions(r['prompt'],live-1-len(r['prompt']),mode=MODES[arm])
            positions.append(ps)
            digits=[f'operand-significance-{d}' for d in range(r['width'],0,-1)]
            labels=['bos',*digits,'plus',*digits,'equals',
                    *(f'answer-slot-{i}' for i in range(r['width']+1)),'unscored-short-row-eos']
            for p,label in zip(ps,labels):roles.setdefault(str(p),Counter())[label]+=200
    hist = Counter(p for ps in positions for p in ps)
    return {'schema': 'trinite.position-study-exposure.v1',
        'arm': arm, 'cycle_identity': identity(json_bytes([r['example_identity'] for r in rows])),
        'cycle_visits': len(rows), 'cycles': 200, 'scheduled_visits': len(rows)*200,
        'scheduled_model_input_tokens': sum(map(len,positions))*200,
        'per_example_predictor_input_tokens': sum(len(e.input_ids)-1 for e in encoded)*200,
        'live_unscored_eos_input_tokens': sum(map(len,positions))*200-sum(len(e.input_ids)-1 for e in encoded)*200,
        'scheduled_target_tokens': sum(sum(e.loss_mask) for e in encoded)*200,
        'training_position_row_lookups': {str(k):v*200 for k,v in sorted(hist.items())},
        'training_position_row_roles':{k:dict(sorted(v.items())) for k,v in sorted(roles.items())},
        'unexposed_position_rows': sorted(set(range(64))-set(hist)),
        'row_scope': 'actual live inputs after shifting each right-padded batch, including unscored short-row EOS; lookup exposure is not gradient support or significance transfer'}


def validate_matching():
    audit_bytes(*dataset())
    if schedule('control') != schedule('candidate'):
        raise ContractError('paired data/schedule differs')
    result = {a:exposure(a) for a in ARMS}
    for e in result.values():
        for k in ('cycle_visits','cycles','scheduled_visits','scheduled_model_input_tokens','scheduled_target_tokens'):
            if e[k] != PROTOCOL[k]: raise ContractError('frozen exposure differs: '+k)
    return result
