"""Admitted exact generated facts; representation pairs never cross family splits."""
from itertools import permutations
from pathlib import Path
from trinite.contracts import ContractError, identity, json_bytes
from trinite.tokenizer import ByteTokenizer
from cyclic_conformance_v1.reference import CyclicElement

ARMS = ('decimal', 'sexagesimal')
SOURCE_ID = 'trinite.cyclic-composition-facts.v1'
REVISION = 'cyclic-learning-generator.v1'
ANCHORS = tuple((a,b) for a in (7,31,59,61,119,179,239,301) for b in (11,47,83))
SPLIT_PROCEDURE = 'canonical zero-sum triples modulo360 under permutations and simultaneous negation; SHA256 rank; first16 train,next4 validation,last4 test'


def orbit(a,b):
    triple=(a,b,(-a-b)%360)
    return sorted(set(permutations(triple)) | set(permutations(tuple((-x)%360 for x in triple))))


def counter(a,b):
    """Independent bounded unit-rotation oracle, without addition modulo."""
    result=a
    for _ in range(b):
        result=0 if result==359 else result+1
    return result


def render(k,arm):
    if arm=='decimal':
        return f'{k:05d}'
    if arm=='sexagesimal':
        return f'{k//60:02d}:{k%60:02d}'
    raise ContractError('unknown encoding')


def prompt(a,b,arm):
    return ('r10' if arm=='decimal' else 'r60')+':rot360['+render(a,arm)+','+render(b,arm)+']='


def parse_prompt(text):
    import re
    match=re.fullmatch(r'r(10|60):rot360\[([0-9:]{5}),([0-9:]{5})\]=',text)
    if match is None:
        raise ContractError('noncanonical prompt')
    radix,left,right=match.groups()
    def decode(raw):
        if radix=='10' and re.fullmatch(r'[0-9]{5}',raw):
            value=int(raw)
        elif radix=='60' and re.fullmatch(r'[0-9]{2}:[0-9]{2}',raw) and int(raw[3:])<60:
            value=60*int(raw[:2])+int(raw[3:])
        else:
            raise ContractError('invalid digit spelling')
        if not 0<=value<360:
            raise ContractError('tick out of range')
        return value
    return decode(left),decode(right)


def generator_sources():
    root=Path(__file__).resolve().parents[2]
    return {name:identity((root/name).read_bytes()) for name in
            ('scripts/cyclic_learning_v1/data.py','scripts/cyclic_conformance_v1/reference.py')}


def examples():
    revision={'id':REVISION,'source':generator_sources()}
    families={}
    for a,b in ANCHORS:
        key=orbit(a,b)[0]
        family=identity(json_bytes(list(key)))
        if family in families:
            raise ContractError('candidate family orbit collision')
        families[family]=orbit(a,b)
    rows=[]
    for rank,family in enumerate(sorted(families)):
        split='train' if rank<16 else 'validation' if rank<20 else 'test'
        for a,b,negative_c in families[family]:
            c=(-negative_c)%360
            if counter(a,b)!=c or CyclicElement(a,360).add(CyclicElement(b,360)).k!=c:
                raise ContractError('independent exact oracles disagree')
            row={'source_id':SOURCE_ID,'generator_revision':revision,
                 'seed':'none; exhaustive fixed anchor orbits','split_procedure':SPLIT_PROCEDURE,
                 'formal_spec':'a,b in Z/360; answer=(a+b) mod360; zero-sum relation orbit',
                 'family':family,'family_rank':rank,'split':split,'a':a,'b':b,
                 'answer':f'{c:03d}','prompts':{arm:prompt(a,b,arm) for arm in ARMS},
                 'verifier':{'outcome':'pass','methods':['bounded repeated unit rotations','PR58 exact CyclicElement.add']},
                 'lineage':{'anchor_orbit':list(families[family][0]),'origin':'new exact generator; no teacher or model output'},
                 'rights_basis':'mathematical facts and minimal formal serialization; generator code under repository license'}
            row['identity']=identity(json_bytes(row))
            rows.append(row)
    return rows


def validate(rows):
    if rows!=examples():
        raise ContractError('item provenance/content differs from exact regeneration')
    seen=set()
    for row in rows:
        if row['identity'] in seen:
            raise ContractError('duplicate row')
        seen.add(row['identity'])
        lengths=[]
        for arm in ARMS:
            text=row['prompts'][arm]
            if parse_prompt(text)!=(row['a'],row['b']):
                raise ContractError('render/parser disagreement')
            encoded=ByteTokenizer(64).encode(text+row['answer'],answer_start=len(text))
            lengths.append((len(text.encode('ascii')),sum(encoded.loss_mask)))
        if lengths[0]!=lengths[1] or lengths[0][1]!=4:
            raise ContractError('paired prompt/target exposure mismatch')


def admission():
    return {'schema':'trinite.cyclic-learning-admission.v1','source_id':SOURCE_ID,
            'origin':'repository-owned deterministic generator of finite mathematical facts',
            'author':'QSOLKCB/Trinite contributors','generator_revision':REVISION,
            'content_identity':identity(json_bytes(examples())),
            'source_identities':generator_sources(),
            'reviewer':'repository assistant prospective policy review; human PR review pending',
            'source_support':'docs/CYCLIC-LEARNING.md; scripts/cyclic_learning_v1/data.py',
            'rights_basis':'exact mathematical facts and minimal formal templates, not AI-generated answers; software license separate',
            'scope':'only this fixed anchor orbit suite and its two canonical representations',
            'outcome':'admitted for bounded exploratory comparison under docs/DATA.md numeric-fact rule',
            'review':'prospective source-specific admission in this pull request, not blanket generated-text approval',
            'date':'2026-10-21','split_procedure':SPLIT_PROCEDURE,
            'candidate_families':24,'accepted_families':24,'excluded_families':0,
            'rejection_policy':'fail closed on orbit collision, incorrect oracle/parser, duplicates or overlength'}


def validate_admission(record):
    if record!=admission():
        raise ContractError('source admission differs from exact generated content/provenance')
