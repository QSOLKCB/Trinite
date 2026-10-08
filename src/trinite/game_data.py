"""Locally generated numeric game foundations; no training integration or imports."""
from itertools import product
from pathlib import Path

from .contracts import ContractError, identity, json_bytes, parse_json
from .game_oracle import solve, parse_prompt
from .tokenizer import ByteTokenizer

POLICY='trinite.game-data.v1'


def source_receipt():
    root=Path(__file__).resolve().parent
    return {n:identity((root/n).read_bytes()) for n in ('game_data.py','game_oracle.py','contracts.py','tokenizer.py')}


def orbit(payoffs):
    """Group row/column relabellings and player exchange (with transpose)."""
    variants=[]
    for row_swap,column_swap,exchange in product((0,1),repeat=3):
        variant=[]
        for row,column in product((0,1),repeat=2):
            r,c=row^row_swap,column^column_swap
            if exchange:r,c=c,r
            pair=payoffs[2*(2*r+c):2*(2*r+c)+2]
            variant.extend(reversed(pair) if exchange else pair)
        variants.append(tuple(variant))
    return min(variants)


def label(spec):
    """Procedural finite unilateral-deviation enumeration, independent of oracle."""
    values=spec['payoffs'];task=spec['task']
    pairs=[[values[:2],values[2:4]],[values[4:6],values[6:]]]
    if task=='payoff': return ','.join(map(str,pairs[spec['row']][spec['column']]))
    if task=='row-best':
        diff=pairs[1][spec['column']][0]-pairs[0][spec['column']][0]
        return '0' if diff<0 else '1' if diff>0 else '0,1'
    if task=='column-best':
        diff=pairs[spec['row']][1][1]-pairs[spec['row']][0][1]
        return '0' if diff<0 else '1' if diff>0 else '0,1'
    if task=='strict-dominant':
        player=spec['player']
        comparisons=[pairs[1][other][0]-pairs[0][other][0] if player==0 else
                     pairs[other][1][1]-pairs[other][0][1] for other in (0,1)]
        return '1' if all(v>0 for v in comparisons) else '0' if all(v<0 for v in comparisons) else '-'
    valid=[]
    for row,column in product((0,1),repeat=2):
        deviations=[(1-row,column,0),(row,1-column,1)]
        improved=False
        for new_row,new_column,player in deviations:
            if pairs[new_row][new_column][player]>pairs[row][column][player]:improved=True
        if not improved:valid.append(str(row)+str(column))
    return ','.join(valid) or '-'


def specs(values):
    base={'payoffs':list(values)}
    for row,column in product((0,1),repeat=2):yield {**base,'task':'payoff','row':row,'column':column}
    for other in (0,1):
        yield {**base,'task':'row-best','column':other}
        yield {**base,'task':'column-best','row':other}
        yield {**base,'task':'strict-dominant','player':other}
    yield {**base,'task':'pure-nash'}


def render(spec):
    tag={'payoff':'P','row-best':'R','column-best':'C','strict-dominant':'D','pure-nash':'N'}[spec['task']]
    keys={'P':('row','column'),'R':('column',),'C':('row',),'D':('player',),'N':()}[tag]
    return tag+':'+''.join(map(str,spec['payoffs']))+(':'+','.join(str(spec[k]) for k in keys) if keys else '')+'='


def dataset():
    matrices=list(product((0,1),repeat=8))
    families={values:'game:'+identity(json_bytes(list(orbit(values)))) for values in matrices}
    ranked=sorted(set(families.values()),key=lambda f:identity(json_bytes({'policy':POLICY,'seed':41,'family':f})))
    train_count=len(ranked)*8//10;validation_count=max(1,len(ranked)//10)
    splits={f:'train' if i<train_count else 'validation' if i<train_count+validation_count else 'test'
            for i,f in enumerate(ranked)}
    rows=[]
    for values in matrices:
        for spec in specs(values):
            answer=label(spec);prompt=render(spec)
            if solve(spec)!=answer or parse_prompt(prompt)!=spec:raise ContractError('game independent oracle mismatch')
            core={'formal':spec,'family_id':families[values],'split':splits[families[values]],
                  'task':spec['task'],'prompt':prompt,'answer':answer,'text':prompt+answer,
                  'source_id':POLICY,'verifier_outcome':'verified'}
            ByteTokenizer(64).encode(core['text'],answer_start=len(prompt.encode()))
            rows.append({**core,'example_identity':identity(json_bytes(core))})
    if len({r['text'] for r in rows})!=len(rows):raise ContractError('duplicate game examples')
    raw=json_bytes({'schema':POLICY,'examples':sorted(rows,key=lambda r:r['example_identity'])})
    evidence={'domain':'all 256 ordered two-player/two-action binary payoff matrices',
              'dataset_identity':identity(raw),'formal_items':len(rows),'families':len(ranked),
              'conventions':'row player first in each ordered pair; actions 0/1; ties retain every best response; dominance is strict against both opposing actions; Nash permits ties; - means empty solution set'}
    admission={'source_id':POLICY,'origin':'local complete finite enumeration of numeric game payoff facts',
        'author':'Codex deterministic generator commissioned by Trent Slade / QSOL-IMC',
        'generation':'dataset enumerates matrices and five task types; label uses unilateral deviations; game_oracle.solve checks labels using payoff maximization and regret; parse_prompt checks semantics; orbit groups all action relabellings and player exchange before family hash splitting',
        'rights_basis':'locally generated numeric facts and minimal symbols; no third-party prose, dialogue, teacher/model output or repository content imported',
        'supporting_reference':{'implementation':source_receipt(),'functions':['dataset','label','orbit','game_oracle.solve','game_oracle.parse_prompt'],
            'evidence_identity':identity(json_bytes(evidence))},'evidence':evidence,
        'scope':'this exact bounded binary payoff corpus only',
        'reviewer':'automated source/rights/oracle/split audit under Trent Slade commissioned generation',
        'reviewed_on':'2026-10-09','outcome':'admitted',
        'limitations':'software-generated numeric facts; not legal certification, a trained model, strategic advantage, fresh evaluation or a claim about human incentives'}
    manifest={'schema':'trinite.game-admission.v1','dataset_identity':identity(raw),'admission':admission,
              'family_splits':splits,'counts':{s:sum(r['split']==s for r in rows) for s in ('train','validation','test')},
              'training_integrated':False,'fresh_evaluation':False}
    return raw,json_bytes(manifest)


def audit_bytes(raw,manifest):
    if type(raw) is not bytes or type(manifest) is not bytes or len(raw)>4*1024*1024 or len(manifest)>128*1024:
        raise ContractError('game admission requires bounded bytes')
    parse_json(raw,canonical=True);parse_json(manifest,canonical=True)
    if (raw,manifest)!=dataset():raise ContractError('game dataset/admission differs from independently checked source')
    return parse_json(manifest,canonical=True)


def validate_admission(record):
    expected=parse_json(dataset()[1],canonical=True)['admission']
    if type(record) is not dict or json_bytes(record)!=json_bytes(expected):
        raise ContractError('game admission contradicts scoped policy/evidence')
