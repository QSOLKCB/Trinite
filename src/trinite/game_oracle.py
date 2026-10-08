"""Independent bounded two-player/two-action payoff and equilibrium oracle."""
import re

from .contracts import ContractError, exact_keys, integer

TASKS = ('payoff','row-best','column-best','strict-dominant','pure-nash')


def checked(spec):
    if type(spec) is not dict or spec.get('task') not in TASKS:
        raise ContractError('unknown game task')
    task=spec['task']
    selectors={'payoff':{'row','column'},'row-best':{'column'},'column-best':{'row'},
               'strict-dominant':{'player'},'pure-nash':set()}[task]
    exact_keys(spec,{'task','payoffs'}|selectors,'game specification')
    if type(spec['payoffs']) is not list or len(spec['payoffs'])!=8:
        raise ContractError('game requires four ordered binary payoff pairs')
    for value in spec['payoffs']: integer(value,'payoff',0,1)
    for key in selectors: integer(spec[key],key,0,1)
    return spec


def solve(spec):
    checked(spec);task=spec['task'];v=spec['payoffs']
    def u(row,column,player): return v[2*(2*row+column)+player]
    if task=='payoff': return f"{u(spec['row'],spec['column'],0)},{u(spec['row'],spec['column'],1)}"
    if task in ('row-best','column-best'):
        player=0 if task=='row-best' else 1
        other=spec['column'] if player==0 else spec['row']
        scores=[u(action,other,0) if player==0 else u(other,action,1) for action in (0,1)]
        return ','.join(str(action) for action,value in enumerate(scores) if value==max(scores))
    if task=='strict-dominant':
        player=spec['player'];winners=[]
        for action in (0,1):
            if all((u(action,other,0)>u(1-action,other,0) if player==0 else
                    u(other,action,1)>u(other,1-action,1)) for other in (0,1)):
                winners.append(str(action))
        return ','.join(winners) or '-'
    equilibria=[]
    for row in (0,1):
        for column in (0,1):
            row_regret=max(u(r,column,0)-u(row,column,0) for r in (0,1))
            column_regret=max(u(row,c,1)-u(row,column,1) for c in (0,1))
            if row_regret==0 and column_regret==0: equilibria.append(f'{row}{column}')
    return ','.join(equilibria) or '-'


def parse_prompt(prompt):
    if type(prompt) is not str or len(prompt)>32:
        raise ContractError('game prompt exceeds bounds')
    match=re.fullmatch(r'([PRCDN]):([01]{8})(?::([01](?:,[01])?))?=',prompt)
    if not match: raise ContractError('invalid game prompt')
    tag,digits,selector=match.groups()
    task={'P':'payoff','R':'row-best','C':'column-best','D':'strict-dominant','N':'pure-nash'}[tag]
    spec={'task':task,'payoffs':list(map(int,digits))}
    values=[] if selector is None else list(map(int,selector.split(',')))
    keys={'P':('row','column'),'R':('column',),'C':('row',),'D':('player',),'N':()}[tag]
    if len(keys)!=len(values): raise ContractError('game prompt selector mismatch')
    spec.update(zip(keys,values));return checked(spec)
