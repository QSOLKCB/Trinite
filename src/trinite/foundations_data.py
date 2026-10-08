"""New numeric/minimal-symbol admissions inspired by pinned planning references."""
from itertools import product
from pathlib import Path

from .contracts import ContractError, identity, json_bytes, parse_json
from .tokenizer import ByteTokenizer
from .foundations_oracle import WORKLOADS, solve, parse_prompt

POLICY = 'trinite.foundations-data.v1'


def source_receipt():
    root = Path(__file__).resolve().parent
    return {n:identity((root/n).read_bytes()) for n in ('foundations_data.py',
        'foundations_oracle.py','contracts.py','tokenizer.py','foundations-protocol.json')}


def _arithmetic(task,a,b):
    def multiply(x,y):
        value=0
        for _ in range(abs(y)): value+=x
        return -value if y<0 else value
    if task=='add': return str(a+b)
    if task=='subtract': return str(a+(-b))
    if task=='multiply': return str(multiply(a,b))
    if b==0: return 'ERR'
    left,right=abs(a),abs(b)
    while right: left,right=right,left%right
    n,d=a//left,b//left
    if d<0:n,d=-n,-d
    return str(n) if d==1 else f'{n}/{d}'


def items(kind):
    if kind not in WORKLOADS: raise ContractError('unknown foundations workload')
    result=[]
    if kind=='arithmetic':
        for a,b,task in product(range(-3,4),range(-3,4),('add','subtract','multiply','divide')):
            spec={'task':task,'a':a,'b':b}
            family=sorted((abs(a),abs(b)))
            result.append((spec,_arithmetic(task,a,b),family))
    elif kind=='fold':
        for length in range(2,5):
            for values in product(range(-2,3),repeat=length):
                count,total,squares=0,0,0
                for value in values:
                    count+=1;total+=value
                    for _ in range(abs(value)): squares+=abs(value)
                spec={'task':'fold','values':list(values)}
                result.append((spec,f'{count} {total} {squares}',sorted(abs(v) for v in values if v)))
    else:
        traversal=[];value=0
        for _ in range(27):
            traversal.append(value);value+=17
            while value>=27:value-=27
        roles=(('Q','R','E'),('O','D','U'),('C','H','X'))
        for coordinates in product(range(3),repeat=3):
            coords=list(coordinates);text='L['+','.join(map(str,coords))+']'
            role=''.join(roles[i][v] for i,v in enumerate(coords))
            for task,answer in (('role-to-address',text),('address-to-role',role),('valid-address','1')):
                result.append(({'task':task,'coordinates':coords},answer,coords))
            corrupt=coords.copy();corrupt[sum(coords)%3]=3
            result.append(({'task':'valid-address','coordinates':corrupt},'0',coords))
            index=traversal.index(9*coords[0]+3*coords[1]+coords[2])
            result.append(({'task':'traversal','index':index},text,coords))
    checked=[]
    for spec,answer,family in result:
        if solve(kind,spec)!=answer: raise ContractError('foundations label failed independent oracle')
        checked.append({'formal':spec,'answer':answer,'family_id':kind+':'+identity(json_bytes(family)),
                        'semantic_identity':identity(json_bytes({'workload':kind,'formal':spec}))})
    return sorted(checked,key=lambda x:x['semantic_identity'])


def render(kind,spec,carrier):
    task=spec['task']
    if kind=='arithmetic':
        tag={'add':'A','subtract':'S','multiply':'M','divide':'D'}[task]
        body=f"{spec['a']},{spec['b']}"
    elif kind=='fold':tag,body='F',','.join(map(str,spec['values']))
    elif task=='traversal':tag,body='T',str(spec['index'])
    else:
        coords=spec['coordinates']
        if task=='role-to-address':
            tag='R';body=''.join(('QRE','ODU','CHX')[i][v] for i,v in enumerate(coords))
        else:tag='V' if task=='valid-address' else 'A';body='L['+','.join(map(str,coords))+']'
    if carrier=='infix':return tag+':'+body+'='
    if carrier=='fields':return tag+'('+body+'):'
    raise ContractError('unknown foundations carrier')


def admission(kind,formal_items,dataset_identity):
    evidence={'formal_items':formal_items,'dataset_identity':dataset_identity,
              'symbols':'bounded decimal integers, reduced fractions, count/sum/squares, L[x,y,z], QRE/ODU/CHX formal role codes, ERR and task delimiters'}
    return {'source_id':POLICY+':'+kind,
        'origin':'local bounded enumeration of mathematical facts and newly rendered minimal symbols',
        'author':'Codex deterministic generator commissioned by Trent Slade / QSOL-IMC',
        'generation':'items enumerates the full declared finite domain; procedural arithmetic/fold/traversal labels; foundations_oracle.solve independently checks labels; parse_prompt independently checks rendered semantics; family hash rank assigns splits',
        'rights_basis':'generated numeric/formal facts with minimal symbolic carriers; no donor code, prose, teacher output or external content imported',
        'supporting_reference':{'implementation':source_receipt(),'functions':['items','render','foundations_oracle.solve','foundations_oracle.parse_prompt'],
                                'evidence_identity':identity(json_bytes(evidence))},
        'evidence':evidence,'scope':'this exact bounded generated foundations dataset only',
        'reviewer':'automated source/rights/oracle/split audit under Trent Slade commissioned protocol',
        'reviewed_on':'2026-10-09','outcome':'admitted',
        'limitations':'software-generated facts; not human legal certification or blanket admission of synthetic data, repository contents, symbolic prose or fresh capability evidence'}


def validate_admission(record,kind):
    if kind not in WORKLOADS:
        raise ContractError('unknown foundations admission')
    expected=parse_json(dataset(kind)[1],canonical=True)['admission']
    if type(record) is not dict or record!=expected:
        raise ContractError('foundations admission contradicts the frozen source policy/evidence')


def dataset(kind):
    formal_items=items(kind)
    families=sorted({item['family_id'] for item in formal_items},key=lambda family:identity(json_bytes(
        {'policy':POLICY,'workload':kind,'seed':31,'family':family})))
    train_count=len(families)*8//10;validation_count=max(1,len(families)//10)
    splits={f:'train' if i<train_count else 'validation' if i<train_count+validation_count else 'test'
            for i,f in enumerate(families)}
    examples=[]
    for item,carrier in product(formal_items,('infix','fields')):
        prompt=render(kind,item['formal'],carrier)
        if parse_prompt(kind,prompt)!=item['formal']: raise ContractError('rendered foundations semantics differ')
        core={**item,'source_ids':[POLICY+':'+kind],'task':item['formal']['task'],
              'split':splits[item['family_id']],'carrier':carrier,'prompt':prompt,
              'text':prompt+item['answer'],'verifier_outcome':'verified',
              'transformation_lineage':{'formal_identity':item['semantic_identity'],'carrier':carrier}}
        ByteTokenizer(64).encode(core['text'],answer_start=len(prompt.encode()))
        examples.append({**core,'example_identity':identity(json_bytes(core))})
    if len({e['text'] for e in examples})!=len(examples):raise ContractError('duplicate foundations text')
    raw=json_bytes({'schema':POLICY,'workload':kind,'examples':sorted(examples,key=lambda e:e['example_identity'])})
    record=admission(kind,formal_items,identity(raw))
    manifest={'schema':'trinite.foundations-admission.v1','workload':kind,'dataset_identity':identity(raw),
        'admission':record,'family_splits':splits,
        'counts':{s:sum(e['split']==s for e in examples) for s in ('train','validation','test')}}
    return raw,json_bytes(manifest)


def audit_bytes(raw,manifest):
    if type(raw) is not bytes or type(manifest) is not bytes or len(raw)>2*1024*1024 or len(manifest)>1024*1024:
        raise ContractError('foundations admission requires bounded bytes')
    data=parse_json(raw,canonical=True)
    kind=data.get('workload') if type(data) is dict else None
    if kind not in WORKLOADS or (raw,manifest)!=dataset(kind):
        raise ContractError('foundations dataset/admission differs from independently verified source')
    return parse_json(manifest,canonical=True)
