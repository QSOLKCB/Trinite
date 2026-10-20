"""Commissioned symbolic facts, family isolation and independent quadrature audit."""
from fractions import Fraction as Q
from pathlib import Path
from trinite.contracts import ContractError, identity, json_bytes
from trinite.tokenizer import ByteTokenizer
from .math import integral, render, evaluate_setup

SOURCE_ID = 'trinite.integral-geometry-formal.v1'
TASKS = ('area', 'volume', 'centroid_x')
ARMS = ('representation_control', 'geometry')


def split(b,c):
    return ('validation' if (b,c) == (1,2) else
            'test' if (b,c) in ((2,1),(2,2)) else 'train')


def polynomial(b,c,h,length,scale):
    # Expand scale*(1+b*u+c*u^2), u=(x-h)/length, into global x.
    return (scale*(1-b*h/length+c*h*h/length**2),
            scale*(b/length-2*c*h/length**2), scale*c/length**2)


def independent_value(task,b,c,h,length,scale):
    # Independent exact Simpson quadrature in normalized coordinates. It is exact
    # for this degree<=2 density and degree<=3 first moment, without antiderivatives.
    def f(u):
        return scale*(1+b*u+c*u*u)
    mass = length*(f(Q(0))+4*f(Q(1,2))+f(Q(1)))/6
    if mass <= 0:
        raise ContractError('nonpositive mass')
    if task != 'centroid_x':
        return mass
    moment = length*(h*f(Q(0))+4*(h+length/2)*f(Q(1,2))+(h+length)*f(Q(1)))/6
    return moment/mass


def examples():
    rows = []
    for b in range(3):
        for c in range(3):
            family = f'normalized-quadratic:{b}:{c}'
            partition = split(b,c)
            transforms = [(Q(0),Q(1),Q(1)), (Q(1),Q(1),Q(2)), (Q(0),Q(2),Q(1))]
            if partition != 'train':
                transforms.append((Q(-1,2),Q(3,2),Q(1,2)))
            for h,length,scale in transforms:
                a,z = h,h+length
                p = polynomial(b,c,h,length,scale)
                for task in TASKS:
                    numerator = (Q(0),*p) if task == 'centroid_x' else p
                    denominator = p if task == 'centroid_x' else None
                    setup = render(a,z,numerator,denominator)
                    value = integral(numerator,a,z)
                    if denominator is not None:
                        value /= integral(denominator,a,z)
                    if value != independent_value(task,b,c,h,length,scale) or evaluate_setup(setup) != value:
                        raise ContractError('independent exact oracle disagreement')
                    # Formal task names specify area under nonnegative height,
                    # volume from supplied cross-sectional area, or x-centroid.
                    prompt = task+'['+str(a)+','+str(z)+';'+','.join(map(str,p))+']='
                    control = ('serialize['+str(a)+';'+str(z)+';'+','.join(map(str,numerator))
                               + ('|'+','.join(map(str,denominator)) if denominator is not None else '')+']=')
                    row = {'source_id':SOURCE_ID,'family':family,'split':partition,'task':task,
                           'normalized':[str(Q(1)),str(Q(b)),str(Q(c))],
                           'transform':{'shift':str(h),'length':str(length),'scale':str(scale)},
                           'prompt':prompt,'control_prompt':control,'setup':setup,'answer':str(value),
                           'novel_transform': length == Q(3,2)}
                    row['ordering_identity'] = identity(json_bytes(row))
                    row['generator_revision'] = {name:identity(Path(__file__).with_name(name).read_bytes())
                                                 for name in ('data.py','math.py')}
                    row['generation_seed'] = None
                    row['generation_procedure'] = 'exhaustive coefficient and transform enumeration; no RNG'
                    row['split_seed'] = None
                    row['split_procedure'] = {
                        'id':'trinite.integral-geometry-coefficient-split.v1',
                        'family_key':'normalized (b,c) before all transforms and task projections',
                        'validation':[[1,2]],'test':[[2,1],[2,2]],
                        'train':'all other enumerated coefficient pairs; no random assignment'}
                    row['verifier_outcome'] = {
                        'antiderivative_vs_normalized_simpson':'pass',
                        'parsed_setup_vs_exact_answer':'pass'}
                    row['identity'] = identity(json_bytes(row))
                    rows.append(row)
    return rows


def validate(rows):
    if rows != examples():
        raise ContractError('formal source differs from complete regenerated admission')
    seen, families = {}, {}
    for r in rows:
        families.setdefault(r['family'], set()).add(r['split'])
        # Task-normalized setup identity groups area/volume duplicates too.
        key = r['setup']
        if key in seen and seen[key] != r['split']:
            raise ContractError('mathematical instance crosses split')
        seen[key] = r['split']
        for prompt,answer in ((r['prompt'],r['setup']), (r['control_prompt'],r['setup']),
                              ('eval['+r['setup']+']=',r['answer'])):
            ByteTokenizer(128).encode(prompt+answer, answer_start=len(prompt))
    if any(len(s) != 1 for s in families.values()):
        raise ContractError('transformation family crosses split')
    return True


def admission():
    rows = examples()
    validate(rows)
    return {'schema':'trinite.integral-geometry-admission.v1','source_id':SOURCE_ID,
            'origin':'deterministic first-party generator commissioned by Trent Slade / QSOL-IMC',
            'rights_basis':'bounded generated numerical facts and minimal formal symbols; no external corpus or teacher output',
            'software_license':'MPL-2.0; software licensing is not the data rights basis',
            'procedure':'global polynomial expansion and antiderivatives; independent normalized Simpson exact check; canonical family split',
            'review':'automated source-specific regeneration audit; not legal certification',
            'date':'2026-10-20','scope':'only complete generated examples identified here',
            'limitations':'public exploratory formal suite, not unseen confirmatory data or natural-language geometry',
            'dataset_identity':identity(json_bytes(rows)), 'examples':len(rows),
            'supporting_sources':{name:identity(Path(__file__).with_name(name).read_bytes())
                                  for name in ('data.py','math.py')},
            'families':{s:sorted({r['family'] for r in rows if r['split']==s}) for s in ('train','validation','test')}}
