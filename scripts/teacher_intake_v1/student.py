"""Explicit teacher-origin CPU experiment using the unchanged native updater."""
from trinite.contracts import ContractError, ModelConfig, identity, json_bytes
from trinite.data import build_dataset, audit_bytes
from trinite.tokenizer import ByteTokenizer, EOS
from .intake import verify, counter
from .spec import source_map

MODEL={'context_length':64,'blocks':1,'width':16,'heads':2,'feed_forward_width':32}
MODEL_ID='trinite.verified-teacher-dense-64.v1'
PROTOCOL={'schema':'trinite.teacher-student-protocol.v1','model_id':MODEL_ID,'model':MODEL,
          'arms':['teacher_selected','oracle_only'],'seeds':[0,1],'steps':256,'batch_size':4,
          'target_tokens':3072,'geometry_loss':'0','lane':'dense',
          'diagnostic':{'seed':0,'steps':256,'rows':4,'distinct_answers':4,'targets':2048,
                        'gate':'all4 raw complete answers with EOS exactly correct; train-only'},
          'selection':'final step only; no tuning or early stopping',
          'control':'independent hash-ranked commutative families per modulus, same counts; overlap allowed and reported',
          'heldout':'all unordered operand pairs of modulus11 validation and13 test; public modulus extrapolation',
          'limits':'five CPU cells at most600s each; controller3600s; no GPU',
          'replay':'full diagnostic and updates plus all tensor/history/scoring identities; timings separate',
          'claims':'exploratory input-selection comparison, not teacher reasoning transfer or ternary benefit'}


def formal(n,a,b,split,origin,revision):
    answer=counter(n,a,b)
    if answer!=(a+b)%n:
        raise ContractError('formal oracle mismatch')
    row={'formal_spec':{'n':n,'a':a,'b':b},'family':[n,min(a,b),max(a,b)],
         'prompt':f'addmod[{n:02d},{a:02d},{b:02d}]=','answer':f'{answer:02d}',
         'split':split,'origin':origin,'generator_revision':revision,'seed':0,
         'split_procedure':'whole modulus5/7/9 train;11 validation;13 test',
         'verifier':'cyclic counter independently checked by modulo',
         'rights_basis':'commissioned deterministic numeric generator; no third-party prose',
         'source_id':'trinite.teacher-control-formal.v1','lineage':{'parent':'formal modular addition'},
         'reviewer':'Trent Slade / QSOLKCB','review_date':'2026-10-21'}
    row['identity']=identity(json_bytes(row))
    return row


def pools(packet):
    verify(packet)
    if packet['admission']!='ADMITTED_FORMAL_ONLY':
        raise ContractError('underfilled intake cannot enter student execution')
    teacher=packet['items'];control=[]
    revision=identity(json_bytes(source_map()))
    counts={n:sum(r['formal_spec']['n']==n for r in teacher) for n in (5,7,9)}
    for n in (5,7,9):
        candidates=[formal(n,a,b,'train','oracle_only',revision) for a in range(n) for b in range(a,n)]
        candidates.sort(key=lambda r:identity(json_bytes({'control_seed':0,'family':r['family']})))
        if counts[n]>len(candidates):
            raise ContractError('insufficient fixed control universe')
        control.extend(candidates[:counts[n]])
    heldout=[formal(n,a,b,split,'oracle_heldout',revision) for n,split in ((11,'validation'),(13,'test'))
             for a in range(n) for b in range(a,n)]
    return {'teacher_selected':teacher,'oracle_only':control,'heldout':heldout}


def diagnostic_rows():
    dataset,manifest=build_dataset(ModelConfig(**MODEL))
    audit_bytes(json_bytes(dataset),json_bytes(manifest))
    selected=[];seen=set()
    for row in dataset['examples']:
        if row['split']=='train' and row['answer'] not in seen:
            selected.append({'identity':row['example_identity'],'prompt':row['prompt'],
                             'answer':row['answer'],'source_item':row,'split':'train'})
            seen.add(row['answer'])
        if len(selected)==4:
            break
    if len(selected)!=4:
        raise ContractError('four distinct admitted training answers required')
    return selected


def request(packet,environment):
    data=pools(packet)
    schedules={}
    for seed in (0,1):
        schedules[str(seed)]={}
        for arm in PROTOCOL['arms']:
            rows=ordered(data[arm],seed)
            schedules[str(seed)][arm]=[rows[i%len(rows)]['identity'] for i in range(1024)]
    return {'schema':'trinite.teacher-student-request.v1','protocol':PROTOCOL,'source':source_map(),
            'environment':environment,'packet':packet,'packet_identity':identity(json_bytes(packet)),
            'pools':data,'diagnostic_rows':diagnostic_rows(),
            'scheduled_visits':schedules,
            'overlap_families':sorted([list(f) for f in
               ({tuple(r['family']) for r in data['teacher_selected']} &
                {tuple(r['family']) for r in data['oracle_only']})]),
            'admission':'explicit trinite.teacher-origin-formal-policy.v1; raw response never consumed'}


def ordered(rows,seed):
    return sorted(rows,key=lambda r:(r.get('formal_spec',{}).get('n',0),
                  identity(json_bytes({'seed':seed,'identity':r['identity']}))))


def create(rows,seed,diagnostic=False):
    from trinite.model import Decoder
    from trinite.training import RunConfig, TrainingState, create_optimizer, evaluate, model_identity, require_environment
    require_environment()
    cfg=RunConfig(model=ModelConfig(**MODEL),seed=seed,steps=256,batch_size=4,
                  validation_every=256,max_seconds=600,max_target_tokens=2048 if diagnostic else 3072)
    encoded=[]
    for row in ordered(rows,seed):
        tokens=ByteTokenizer(64).encode(row['prompt']+row['answer'],answer_start=len(row['prompt']))
        if sum(tokens.loss_mask)!=(2 if diagnostic else 3):
            raise ContractError('unexpected target width')
        encoded.append({'identity':row['identity'],'ids':tokens.input_ids,
                        'mask':tokens.attention_mask,'loss':tokens.loss_mask})
    data={'train':encoded,'validation':encoded}
    model=Decoder(cfg.model,lane='dense',seed=seed)
    return TrainingState(cfg,'dense',model,create_optimizer(model,cfg),data,
                         identity(json_bytes(rows)),identity(json_bytes(PROTOCOL)),model_identity(model),
                         evaluate(model,encoded,cfg.batch_size))


def score(model,rows):
    from trinite.comparison import greedy
    results=[]
    for row in rows:
        tokens=greedy(model,row['prompt'],len(row['answer'])+1)
        expected=list(row['answer'].encode('ascii'))+[EOS]
        results.append({'identity':row['identity'],'split':row['split'],'tokens':tokens,
                        'exact':tokens==expected})
    return {'records':results,'counts':{s:{'rows':sum(r['split']==s for r in results),
                                         'exact':sum(r['exact'] for r in results if r['split']==s)}
                                        for s in ('train','validation','test')}}


def snapshot(state):
    from trinite.inspection import tensor_identity
    from trinite.training import model_identity
    return {'initial_model_identity':state.initialization_identity,'final_model_identity':model_identity(state.model),
            'config':state.config.to_dict(),'steps':state.step,'target_tokens':state.target_tokens,
            'scheduled_visits':state.cursor,'history':state.history,'train_only_diagnostics':state.validation,
            'initial_train_loss':state.initial_train_loss,
            'optimizer':{name:{k:tensor_identity(v) for k,v in state.optimizer.state[p].items()}
                         for name,p in state.model.named_parameters()}}


def report(*,expected_request,on_cell=None):
    from trinite.model import Decoder
    from trinite.training import require_environment, run_steps, model_identity
    frozen=request(expected_request['packet'],require_environment())
    if json_bytes(frozen)!=json_bytes(expected_request):
        raise ContractError('student request/source/environment mismatch')
    rid=identity(json_bytes(frozen))
    rows=frozen['diagnostic_rows'];state=create(rows,0,True)
    run_steps(state)
    if state.target_tokens!=2048:
        raise ContractError('diagnostic target budget mismatch')
    scores=score(state.model,rows)
    diagnostic={**snapshot(state),'scores':scores,'passed':scores['counts']['train']['exact']==4}
    if on_cell:on_cell({'phase':'diagnostic_completed','request_identity':rid,'diagnostic':diagnostic})
    cells=[]
    if diagnostic['passed']:
        trained=[]
        for seed in (0,1):
            paired=None
            for arm in PROTOCOL['arms']:
                state=create(frozen['pools'][arm],seed)
                if paired is not None and state.initialization_identity!=paired:
                    raise ContractError('paired initialization mismatch')
                paired=state.initialization_identity
                run_steps(state)
                if state.target_tokens!=3072:
                    raise ContractError('comparison target budget mismatch')
                record={'phase':'training_completed','seed':seed,'arm':arm,'state':snapshot(state)}
                if on_cell:on_cell(record)
                trained.append((seed,arm,state))
        # Heldout scoring starts only after every comparison update completed.
        for seed,arm,state in trained:
            rows=frozen['pools'][arm]+frozen['pools']['heldout']
            initial=Decoder(state.config.model,lane='dense',seed=seed)
            if model_identity(initial)!=state.initialization_identity:
                raise ContractError('untrained reconstruction mismatch')
            cell={'seed':seed,'arm':arm,**snapshot(state),
                  'initial_scores':score(initial,rows),'final_scores':score(state.model,rows)}
            cells.append(cell)
            if on_cell:on_cell({'phase':'scoring_completed','cell':cell})
    if json_bytes(request(expected_request['packet'],require_environment()))!=json_bytes(frozen):
        raise ContractError('source/environment changed during experiment')
    return {'schema':'trinite.teacher-student-report.v1','request_identity':rid,
            'diagnostic':diagnostic,'cells':cells,
            'outcome':'completed' if diagnostic['passed'] else 'blocked_train_only_learnability',
            'scope':'exploratory teacher input-selection comparison; not reasoning transfer'}
