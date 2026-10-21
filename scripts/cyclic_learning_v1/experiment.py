"""Unchanged native dense CPU updater; heldout labels used only after fixed training."""
from trinite.contracts import ContractError, ModelConfig, identity, json_bytes
from trinite.model import Decoder
from trinite.inspection import tensor_identity
from trinite.tokenizer import ByteTokenizer, EOS
from trinite.training import (RunConfig, TrainingState, create_optimizer, evaluate, model_identity,
                              require_environment, run_steps)
from trinite.comparison import greedy
from .data import ARMS, examples, admission, validate_admission
from .spec import MODEL_ID, PROTOCOL, request, schedule, targets


def config(seed):
    schedule(seed)
    return RunConfig(model=ModelConfig(**PROTOCOL['model']),seed=seed,steps=32,batch_size=4,
                     validation_every=32,max_seconds=180,max_target_tokens=512)


def training_data(seed,arm):
    if arm not in ARMS:
        raise ContractError('unknown encoding arm')
    result=[]
    for row in schedule(seed):
        prompt=row['prompts'][arm]
        enc=ByteTokenizer(64).encode(prompt+row['answer'],answer_start=len(prompt))
        result.append({'identity':row['identity'],'ids':enc.input_ids,
                       'mask':enc.attention_mask,'loss':enc.loss_mask})
    # This native updater field is only a unique TRAIN diagnostic, never heldout data.
    return {'train':result,'validation':result}


def create(seed,arm):
    require_environment()
    validate_admission(admission())
    cfg=config(seed); data=training_data(seed,arm)
    model=Decoder(cfg.model,lane='dense',seed=seed)
    return TrainingState(cfg,'dense',model,create_optimizer(model,cfg),data,
                         identity(json_bytes(examples())),identity(json_bytes(admission())),
                         model_identity(model),evaluate(model,data['train'],cfg.batch_size))


def grade(tokens,answer):
    canonical=(len(tokens)==4 and tokens[-1]==EOS and
               all(type(t) is int and 48<=t<=57 for t in tokens[:3]))
    return {'canonical':canonical,'exact':canonical and bytes(tokens[:3]).decode('ascii')==answer}


def score(model,arm):
    records=[]
    for row in examples():
        tokens=greedy(model,row['prompts'][arm],4)
        records.append({'identity':row['identity'],'family':row['family'],'split':row['split'],
                        'tokens':tokens,**grade(tokens,row['answer'])})
    return {'records':records,'groups':{s:{'rows':sum(r['split']==s for r in records),
              **{k:sum(r[k] for r in records if r['split']==s) for k in ('exact','canonical')}}
             for s in ('train','validation','test')}}



def untrained_score(state,arm):
    """Recreate native initialization with its private RNG, without an optimizer."""
    model=Decoder(state.config.model,lane='dense',seed=state.config.seed)
    initial=model_identity(model)
    if initial!=state.initialization_identity:
        raise ContractError('recreated initial model differs from recorded initialization')
    return {'model_identity':initial,'scores':score(model,arm)}


def score_changes(final,initial):
    if ([r['identity'] for r in final['records']] !=
            [r['identity'] for r in initial['records']]):
        raise ContractError('baseline and trained scoring rows differ')
    return {split:{key:final['groups'][split][key]-initial['groups'][split][key]
                   for key in ('exact','canonical')}
            for split in ('train','validation','test')}

def report(*,expected_request,on_cell=None):
    frozen=request(require_environment())
    if frozen!=expected_request:
        raise ContractError('experiment differs from validated supplied request')
    # Complete every fixed update before any heldout scoring; no model selection.
    states=[]
    for seed in PROTOCOL['seeds']:
        paired=None
        for arm in ARMS:
            state=create(seed,arm)
            if paired is not None and paired!=state.initialization_identity:
                raise ContractError('paired initialization mismatch')
            paired=state.initialization_identity
            run_steps(state)
            if state.target_tokens!=targets(seed):
                raise ContractError('scheduled target count mismatch')
            if on_cell:
                on_cell({'phase':'training_completed','seed':seed,'arm':arm,
                         'request_identity':identity(json_bytes(frozen)),
                         'final_model_identity':model_identity(state.model),
                         'steps':state.step,'target_tokens':state.target_tokens,'history':state.history})
            states.append((seed,arm,state))
    cells=[]
    for seed,arm,state in states:
        baseline=untrained_score(state,arm)
        final_scores=score(state.model,arm)
        cell={'seed':seed,'arm':arm,'model_id':MODEL_ID,'config':state.config.to_dict(),
              'parameters':sum(p.numel() for p in state.model.parameters()),
              'initial_model_identity':state.initialization_identity,'final_model_identity':model_identity(state.model),
              'final_optimizer':{name:{key:tensor_identity(value) for key,value in state.optimizer.state[p].items()}
                                 for name,p in state.model.named_parameters()},
              'steps':state.step,'target_tokens':state.target_tokens,'history':state.history,
              'train_only_diagnostics':state.validation,'unique_train_rows':len(state.data['train']),
              'scheduled_visits':state.cursor,'initial_train_loss':state.initial_train_loss,
              'scores':final_scores,'untrained':baseline,
              'final_minus_untrained_counts':score_changes(final_scores,baseline['scores'])}
        if on_cell:
            on_cell({'phase':'scoring_completed','cell':cell})
        cells.append(cell)
    if frozen!=request(require_environment()):
        raise ContractError('source/input/environment changed during run')
    return {'schema':'trinite.cyclic-learning-report.v1','request_identity':identity(json_bytes(frozen)),
            'cells':cells,'scope':'exploratory matched encoding comparison, not geometric reasoning or base superiority'}
