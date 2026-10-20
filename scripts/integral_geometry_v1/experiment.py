"""Compose the unchanged CPU decoder, updater, tokenizer and greedy generator."""
import torch
from trinite.contracts import ContractError, ModelConfig, identity, json_bytes
from trinite.model import Decoder
from trinite.inspection import tensor_identity
from trinite.tokenizer import ByteTokenizer, EOS
from trinite.training import (RunConfig, TrainingState, create_optimizer, evaluate, model_identity,
                              require_environment, run_steps)
from trinite.comparison import greedy
from .data import ARMS, examples, admission
from .math import parse_setup, setup_correct, answer_correct
from .spec import MODEL_ID, PROTOCOL, request, schedule, targets


def config(seed):
    if type(seed) is not int or seed not in PROTOCOL['seeds']:
        raise ContractError('unknown seed')
    return RunConfig(model=ModelConfig(**PROTOCOL['model']), seed=seed,
                     steps=PROTOCOL['steps'], batch_size=PROTOCOL['batch_size'],
                     validation_every=PROTOCOL['steps'], max_seconds=PROTOCOL['cell_seconds'],
                     max_target_tokens=PROTOCOL['max_target_tokens'])


def training_data(seed, arm):
    if arm not in ARMS:
        raise ContractError('unknown curriculum arm')
    tokenizer = ByteTokenizer(128)
    result = []
    for row,kind in schedule(seed):
        if kind == 'setup':
            prompt = row['prompt' if arm == 'geometry' else 'control_prompt']
            answer = row['setup']
        else:
            prompt,answer = 'eval['+row['setup']+']=',row['answer']
        enc = tokenizer.encode(prompt+answer, answer_start=len(prompt))
        result.append({'identity':identity(json_bytes({'row':row['identity'],'component':kind})),
                       'ids':enc.input_ids,'mask':enc.attention_mask,'loss':enc.loss_mask})
    # The existing update loop's periodic diagnostic is deliberately TRAIN-ONLY.
    # No validation or test labels reach state.data or update-time diagnostics.
    return {'train':result,'validation':result}


def create(seed, arm):
    require_environment()
    cfg = config(seed)
    data = training_data(seed,arm)
    model = Decoder(cfg.model,lane='dense',seed=seed)
    return TrainingState(cfg,'dense',model,create_optimizer(model,cfg),data,
                         identity(json_bytes(examples())),identity(json_bytes(admission())),
                         model_identity(model),evaluate(model,data['train'],cfg.batch_size))


def decoded(tokens):
    if not tokens or tokens[-1] != EOS or any(type(x) is not int or not 0<=x<=255 for x in tokens[:-1]):
        return None
    try:
        return bytes(tokens[:-1]).decode('ascii')
    except UnicodeError:
        return None


def score(model):
    records=[]
    for row in examples():
        setup_tokens=greedy(model,row['prompt'],PROTOCOL['setup_cap'])
        setup=decoded(setup_tokens)
        conditional_tokens=greedy(model,'eval['+row['setup']+']=',PROTOCOL['answer_cap'])
        conditional=decoded(conditional_tokens)
        end_tokens=[]
        if setup is not None:
            try:
                parse_setup(setup)
            except (ContractError,ValueError,ZeroDivisionError):
                pass
            else:
                # Generated setup only, never an oracle value or gold correction.
                end_tokens=greedy(model,'eval['+setup+']=',PROTOCOL['answer_cap'])
        end=decoded(end_tokens)
        records.append({'identity':row['identity'],'family':row['family'],'task':row['task'],
                        'split':row['split'],'novel_transform':row['novel_transform'],
                        'setup_tokens':setup_tokens,'conditional_tokens':conditional_tokens,
                        'end_to_end_tokens':end_tokens,
                        'setup_semantic':setup_correct(setup,row['setup']),
                        'setup_canonical':setup==row['setup'],
                        'oracle_setup_evaluation':answer_correct(conditional,row['answer']),
                        'end_to_end':answer_correct(end,row['answer'])})
    groups={}
    for split in ('train','validation','test'):
        for task in ('all','area','volume','centroid_x'):
            selected=[r for r in records if r['split']==split and (task=='all' or r['task']==task)]
            groups[split+'/'+task]={'examples':len(selected),**{k:sum(r[k] for r in selected) for k in PROTOCOL['scores']}}
    return {'groups':groups,'records':records}


def report(*, expected_request, on_cell=None):
    frozen=request(require_environment())
    if frozen != expected_request:
        raise ContractError('experiment differs from validated supplied request')
    cells=[]
    for seed in PROTOCOL['seeds']:
        initial=None
        for arm in ARMS:
            state=create(seed,arm)
            if initial is None:
                initial={'model_identity':state.initialization_identity,'scores':score(state.model)}
            elif initial['model_identity']!=state.initialization_identity:
                raise ContractError('paired initialization mismatch')
            run_steps(state)
            if state.target_tokens != targets(seed):
                raise ContractError('scheduled target count mismatch')
            cell={'seed':seed,'arm':arm,'model_id':MODEL_ID,'config':state.config.to_dict(),
                  'parameters':sum(p.numel() for p in state.model.parameters()),
                  'initial_model_identity':state.initialization_identity,
                  'final_model_identity':model_identity(state.model),
                  'final_optimizer':{name:{key:tensor_identity(value) for key,value in state.optimizer.state[p].items()}
                                     for name,p in state.model.named_parameters()},
                  'steps':state.step,'target_tokens':state.target_tokens,
                  'history':state.history,'train_only_diagnostics':state.validation,
                  'initial_train_loss':state.initial_train_loss,'scores':score(state.model),'untrained':initial}
            if on_cell:
                on_cell(cell)
            cells.append(cell)
    if frozen != request(require_environment()):
        raise ContractError('source/input/environment changed during run')
    return {'schema':'trinite.integral-geometry-report.v1','request_identity':identity(json_bytes(frozen)),
            'cells':cells,'scope':'exploratory curriculum comparison; no capability or geometry-loss gate'}
