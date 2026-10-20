"""Pinned inference-only comparison; generation has no label/data interface."""
import gzip
import io
import time
from pathlib import Path
import torch
from trinite.contracts import ContractError, ModelConfig, exact_keys, identity, json_bytes, parse_json
from trinite.checkpoint import hex_value
from trinite.tokenizer import BOS, EOS
from trinite.training import admitted_data, model_identity, require_environment
from scale_refit_training_v1.model import RefitDecoder
from scale_refit_training_v1.experiment import configuration
from scale_refit_training_v1.spec import inputs, sources as historical_sources

ROOT = Path(__file__).resolve().parents[2]
REPORT_ID = 'sha256:5e3614ea4a34248afe656b867cc59fca2ea64b8198d372e8d58bf7aa6e4e04c0'
MODES = ('greedy', 'syntax')
CELLS = ((0,'absmean'), (0,'refit'), (1,'absmean'), (1,'refit'))
CAP = 2


def original():
    path = ROOT/'fixtures/scale-refit-training-v1/report.json.gz'
    if path.is_symlink() or not path.is_file():
        raise ContractError('original report must be regular file')
    with path.open('rb') as f:
        compressed=f.read(16_777_217)
    if len(compressed)>16_777_216:
        raise ContractError('compressed original report exceeds bound')
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as f:
        raw = f.read(16_777_217)
    if len(raw)>16_777_216 or identity(raw)!=REPORT_ID:
        raise ContractError('original report identity/size mismatch')
    obj = parse_json(raw, canonical=True)
    if obj['source'] != historical_sources() or [(c['seed'],c['arm']) for c in obj['cells']] != list(CELLS):
        raise ContractError('historical source/cell inventory mismatch')
    return obj


def load_model(cell):
    """Only one of four exact pinned final endpoints, never arbitrary tensors."""
    trusted = original()
    matches = [c for c in trusted['cells'] if c['seed']==cell.get('seed') and c['arm']==cell.get('arm')]
    if len(matches)!=1 or json_bytes(matches[0]) != json_bytes(cell):
        raise ContractError('endpoint differs from immutable original report')
    cp = cell['final_checkpoint']
    config = configuration(cell['seed'])
    if cp['config']!=config.to_dict() or cp['step']!=60 or cp['source']!=historical_sources() or cp['environment']!=trusted['environment'] or cp['protocol_identity']!=trusted['protocol_identity']:
        raise ContractError('frozen checkpoint configuration/source mismatch')
    dataset, manifest = inputs()
    admitted_data(dataset, manifest, config)
    if cp['dataset_identity']!=identity(dataset) or cp['manifest_identity']!=identity(manifest):
        raise ContractError('frozen checkpoint inputs mismatch')
    model = RefitDecoder(config.model, arm=cell['arm'], seed=cell['seed'])
    if model.inventory()!=cp['inventory']:
        raise ContractError('frozen checkpoint model inventory mismatch')
    named = dict(model.named_parameters())
    exact_keys(cp['model_hex'],set(named),'frozen model tensors')
    with torch.no_grad():
        for name,p in named.items():
            vals=cp['model_hex'][name]
            if type(vals) is not list or len(vals)!=p.numel():
                raise ContractError('frozen tensor shape mismatch')
            tensor=torch.tensor([hex_value(v,'model value') for v in vals],dtype=torch.float32)
            if not bool(torch.isfinite(tensor).all()) or [float(v).hex() for v in tensor.tolist()]!=vals:
                raise ContractError('frozen tensor not exact finite float32')
            p.copy_(tensor.reshape(p.shape))
    model.eval();model.validate_state()
    return model


def generate(model, prompt, mode):
    """Fixed workload-wide caps; syntax is [0-9] EOS, never arithmetic."""
    if type(prompt) is not str or mode not in MODES:
        raise ContractError('invalid generation prompt/mode')
    ids=[BOS,*prompt.encode('utf-8')]; emitted=[];calls=0
    with torch.no_grad():
        for position in range(CAP):
            if len(ids)>model.config.context_length:
                break
            tensor=torch.tensor([ids],dtype=torch.int64,device='cpu')
            logits=model(tensor,torch.ones_like(tensor,dtype=torch.bool)).logits[0,-1]
            calls+=1
            if logits.shape!=(259,) or not bool(torch.isfinite(logits).all()):
                raise ContractError('invalid/nonfinite generation logits')
            if mode=='syntax':
                allowed=list(range(ord('0'),ord('9')+1)) if position==0 else [EOS]
                # Sorted IDs preserve native argmax's lowest-token tie resolution.
                token=allowed[int(logits[allowed].argmax())]
            else:
                token=int(logits.argmax())
            emitted.append(token)
            if token==EOS:break
            ids.append(token)
    return {'generated_tokens':emitted,'forward_calls':calls,'generated_token_count':len(emitted),
            'stop_reason':'eos' if emitted and emitted[-1]==EOS else ('token_cap' if len(emitted)==CAP else 'context_cap')}


def score(tokens, answer):
    # Same authoritative formal complete answer+EOS criterion as comparison.score_model.
    correct=tokens==[*answer.encode('utf-8'),EOS]
    valid=len(tokens)==2 and 48<=tokens[0]<=57 and tokens[1]==EOS
    return {'correct':correct,'format_valid':valid,'malformed':not valid,'semantic_error':valid and not correct}


def sources():
    paths=list((ROOT/'scripts/frozen_decoding_v1').glob('*.py'))+list((ROOT/'tests/frozen_decoding_v1').glob('*.py'))
    paths += [ROOT/'scripts/frozen_decoding_v1/protocol.json',ROOT/'scripts/run_frozen_decoding.py',ROOT/'scripts/check_frozen_decoding_cpu.py',ROOT/'.github/workflows/frozen-decoding.yml']
    return {**historical_sources(),**{p.relative_to(ROOT).as_posix():identity(p.read_bytes()) for p in sorted(paths)}}


def request():
    protocol=parse_json((ROOT/'scripts/frozen_decoding_v1/protocol.json').read_bytes(),canonical=True)
    prior=original()
    return {'schema':'trinite.frozen-decoding-request.v1','protocol':protocol,'source':sources(),
            'environment':require_environment(),'original_report_identity':REPORT_ID,
            'checkpoint_identities':[identity(json_bytes(c['final_checkpoint'])) for c in prior['cells']]}


def report(on_cell=None, timings=None):
    binding=request();prior=original();rows=parse_json(inputs()[0],canonical=True)['examples'];cells=[]
    for cell in prior['cells']:
        model=load_model(cell); before=model_identity(model); modes={}
        for mode in MODES:
            scored=[]
            for split in ('train','validation'):
                for item in rows:
                    if item['split']!=split:continue
                    start=time.perf_counter_ns(); prediction=generate(model,item['prompt'],mode);elapsed=time.perf_counter_ns()-start
                    if timings is not None:timings.append({'seed':cell['seed'],'arm':cell['arm'],'mode':mode,'example_identity':item['example_identity'],'elapsed_ns':elapsed})
                    if mode=='greedy':
                        old=next(r for r in cell['quality'][split]['predictions'] if r['example_identity']==item['example_identity'])
                        if prediction['generated_tokens']!=old['generated_tokens']:
                            raise ContractError('ordinary decoding differs from historical prediction')
                    scored.append({'example_identity':item['example_identity'],'family_id':item['family_id'],'split':split,
                        'prompt_identity':identity(item['prompt'].encode()),**prediction,**score(prediction['generated_tokens'],item['answer'])})
            if model_identity(model)!=before or any(p.grad is not None for p in model.parameters()):
                raise ContractError('frozen model changed during generation')
            modes[mode]={'predictions':scored,'splits':{s:{k:sum(int(r[k]) for r in scored if r['split']==s) for k in ('correct','format_valid','malformed','semantic_error','generated_token_count','forward_calls')} for s in ('train','validation')}}
        result={'seed':cell['seed'],'arm':cell['arm'],'checkpoint_identity':identity(json_bytes(cell['final_checkpoint'])),'model_identity':before,'modes':modes}
        cells.append(result)
        if on_cell is not None:on_cell(result)
    if request()!=binding:raise ContractError('comparison bindings changed during generation')
    return {'schema':'trinite.frozen-decoding-report.v1','request':binding,'cells':cells,'training_updates':0,'test_evaluated':False,
            'scope':'visible development fixture; task-format prior, not learned validity or fresh generalization','speed_claim':False}
