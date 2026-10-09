"""Explicit CPU instrumentation, detached prompt states and greedy outputs."""
import hashlib
import json
from pathlib import Path
import sys

import torch

from .contracts import ContractError, identity, json_bytes
from .model import CaptureSpec
from .tokenizer import ByteTokenizer, BOS, EOS
from .reference_geometry import encode_vector
from .reference_inputs import anchors, read, validate, CONDITIONS


def tensor_manifest(model):
    if sys.byteorder != 'little': raise ContractError('reference profile requires little endian')
    records = []
    # Separate external profile streams large tensors; native inspection caps stay unchanged.
    for name,t in list(model.named_parameters())+list(model.named_buffers()):
        if t.device.type != 'cpu' or t.layout != torch.strided: raise ContractError('CPU strided tensors required')
        raw = t.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        sha = hashlib.sha256()
        for offset in range(0, len(raw), 1024*1024): sha.update(memoryview(raw[offset:offset+1024*1024]))
        records.append({'name':name,'shape':list(t.shape),'dtype':str(t.dtype),
                        'content_identity':'sha256:'+sha.hexdigest()})
    return {'tensors':records,'unique_parameters':sum(t.numel() for t in model.parameters()),
            'identity':identity(json_bytes(records))}


def native_model(request, locations):
    from .scalar_checkpoint import restore
    from .scalar_training import read_request
    prior = Path(locations['native_run'])
    read_request(prior/'request.json',request['native']['request_identity'])
    cell = prior/'arithmetic-0-dense'
    return restore(read(cell/'tensors.safetensors'),read(cell/'metadata.json'),
                   'arithmetic',0,'dense',request['native']['request_identity'],
                   (request['native']['tensors_identity'],request['native']['metadata_identity'])).model


def qwen_model(directory):
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    config = json.loads(read(Path(directory)/'config.json'))
    if config.get('model_type') != 'qwen3' or config.get('auto_map'):
        raise ContractError('only builtin Qwen3 architecture supported')
    tokenizer = AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
    if not tokenizer.is_fast: raise ContractError('offset-capable pinned tokenizer required')
    model = Qwen3ForCausalLM.from_pretrained(directory,local_files_only=True,trust_remote_code=False,
              use_safetensors=True,torch_dtype=torch.float32,attn_implementation='eager')
    return model.cpu(), tokenizer


def native_prompt(model, prompt):
    encoding = ByteTokenizer(model.config.context_length).encode(prompt)
    ids = list(encoding.input_ids); position = len(ids)-2
    return {'rendered':prompt,'input_ids':ids,'offsets':encoding.to_dict()['byte_spans'],
            'user_span':[0,len(prompt)],'selected_position':position}


def reference_prompt(tokenizer, prompt, system):
    messages = ([{'role':'system','content':system}] if system else [])+[{'role':'user','content':prompt}]
    rendered = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    marker = '<|im_start|>user\n'; start = rendered.index(marker)+len(marker); end = start+len(prompt)
    if rendered[start:end] != prompt or rendered[end:end+10] != '<|im_end|>':
        raise ContractError('unsupported chat-template user span')
    encoded = tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
    offsets = [list(x) for x in encoded['offset_mapping']]
    choices = [i for i,(a,b) in enumerate(offsets) if a < end and b == end and b > start]
    if len(choices) != 1 or not 1 <= len(encoded['input_ids']) <= 512:
        raise ContractError('unaligned user anchor or token budget')
    return {'rendered':rendered,'input_ids':encoded['input_ids'],'offsets':offsets,
            'user_span':[start,end],'selected_position':choices[0]}


def hidden(model, encoded, native):
    ids = torch.tensor([encoded['input_ids']],dtype=torch.long,device='cpu'); p = encoded['selected_position']
    with torch.no_grad():
        if native:
            output = model(ids,torch.ones_like(ids,dtype=torch.bool),capture=CaptureSpec(('block.0','final'),(p,)))
            values = {k:v[0,0].tolist() for k,v in output.captures.items()}
        else:
            output = model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,
                           output_hidden_states=True,return_dict=True)
            values = {'block.0':output.hidden_states[1][0,p].tolist(),
                      'final':output.hidden_states[-1][0,p].tolist()}
    return {k:encode_vector(v) for k,v in values.items()}


def generate(model, tokenizer, encoded, native):
    ids = encoded['input_ids'][:-1] if native else encoded['input_ids'][:]
    output = []; eos = [EOS] if native else model.generation_config.eos_token_id
    if type(eos) is int: eos = [eos]
    reason = 'token-cap'
    with torch.no_grad():
        for _ in range(32):
            if native and len(ids) >= model.config.context_length:
                reason = 'context-cap'; break
            values = torch.tensor([ids],dtype=torch.long,device='cpu')
            result = model(values,torch.ones_like(values,dtype=torch.bool)) if native else model(
                input_ids=values,attention_mask=torch.ones_like(values),use_cache=False,return_dict=True)
            token = int(result.logits[0,-1].argmax()); output.append(token); ids.append(token)
            if token in eos: reason = 'eos'; break
    if native:
        content = output[:-1] if reason == 'eos' else output
        valid = all(0 <= t < 256 for t in content)
        raw = bytes(content) if valid else b''
        try: text = raw.decode('utf-8') if valid else None
        except UnicodeError: text = None
        byte_hex = raw.hex() if valid else None
    else:
        # Strip only the terminal EOS; retain all other special output tokens.
        content = output[:-1] if reason == 'eos' else output
        text = tokenizer.decode(content,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        byte_hex = text.encode('utf-8').hex()
    return {'token_ids':output,'text':text,'utf8_hex':byte_hex,'stop_reason':reason}


def capture(directory, condition):
    # Include initialization/seeding as well as inference in caller isolation.
    # CPU is the sole supported lane; do not initialize a CUDA RNG context.
    with torch.random.fork_rng(devices=[]):
        return _capture(directory, condition)


def _capture(directory, condition):
    if condition not in CONDITIONS: raise ContractError('unknown reference condition')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(17)
    request, locations = validate(directory)
    native = condition == 'native'; tokenizer = None
    if native: model = native_model(request,locations)
    else: model,tokenizer = qwen_model(locations['model'])
    model.eval(); before = tensor_manifest(model); rng = torch.get_rng_state().clone()
    system = request['protocol']['system'] if condition == 'modelfile' else None
    rows = []
    for probe in request['probes']:
        states = []
        for endpoint in anchors(probe['prompt']):
            prefix = probe['prompt'][:endpoint]
            encoded = native_prompt(model,prefix) if native else reference_prompt(tokenizer,prefix,system)
            states.append({'endpoint':endpoint,'encoding':encoded,'vectors':hidden(model,encoded,native)})
        full = native_prompt(model,probe['prompt']) if native else reference_prompt(tokenizer,probe['prompt'],system)
        output = generate(model,tokenizer,full,native)
        rows.append({'example_identity':probe['example_identity'],'anchors':states,'generation_input':full,
                     'output':output,'exact_visible_answer':output['text'] == probe['answer']})
    if before != tensor_manifest(model) or not torch.equal(rng,torch.get_rng_state()):
        raise ContractError('capture changed model tensors or caller RNG')
    validate(directory)
    return {'schema':'trinite.reference-capture.v1','condition':condition,
            'request_identity':identity(read(Path(directory)/'request.json')),
            'model':before,'rng_identity':identity(bytes(rng.tolist())),
            'layers':{'block.0':'first transformer block output','final':'final normalized prompt state'},
            'rows':rows}
