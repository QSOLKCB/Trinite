"""Versioned dense device adapter; historical CPU decoder remains unchanged.

Reuse native initialization and block operations, not its CPU-only forward or
validation. This adapter is admitted only by the separate device trial.
"""
from time import monotonic
import torch
from torch import nn
from torch.nn import functional as F

from trinite.contracts import ContractError, ModelConfig, integer
from trinite.model import Decoder, Block, Linear, RMSNorm, ModelOutput, MAX_BATCH, MAX_ATTENTION_CELLS
from trinite.inspection import tensor_identity
from trinite.contracts import identity, json_bytes


def device_for(backend):
    if type(backend) is not str or backend not in ('cpu', 'cuda'):
        raise ContractError('fixed cpu/cuda backend required; no fallback')
    return torch.device('cuda:0' if backend == 'cuda' else 'cpu')


class DeviceDecoder(nn.Module):
    def __init__(self, config, seed=17, device='cpu'):
        super().__init__()
        target = device_for(device)
        if not isinstance(config, ModelConfig):
            raise ContractError('validated model config required')
        if device == 'cuda':torch.cuda.synchronize(0)
        started = monotonic()
        original = Decoder(config, lane='dense', seed=seed)
        if device == 'cuda':torch.cuda.synchronize(0)
        self.cpu_initialization_seconds_hex = (monotonic()-started).hex()
        self.config, self.seed, self.lane, self.backend = config, seed, 'dense', device
        self._construction_config = config
        self.token_embedding = original.token_embedding
        self.position_embedding = original.position_embedding
        self.blocks = original.blocks
        self.final_norm = original.final_norm
        self.output_weight = self.token_embedding
        self._parameter_shapes = dict(original._parameter_shapes)
        if device == 'cuda':torch.cuda.synchronize(0)
        started = monotonic()
        self.to(target)
        # Explicitly restore and verify sharing after module device conversion.
        self.output_weight = self.token_embedding
        if device == 'cuda':torch.cuda.synchronize(0)
        self.model_transfer_seconds_hex = (monotonic()-started).hex()
        self.validate_state()

    def validate_state(self):
        target = device_for(self.backend)
        if self.config != self._construction_config or self.lane != 'dense':
            raise ContractError('device adapter construction context differs')
        parameters = dict(self.named_parameters())
        if set(parameters) != set(self._parameter_shapes) or self.output_weight is not self.token_embedding:
            raise ContractError('device adapter inventory/tied weight differs')
        if list(self.named_buffers()):
            raise ContractError('device adapter has unexpected buffers')
        for name, p in parameters.items():
            if (p.device != target or p.dtype != torch.float32 or p.layout != torch.strided
                    or tuple(p.shape) != self._parameter_shapes[name] or not bool(torch.isfinite(p).all())):
                raise ContractError('device adapter parameter shape/device/dtype/finite mismatch')
        if len(self.blocks) != self.config.blocks or any(type(b) is not Block for b in self.blocks):
            raise ContractError('native block inventory differs')
        if any(m.lane != 'dense' for m in self.modules() if isinstance(m, Linear)):
            raise ContractError('dense native linear operations required')
        if any(type(m.epsilon) is not float or m.epsilon != float(self.config.rms_epsilon)
               for m in self.modules() if isinstance(m, RMSNorm)):
            raise ContractError('native normalization configuration differs')
        if any(b.heads != self.config.heads or b.head_width != self.config.width//self.config.heads for b in self.blocks):
            raise ContractError('native head layout differs')

    def forward(self, input_ids, attention_mask):
        target = device_for(self.backend)
        if (not isinstance(input_ids, torch.Tensor) or input_ids.device != target
                or input_ids.dtype != torch.int64 or input_ids.layout != torch.strided or input_ids.ndim != 2):
            raise ContractError('device adapter requires explicit-device int64 IDs')
        batch, length = input_ids.shape
        integer(batch, 'batch', 1, MAX_BATCH); integer(length, 'length', 1, self.config.context_length)
        if batch*self.config.heads*length*length > MAX_ATTENTION_CELLS:
            raise ContractError('attention allocation bound exceeded')
        if (not isinstance(attention_mask, torch.Tensor) or attention_mask.device != target
                or attention_mask.dtype != torch.bool or attention_mask.layout != torch.strided
                or attention_mask.shape != input_ids.shape or not bool(attention_mask[:,0].all())
                or bool((~attention_mask[:,:-1] & attention_mask[:,1:]).any())
                or bool((input_ids < 0).any()) or bool((input_ids >= self.config.vocabulary_size).any())):
            raise ContractError('invalid bounded right-padded device inputs')
        if torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('cuda'):
            raise ContractError('autocast is outside float32 device trial')
        self.validate_state()
        live = attention_mask.unsqueeze(-1).to(torch.float32)
        positions = torch.arange(length, device=target)
        value = (F.embedding(input_ids,self.token_embedding)+F.embedding(positions,self.position_embedding))*live
        causal = torch.ones((length,length),device=target,dtype=torch.bool).tril()
        allowed = causal[None,None,:,:] & attention_mask[:,None,None,:]
        for block in self.blocks:
            value = block(value,allowed,live)
        value = self.final_norm(value)*live
        logits = F.linear(value,self.output_weight)*live
        if logits.device != target or logits.dtype != torch.float32 or not bool(torch.isfinite(logits).all()):
            raise ContractError('invalid device logits')
        return ModelOutput(logits,{})


def model_identity(model):
    model.validate_state()
    return identity(json_bytes({n:tensor_identity(p.detach().to('cpu').contiguous())
                                for n,p in model.named_parameters()}))


def batch(examples, backend):
    # Preserve the native padding/masks, then explicitly transfer every tensor.
    from trinite.training import batch as cpu_batch
    return tuple(t.to(device_for(backend)) for t in cpu_batch(examples))


def target_loss(model, examples):
    ids,mask,scored = batch(examples,model.backend)
    logits = model(ids[:,:-1],mask[:,:-1]).logits
    weights = scored[:,1:]; tokens = int(weights.sum())
    if not tokens:
        raise ContractError('no answer/EOS targets')
    losses = F.cross_entropy(logits.reshape(-1,259),ids[:,1:].reshape(-1),reduction='none')
    result = (losses.reshape(weights.shape)*weights).sum()/tokens
    if not bool(torch.isfinite(result)):
        raise ContractError('nonfinite device loss')
    return result,tokens


def evaluate(model, examples, batch_size):
    total,tokens=0.,0
    with torch.no_grad():
        for start in range(0,len(examples),batch_size):
            loss,count=target_loss(model,examples[start:start+batch_size]);total+=float(loss)*count;tokens+=count
    return (total/tokens).hex()


def greedy(model,prompt,cap):
    if type(prompt) is not str or not prompt.isascii() or cap != 64 or 1+len(prompt)+cap > model.config.context_length:
        raise ContractError('fixed prompt-only device greedy boundary')
    ids=[256,*prompt.encode('ascii')]; emitted=[]
    with torch.no_grad():
        for _ in range(cap):
            tensor=torch.tensor([ids],dtype=torch.int64,device=device_for(model.backend))
            logits=model(tensor,torch.ones_like(tensor,dtype=torch.bool)).logits
            token=int(logits[0,-1].argmax());emitted.append(token)
            if token==257:break
            ids.append(token)
    return emitted
