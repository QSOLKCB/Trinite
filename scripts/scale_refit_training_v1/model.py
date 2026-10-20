"""Versioned fixed-code refit decoder using unchanged native decoder operations."""
import torch
from torch import nn
from torch.nn import functional as F
from trinite.contracts import ContractError, ModelConfig, integer
from trinite.model import Decoder, Linear, MAX_PARAMETERS
from trinite.quantizer import quantize
from scale_refit.quantizer import refit, MAX_ELEMENTS, QUANTIZER_ID

MODEL_ID = 'trinite.fixed-code-refit-decoder.pilot.v1'
ARMS = ('absmean', 'refit')


class RefitLinear(Linear):
    def __init__(self, original, arm):
        nn.Module.__init__(self)
        self.weight, self.arm, self.lane = original.weight, arm, 'ternary'

    def forward(self, value):
        selected = refit(self.weight).quantized if self.arm == 'refit' else quantize(self.weight)
        return F.linear(value, selected.weight)


class RefitDecoder(Decoder):
    def __init__(self, config, *, arm, seed=0, parameter_budget=2_000_000):
        if type(arm) is not str or arm not in ARMS or not isinstance(config, ModelConfig):
            raise ContractError('validated refit configuration and arm required')
        integer(parameter_budget, 'parameter budget', 1, MAX_PARAMETERS)
        if max(config.width**2, config.width*config.feed_forward_width) > MAX_ELEMENTS:
            raise ContractError('refit matrix bound exceeded before allocation')
        super().__init__(config, lane='ternary', seed=seed, parameter_budget=parameter_budget)
        self.arm, self.model_id = arm, MODEL_ID
        for block in self.blocks:
            for name in ('q', 'k', 'v', 'o', 'up', 'down'):
                setattr(block, name, RefitLinear(getattr(block, name), arm))
        self.validate_state()

    def validate_state(self):
        super().validate_state()
        if self.model_id != MODEL_ID or self.arm not in ARMS or self.lane != 'ternary':
            raise ContractError('invalid refit model identity')
        matrices = [m for m in self.modules() if isinstance(m, Linear)]
        if len(matrices) != 6*self.config.blocks or any(type(m) is not RefitLinear or m.arm != self.arm for m in matrices):
            raise ContractError('refit model requires declared versioned matrix inventory')
        if any(not p.requires_grad for p in self.parameters()):
            raise ContractError('all latent refit parameters must remain trainable')

    def inventory(self):
        self.validate_state()
        return {'model_id': MODEL_ID, 'arm': self.arm, 'quantizer_id': QUANTIZER_ID if self.arm == 'refit' else 'trinite.absmean-ste.v0',
                'parameters': sum(p.numel() for p in self.parameters()), 'scale_parameters': 0,
                'tied_output': self.output_weight is self.token_embedding, 'architecture': self.config.to_dict()}
