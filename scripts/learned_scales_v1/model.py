"""New model identity wrapping unchanged native operations and initialization."""
import torch
from torch import nn
from torch.nn import functional as F
from trinite.contracts import ContractError, ModelConfig, integer
from trinite.model import Decoder, Linear, MAX_PARAMETERS
from .quantizer import initial_scale, quantize, require_scale, project_scale_, MAX_ELEMENTS

MODEL_ID = 'trinite.learned-ternary-decoder.engineering.v1'
ARMS = ('learned', 'frozen')


class ScaleLinear(Linear):
    def __init__(self, original, arm):
        # Do not invoke Linear initialization: consume no new random draws.
        nn.Module.__init__(self)
        self.weight = original.weight
        self.lane = 'dense'  # Native structural validation; numerical ID is above.
        self.scale = nn.Parameter(initial_scale(self.weight), requires_grad=(arm == 'learned'))

    def forward(self, value):
        return F.linear(value, quantize(self.weight, self.scale).weight)


class ScaleDecoder(Decoder):
    def __init__(self, config, *, arm, seed=0, parameter_budget=2_000_000):
        if type(arm) is not str or arm not in ARMS:
            raise ContractError('unknown learned-scale arm')
        if not isinstance(config, ModelConfig):
            raise ContractError('validated architecture required')
        integer(parameter_budget, 'parameter budget', 1, MAX_PARAMETERS)
        if max(config.width**2, config.width*config.feed_forward_width) > MAX_ELEMENTS:
            raise ContractError('matrix exceeds engineering element bound before allocation')
        if config.parameter_count + 6*config.blocks > parameter_budget:
            raise ContractError('learned-scale model exceeds parameter budget before allocation')
        super().__init__(config, lane='dense', seed=seed, parameter_budget=parameter_budget)
        self.arm = arm
        self.model_id = MODEL_ID
        for block in self.blocks:
            for name in ('q', 'k', 'v', 'o', 'up', 'down'):
                setattr(block, name, ScaleLinear(getattr(block, name), arm))
        self._parameter_shapes = {n: tuple(p.shape) for n, p in self.named_parameters()}
        if sum(p.numel() for p in self.parameters()) != config.parameter_count + 6*config.blocks:
            raise ContractError('actual scale parameter inventory mismatch')
        self.validate_state()

    def validate_state(self):
        super().validate_state()
        if self.arm not in ARMS or self.model_id != MODEL_ID or self.lane != 'dense':
            raise ContractError('invalid learned-scale model identity')
        matrices = [m for m in self.modules() if isinstance(m, Linear)]
        if len(matrices) != 6*self.config.blocks or any(type(m) is not ScaleLinear for m in matrices):
            raise ContractError('all attention/FF matrices require versioned scale modules')
        for name, parameter in self.named_parameters():
            scale = name.endswith('.scale')
            if parameter.requires_grad != (not scale or self.arm == 'learned'):
                raise ContractError('trainability differs from declared scale arm')
            if scale:
                require_scale(parameter)

    def project_scales_(self):
        for module in self.modules():
            if isinstance(module, ScaleLinear):
                project_scale_(module.scale)
        self.validate_state()

    def inventory(self):
        self.validate_state()
        return {'model_id': MODEL_ID, 'arm': self.arm,
                'parameters': sum(p.numel() for p in self.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'scale_parameters': sum(p.numel() for n, p in self.named_parameters() if n.endswith('.scale')),
                'tied_output': self.output_weight is self.token_embedding,
                'architecture': self.config.to_dict()}
