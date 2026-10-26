"""Versioned explicit-position forward; historical Decoder bytes stay frozen."""
import torch
from torch.nn import functional as F

from trinite.contracts import ContractError
from trinite.model import CaptureSpec, Decoder, ModelOutput
from trinite.quantizer import require_float_tensor

MODEL_ID = 'trinite.explicit-position-decoder.engineering.v1'


class PositionDecoder(Decoder):
    """Reuse native initialization, state validation and every numerical block.

    Only embedding address orchestration is new. No optimizer, checkpoint
    importer, generation loop or scientific training protocol is implemented.
    """
    def __init__(self, config, *, seed=0, parameter_budget=2_000_000):
        super().__init__(config, lane='dense', seed=seed, parameter_budget=parameter_budget)

    def forward(self, input_ids, attention_mask, *, position_ids, capture=None):
        self._validate_inputs(input_ids, attention_mask)
        if self.lane != 'dense':
            raise ContractError('position engineering lane is dense only')
        if (not isinstance(position_ids, torch.Tensor) or position_ids.device.type != 'cpu'
                or position_ids.dtype != torch.int64 or position_ids.layout != torch.strided
                or position_ids.shape != input_ids.shape
                or bool((position_ids < 0).any())
                or bool((position_ids >= self.config.context_length).any())):
            raise ContractError('position IDs require bounded CPU int64 with the input shape')
        batch, length = input_ids.shape
        if capture is not None:
            if not isinstance(capture, CaptureSpec):
                raise ContractError('capture requires a CaptureSpec')
            capture.validate(self.config, batch, length)
        captures = {}

        def retain(name, value):
            if capture is not None and name in capture.layers:
                captures[name] = value[:, capture.positions, :].detach().clone()

        live = attention_mask.unsqueeze(-1).to(torch.float32)
        value = (F.embedding(input_ids, self.token_embedding)
                 + F.embedding(position_ids, self.position_embedding)) * live
        retain('embedding', value)
        # Actual sequence order is causal. Position IDs may repeat or descend.
        causal = torch.ones((length, length), device='cpu', dtype=torch.bool).tril()
        allowed = causal[None, None, :, :] & attention_mask[:, None, None, :]
        for index, block in enumerate(self.blocks):
            value = block(value, allowed, live)
            retain(f'block.{index}', value)
        value = self.final_norm(value) * live
        retain('final', value)
        logits = F.linear(value, self.output_weight) * live
        require_float_tensor(logits, 'decoder logits')
        return ModelOutput(logits, captures)
