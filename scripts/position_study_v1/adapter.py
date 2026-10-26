"""Stateless positions from actual live input tokens; never from answer values."""
import torch

from trinite.contracts import ContractError
from trinite.tokenizer import BOS
from position_coupling_v1.model import PositionDecoder, MODEL_ID
from position_coupling_v1.positions import prefix_positions, operand_width, MODES
from .protocol import POSITION_CONFIG


def addressing(mode):
    if type(mode) is not str or mode not in MODES:
        raise ContractError('unknown position mode')
    return {**POSITION_CONFIG, 'mode': mode, 'model': MODEL_ID}


def live_positions(ids, mask, *, mode, context_length):
    """Called after native input validation; padding never enters prompt parsing.

    Only the bytes through the first delimiter are parsed. Emitted values,
    including EOS/PAD/BOS or extra delimiters, are counted, never interpreted.
    """
    addressing(mode)
    result = torch.zeros_like(ids)
    for index in range(ids.shape[0]):
        tokens = ids[index, mask[index]].tolist()
        if not tokens or tokens[0] != BOS:
            raise ContractError('live study prefix requires BOS')
        try:
            end = tokens.index(ord('='), 1)
        except ValueError as error:
            raise ContractError('live study prefix requires complete prompt') from error
        prompt_ids = tokens[1:end+1]
        if any(not 0 <= t <= 127 for t in prompt_ids):
            raise ContractError('prompt requires ASCII grammar')
        prompt = ''.join(map(chr,prompt_ids))
        positions = prefix_positions(prompt,len(tokens)-end-1,mode=mode,
                                     context_length=context_length,offset=0)
        result[index,:len(tokens)] = torch.tensor(positions,dtype=torch.int64)
    return result


def generation_cap(prompt):
    return operand_width(prompt)+2


class StudyDecoder(PositionDecoder):
    """No added parameters/buffers/state keys, RNG draws or prefix sidecars."""
    def __init__(self, config, *, mode, seed=0):
        addressing(mode)
        super().__init__(config,seed=seed)
        self.position_mode = mode

    @property
    def addressing(self):
        return addressing(self.position_mode)

    def positions(self, input_ids, attention_mask):
        self._validate_inputs(input_ids,attention_mask)
        return live_positions(input_ids,attention_mask,mode=self.position_mode,
                              context_length=self.config.context_length)

    def forward(self, input_ids, attention_mask, *, capture=None):
        positions = self.positions(input_ids,attention_mask)
        return super().forward(input_ids,attention_mask,position_ids=positions,capture=capture)
