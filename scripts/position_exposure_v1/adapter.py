"""Explicit cycle-scoped offsets; inference and diagnostics always use offset zero."""
from contextlib import contextmanager

import torch

from trinite.contracts import ContractError, integer
from trinite.tokenizer import BOS
from position_coupling_v1.model import PositionDecoder, MODEL_ID
from position_coupling_v1.positions import prefix_positions
from position_study_v1.adapter import generation_cap
from .protocol import ARMS, MODES, POLICIES, OFFSETS, PROTOCOL


def addressing(arm):
    if type(arm) is not str or arm not in ARMS:
        raise ContractError('unknown position exposure arm')
    return {'schema': 'trinite.position-exposure-addressing.v1',
        'arm': arm, 'mode': MODES[arm], 'model': MODEL_ID,
        'slots': 'trinite.arithmetic-position-slots.v1',
        'offset_policy': POLICIES[arm],
        'training_offsets': list(OFFSETS) if arm.endswith('-balanced') else [0],
        'evaluation_offset': 0, 'schedule': PROTOCOL['schedule'],
        'cap_policy': 'prompt-operand-width-plus-two.v1',
        'adapter': 'trinite.position-exposure-live-prefix.v1'}


def live_positions(ids, mask, *, mode, context_length, offset=0):
    """Parse only complete input prompt; generated values never select positions."""
    integer(offset, 'position exposure offset', 0, max(OFFSETS))
    result = torch.zeros_like(ids)
    for index in range(ids.shape[0]):
        tokens = ids[index, mask[index]].tolist()
        if not tokens or tokens[0] != BOS:
            raise ContractError('live exposure prefix requires BOS')
        try:
            end = tokens.index(ord('='), 1)
        except ValueError as error:
            raise ContractError('live exposure prefix requires complete prompt') from error
        prompt_ids = tokens[1:end+1]
        if any(not 0 <= token <= 127 for token in prompt_ids):
            raise ContractError('prompt requires ASCII grammar')
        prompt = ''.join(map(chr, prompt_ids))
        positions = prefix_positions(prompt, len(tokens)-end-1, mode=mode,
                                     context_length=context_length, offset=offset)
        result[index, :len(tokens)] = torch.tensor(positions, dtype=torch.int64)
    return result


class ExposureDecoder(PositionDecoder):
    """No added parameter/buffer or RNG. Context is explicit, transient and reset.

    The sole updater evaluates its diagnostics inside torch.no_grad(). Those
    forwards, and native greedy generation, deliberately use offset zero even
    while a training cycle context exists. No forward increments a counter.
    """
    def __init__(self, config, *, arm, seed=0):
        addressing(arm)
        super().__init__(config, seed=seed)
        self.arm = arm
        self._training_offset = None

    @property
    def addressing(self):
        return addressing(self.arm)

    def require_idle(self):
        if self._training_offset is not None:
            raise ContractError('exposure context must be idle at evidence boundary')

    @contextmanager
    def training_offset(self, offset):
        self.require_idle()
        integer(offset, 'training position offset', 0, max(OFFSETS))
        if self.arm.endswith('-fixed') and offset != 0:
            raise ContractError('fixed arm cannot select shifted offset')
        self._training_offset = offset
        try:
            yield
        finally:
            self._training_offset = None

    def positions(self, input_ids, attention_mask):
        self._validate_inputs(input_ids, attention_mask)
        offset = self._training_offset if torch.is_grad_enabled() and self._training_offset is not None else 0
        return live_positions(input_ids, attention_mask, mode=MODES[self.arm],
                              context_length=self.config.context_length, offset=offset)

    def forward(self, input_ids, attention_mask, *, capture=None):
        positions = self.positions(input_ids, attention_mask)
        return super().forward(input_ids, attention_mask, position_ids=positions, capture=capture)
