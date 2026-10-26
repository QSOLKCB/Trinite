"""Bounded synthetic forward/gradient checks, not a learning experiment."""
import torch

from trinite.contracts import ContractError, ModelConfig, identity, json_bytes
from trinite.inspection import tensor_identity
from trinite.model import CaptureSpec, Decoder
from trinite.tokenizer import ByteTokenizer
from trinite.training import model_identity, require_environment
from .model import MODEL_ID, PositionDecoder
from .positions import POSITION_SCHEMA, prefix_positions


CONFIG = ModelConfig(context_length=64, blocks=2, width=64, heads=4,
                     feed_forward_width=128)


def _equal(left, right, label):
    if not torch.equal(left, right):
        raise ContractError(label + ' differs')


def run_checks():
    """Three fresh seeds, no optimizer, no checkpoint or data-file intake."""
    environment = require_environment()
    tokenizer = ByteTokenizer(64)
    prompts = ('47+58=', '12+03=')
    # Visible engineering examples, never fresh held-out research facts.
    answers = ('501', '510')
    encoded = [tokenizer.encode(prompt + answer, answer_start=len(prompt), pad_to=16)
               for prompt, answer in zip(prompts, answers)]
    ids = torch.tensor([row.input_ids for row in encoded], dtype=torch.int64)
    mask = torch.tensor([row.attention_mask for row in encoded], dtype=torch.bool)
    sequence = torch.arange(ids.shape[1], dtype=torch.int64).expand_as(ids)
    coupled_rows = [prefix_positions(prompt, len(answer) + 1, mode='coupled')
                    for prompt, answer in zip(prompts, answers)]
    coupled = torch.tensor([(*row, *([0] * (ids.shape[1] - len(row))))
                            for row in coupled_rows], dtype=torch.int64)
    captured = CaptureSpec(('embedding', 'block.0', 'final'), (0, 3, 6, 9))
    records = []
    for seed in (0, 1, 2):
        rng = torch.random.get_rng_state().clone()
        native = Decoder(CONFIG, lane='dense', seed=seed)
        explicit = PositionDecoder(CONFIG, seed=seed)
        if model_identity(native) != model_identity(explicit):
            raise ContractError('initial tensor identities differ')
        reference = native(ids, mask, capture=captured)
        actual = explicit(ids, mask, position_ids=sequence, capture=captured)
        _equal(reference.logits, actual.logits, 'ordinary-position logits')
        for name in reference.captures:
            _equal(reference.captures[name], actual.captures[name], 'ordinary capture ' + name)
        # Identical same-shape scalar probe gives exact full backward parity.
        reference.logits.square().sum().backward()
        actual.logits.square().sum().backward()
        for (name, param), (other, value) in zip(native.named_parameters(), explicit.named_parameters()):
            if name != other or param.grad is None or value.grad is None:
                raise ContractError('gradient inventory mismatch')
            _equal(param.grad, value.grad, 'ordinary-position gradient ' + name)
        before = model_identity(explicit)
        with torch.no_grad():
            baseline = explicit(ids, mask, position_ids=coupled).logits
            changed = ids.clone()
            changed[:, 8:11] = ord('9')
            future = explicit(changed, mask, position_ids=coupled).logits
            _equal(baseline[:, :8], future[:, :8], 'coupled causal prefix')
            padded_ids = ids.clone()
            padded_positions = coupled.clone()
            padded_ids[~mask] = 17
            padded_positions[~mask] = 63
            padded = explicit(padded_ids, mask, position_ids=padded_positions).logits
            _equal(baseline, padded, 'masked token/position isolation')
        if before != model_identity(explicit):
            raise ContractError('forward observation changed model tensors')
        _equal(rng, torch.random.get_rng_state(), 'caller RNG')
        records.append({'seed': seed, 'initial_tensor_identity': before,
                        'ordinary_logits_identity': tensor_identity(actual.logits),
                        'coupled_logits_identity': tensor_identity(baseline),
                        'ordinary_logits_captures_gradients_exact': True,
                        'causal_prefix_exact': True, 'right_padding_exact': True,
                        'caller_rng_unchanged': True, 'parameter_count': CONFIG.parameter_count})
    if require_environment() != environment:
        raise ContractError('environment changed during conformance')
    return {'model_id': MODEL_ID, 'position_schema': POSITION_SCHEMA,
            'modes_checked': ['sequence', 'coupled'], 'offset': 0,
            'architecture': CONFIG.to_dict(), 'synthetic_prompts': list(prompts),
            'synthetic_answers_reversed_padded': list(answers),
            'input_ids': ids.tolist(), 'attention_mask': mask.tolist(),
            'sequence_position_ids': sequence.tolist(), 'coupled_position_ids': coupled.tolist(),
            'position_binding': identity(json_bytes({'schema': POSITION_SCHEMA,
                'sequence': sequence.tolist(), 'coupled': coupled.tolist()})),
            'environment': environment, 'cells': records,
            'target_training_run': False, 'checkpoint_import': False,
            'learning_claim': False, 'closed_custody': False}
