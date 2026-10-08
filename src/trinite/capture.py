"""Bounded prompt-only float32 capture artifacts and independent byte/span audit."""
import math
import struct

import torch

from .contracts import ContractError, exact_keys, identity, json_bytes, parse_json, require_identity
from .inspection import inventory, tensor_identity
from .model import CaptureSpec, Decoder
from .tokenizer import ByteTokenizer

CAPTURE_SCHEMA = 'trinite.prompt-capture.v1'
MAX_BYTES = 2 * 1024 * 1024


def capture_prompt(model: Decoder, prompt: str, *, layers=('final', 'block.0')):
    if type(layers) is not tuple or layers != ('final', 'block.0'):
        raise ContractError('Phase 4 instrument requires the frozen final/block.0 selection')
    encoding = ByteTokenizer(model.config.context_length).encode(prompt)
    positions = tuple(range(1, len(encoding.input_ids)-1))
    if not positions: raise ContractError('capture requires nonempty prompt bytes')
    spec = CaptureSpec(layers, positions, 65536)
    ids = torch.tensor([encoding.input_ids], dtype=torch.int64, device='cpu')
    mask = torch.tensor([encoding.attention_mask], dtype=torch.bool, device='cpu')
    with torch.no_grad(): output = model(ids, mask, capture=spec)
    record = {'schema': CAPTURE_SCHEMA, 'prompt': prompt, 'prompt_identity': identity(prompt.encode()),
              'config': model.config.to_dict(), 'config_identity': model.config.content_identity,
              'lane': model.lane, 'model_inventory_identity': inventory(model)['inventory_identity'],
              'tokenizer': encoding.to_dict(), 'positions': list(positions), 'layers': list(layers),
              'context_mode': 'cumulative-causal-prefix', 'pooling': 'none', 'transform': 'identity',
              'step_unit': 'UTF-8-byte-token', 'dtype': 'float32', 'device': 'cpu',
              'logits_identity': tensor_identity(output.logits),
              'trajectories': {}}
    for layer, tensor in output.captures.items():
        if tensor.requires_grad or tensor.grad_fn is not None: raise ContractError('capture is not detached')
        value = tensor[0].contiguous()
        record['trajectories'][layer] = {'shape': list(value.shape), 'tensor_identity': tensor_identity(value),
                                        'coordinates_hex': [[float(x).hex() for x in row] for row in value.tolist()]}
    raw = json_bytes(record)
    audit_capture(raw)
    return raw


def audit_capture(raw: bytes):
    if type(raw) is not bytes or len(raw) > MAX_BYTES: raise ContractError('capture exceeds 2 MiB')
    r = exact_keys(parse_json(raw, canonical=True), {'schema', 'prompt', 'prompt_identity', 'config', 'config_identity',
        'tokenizer', 'positions', 'layers', 'context_mode', 'pooling', 'transform', 'step_unit', 'dtype', 'device',
        'logits_identity', 'trajectories', 'lane', 'model_inventory_identity'}, 'capture')
    from .contracts import ModelConfig
    config = ModelConfig.from_dict(r['config'])
    encoding = ByteTokenizer(config.context_length).encode(r['prompt'])
    expected = {'schema': CAPTURE_SCHEMA, 'prompt_identity': identity(r['prompt'].encode()),
                'config_identity': config.content_identity, 'tokenizer': encoding.to_dict(),
                'positions': list(range(1, len(encoding.input_ids)-1)), 'layers': ['final', 'block.0'],
                'context_mode': 'cumulative-causal-prefix', 'pooling': 'none', 'transform': 'identity',
                'step_unit': 'UTF-8-byte-token', 'dtype': 'float32', 'device': 'cpu'}
    if any(json_bytes(r[k]) != json_bytes(v) for k, v in expected.items()) or not r['positions']:
        raise ContractError('capture selection/config/span/context mismatch')
    if config.blocks < 1: raise ContractError('block.0 capture requires one block')
    require_identity(r['logits_identity'])
    require_identity(r['model_inventory_identity'])
    if r['lane'] not in ('dense', 'ternary'): raise ContractError('invalid capture lane')
    exact_keys(r['trajectories'], set(r['layers']), 'layer captures')
    trajectories = {}
    for layer, t in r['trajectories'].items():
        exact_keys(t, {'shape', 'tensor_identity', 'coordinates_hex'}, 'trajectory')
        shape = [len(r['positions']), config.width]
        rows = t['coordinates_hex']
        if (json_bytes(t['shape']) != json_bytes(shape) or type(rows) is not list or len(rows) != shape[0]
                or any(type(row) is not list or len(row) != shape[1] for row in rows)):
            raise ContractError('capture shape mismatch')
        points, payload = [], bytearray()
        for row in rows:
            values = []
            for text in row:
                try:
                    if type(text) is not str: raise ValueError('not a float hex string')
                    value = float.fromhex(text)
                    packed = struct.pack('<f', value)
                    if not math.isfinite(value) or value.hex() != text or struct.unpack('<f', packed)[0] != value:
                        raise ValueError('not exact finite float32')
                except (ValueError, OverflowError, struct.error) as error:
                    raise ContractError('invalid capture coordinate') from error
                values.append(value); payload.extend(packed)
            points.append(values)
        if identity(bytes(payload)) != t['tensor_identity']: raise ContractError('capture tensor identity mismatch')
        trajectories[layer] = points
    return r, trajectories
