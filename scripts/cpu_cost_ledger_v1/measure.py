"""Observe the unchanged decoder, keeping timing out of parity bytes."""
import resource
import time

import torch

from trinite.comparison import greedy
from trinite.comparison_checkpoint import load
from trinite.contracts import ContractError, ModelConfig, identity, json_bytes
from trinite.model import Decoder
from trinite.training import model_identity
from .request import LIMITS, plan, read


def clock():
    return time.perf_counter_ns(), time.process_time_ns()


def elapsed(start):
    wall, cpu = clock()
    return {'wall_ns': wall-start[0], 'process_cpu_ns': cpu-start[1]}


def load_model(path, value):
    """Exact new model-only payload; no optimizer construction or pickle."""
    plan(value)  # predictive bounds before file read, tensor decode or model allocation
    raw = read(path, LIMITS['checkpoint_bytes'])
    if identity(raw) != value['checkpoint_identity']:
        raise ContractError('checkpoint identity differs')
    config = ModelConfig.from_dict(value['config'])
    model = Decoder(config, lane=value['lane'], seed=0)
    # Authoritative safe named-tensor reader, never torch.load.
    tensors = load(raw)
    parameters = dict(model.named_parameters())
    if set(tensors) != {'model.'+n for n in parameters}:
        raise ContractError('model-only checkpoint inventory differs; optimizer payloads are not accepted')
    with torch.no_grad():
        for name, parameter in parameters.items():
            tensor = tensors['model.'+name]
            if (tensor.device.type != 'cpu' or tensor.dtype != torch.float32
                    or tensor.layout != torch.strided or tensor.shape != parameter.shape
                    or not bool(torch.isfinite(tensor).all())):
                raise ContractError('invalid finite CPU model tensor: '+name)
            parameter.copy_(tensor)
    model.eval(); model.validate_state()
    if model_identity(model) != value['model_identity']:
        raise ContractError('checkpoint tensor/model identity differs')
    return model, len(raw)


def unique_storage_bytes(model):
    storages = {p.untyped_storage().data_ptr(): p.untyped_storage().nbytes() for p in model.parameters()}
    return sum(storages.values())


def prediction(tokens, cap):
    return {'tokens': tokens, 'output_tokens': len(tokens), 'forward_calls': len(tokens),
            'stop_reason': 'eos' if tokens and tokens[-1] == 257 else 'output_cap' if len(tokens) == cap else 'context_cap'}


class Meter:
    """Transparent callable proxy; no monkeypatch, hooks or changed tensor math."""
    def __init__(self, model, checkpoint_identity, maximum_calls, maximum_positions):
        self.model, self.config = model, model.config
        self.checkpoint_identity = checkpoint_identity
        self.maximum_calls, self.maximum_positions = maximum_calls, maximum_positions
        self.rows = []; self.positions = 0
        self.audit_wall_ns = 0; self.audit_cpu_ns = 0

    def __call__(self, ids, mask):
        start = clock()
        n = ids.numel()
        if len(self.rows) >= self.maximum_calls or self.positions+n > self.maximum_positions:
            raise ContractError('forward/processed-position budget exceeded before forward')
        prefix = ids[0].tolist()
        binding = {'checkpoint_identity': self.checkpoint_identity,
                   'prefix_identity': identity(json_bytes(prefix)),
                   'mask_identity': identity(json_bytes(mask.tolist())),
                   'absolute_positions': list(range(ids.shape[1])), 'input_tokens': n,
                   'phase': 'prompt_forward' if not self.rows else 'decode_full_prefix'}
        overhead = elapsed(start)
        self.audit_wall_ns += overhead['wall_ns']; self.audit_cpu_ns += overhead['process_cpu_ns']
        start = clock()
        output = self.model(ids, mask)
        timing = elapsed(start)
        start = clock()
        if not bool(torch.isfinite(output.logits).all()):
            raise ContractError('nonfinite model logits')
        self.positions += n
        self.rows.append({**binding, **timing})
        overhead = elapsed(start)
        self.audit_wall_ns += overhead['wall_ns']; self.audit_cpu_ns += overhead['process_cpu_ns']
        return output


def measure(model, value, on_row=None):
    bounds = plan(value)
    if model.config.to_dict() != value['config'] or model.lane != value['lane']:
        raise ContractError('model config/lane differs')
    cap = value['output_tokens']; rows = []; calls = positions = 0
    start = clock(); initial = model_identity(model); identity_cost = elapsed(start)
    if initial != value['model_identity'] or any(p.grad is not None for p in model.parameters()):
        raise ContractError('model identity/gradient differs before inference')
    for index, prompt in enumerate(value['prompts']):
        length = 1 + len(prompt.encode('utf-8'))
        for repetition in range(value['repetitions']):
            # Admit the complete trio before any forward; early EOS only reduces actual use.
            worst_calls = 3 * cap
            worst_positions = 3 * (cap*length + cap*(cap-1)//2)
            if (calls+worst_calls > bounds['maximum_forward_calls'] or
                    positions+worst_positions > bounds['maximum_processed_positions']):
                raise ContractError('request compute budget exceeded')
            before_start = clock(); before = greedy(model, prompt, cap); before_time = elapsed(before_start)
            meter = Meter(model, value['checkpoint_identity'], cap, cap*length + cap*(cap-1)//2)
            measured_start = clock(); observed = greedy(meter, prompt, cap); measured_time = elapsed(measured_start)
            after_start = clock(); after = greedy(model, prompt, cap); after_time = elapsed(after_start)
            trio = [prediction(t, cap) for t in (before, observed, after)]
            actual_calls = sum(len(t) for t in (before, observed, after))
            actual_positions = sum(len(t)*length + len(t)*(len(t)-1)//2 for t in (before, observed, after))
            calls += actual_calls; positions += actual_positions
            row = {'prompt_index': index, 'repetition': repetition, 'prompt_binding': bounds['prompt_bindings'][index],
                   'before': trio[0], 'measured': trio[1], 'after': trio[2],
                   'parity': trio[0] == trio[1] == trio[2],
                   'first_parity_before_generation_in_worker': index == 0 and repetition == 0,
                   'measured_cache_state': 'warmed by preceding parity pass; OS/CPU caches uncontrolled',
                   'instrumented_end_to_end': measured_time, 'parity_before': before_time, 'parity_after': after_time,
                   'forward_observations': meter.rows,
                   'instrumentation_audit': {'wall_ns': meter.audit_wall_ns, 'process_cpu_ns': meter.audit_cpu_ns},
                   'all_passes_forward_calls': actual_calls, 'all_passes_processed_positions': actual_positions}
            if on_row: on_row(row)  # preserve mismatch and earlier rows before failing
            if not row['parity'] or len(meter.rows) != len(observed):
                raise ContractError('instrumented/uninstrumented output parity failed')
            rows.append(row)
    start = clock(); final = model_identity(model); final_identity_cost = elapsed(start)
    if final != initial or any(p.grad is not None for p in model.parameters()):
        raise ContractError('inference changed model/gradients')
    return {'rows': rows, 'all_passes_forward_calls': calls, 'all_passes_processed_positions': positions,
            'model_identity': initial, 'identity_audit_before': identity_cost, 'identity_audit_after': final_identity_cost,
            'unique_resident_parameter_storage_bytes': unique_storage_bytes(model),
            'worker_lifetime_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'rss_scope': 'Linux ru_maxrss whole fresh worker lifetime so far, including imports/load/parity/audit; not forward transient peak or enforced ceiling',
            'transient_forward_peak_bytes': None, 'transient_forward_peak_reason': 'not measured; lifetime RSS cannot isolate transient allocations',
            'sampled_process_tree_rss_bytes': None, 'process_tree_reason': 'not sampled; single worker only',
            'runtime_overhead_bytes': None, 'runtime_overhead_reason': 'not isolated from overlapping process RSS',
            'native_kv_payload_bytes': None, 'native_kv_reason': 'unsupported by unchanged decoder',
            'cache_mode': 'unsupported', 'memory_ceiling_bytes': None,
            'training_updates': 0, 'speed_claim': False, 'learning_adequate': False}
