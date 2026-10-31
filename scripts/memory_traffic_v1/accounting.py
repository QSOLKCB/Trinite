"""Integer-only tensor accounting for the pinned uncached native decoder."""
from pathlib import Path
import sys
from trinite.contracts import ContractError, ModelConfig, exact_keys, identity, integer, json_bytes

SCHEMA = 'trinite.memory-traffic.v1'
ROOT = Path(__file__).resolve().parents[2]
BASELINE = {
    'src/trinite/model.py': 'e8f17ea196c253d390437c24eb51750b91a455e84685d0c905569bce548f0350',
    'src/trinite/quantizer.py': '2ac17f3a204df03870e5a21f454eb04d8c31e2ce7ad5ee243df823d1bccc9be9',
    'src/trinite/contracts.py': '36562611c6d60b4426a28e0974cda77b81a305186768f69d85bbbaf12a355d24',
}


def source_binding():
    """Fail closed if formula-bearing native sources change; never refresh pins."""
    result = {}
    for module_name in ('trinite', 'trinite.contracts', 'trinite.model', 'trinite.quantizer'):
        module = sys.modules.get(module_name)
        if module is not None:
            suffix = '__init__.py' if module_name == 'trinite' else module_name.split('.')[-1] + '.py'
            if getattr(module, '__file__', None) != str(ROOT / 'src/trinite' / suffix):
                raise ContractError('foreign imported native module: ' + module_name)
    for name, digest in BASELINE.items():
        path = ROOT / name
        if path.is_symlink() or identity(path.read_bytes()) != 'sha256:' + digest:
            raise ContractError('unsupported native source: ' + name)
        result[name] = 'sha256:' + digest
    for name in ('scripts/memory_traffic_v1/__init__.py',
                 'scripts/memory_traffic_v1/accounting.py',
                 'scripts/memory_traffic_v1/native.py',
                 'scripts/report_memory_traffic.py', 'src/trinite/__init__.py',
                 'requirements.lock', 'docs/MEMORY-TRAFFIC.md'):
        path = ROOT / name
        if path.is_symlink():
            raise ContractError('linked accounting source: ' + name)
        result[name] = identity(path.read_bytes())
    return result


def admit(value):
    exact_keys(value, {'config', 'lane', 'batch', 'prompt_tokens', 'forward_calls'}, 'memory plan')
    config = ModelConfig.from_dict(value['config'])
    if type(value['lane']) is not str or value['lane'] not in ('dense', 'ternary', 'four-state'):
        raise ContractError('unsupported lane')
    integer(value['batch'], 'batch', 1, 8)
    integer(value['prompt_tokens'], 'prompt_tokens including BOS', 1, config.context_length)
    integer(value['forward_calls'], 'forward_calls', 1, 16)
    last = value['prompt_tokens'] + value['forward_calls'] - 1
    if last > config.context_length:
        raise ContractError('full prefix schedule exceeds native context')
    if config.parameter_count > 5_000_000:
        raise ContractError('native parameter budget exceeded')
    if max(config.width**2, config.width*config.feed_forward_width) > 2_000_000:
        raise ContractError('native quantizer matrix budget exceeded')
    if value['batch'] * config.heads * last**2 > 4_194_304:
        raise ContractError('native attention cell budget exceeded')
    return config


def parameter_accounting(config, lane):
    d, f, n = config.width, config.feed_forward_width, config.blocks
    linear = n * (4*d*d + 2*d*f)
    other = config.parameter_count - linear
    packed = n * (4*((d*d+3)//4) + 2*((d*f+3)//4))
    return {
        'evidence_class': 'analytic_tensor_payload_not_observation',
        'unique_parameters': config.parameter_count,
        'linear_parameters': linear,
        'other_parameters': other,
        'native_float32_parameter_bytes': 4*config.parameter_count,
        'native_linear_parameter_bytes': 4*linear,
        'native_other_parameter_bytes': 4*other,
        'tied_embedding_count': 1,
        'hypothetical_fixed_2bit_ternary': None if lane != 'ternary' else {
            'matrix_code_bytes': packed, 'float32_scale_bytes': 6*n*4,
            'other_float32_bytes': 4*other,
            'payload_bytes': packed + 6*n*4 + 4*other,
            'scope': 'per-matrix ceil(elements/4), one scale per matrix; excludes format metadata; not resident native storage',
        },
    }


def forward_accounting(config, batch, length, lane):
    """Selected logical payloads; overlap/lifetimes preclude adding into a peak."""
    d, f, n, h = config.width, config.feed_forward_width, config.blocks, config.heads
    hidden = batch*length*d*4
    square = batch*h*length*length*4
    linear = n*(4*d*d+2*d*f)
    return {
        'prefix_tokens': length,
        'processed_positions': batch*length,
        'selected_activation_payload_bytes': {
            'q_outputs_all_blocks': n*hidden,
            'k_outputs_all_blocks': n*hidden,
            'v_outputs_all_blocks': n*hidden,
            'o_outputs_all_blocks': n*hidden,
            'ffn_down_outputs_all_blocks': n*hidden,
            'attention_scores_all_blocks': n*square,
            'attention_probabilities_all_blocks': n*square,
            'ffn_up_outputs_all_blocks': n*batch*length*f*4,
            'ffn_gelu_outputs_all_blocks': n*batch*length*f*4,
            'logits': batch*length*config.vocabulary_size*4,
        },
        'logical_operand_payload_bytes': {
            'linear_weight_once_per_operator': linear*4,
            'tied_output_weight_once': config.vocabulary_size*d*4,
            'qk_key_operand_once': n*hidden,
            'pv_value_operand_once': n*hidden,
        },
        'quantizer_materializations': {
            'effective_float32_weights_all_calls_bytes': 0 if lane == 'dense' else linear*4,
            'int8_codes_all_calls_bytes': 0 if lane == 'dense' else linear,
            'float32_scales_all_calls_bytes': 0 if lane == 'dense' else 6*n*4,
            'complete_temporary_peak_bytes': None,
            'reason': 'Selected products only; float64 reduction, unit/mask/conversion intermediates and lifetimes are not modeled.',
        },
        'native_cache': {'mode': 'unsupported', 'resident_payload_bytes': None,
                         'read_bytes': None, 'write_bytes': None},
    }


def report(value):
    config = admit(value)
    sources = source_binding()
    batch, start, calls = value['batch'], value['prompt_tokens'], value['forward_calls']
    rows, cache = [], []
    for step in range(calls):
        length = start + step
        rows.append(forward_accounting(config, batch, length, value['lane']))
        # Hypothetical prefill produces all prompt K/V; subsequent calls append
        # exactly one token, then attend to every cached key/value including it.
        appended = start if step == 0 else 1
        cache.append({
            'phase': 'prefill' if step == 0 else 'incremental_decode',
            'cached_tokens': length, 'appended_tokens': appended,
            'occupied_kv_payload_bytes': 2*config.blocks*batch*length*config.width*4,
            'kv_append_payload_bytes': 2*config.blocks*batch*appended*config.width*4,
            'dense_kv_operand_read_once_bytes': 2*config.blocks*batch*length*config.width*4,
        })
    return {
        'schema': SCHEMA,
        'evidence_class': 'source_bound_analytic_accounting_not_hardware_measurement',
        'plan': value, 'plan_identity': identity(json_bytes(value)), 'source_identities': sources,
        'parameters': parameter_accounting(config, value['lane']),
        'native_full_prefix_forwards': rows,
        'schedule': {'forward_calls': calls,
                     'processed_positions': sum(r['processed_positions'] for r in rows),
                     'k_output_bytes_all_calls': sum(r['selected_activation_payload_bytes']['k_outputs_all_blocks'] for r in rows),
                     'v_output_bytes_all_calls': sum(r['selected_activation_payload_bytes']['v_outputs_all_blocks'] for r in rows)},
        'hypothetical_dense_float32_cache': {
            'implemented': False, 'element_bytes': 4, 'rows': cache,
            'scope': 'Ordinary multi-head attention; current token included in reads; operand once, not per-query/physical traffic. Excludes metadata, capacity slack, copies, workspace and cache maintenance.',
        },
        'physical_dram_traffic_bytes': None,
        'peak_live_activation_bytes': None,
        'limits': [
            'No cache exists in this native path. No model, checkpoint or tensor is loaded by this report.',
            'All schedule rows are planned full-prefix forwards, not observed generation or EOS behavior.',
            'Logical tensor/operand bytes are not hardware traffic, elapsed time, energy or peak memory.',
            'Selected activation categories overlap and exclude many intermediates; do not sum into RSS or a peak.',
            'Padding does not shrink allocated shapes. Native weights and activations remain float32 in every lane.',
            'No sparse page/token selection, cache quantization, eviction, packed execution or SPARTA speedup is implemented.',
        ],
    }
