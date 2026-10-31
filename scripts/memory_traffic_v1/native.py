"""Opt-in metadata observation around one unchanged CPU inference forward."""
import torch
from trinite.contracts import ContractError
from trinite.model import Decoder, Linear
from .accounting import admit, forward_accounting, source_binding


def observe_forward(model, input_ids, attention_mask):
    """Return (native output, metadata). Caller owns model/input authorization.

    Does not change eval/grad mode, invoke an extra forward, retain tensor outputs
    in hooks, or load a checkpoint. Not a timing/DRAM profiler or run receipt.
    """
    sources = source_binding()
    if type(model) is not Decoder or model.training or torch.is_grad_enabled():
        raise ContractError('observation requires exact native Decoder in eval and no-grad/inference mode')
    # Use authoritative input/state admission before installing hooks.
    model._validate_inputs(input_ids, attention_mask)
    batch, length = input_ids.shape
    admit({'config': model.config.to_dict(), 'lane': model.lane,
           'batch': batch, 'prompt_tokens': length, 'forward_calls': 1})
    parameters = list(model.named_parameters())
    storages = {}
    for _, tensor in parameters:
        storage = tensor.untyped_storage()
        storages[storage.data_ptr()] = storage.nbytes()
    outputs, handles = [], []

    def hook(name):
        def record(module, args, output):
            # Scalar metadata only; never save output or detach/copy its payload.
            outputs.append({'name': name, 'shape': list(output.shape),
                            'dtype': str(output.dtype),
                            'logical_output_bytes': output.numel()*output.element_size()})
        return record

    try:
        for name, module in model.named_modules():
            if type(module) is Linear:
                handles.append(module.register_forward_hook(hook(name)))
        output = model(input_ids, attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    return output, {
        'schema': 'trinite.memory-traffic-observation.v1',
        'source_identities': sources, 'torch_version': str(torch.__version__),
        'evidence_class': 'in_process_tensor_metadata_not_hardware_traffic',
        'unique_parameter_storage_bytes': sum(storages.values()),
        'unique_parameter_tensor_payload_bytes': sum(p.numel()*p.element_size() for _, p in parameters),
        'linear_outputs': outputs,
        'logits_logical_bytes': output.logits.numel()*output.logits.element_size(),
        'shape_accounting': forward_accounting(model.config, batch, length, model.lane),
        'physical_dram_traffic_bytes': None,
        'peak_live_activation_bytes': None,
        'native_cache_mode': 'unsupported',
        'native_cache_payload_bytes': None,
    }
