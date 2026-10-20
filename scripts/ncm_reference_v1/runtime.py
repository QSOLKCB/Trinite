"""Explicit optional CUDA boundary. Never imported by native Trinite or CPU CI."""
import io
import importlib.metadata
import json
import math
import platform
import sys
import subprocess
import time
from pathlib import Path
from .core import ROOT, LEVELS, digest, encode_board, read_bytes, score, verify_upstream, verify_weights

VERSIONS = {'torch': '2.8.0+cu128', 'numpy': '2.3.5', 'omegaconf': '2.3.0', 'hydra-core': '1.3.2'}


def preflight(torch, protocol):
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    if versions != VERSIONS:
        raise ValueError('external dependency version mismatch: ' + repr(versions))
    if sys.version_info[:2] != (3, 11):
        raise ValueError('external lane requires Python 3.11')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('exactly one visible CUDA GPU required; no CPU fallback')
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    if protocol['target_device_name_contains'] not in props.name:
        raise ValueError('wrong GPU: ' + props.name)
    if total < protocol['minimum_total_bytes'] or free < protocol['minimum_free_bytes']:
        raise ValueError('insufficient GPU memory')
    if tuple(torch.cuda.get_device_capability(0)) != (12, 0):
        raise ValueError('expected Blackwell sm_120')
    if 'sm_120' not in torch.cuda.get_arch_list():
        raise ValueError('installed torch has no native sm_120 support')
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(protocol['max_allocator_bytes']/total, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(protocol['seed'])
    torch.cuda.manual_seed_all(protocol['seed'])
    driver = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    if not driver:
        raise ValueError('missing driver identification')
    return dict(driver=driver, installed_packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
                versions=versions, python=sys.version, platform=platform.platform(),
                torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                device=props.name, capability=list(torch.cuda.get_device_capability(0)),
                total_memory=total, torch_build=torch.__config__.show())


def safe_state(torch, path, pin):
    data = read_bytes(path, pin['size'])
    if len(data) != pin['size'] or digest(data) != pin['sha256']:
        raise ValueError('checkpoint changed before load')
    # No unsafe fallback, custom allowlist, or arbitrary class construction.
    obj = torch.load(io.BytesIO(data), weights_only=True, map_location='cpu')
    if type(obj) is not dict or 'model' not in obj or not isinstance(obj['model'], dict):
        raise ValueError('expected checkpoint model state dictionary')
    state = obj['model']
    if not state or len(state) > 5000:
        raise ValueError('invalid state cardinality')
    for name, tensor in state.items():
        if type(name) is not str or not isinstance(tensor, torch.Tensor):
            raise ValueError('non-tensor state entry')
        if tensor.numel() > 50_000_000 or tensor.layout != torch.strided:
            raise ValueError('unbounded or nonstrided tensor')
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError('nonfinite checkpoint tensor')
    return state


def load_models(torch, paths):
    pin = verify_upstream()
    vendor = ROOT/'vendor'
    if list(vendor.rglob('*.pyc')):
        raise ValueError('unverified upstream bytecode cache')
    if any(n == 'models' or n.startswith('models.') or n == 'utils' or n.startswith('utils.') for n in sys.modules):
        raise ValueError('upstream import namespace already occupied')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(vendor))
    from omegaconf import OmegaConf
    from models.vqvae import VQVAE
    from models.next_concept import NextConceptTransformer
    cfg = OmegaConf.load(ROOT/'config.yaml')
    codec = VQVAE(**cfg.tokenizer.vqvae)
    codec.load_state_dict(safe_state(torch, paths['tokenizer.pt'], pin['weights']['tokenizer.pt']), strict=True)
    net = NextConceptTransformer(vqvae_local=codec, config=cfg.model)
    net.load_state_dict(safe_state(torch, paths['generator.pt'], pin['weights']['generator.pt']), strict=True)
    for name, module in list(sys.modules.items()):
        if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
            file = Path(module.__file__).resolve()
            relative = str(file.relative_to(vendor.resolve()))
            if relative not in pin['files'] or digest(read_bytes(file, 100000)) != pin['files'][relative]:
                raise ValueError('unverified imported upstream module')
    if tuple(codec.vq.levels) != LEVELS:
        raise ValueError('upstream level mismatch')
    for model in (codec, net):
        model.eval().requires_grad_(False).to(device='cuda:0', dtype=torch.float32)
    return net, codec


def infer(torch, net, codec, clues):
    """Only clues cross this boundary. No target, solver, repair or retries."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    prefix = [11] + encode_board(clues) + [11]
    x = torch.tensor([prefix], dtype=torch.long, device='cuda:0')
    torch.cuda.synchronize()
    prepared = time.perf_counter()
    latent = net.generate(None, x, temperature=0.0, top_k=0, top_p=1.0,
                          cfg_scale=0.0, cfg_null_eos=None, use_kv_cache=False)
    if tuple(latent.shape) != (1, sum(LEVELS)):
        raise ValueError('unexpected latent shape')
    if bool(((latent < 0) | (latent >= 16)).any()):
        raise ValueError('invalid latent codes')
    torch.cuda.synchronize()
    generated = time.perf_counter()
    _, levels = codec.vq.idx_to_zhats(latent, return_zlevels=True)
    logits = codec.decoder(levels[-1])
    if tuple(logits.shape) != (1, 89, 12) or not bool(torch.isfinite(logits).all()):
        raise ValueError('invalid decoder output')
    output = logits.argmax(-1).cpu().tolist()[0]
    codes = latent.cpu().tolist()[0]
    torch.cuda.synchronize()
    finished = time.perf_counter()
    return dict(raw_ids=output, latent_ids=codes), dict(
        prepare_seconds=prepared-start, generator_seconds=generated-prepared,
        codec_and_transfer_seconds=finished-generated, total_seconds=finished-start)


def execute(req, checkpoint_dir, progress=lambda value: None):
    protocol = req['protocol']
    paths = verify_weights(checkpoint_dir)
    import torch
    environment = preflight(torch, protocol)
    torch.cuda.synchronize()
    begin = time.perf_counter()
    net, codec = load_models(torch, paths)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter()-begin
    # Keep construction/loading peaks as well as measured inference allocations.
    if torch.cuda.max_memory_reserved() > protocol['max_allocator_bytes']:
        raise ValueError('GPU allocator budget exceeded during load')
    progress(dict(stage='loaded', environment=environment, load_seconds=load_seconds))
    outcomes, timings, reconstructions = [], [], []
    with torch.inference_mode():
        for phase, item in [('cold', req['inputs'][0]), ('warmup', req['inputs'][0])] + [('warm', x) for x in req['inputs']]:
            output, timing = infer(torch, net, codec, item['clues'])
            outcomes.append(dict(phase=phase, id=item['id'], **output,
                                 metrics=score(output['raw_ids'], item['clues'], item['target'])))
            timings.append(dict(phase=phase, id=item['id'], **timing))
            progress(dict(outcome=outcomes[-1], timing=timings[-1]))
        # Codec diagnostic AFTER all clue-only generation. Never feed gold codes to net.
        for item in req['inputs']:
            torch.cuda.synchronize()
            begin = time.perf_counter()
            gold = torch.tensor([encode_board(item['target'])], dtype=torch.long, device='cuda:0')
            codes = codec.encode(gold)
            _, levels = codec.vq.idx_to_zhats(codes, return_zlevels=True)
            logits = codec.decoder(levels[-1])
            if tuple(logits.shape) != (1, 89, 12) or not bool(torch.isfinite(logits).all()):
                raise ValueError('invalid reconstruction logits')
            ids = logits.argmax(-1).cpu().tolist()[0]
            torch.cuda.synchronize()
            timings.append(dict(phase='gold-codec-diagnostic', id=item['id'], total_seconds=time.perf_counter()-begin))
            reconstructions.append(dict(id=item['id'], raw_ids=ids,
                                       metrics=score(ids, item['clues'], item['target'])))
            progress(dict(reconstruction= reconstructions[-1], timing=timings[-1]))
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    if max(allocated, reserved) > protocol['max_allocator_bytes']:
        raise ValueError('GPU allocator budget exceeded')
    if any(p.grad is not None or p.requires_grad for model in (codec, net) for p in model.parameters()):
        raise ValueError('inference immutability flag failure')
    return dict(schema='ncm-reference-result-v1', environment=environment, outcomes=outcomes,
                reconstructions=reconstructions, evidence_class='actual-local-cuda-inference'), dict(
        load_seconds=load_seconds, measurements=timings, max_allocated_bytes=allocated,
        max_reserved_bytes=reserved, memory_scope='PyTorch allocator, not whole-device VRAM')
