"""NONCERTIFYING fixed public-fit forward diagnostic; never a native restore.

Only reviewed local Decoder source executes. Evidence ZIPs are data, never imports.
This deliberately does not import native training/checkpoint/gate controllers.
"""
import argparse
import hashlib
import io
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import stat
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PINS = Path(__file__).with_suffix('.inputs.json')
PINS_SHA256 = '09c23ba7a6053c5bf4d744bc6e3c50d512fc62c6f3698d859fd313793bef1d03'
SCOPE = ('NONCERTIFYING exploratory foreign-host forward comparison of eight already '
         'exposed public training rows. Cannot authorize fit gate, main runs or research '
         'release. Not optimizer replay, blindness, generalization or a causal claim.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw):
    require(type(raw) is bytes and len(raw) <= 1 << 20, 'JSON byte bound')
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('nonfinite JSON constant')
    def number(value):
        result = float(value)
        require(math.isfinite(result), 'nonfinite JSON number')
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=number)


def keys(value, expected):
    require(type(value) is dict and set(value) == set(expected), 'closed schema keys')


def read_file(path, maximum):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), 'regular non-symlink file required')
    with path.open('rb') as handle:
        require(stat.S_ISREG(os.fstat(handle.fileno()).st_mode), 'regular file required')
        raw = handle.read(maximum + 1)
    require(len(raw) <= maximum, 'file byte bound')
    return raw


def pins():
    raw = read_file(PINS, 1 << 20)
    require(digest(raw) == PINS_SHA256, 'fixed input specification hash mismatch')
    value = strict_json(raw)
    keys(value, ('schema', 'prior', 'current', 'rows', 'runtime_sources'))
    require(value['schema'] == 'trinite.forward-checkpoint-inputs.v1', 'input schema')
    for label in ('prior', 'current'):
        keys(value[label], ('payload_sha256', 'members', 'model_config'))
    for name, expected in value['runtime_sources'].items():
        require(digest(read_file(ROOT / name, 1 << 20)) == expected, 'runtime source mismatch: ' + name)
    validate_rows(value['rows'])
    return value


def validate_rows(rows):
    require(type(rows) is list and len(rows) == 8, 'exact eight train fit rows')
    require(len({r.get('identity') for r in rows if type(r) is dict}) == 8, 'distinct row IDs')
    for row in rows:
        keys(row, ('identity', 'answer', 'ids', 'mask', 'loss'))
        require(type(row['identity']) is str and row['answer'] in ('-1', '0', '1'), 'row identity/answer')
        ids, mask, loss = (row[n] for n in ('ids', 'mask', 'loss'))
        require(all(type(v) is list for v in (ids, mask, loss)), 'row arrays')
        require(4 <= len(ids) <= 256 and len(ids) == len(mask) == len(loss), 'row lengths')
        require(all(type(v) is int and 0 <= v < 259 for v in ids), 'token type/range')
        require(all(type(v) is int and v == 1 for v in mask), 'unmasked row required')
        target = [*row['answer'].encode('ascii'), 257]
        count = len(target)
        require(ids[0] == 256 and ids[-count:] == target, 'answer+EOS tokens')
        require(loss == [0] * (len(ids)-count) + [1] * count
                and all(type(v) is int for v in loss), 'target-only loss mask')
        require(all(v < 256 for v in ids[1:-count]) and len(ids)-count+3 <= 256, 'prompt/context')


def read_deposit(path, pin):
    raw = read_file(path, 8 << 20)
    require(digest(raw) == pin['payload_sha256'], 'pinned deposit hash mismatch')
    result = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        names = [i.filename for i in infos]
        require(len(names) <= 1024 and len(names) == len(set(names)), 'ZIP inventory bound/duplicates')
        require(sum(i.file_size for i in infos) <= 32 << 20, 'ZIP expanded byte bound')
        for info in infos:
            p = PurePosixPath(info.filename)
            require(bool(p.parts) and not p.is_absolute() and '..' not in p.parts and '\\' not in info.filename
                    and ':' not in info.filename and '\x00' not in info.filename
                    and str(p) == info.filename.rstrip('/'), 'unsafe ZIP path')
            require(not stat.S_ISLNK(info.external_attr >> 16), 'ZIP symlink')
            require(info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                    and not info.flag_bits & 1, 'ZIP compression/encryption')
        # No extraction; no source, oracle binary, public inventory or heldout file is read.
        for name, expected in pin['members'].items():
            require(name in names and archive.getinfo(name).file_size <= 1 << 20, 'member byte bound')
            data = archive.read(name)
            require(digest(data) == expected, 'checkpoint member hash mismatch')
            result[name] = data
    return result


def bind_imports():
    # Refuse a preloaded foreign package; choose reviewed local code, never ZIP code.
    pins()
    for name in ('trinite', 'trinite.contracts', 'trinite.model', 'trinite.quantizer'):
        expected = ROOT / 'src' / (name.replace('.', '/') + ('/__init__.py' if name == 'trinite' else '.py'))
        loaded = sys.modules.get(name)
        require(loaded is None or Path(loaded.__file__).resolve() == expected.resolve(),
                'foreign preloaded numerical module: ' + name)
    sys.path.insert(0, str(ROOT / 'src'))
    for name in ('trinite', 'trinite.contracts', 'trinite.quantizer', 'trinite.model'):
        loaded = importlib.import_module(name)
        expected = ROOT / 'src' / (name.replace('.', '/') + ('/__init__.py' if name == 'trinite' else '.py'))
        require(Path(loaded.__file__).resolve() == expected.resolve(), 'numerical import origin')


def configure():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.mkldnn.enabled = False
    require(not torch.is_autocast_enabled('cpu'), 'CPU autocast prohibited')
    require(torch.__version__ == '2.8.0+cpu' and torch.version.cuda is None, 'pinned CPU torch required')
    require(__import__('numpy').__version__ == '2.3.5' and __import__('safetensors').__version__ == '0.6.2',
            'pinned numeric-only dependencies required')
    with open('/proc/cpuinfo', 'rb') as handle:
        first = handle.read(65536).decode('ascii').split('\n\n', 1)[0]
    cpu = {k.strip():v.strip() for k, v in (line.split(':', 1) for line in first.splitlines() if ':' in line)}
    return dict(python=platform.python_version(), platform=platform.platform(),
                machine=platform.machine(), processor=platform.processor(),
                cpu={k:cpu.get(k, 'unreported') for k in ('vendor_id', 'model name', 'flags')},
                cpu_kernel_capability=torch.backends.cpu.get_cpu_capability(),
                default_dtype=str(torch.get_default_dtype()), default_device=str(torch.get_default_device()),
                packages={p:importlib.metadata.version(p) for p in ('torch','numpy','safetensors','packaging',
                    'filelock','typing-extensions','setuptools','sympy','networkx','jinja2','fsspec','mpmath','markupsafe')},
                torch=torch.__version__, numpy=__import__('numpy').__version__,
                safetensors=__import__('safetensors').__version__,
                torch_build=torch.__config__.show(), device='cpu', dtype='float32',
                threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                deterministic=torch.are_deterministic_algorithms_enabled(),
                mkldnn=torch.backends.mkldnn.enabled, matmul_precision=torch.get_float32_matmul_precision())


def load_model(payload, metadata, config):
    """Separate model-only foreign-host adapter. No TrainingState or optimizer.

    safetensors is the existing numeric-only reader, not pickle or a parallel parser.
    Validate ALL model and discarded optimizer tensors before copying model values.
    """
    import torch
    from safetensors.torch import load
    bind_imports()
    from trinite.contracts import ModelConfig
    from trinite.model import Decoder
    require(type(payload) is bytes and 8 <= len(payload) <= 1 << 20, 'tensor byte bound')
    header_length = int.from_bytes(payload[:8], 'little')
    require(1 <= header_length <= 65536 and 8 + header_length <= len(payload), 'tensor header bound')
    strict_json(payload[8:8 + header_length])
    meta = strict_json(metadata)
    keys(meta, ('schema', 'binding', 'sources', 'environment', 'plan', 'dataset_identity',
                'initialization_identity', 'initial_train_loss', 'model_identity', 'step',
                'cursor', 'target_tokens', 'history', 'train_diagnostics'))
    require(meta['schema'] == 'trinite.context-symbol-checkpoint.v1', 'checkpoint schema')
    require(meta['plan']['model'] == config, 'architecture not allowlisted')
    require(type(meta['step']) is int and meta['step'] in (0, 128), 'fixed or zero-step checkpoint only')
    model_config = ModelConfig(context_length=256, blocks=1, width=32, heads=4, feed_forward_width=64)
    require(model_config.to_dict() == config, 'only initial fixed architecture is supported')
    model = Decoder(model_config, lane='dense', seed=17, parameter_budget=24768)
    require(sum(p.numel() for p in model.parameters()) == 24768, 'parameter budget')
    parameters = dict(model.named_parameters())
    expected = {'model.' + n: tuple(p.shape) for n, p in parameters.items()}
    if meta['step']:
        for n, p in parameters.items():
            for slot in ('step', 'exp_avg', 'exp_avg_sq'):
                expected['optimizer.' + n + '.' + slot] = () if slot == 'step' else tuple(p.shape)
    tensors = load(payload)
    require(set(tensors) == set(expected), 'closed tensor inventory')
    for n, tensor in tensors.items():
        require(tensor.device.type == 'cpu' and tensor.dtype == torch.float32
                and tuple(tensor.shape) == expected[n] and bool(torch.isfinite(tensor).all()),
                'tensor shape/dtype/finite: ' + n)
        if n.startswith('optimizer.'):
            if n.endswith('.step'):
                require(float(tensor) == meta['step'], 'optimizer step metadata')
            if n.endswith('.exp_avg_sq'):
                require(bool((tensor >= 0).all()), 'negative second moment')
    with torch.no_grad():
        for n, p in parameters.items():
            p.copy_(tensors['model.' + n])
            p.requires_grad_(False)
    model.eval()
    model.validate_state()
    from trinite.contracts import identity, json_bytes
    actual_identity = identity(json_bytes({n:'sha256:' + h for n,h in tensor_inventory(model).items()}))
    require(actual_identity == meta['model_identity'], 'model identity mismatch')
    return model, meta


def tensor_inventory(model):
    return {n: digest(p.detach().contiguous().numpy().tobytes()) for n, p in model.named_parameters()}


def forward_rows(model, rows):
    import torch
    validate_rows(rows)
    before = tensor_inventory(model)
    result = []
    with torch.inference_mode():
        for row in rows:
            first = row['loss'].index(1)
            prefix = row['ids'][:first]
            emitted = []
            for _ in range(3):
                ids = torch.tensor([prefix + emitted], dtype=torch.int64, device='cpu')
                logits = model(ids, torch.ones_like(ids, dtype=torch.bool)).logits[0, -1]
                token = int(logits.argmax())
                emitted.append(token)
                if token == 257:
                    break
            ids = torch.tensor([row['ids'][:-1]], dtype=torch.int64, device='cpu')
            logits = model(ids, torch.ones_like(ids, dtype=torch.bool)).logits[0]
            targets = []
            for i in range(first, len(row['ids'])):
                target = row['ids'][i]
                vector = logits[i-1]
                correct = float(vector[target])
                other = vector.clone()
                other[target] = -torch.inf
                competitor = int(other.argmax())
                rival = float(vector[competitor])
                targets.append(dict(position=i, target=target, competitor=competitor,
                    correct_logit=correct, competitor_logit=rival, margin=correct-rival,
                    correct_logit_hex=correct.hex(), competitor_logit_hex=rival.hex()))
            result.append(dict(id=row['identity'], expected=row['answer'], greedy_tokens=emitted,
                               exact=emitted == row['ids'][first:], teacher_forced=targets))
    require(before == tensor_inventory(model), 'forward changed weights')
    require(all(p.grad is None and not p.requires_grad for p in model.parameters()), 'gradient state')
    return result


def differences(a, b):
    import numpy as np
    result = []
    for (name, p), (other, q) in zip(a.named_parameters(), b.named_parameters()):
        require(name == other and p.shape == q.shape, 'comparison inventory')
        x, y = p.detach().numpy(), q.detach().numpy()
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        den = np.maximum(np.abs(x.astype(np.float64)), np.abs(y.astype(np.float64)))
        rel = np.divide(d, den, out=np.zeros_like(d), where=den != 0)
        result.append(dict(name=name, shape=list(x.shape), count=x.size,
            prior_sha256=digest(x.tobytes()), current_sha256=digest(y.tobytes()),
            bitwise_different_elements=int(np.count_nonzero(x.view(np.uint32) != y.view(np.uint32))),
            max_abs=float(d.max()), symmetric_relative_max=float(rel.max()),
            l2_difference=float(np.linalg.norm(d))))
    return result


def compare(prior, current):
    inputs = pins()
    materials = [read_deposit(path, inputs[label]) for label, path in [('prior', prior), ('current', current)]]
    environment = configure()
    models, metadata, results = [], [], []
    for label, material in zip(('prior', 'current'), materials):
        model, meta = load_model(material['run/fit-17/tensors.safetensors'],
                                 material['run/fit-17/metadata.json'], inputs[label]['model_config'])
        # Forward implementation is exactly the historical pure numerical closure.
        for name, expected in inputs['runtime_sources'].items():
            require(meta['sources'].get(name) == expected, 'historical numerical source mismatch')
        models.append(model)
        metadata.append(meta)
        results.append(forward_rows(model, inputs['rows']))
    for label, path in (('prior', prior), ('current', current)):
        require(digest(read_file(path, 8 << 20)) == inputs[label]['payload_sha256'], 'input changed during forward')
    pins()  # Source/input closure must still match after execution.
    source_names = [Path(__file__), PINS, *(ROOT / n for n in inputs['runtime_sources'])]
    return dict(schema='trinite.forward-checkpoint-diagnostic.v1', scope=SCOPE,
        source_sha256={str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in source_names},
        input_pins={k:inputs[k] for k in ('prior', 'current')}, observed_host=environment,
        original_metadata=dict(zip(('prior', 'current'), metadata)),
        original_vs_observed_environment={label:{k:dict(original=v, observed=environment[k])
            for k,v in meta['environment'].items() if k in environment and v != environment[k]}
            for label,meta in zip(('prior','current'),metadata)},
        original_environment_differences={k:dict(prior=metadata[0]['environment'].get(k),
            current=metadata[1]['environment'].get(k)) for k in
            sorted(set(metadata[0]['environment']) | set(metadata[1]['environment']))
            if metadata[0]['environment'].get(k) != metadata[1]['environment'].get(k)},
        environment_note='Original metadata is reported unchanged, not overridden or accepted as native-host certification. Observed host uses a distinct explicit diagnostic configuration.',
        rows=dict(zip(('prior', 'current'), results)), tensor_differences=differences(*models),
        row_comparison=[dict(id=a['id'], greedy_equal=a['greedy_tokens'] == b['greedy_tokens'],
            teacher_forced_deltas=[dict(position=x['position'],
                correct_logit_delta_current_minus_prior=y['correct_logit']-x['correct_logit'],
                margin_delta_current_minus_prior=y['margin']-x['margin'])
                for x,y in zip(a['teacher_forced'],b['teacher_forced'])])
            for a,b in zip(*results)],
        input_rows_note='Exact eight public train fit rows copied identically from both pinned worker-input.json train arrays; input specification hash-locked. No oracle used by this tool.',
        accounting=dict(optimizer_objects_created=0, optimizer_steps=0, backward_calls=0,
                        heldout_rows_executed=0, fit_rows_per_checkpoint=8, max_greedy_tokens=3,
                        model_bytes_unchanged=True, input_payload_bytes_unchanged=True),
        error_accounting='Observed float32 CPU logits and float64 descriptive tensor differences only; no interval/formal rounding bound. Relative difference is abs(a-b)/max(abs(a),abs(b)), zero for two zeros. Teacher forcing uses gold prefixes; margins may describe prefixes not reached by free generation. Argmax ties use first token index. Cross-checkpoint differences cannot identify training divergence step or cause.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', required=True, type=Path)
    parser.add_argument('--current', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.is_symlink(), 'fresh output required')
    require(not any(parent == evidence or evidence in parent.parents
                    for parent in [args.output.parent.resolve()]
                    for evidence in (args.prior.parent.resolve(), args.current.parent.resolve())),
            'output must not enter an evidence directory')
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    report = compare(args.prior, args.current)
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n'
    with args.output.open('x') as handle:
        handle.write(encoded)
    print(json.dumps({'scope': SCOPE, 'output': str(args.output),
                      'exact': {k:sum(r['exact'] for r in rows) for k, rows in report['rows'].items()}}))


if __name__ == '__main__':
    main()
