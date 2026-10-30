"""Versioned quantized-allocator CUDA-wheel device trial admission, never a native CPU-lane fallback.

No model construction, tensor allocation or network access occurs here. CUDA
metadata/context and allocator configuration are deliberate startup overhead.
"""
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import sys

PACKAGES = {'torch': '2.8.0+cu128', 'numpy': '2.3.5', 'safetensors': '0.6.2',
    'packaging': '25.0', 'filelock': '3.32.3', 'typing-extensions': '4.16.0',
    'setuptools': '78.1.0', 'sympy': '1.14.0', 'networkx': '3.6.1',
    'jinja2': '3.1.6', 'fsspec': '2026.7.0', 'mpmath': '1.3.0', 'markupsafe': '3.0.3',
    'nvidia-cuda-nvrtc-cu12': '12.8.93', 'nvidia-cuda-runtime-cu12': '12.8.90',
    'nvidia-cuda-cupti-cu12': '12.8.90', 'nvidia-cudnn-cu12': '9.10.2.21',
    'nvidia-cublas-cu12': '12.8.4.1', 'nvidia-cufft-cu12': '11.3.3.83',
    'nvidia-curand-cu12': '10.3.9.90', 'nvidia-cusolver-cu12': '11.7.3.90',
    'nvidia-cusparse-cu12': '12.5.8.93', 'nvidia-cusparselt-cu12': '0.7.1',
    'nvidia-nccl-cu12': '2.27.3', 'nvidia-nvtx-cu12': '12.8.90',
    'nvidia-nvjitlink-cu12': '12.8.93', 'nvidia-cufile-cu12': '1.13.1.3',
    'triton': '3.4.0'}
ALLOCATOR_BYTES = 8 * 1024**3
WORKSPACE = ':4096:8'


class UnsupportedRuntime(ValueError):
    """The separately frozen device trial cannot run on this environment."""


def _backend(backend):
    if type(backend) is not str or backend not in ('cpu', 'cuda'):
        raise UnsupportedRuntime('fixed device trial requires cpu or cuda, never a fallback selector')


def _torch():
    try:
        import torch
    except (ImportError, OSError, RuntimeError) as error:
        raise UnsupportedRuntime('pinned CUDA wheel or shared-library runtime unavailable') from error
    return torch


def _packages():
    try:
        return {name: importlib.metadata.version(name) for name in PACKAGES}
    except importlib.metadata.PackageNotFoundError as error:
        raise UnsupportedRuntime('device trial requires its separate complete CUDA hash lock') from error


def _host():
    if (platform.python_implementation() != 'CPython' or sys.version_info[:2] != (3, 12)
            or platform.system() != 'Linux' or platform.machine() != 'x86_64'):
        raise UnsupportedRuntime('device trial supports Linux x86_64 CPython 3.12 only')
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'machine': platform.machine(), 'implementation': platform.python_implementation()}


def _cpu():
    with Path('/proc/cpuinfo').open('rb') as stream:
        first = stream.read(65536).decode('ascii').split('\n\n', 1)[0]
    fields = {k.strip(): v.strip() for k, v in
              (line.split(':', 1) for line in first.splitlines() if ':' in line)}
    return {key: fields.get(key, 'unreported') for key in ('vendor_id', 'model name', 'flags')}


def _driver():
    """Record kernel driver text locally; no executable or arbitrary shell lookup."""
    try:
        with Path('/proc/driver/nvidia/version').open('rb') as stream:
            value = stream.read(4097)
        if len(value) > 4096:
            raise UnsupportedRuntime('NVIDIA driver identity exceeds bound')
        return value.decode('ascii').strip()
    except OSError as error:
        raise UnsupportedRuntime('NVIDIA kernel-driver identity unavailable') from error


def _hardware(torch):
    if (str(torch.__version__) != PACKAGES['torch'] or torch.version.cuda != '12.8'
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise UnsupportedRuntime('both trial cells require the pinned CUDA wheel and one available GPU')
    properties = torch.cuda.get_device_properties(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    architectures = sorted(torch.cuda.get_arch_list())
    total = properties.total_memory
    if (capability != (12, 0) or 'sm_120' not in architectures
            or type(total) is not int or not 14 * 1024**3 <= total <= 18 * 1024**3):
        raise UnsupportedRuntime('trial requires native sm_120 support and a 16-GiB-class device')
    return {'index': 0, 'name': str(properties.name), 'capability': list(capability),
            'total_memory': total, 'multiprocessors': int(properties.multi_processor_count),
            'uuid': str(getattr(properties, 'uuid', 'unreported')),
            'compiled_architectures': architectures, 'driver': _driver(),
            'cuda_runtime': str(torch.version.cuda), 'cudnn': torch.backends.cudnn.version(),
            'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}


def _environment_selectors():
    # These variables can override precision, allocator or timing policy outside
    # the visible Torch flags. Do not silently repair a conflicting process.
    names = ('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', 'NVIDIA_TF32_OVERRIDE',
             'CUDA_LAUNCH_BLOCKING')
    if any(os.environ.get(name) not in (None, '0') for name in names):
        raise UnsupportedRuntime('precision/timing environment override is unsupported')
    if any(os.environ.get(name) not in (None, '') for name in
           ('PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF')):
        raise UnsupportedRuntime('alternate CUDA allocator configuration is unsupported')
    return {name: os.environ.get(name, 'unset') for name in (*names,
            'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF')}


def configure():
    """Set declared startup policy once; never repair drift after work starts."""
    _environment_selectors()
    existing = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if existing not in (None, WORKSPACE):
        raise UnsupportedRuntime('conflicting CUBLAS_WORKSPACE_CONFIG; no automatic environment repair')
    # This precedes importing Torch and any CUDA metadata/context initialization.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = WORKSPACE
    _host()
    torch = _torch()
    if _packages() != PACKAGES:
        raise UnsupportedRuntime('device trial package versions differ from the separate CUDA lock')
    if torch.cuda.is_initialized():
        if existing is None:
            raise UnsupportedRuntime('CUDA was initialized before the required workspace policy')
        # Reentrant admission checks do not mutate an already-used runtime.
        require_environment('cuda')
        return
    hardware = _hardware(torch)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    if str(torch.get_default_device()) != 'cpu':
        raise UnsupportedRuntime('default device must remain CPU; transfers are explicit')
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.set_device(0)
    requested, _, _ = allocator_policy(_allocator_total(torch))
    torch.cuda.memory.set_per_process_memory_fraction(requested, 0)
    require_environment('cuda')


def _allocator_total(torch):
    # The pinned allocator uses cudaMemGetInfo, not device-properties total.
    # Free memory is volatile and deliberately excluded from request identity.
    info = torch.cuda.mem_get_info(0)
    if (type(info) is not tuple or len(info) != 2
            or any(type(v) is not int for v in info)
            or not 0 <= info[0] <= info[1]
            or not 14 * 1024**3 <= info[1] <= 18 * 1024**3):
        raise UnsupportedRuntime('invalid allocator memory information')
    return info[1]


def allocator_policy(total):
    """Pinned PyTorch 2.8 native caching-allocator setter/getter arithmetic.

    https://github.com/pytorch/pytorch/blob/v2.8.0/c10/cuda/CUDACachingAllocator.cpp
    casts fraction * total to size_t, then returns
    that integer divided by total. The single expected quantized cap is not
    a tolerance interval: genuine one-byte policy drift remains rejected.
    """
    if type(total) is not int or not 14 * 1024**3 <= total <= 18 * 1024**3:
        raise UnsupportedRuntime('invalid admitted allocator total')
    requested_fraction = ALLOCATOR_BYTES / total
    effective = int(requested_fraction * total)
    if not ALLOCATOR_BYTES - 1 <= effective <= ALLOCATOR_BYTES:
        raise UnsupportedRuntime('allocator quantization outside fixed requested byte cap')
    return requested_fraction, effective, effective / total


def require_environment(backend):
    """Recheck stable policy and hardware; CPU arm still requires this CUDA trial."""
    _backend(backend)
    host = _host()
    selectors = _environment_selectors()
    torch = _torch()
    packages = _packages()
    if packages != PACKAGES:
        raise UnsupportedRuntime('device trial package versions differ from the separate CUDA lock')
    hardware = _hardware(torch)
    fraction = torch.cuda.memory.get_per_process_memory_fraction(0)
    allocator_total = _allocator_total(torch)
    requested, effective, expected_fraction = allocator_policy(allocator_total)
    if type(fraction) is not float or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise UnsupportedRuntime('invalid allocator fraction')
    if (os.environ.get('CUBLAS_WORKSPACE_CONFIG') != WORKSPACE
            or torch.get_num_threads() != 1 or str(torch.get_default_dtype()) != 'torch.float32'
            or str(torch.get_default_device()) != 'cpu'
            or not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()
            or torch.backends.cudnn.benchmark or not torch.backends.cudnn.deterministic
            or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
            or torch.get_float32_matmul_precision() != 'highest'
            or torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('cuda')
            or torch.cuda.current_device() != 0
            or fraction != expected_fraction
            or torch.cuda.current_stream(0) != torch.cuda.default_stream(0)):
        raise UnsupportedRuntime('device trial deterministic FP32/default-stream/allocator policy differs')
    return {**host, 'backend': backend, 'packages': packages, 'hardware': hardware,
            'environment_selectors': selectors,
            'cpu': _cpu(), 'cpu_kernel_capability': torch.backends.cpu.get_cpu_capability(),
            'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
            'default_dtype': str(torch.get_default_dtype()), 'default_device': str(torch.get_default_device()),
            'deterministic': True, 'warn_only': False, 'tf32': False, 'autocast': False,
            'compile': False, 'cudnn_benchmark': False, 'cudnn_deterministic': True,
            'float32_matmul_precision': 'highest', 'cublas_workspace_config': WORKSPACE,
            'allocator_total_bytes': allocator_total, 'allocator_limit_bytes': effective, 'allocator_requested_limit_bytes': ALLOCATOR_BYTES,
            'allocator_effective_limit_bytes': effective,
            'allocator_requested_fraction_hex': requested.hex(),
            'allocator_fraction_hex': fraction.hex(),
            'allocator_quantization': 'pytorch-2.8-native-size_t(fraction*total)/total; exact expected getter',
            'allocator_scope': 'PyTorch caching allocator only; not whole-process or whole-device memory',
            'optimizer': 'AdamW-single-tensor; foreach=False; fused=False',
            'comparison_scope': 'same CUDA-wheel environment; no CPU/GPU bitwise equality guarantee'}


def synchronize(backend):
    _backend(backend)
    if backend == 'cuda':
        _torch().cuda.synchronize(0)
