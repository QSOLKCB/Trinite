"""TT-003 optional RTX 50-series CUDA compatibility gate (not a GPU backend)."""
from __future__ import annotations

import ctypes
import re

from trinite.contracts import ContractError

MIN_CUDA = (12, 8)
TARGET_CAPABILITY = (12, 0)


def version(text: str) -> tuple[int, int]:
    if type(text) is not str or not re.fullmatch(r"[0-9]{1,2}\.[0-9]{1,2}", text):
        raise ContractError("invalid CUDA toolkit/runtime version string")
    return tuple(int(p) for p in text.split("."))


def validate(observed: dict) -> dict:
    """Pure contract validation; fakes do not establish a real device result."""
    expected = {"cuda_build", "driver_api", "gpu_name", "capability",
                "arch_list", "kernel_executed", "kernel_correct"}
    if type(observed) is not dict or set(observed) != expected:
        raise ContractError("CUDA preflight observation schema mismatch")
    if version(observed["cuda_build"]) < MIN_CUDA:
        raise ContractError("CUDA 12.8+ build required for RTX 50 native sm_120")
    driver = observed["driver_api"]
    if type(driver) is not int or driver < 12080:
        raise ContractError("CUDA driver API is below 12.8 compatibility")
    if (type(observed["gpu_name"]) is not str or
            re.search(r"\bRTX 50[0-9]{2}\b", observed["gpu_name"]) is None):
        raise ContractError("RTX 50-series hardware identity required")
    if (type(observed["capability"]) is not list or
            observed["capability"] != [12, 0] or
            any(type(v) is not int for v in observed["capability"])):
        raise ContractError("actual RTX 50 device must report compute capability sm_120")
    if (type(observed["arch_list"]) is not list or
            any(type(x) is not str for x in observed["arch_list"]) or
            "sm_120" not in observed["arch_list"]):
        raise ContractError("backend lacks native sm_120 kernels")
    if (observed["kernel_executed"] is not True or
            observed["kernel_correct"] is not True):
        raise ContractError("real CUDA kernel execution and parity are required")
    return {"eligible_preflight_only": True, "gpu_training_supported": False,
            "observed": observed}


def driver_api_version() -> int:
    # Read CUDA Driver API from libcuda.so.1; nvidia-smi is not used as evidence.
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        result = ctypes.c_int()
        lib.cuInit.argtypes = [ctypes.c_uint]
        lib.cuInit.restype = ctypes.c_int
        if lib.cuInit(0) != 0:
            raise ContractError("CUDA driver initialization failed")
        lib.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.cuDriverGetVersion.restype = ctypes.c_int
        if lib.cuDriverGetVersion(ctypes.byref(result)) != 0:
            raise ContractError("CUDA driver API version query failed")
        return result.value
    except OSError as exc:
        raise ContractError("libcuda.so.1 missing; no CUDA fallback") from exc


def live(device_index: int = 0) -> dict:
    """Actual GPU check; never runs on CI's CPU-only software doubles."""
    if type(device_index) is not int or device_index < 0:
        raise ContractError("invalid explicit CUDA device index")
    try:
        import torch
        if (torch.version.cuda is None or not torch.cuda.is_available() or
                device_index >= torch.cuda.device_count()):
            raise ContractError("CUDA-enabled PyTorch and selected GPU required")
        device = torch.device("cuda", device_index)
        props = torch.cuda.get_device_properties(device_index)
        arch_list = list(torch.cuda.get_arch_list())
        candidate = {
            "cuda_build": str(torch.version.cuda),
            "driver_api": driver_api_version(),
            "gpu_name": str(props.name),
            "capability": [int(props.major), int(props.minor)],
            "arch_list": arch_list,
            "kernel_executed": False, "kernel_correct": False,
        }
        # Fail before any allocation if CUDA version/architecture is unsupported.
        validate(dict(candidate, kernel_executed=True, kernel_correct=True))
        with torch.cuda.device(device):
            value = torch.tensor([0., 1., 2., 3.], dtype=torch.float32, device=device)
            result = value * 3.0 + 2.0
            torch.cuda.synchronize(device)
            ok = result.cpu().tolist() == [2.0, 5.0, 8.0, 11.0]
        return validate(dict(candidate, kernel_executed=True, kernel_correct=ok))
    except ContractError:
        raise
    except (RuntimeError, AttributeError, OSError, ValueError) as exc:
        raise ContractError("CUDA preflight execution failed; no CPU fallback") from exc
