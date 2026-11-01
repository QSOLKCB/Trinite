"""Fixed offline module builds and bounded, source-bound oracle transport.

Build records are local caches, not trusted attestations. An unchanging checkout,
compiler/system libraries and build directory remain operator trust boundaries.
Nothing here accepts an executable path, loads a model or runs an optimizer.
"""
from __future__ import annotations

import base64
from pathlib import Path
import subprocess

import queue_evaluation_v1 as gate
import tombstone_adapter_v1 as tombstone

ROOT = Path(__file__).resolve().parents[2]
MODULES = {
    'ternary': ('ternary-evidence-v1', ('score',)),
    'context': ('context-symbol-relation-v1', ('observe', 'evaluate-privileged')),
    'tombstone': ('tombstone-adapter-v1', ('observe', 'evaluate-privileged')),
}
TIMEOUT = 5.0
OUTPUT_CAP = 16384  # hard combined cap via 8192 bytes per stream


class InfrastructureError(gate.GateError):
    """A failed oracle/build is never an admitted negative semantic label."""


def _paths(module):
    gate.require(type(module) is str and module in MODULES, 'unknown module')
    crate = ROOT / 'external' / MODULES[module][0]
    return (crate, crate / 'target/release' / ('trinite-' + MODULES[module][0]),
            crate / 'target/procedural-build-record.json')


def sources(module):
    crate, _, _ = _paths(module)
    paths = {ROOT / 'scripts/procedural_dataset_v1/transport.py',
             ROOT / 'scripts/queue_evaluation_v1.py',
             ROOT / 'scripts/tombstone_adapter_v1.py'}
    # Include license, pin and complete vendored C/Rust closure; never build caches.
    paths.update(p for p in crate.rglob('*') if p.is_file()
                 and not {'target', 'tests'}.intersection(p.relative_to(crate).parts))
    return {p.relative_to(ROOT).as_posix(): gate.digest(gate.read(p, 4 * 1024 * 1024))
            for p in sorted(paths)}


def _identity(module):
    _, binary, _ = _paths(module)
    return {'schema': 'trinite.procedural-module-build.v1', 'module': module,
            'sources': sources(module),
            'binary_sha256': gate.digest(gate.read(binary, 64 * 1024 * 1024, allow_links=True))}


def build():
    """Explicit operator build only; offline locked compiler actions, no downloads."""
    version = subprocess.run(['rustc', '--version'], check=True, capture_output=True,
                             timeout=10).stdout.decode('ascii').strip()
    gate.require(version.startswith('rustc 1.90.0 '), 'Rust 1.90.0 required')
    for module in MODULES:
        crate, _, record = _paths(module)
        before = sources(module)
        gate.safe_path(crate / 'target')
        if module == 'tombstone':
            tombstone.build()  # crate rust-toolchain.toml fixes 1.90.0; checks upstream pin
        else:
            subprocess.run(['cargo', 'build', '--release', '--offline', '--locked'],
                           cwd=crate, check=True, timeout=120)
        gate.require(sources(module) == before, 'sources changed during module build')
        # Disposable build record only, never historical evidence. Reject link paths.
        gate.safe_path(record)
        temporary = record.with_suffix('.tmp')
        gate.write(temporary, gate.encode(_identity(module)))
        temporary.replace(record)
    return identities()


def identities():
    """Rehash every source and executable on each call; no success cache."""
    result = {}
    try:
        tombstone.verify_build()
        for module in MODULES:
            _, _, record = _paths(module)
            expected = _identity(module)
            gate.require(gate.decode(gate.read(record)) == expected,
                         'module source/binary build identity mismatch')
            result[module] = expected
    except (OSError, ValueError, tombstone.InfrastructureError) as error:
        raise InfrastructureError('module build unavailable or identity mismatch') from error
    return result


def run(module, operation, request):
    """Validate fixed dispatch, execute actual oracle, return a strict JSON object.

    ProcessError retains bounded stdout/stderr and exit/timeout evidence for the
    enclosing admission attempt. Exit 2 (malformed input) is still a failure,
    never an oracle label; exit 3 and all transport errors likewise fail closed.
    """
    _, binary, _ = _paths(module)
    gate.require(operation in MODULES[module][1], 'unknown module operation')
    gate.require(type(request) is dict, 'request must be an object')
    raw = gate.encode(request)
    gate.require(len(raw) <= (1024 if module == 'tombstone' else 16384),
                 'module request byte limit')
    before = identities()
    output, evidence = gate.bounded_process([str(binary), operation], raw,
                                           timeout=TIMEOUT, cap=OUTPUT_CAP // 2)
    try:
        stderr = base64.b64decode(evidence['stderr_b64'], validate=True)
        gate.require(not stderr and len(output) + len(stderr) <= OUTPUT_CAP,
                     'unexpected oracle diagnostics or output limit')
        gate.require(output.endswith(b'\n') and output.count(b'\n') == 1,
                     'expected one oracle response line')
        value = gate.decode(output)
        gate.require(type(value) is dict, 'expected oracle object')
        gate.require(identities() == before, 'module identity changed during execution')
    except (ValueError, OSError) as error:
        # Preserve the already bounded actual response even for protocol/identity
        # failures. The caller retains it privately, never as a learner label.
        raise gate.ProcessError(dict(evidence, failure='invalid_oracle_response')) from error
    return value
