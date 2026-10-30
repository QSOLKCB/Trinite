# Read-only CUDA availability observations v1

This separate owner-local diagnostic helps distinguish **what can currently be
observed**, without repeating an experiment or changing the system. It does not
explain why CUDA disappeared, certify a healthy GPU, admit a device trial, or
supersede the earlier [CPU/CUDA trial](MUSIC-DEVICE-TRIAL.md). Root cause remains
`unknown`, including when a query succeeds. These tools were tested with software
doubles and harmless subprocesses; no current owner-hardware observation is
inferred from the implementation environment.

## Owner-local commands

Only run these diagnostics when the owner authorizes local observation. Use the
existing trusted virtual environment and a clean reviewed checkout. Do not
rebuild an environment simply to diagnose it. The absolute **lexical** venv
`bin/python` path may be its ordinary symlink: invoke it directly, without
resolving it to a different system interpreter.

```sh
"$PY" -I -B scripts/check_cuda_availability.py
# Explicitly permit Torch CUDA metadata queries, which may initialize the driver:
"$PY" -I -B scripts/check_cuda_availability.py --query-cuda
```

The default command never imports Torch. It reports Python version,
implementation, whether a venv is active and whether the executable is a symlink,
without printing the executable or venv path. An isolated standard-library child
reads distribution versions for `torch`, `numpy` and `safetensors`; an absent
distribution is distinct from a metadata-inspection error. Installed version strings
are not exact wheel/hash-lock verification or runtime admission. A compiled CUDA
version can be present even when CUDA is unavailable; that does not identify the
cause.

The default also invokes **only** the root-controlled `/usr/bin/nvidia-smi`
installation, with this fixed read-only query:

```text
--query-gpu=driver_version,name,memory.total --format=csv,noheader,nounits
```

There is no PATH/current-directory search or arbitrary executable/argument option.
A root-controlled symlink target is checked and executed by its resolved trusted
path; an unsafe or missing installation is reported, not replaced. This first
version supports Linux only. Other operating systems or NVIDIA utilities
installed elsewhere are unsupported; the tool does not search for a substitute.
In particular, WSL may expose `/usr/lib/wsl/lib/nvidia-smi` instead.
`tool_unavailable` means this fixed `/usr/bin/nvidia-smi` candidate is missing,
not that no NVIDIA utility exists anywhere on the system.
[NVIDIA documents selective queries and CSV formatting](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

`--query-cuda` additionally starts one isolated child using the **same lexical
Python executable**, `-I -B`, and fixed code. That child imports installed Torch,
reads its version and compiled CUDA version, then calls `torch.cuda.is_available()`,
`device_count()` and (only when available) basic device-property queries. It
requests no tensor/model allocation, forward pass, optimizer, training, inference,
compilation or parity test. These metadata APIs and NVIDIA's utility can
initialize or interact with the driver. “Read-only diagnostic” does not promise
that library/driver initialization has no incidental operating-system effects.
See the official [availability API](https://docs.pytorch.org/docs/2.8/generated/torch.cuda.is_available.html)
and [CUDA semantics](https://docs.pytorch.org/docs/2.8/notes/cuda.html).

## Bounded and sanitized evidence

Each child has a **10-second wall deadline** and **16 KiB combined stdout/stderr
ceiling**. Both pipes are drained concurrently; timeout or excess output kills
the process group and waits for termination. Default execution has two fixed
queries; the explicit CUDA query adds a third, at most 30 query-seconds plus
bounded setup/cleanup overhead. This is not a sandbox for hostile executables
that escape their process group, nor a guarantee against uninterruptible kernel
I/O. Trusted system executables, interpreter/libraries and a quiescent filesystem
remain assumptions.

Only fixed typed fields are exported. No raw stderr, exception message,
interpreter/home path, process list, GPU UUID/serial, environment values or
arbitrary driver diagnostics appear in the report. `CUDA_VISIBLE_DEVICES` is
classified only as `unset`, `empty`, `minus-one`, `index-list`, or `other-redacted`;
actual selectors are withheld. This classification is a clue, not a diagnosis.
Device names, versions, counts and memory fields have strict size/type/character
limits. A malformed or noisy response is retained as an error category, never
silently repaired or copied into the public receipt.

Reported categories distinguish:

- missing or unsafe `nvidia-smi`, unsupported platform, spawn/command failure;
- timeout, output limit or invalid response;
- absent distribution versus Torch import unavailable;
- Torch query failure versus `no_visible_cuda` versus `visible_cuda`.

An NVIDIA command failure is not automatically a broken GPU or driver. A Torch
import failure can have multiple causes. Successful availability/property queries
are not workload compatibility, deterministic-operation, free-memory, reliability
or trial-admission tests. No unsupported operation triggers a fallback or retry.

JSON stdout contains hashes of the three exact running diagnostic source files,
observations, fixed bounds and a `report_identity` computed from canonical JSON
of the `report` object (sorted keys, compact separators, ASCII and one newline).
Sources are read with bounded non-following checks before and after observation.
This binds the receipt to actual source bytes under the trusted launcher; it does
not prove those bytes are a reviewed Git commit, attest hardware, or certify the
installed dependency contents. Preserve the Git commit separately when collecting
owner-local evidence. A source mutation fails the diagnostic rather than issuing
a bound receipt. The strict `-I -B` guard precedes project imports; it cannot undo
interpreter startup/site code that has already executed.

Exit zero means a diagnostic receipt was produced, **not** that CUDA works. Query
failures remain visible inside that receipt. CLI/source failures return nonzero
with a fixed sanitized error, not a private traceback. Capture exit status and
receipt without overwriting earlier evidence; do not use failure as permission
to install packages, change environment variables, alter drivers/security,
restart services, reboot, or rerun training. This command implements none of
those actions and writes no output files itself.

## Conformance only

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts python -B -m unittest discover -s tests/cuda_availability_v1 -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts python -O -B -m unittest discover -s tests/cuda_availability_v1 -v
```

The suite does not invoke actual CUDA or `nvidia-smi`. It uses fixed protocol
responses and harmless local Python subprocesses to exercise bounded noisy
output, timeout/cleanup, privacy filtering, flag guards and lexical venv handling.
Affected TRI-I01/05/07/08/12/13/14. Historical source closures, checkpoints,
requests, numerical protocols and results remain unchanged. No system repair,
new target experiment or GPU-health claim is authorized by this implementation.
