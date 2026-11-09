#!/usr/bin/env python3
"""OLLAMA-LOG-001: offline supplementary operational log receipt, NOT inference proof."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat

SCHEMA = "trinite.ollama-log.receipt.v1"
MAX_LOG = 64 * 1024 * 1024
MAX_LINE = 64 * 1024
MAX_MANIFEST = 1024 * 1024
CHUNK = 64 * 1024
SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
LABEL = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}\Z")
STAMP = re.compile(r"^time=(\S+)")
GIN = re.compile(r"^\[GIN\] (\d{4}/\d{2}/\d{2} - \d{2}:\d{2}:\d{2})")
LISTEN = re.compile(r"^Listening on (\S+) \(version ([a-zA-Z0-9._-]+)\)$")
CONTRACT_PATHS = (
    "AGENTS.md", "scripts/AGENTS.md", "tests/AGENTS.md", "docs/AGENTS.md",
    "docs/INVARIANTS.md", "docs/DATA.md",
    "docs/AGENT-IBAE-GOVERNANCE.md",
    "scripts/ollama_log_receipt_001.py",
    "tests/ollama_log_receipt_001/test_receipt.py",
    "docs/OLLAMA-LOG-001.md",
    ".github/workflows/ollama-log-001-contract.yml",
)


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, ensure_ascii=True, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("ascii")


def sha(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def private_parent(path):
    parent = Path(path).parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("private real parent directory required")
    info = parent.stat()
    if os.name == "posix" and (info.st_uid != os.geteuid() or
                                stat.S_IMODE(info.st_mode) & 0o077):
        raise ValueError("parent must be owner-controlled mode 0700")


def source_closure():
    root = Path(__file__).resolve().parents[1]
    result = {}
    for name in CONTRACT_PATHS:
        path = root / name
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("invalid source closure file: " + name)
        result[name] = sha(path.read_bytes())
    return result


def open_private_regular(path):
    path = Path(path)
    private_parent(path)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) |
                 getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or
                (os.name == "posix" and (
                    info.st_uid != os.geteuid() or
                    stat.S_IMODE(info.st_mode) & 0o077))):
            raise ValueError("private single-link regular file required")
    except BaseException:
        os.close(fd)
        raise
    return fd, info


def read_manifest(path):
    fd, before = open_private_regular(path)
    with os.fdopen(fd, "rb") as source:
        if not 1 <= before.st_size <= MAX_MANIFEST:
            raise ValueError("experiment manifest exceeds receipt limit")
        raw = source.read(MAX_MANIFEST + 1)
        after = os.fstat(source.fileno())
    if (len(raw) != before.st_size or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns):
        raise ValueError("experiment manifest changed during read")
    def unique_pairs(pairs):
        output = {}
        for k, v in pairs:
            if k in output:
                raise ValueError("duplicate JSON field in experiment manifest")
            output[k] = v
        return output
    try:
        data = json.loads(raw, object_pairs_hook=unique_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(
                              ValueError("nonfinite JSON")))
    except (UnicodeError, ValueError, RecursionError):
        raise ValueError("invalid experiment manifest JSON") from None
    if (type(data) is not dict or type(data.get("schema")) is not str or
            not data["schema"].startswith("trinite.") or
            type(data.get("members")) is not dict or
            not data["members"] or
            any(type(k) is not str or type(v) is not str or SHA.fullmatch(v) is None
                for k, v in data["members"].items())):
        raise ValueError("unsupported experiment manifest envelope")
    return {
        "basename": Path(path).name,
        "sha256": sha(raw),
        "size_bytes": len(raw),
        "schema_observed": data["schema"],
        "association": "operator_supplied_reference_only",
        "experiment_verified_by_this_tool": False,
    }


def aware_timestamp(text):
    try:
        stamp = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (stamp.astimezone(dt.timezone.utc) if stamp.tzinfo is not None
            else None)


def observed_fields(line):
    try:
        return dict(pair.split("=", 1) for pair in shlex.split(line)
                    if "=" in pair)
    except ValueError:
        return {}


def scan_log(path):
    fd, before = open_private_regular(path)
    if before.st_size < 1 or before.st_size > MAX_LOG:
        os.close(fd)
        raise ValueError("log must be nonempty and at most 64 MiB")
    digest = hashlib.sha256()
    size = 0
    lines = 0
    invalid_utf8 = 0
    offset_times = []
    unzoned_times = []
    listens = []
    gpu = []
    cloud_flags = []
    http_total = 0
    http_chat = 0
    last_line_newline = False
    with os.fdopen(fd, "rb", buffering=CHUNK) as stream:
        while True:
            raw = stream.readline(MAX_LINE + 1)
            if not raw:
                break
            if len(raw) > MAX_LINE:
                raise ValueError("oversized Ollama log line")
            digest.update(raw)
            size += len(raw)
            lines += 1
            if size > MAX_LOG:
                raise ValueError("Ollama log exceeds size limit")
            last_line_newline = raw.endswith(b"\n")
            try:
                line = raw.decode("utf-8").rstrip("\r\n")
            except UnicodeDecodeError:
                invalid_utf8 += 1
                continue
            found = STAMP.match(line)
            if found is not None:
                time = aware_timestamp(found.group(1))
                if time is not None:
                    offset_times.append(time)
            access = GIN.match(line)
            if access is not None:
                # GIN time has no UTC offset; do not silently convert it.
                unzoned_times.append(access.group(1))
                http_total += 1
                if '"/api/chat"' in line:
                    http_chat += 1
            if "msg=" not in line:
                continue
            fields = observed_fields(line)
            msg = fields.get("msg")
            if not isinstance(msg, str):
                continue
            detected = LISTEN.fullmatch(msg)
            if detected is not None:
                listens.append({"host": detected.group(1), "version": detected.group(2)})
            if msg.startswith("Ollama cloud disabled: "):
                if msg.endswith("true"):
                    cloud_flags.append(True)
                elif msg.endswith("false"):
                    cloud_flags.append(False)
            if msg == "inference compute":
                entry = {
                    name: fields[name][:160] for name in (
                        "id", "library", "description", "driver", "total", "available",
                    ) if name in fields
                }
                if entry:
                    gpu.append(entry)
        after = os.fstat(stream.fileno())
    if (not last_line_newline or size != before.st_size or
            before.st_mtime_ns != after.st_mtime_ns or
            before.st_ctime_ns != after.st_ctime_ns):
        raise ValueError("incomplete or concurrently changing log")
    timestamps = [x.isoformat(timespec="microseconds") for x in offset_times]
    return {
        "basename": Path(path).name,
        "sha256": "sha256:" + digest.hexdigest(),
        "size_bytes": size,
        "line_count": lines,
        "invalid_utf8_line_count": invalid_utf8,
        "offset_timestamp_interval": {
            "basis": "explicit_offset_time_equals_entries_only",
            "count": len(timestamps),
            "first_utc": min(timestamps) if timestamps else None,
            "last_utc": max(timestamps) if timestamps else None,
            "not_full_uptime": True,
        },
        "unzoneds_access_interval": {
            "basis": "GIN_local_clock_no_offset",
            "count": len(unzoned_times),
            "first_literal": min(unzoned_times) if unzoned_times else None,
            "last_literal": max(unzoned_times) if unzoned_times else None,
        },
        "listen_events": listens,
        "gpu_discovery_events": gpu,
        "cloud_disabled_events": {
            "observed_true": cloud_flags.count(True),
            "observed_false": cloud_flags.count(False),
        },
        "http_observations": {
            "gin_requests": http_total, "gin_chat_requests": http_chat
        },
    }


def receipt_value(log, experiment, experiment_identity, label):
    if type(experiment_identity) is not str or SHA.fullmatch(experiment_identity) is None:
        raise ValueError("expected external experiment sha256 identity")
    if type(label) is not str or LABEL.fullmatch(label) is None:
        raise ValueError("invalid session label")
    if Path(log).resolve(strict=True) == Path(experiment).resolve(strict=True):
        raise ValueError("log and experiment manifest must be different files")
    related = read_manifest(experiment)
    if related["sha256"] != experiment_identity:
        raise ValueError("experiment manifest external identity mismatch")
    observed = scan_log(log)
    return {
        "schema": SCHEMA,
        "source_closure": source_closure(),
        "session_label": label,
        "log": observed,
        "experiment_manifest": related,
        "finalization": {
            "owner_declared": "log_stopped_or_rotated",
            "machine_attested": False,
        },
        "claim_scope": "supplementary_operational_context_not_inference_proof",
        "operator_association_is_execution_evidence": False,
        "training_admission": False,
        "rights_status": "QUARANTINED_NOT_ADMITTED",
    }


def write_new(path, raw):
    private_parent(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                 getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())


def seal_command(args):
    if not args.confirm_finalized:
        raise ValueError("owner confirmation log is stopped/rotated required")
    record = receipt_value(args.log, args.experiment_manifest,
                           args.experiment_identity, args.session_label)
    raw = canonical(record)
    write_new(args.out, raw)
    print("ollama_log_receipt_identity=" + sha(raw))
    print("observed_log_sha256=" + record["log"]["sha256"])
    print("observed_log_bytes=" + str(record["log"]["size_bytes"]))
    print("association=operator_reference_only; execution_attestation=false")
    print("training_admission=0; raw_log_not_copied")


def verify_command(args):
    raw_fd, info = open_private_regular(args.receipt)
    with os.fdopen(raw_fd, "rb") as source:
        if not 1 <= info.st_size <= MAX_MANIFEST:
            raise ValueError("receipt exceeds limit")
        raw = source.read(MAX_MANIFEST + 1)
        after = os.fstat(source.fileno())
    if (len(raw) != info.st_size or info.st_mtime_ns != after.st_mtime_ns
            or info.st_ctime_ns != after.st_ctime_ns):
        raise ValueError("receipt changed during verification")
    if type(args.receipt_identity) is not str or sha(raw) != args.receipt_identity:
        raise ValueError("external receipt identity mismatch")
    try:
        saved = json.loads(raw, object_pairs_hook=lambda entries:
                           no_duplicate_keys(entries))
    except (UnicodeError, RecursionError, ValueError):
        raise ValueError("invalid receipt JSON") from None
    if (type(saved) is not dict or set(saved) != {
        "schema", "source_closure", "session_label", "log", "experiment_manifest",
        "finalization", "claim_scope", "operator_association_is_execution_evidence",
        "training_admission", "rights_status",
    } or saved["schema"] != SCHEMA):
        raise ValueError("unexpected receipt schema or fields")
    fresh = receipt_value(args.log, args.experiment_manifest,
                          saved["experiment_manifest"]["sha256"], saved["session_label"])
    if saved != fresh or raw != canonical(fresh):
        raise ValueError("log, experiment or source closure differs from receipt")
    print("verified_ollama_log_receipt=" + sha(raw))
    print("associated_experiment_manifest=" + fresh["experiment_manifest"]["sha256"])
    print("claim=log_integrity_and_metadata_only; no_inference_attestation")


def no_duplicate_keys(entries):
    obj = {}
    for key, val in entries:
        if key in obj:
            raise ValueError("duplicate receipt field")
        obj[key] = val
    return obj


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="OLLAMA-LOG-001 offline supplementary operational receipt")
    sub = parser.add_subparsers(dest="command", required=True)
    seal = sub.add_parser("seal")
    seal.add_argument("--log", type=Path, required=True)
    seal.add_argument("--experiment-manifest", type=Path, required=True)
    seal.add_argument("--experiment-identity", required=True)
    seal.add_argument("--session-label", required=True)
    seal.add_argument("--out", type=Path, required=True)
    seal.add_argument("--confirm-finalized", action="store_true")
    verify = sub.add_parser("verify")
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--receipt-identity", required=True)
    verify.add_argument("--log", type=Path, required=True)
    verify.add_argument("--experiment-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        {"seal": seal_command, "verify": verify_command}[args.command](args)
    except (OSError, TypeError, ValueError, KeyError, RecursionError) as exc:
        parser.exit(2, "OLLAMA-LOG-001: " + type(exc).__name__ + ": " +
                    str(exc)[:160] + "\n")


if __name__ == "__main__":
    main()
