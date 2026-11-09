#!/usr/bin/env python3
"""MTC-LFM3-002: bounded local 1-5 teacher capture. NEVER training admission."""
from __future__ import annotations

import argparse
import base64
from contextlib import closing
import datetime as dt
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import re
import socket
import stat
import time

SCHEMA = "trinite.mtc-lfm3.plan.v3"
MANIFEST_SCHEMA = "trinite.mtc-lfm3.manifest.v3"
URL_HOST, URL_PORT = "127.0.0.1", 11434
MIN_MODELS = 1
MAX_MODELS = 5
COUNT_ITEMS = 6
MAX_HTTP = 65536
DEADLINE = 70.0
CONTRACT_PATHS = (
    "AGENTS.md", ".github/AGENTS.md", "scripts/AGENTS.md",
    "tests/AGENTS.md", "docs/AGENTS.md", "docs/AGENT-IBAE-GOVERNANCE.md",
    "docs/DATA.md", "docs/INVARIANTS.md", "docs/MASTER-TEACHER-CORPUS-V1.md",
    "docs/MTC-LFM3-001.md", "scripts/mtc_lfm3_001.py",
    "scripts/mtc_lfm3_owner_job.py",
    "tests/mtc_lfm3_001/test_capture.py",
    ".github/workflows/mtc-lfm3-001-contract.yml",
    ".github/workflows/mtc-lfm3-001-owner-local.yml",
)
SETTINGS = {"temperature": 0, "seed": 17, "num_predict": 96, "num_ctx": 2048}
# Must be top-level /api/chat fields: otherwise the model may exhaust its
# token budget producing only message.thinking, with no final content.
RESPONSE_POLICY = {"think": False, "format": "json"}
NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._:/+-]{0,127}\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("ascii")


def sha(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def checked_file(path):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("expected single-link regular file")
    if os.name == "posix" and (info.st_uid != os.geteuid() or
                               stat.S_IMODE(info.st_mode) & 0o077):
        raise ValueError("unsafe file ownership or permissions")
    return path.read_bytes()


def write_new(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                 getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def append_line(path, row):
    fd = os.open(path, os.O_WRONLY | os.O_APPEND |
                 getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "ab") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("unsafe attempt journal")
        if os.name == "posix" and (info.st_uid != os.geteuid() or
                                   stat.S_IMODE(info.st_mode) & 0o077):
            raise ValueError("unsafe attempt journal permissions")
        stream.write(canonical(row))
        stream.flush()
        os.fsync(stream.fileno())


def create_private(path):
    p = Path(path)
    if not p.parent.is_dir() or p.is_symlink():
        raise ValueError("missing parent or symlink output")
    if os.name == "posix":
        info = p.parent.stat()
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError("private owner-controlled parent required")
    p.mkdir(mode=0o700, exist_ok=False)
    return p


def source_bytes(path):
    """Reviewable repository sources may be 0644; private captures may not."""
    info = Path(path).lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("invalid source closure member")
    return Path(path).read_bytes()


def closure():
    root = Path(__file__).resolve().parents[1]
    return {name: sha(source_bytes(root / name)) for name in CONTRACT_PATHS}


def tasks():
    # A deliberately small shared *text* probe, not multimodal or a benchmark.
    # Structural families are frozen before teachers respond.
    cases = (
        ("multiply-17-19", "train", "mul", 17, 19, None),
        ("gf3-add-2-2", "train", "addmod", 2, 2, 3),
        ("mod11-add-8-15", "train", "addmod", 8, 15, 11),
        ("mod13-mul-12-7", "train", "mulmod", 12, 7, 13),
        ("mod17-add-23-31", "development", "addmod", 23, 31, 17),
        ("mod7-mul-9-9", "locked_test", "mulmod", 9, 9, 7),
    )
    out = []
    for ident, split, op, a, b, modulus in cases:
        expr = (f"{a} times {b}" if op == "mul" else
                f"({a} + {b}) modulo {modulus}" if op == "addmod" else
                f"({a} times {b}) modulo {modulus}")
        out.append({"id": ident, "split": split, "task": op,
                    "a": a, "b": b, "modulus": modulus,
                    "prompt": 'Calculate ' + expr +
                    '. Return only one JSON object with exactly one string key "answer".'})
    return out


def oracle(item):
    if item["task"] == "mul":
        return str(item["a"] * item["b"])
    if item["task"] == "addmod":
        return str((item["a"] + item["b"]) % item["modulus"])
    if item["task"] == "mulmod":
        return str((item["a"] * item["b"]) % item["modulus"])
    raise ValueError("unknown task")


def strict_json(raw):
    def pairs(entries):
        output = {}
        for k, v in entries:
            if k in output:
                raise ValueError("duplicate JSON key")
            output[k] = v
        return output
    def reject_constant(_):
        raise ValueError("nonfinite JSON")
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject_constant)
    except (UnicodeError, RecursionError, ValueError):
        raise ValueError("malformed_json") from None


def validate_models(value):
    names = [x.strip() for x in value.split(",")]
    if not MIN_MODELS <= len(names) <= MAX_MODELS or len(set(names)) != len(names):
        raise ValueError("one to five distinct installed model tags required")
    for name in names:
        if not NAME.fullmatch(name) or "lfm3" not in name.lower() or "d1" in name.lower():
            raise ValueError("each tag must name a text-capable LFM3 variant, not d1")
    return names


class Reader(io.RawIOBase):
    def __init__(self, sock, deadline):
        super().__init__()
        self.sock, self.deadline = sock, deadline

    def readable(self):
        return True

    def readinto(self, buffer):
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("ollama_deadline")
        self.sock.settimeout(left)
        return self.sock.recv_into(buffer)


class Response(http.client.HTTPResponse):
    def __init__(self, sock, *, deadline, **kw):
        super().__init__(sock, **kw)
        self.keep_socket = self.fp
        self.fp = io.BufferedReader(Reader(sock, deadline))

    def close(self):
        try:
            super().close()
        finally:
            self.keep_socket.close()


class TransportError(Exception):
    def __init__(self, code, response=b"", status=None, truncated=False):
        self.code, self.response, self.status, self.truncated = (
            code, response, status, truncated
        )
        super().__init__(code)


def api(path, body=None):
    """Only direct Ollama loopback; no proxy, redirects or external egress."""
    if path not in ("/api/tags", "/api/chat", "/api/version"):
        raise ValueError("endpoint denied")
    conn = http.client.HTTPConnection(URL_HOST, URL_PORT, timeout=DEADLINE)
    deadline = time.monotonic() + DEADLINE
    conn.response_class = lambda sock, **kwargs: Response(sock, deadline=deadline, **kwargs)
    resp = None
    try:
        data = canonical(body) if body is not None else None
        conn.request("POST" if data is not None else "GET", path, body=data,
                     headers={"Content-Type": "application/json"} if data else {})
        resp = conn.getresponse()
        raw = resp.read(MAX_HTTP + 1)
        overflow = len(raw) > MAX_HTTP
        raw = raw[:MAX_HTTP]
        if overflow:
            raise TransportError("response_truncated", raw, resp.status, True)
        if resp.status != 200:
            raise TransportError("http_failure", raw, resp.status)
        try:
            return strict_json(raw), raw
        except ValueError:
            raise TransportError("invalid_json", raw, resp.status) from None
    except (socket.timeout, TimeoutError):
        raise TransportError("deadline") from None
    except (OSError, http.client.HTTPException):
        if time.monotonic() >= deadline:
            raise TransportError("deadline") from None
        raise TransportError("transport_failure") from None
    finally:
        if resp is not None:
            resp.close()
        conn.close()


def inventory():
    data, _ = api("/api/tags")
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise ValueError("invalid Ollama inventory")
    return data["models"]


def model_sha(models, name):
    found = [m for m in models if isinstance(m, dict) and
             (m.get("name") == name or m.get("model") == name)]
    if len(found) != 1 or not isinstance(found[0].get("digest"), str):
        raise ValueError("missing or ambiguous installed model: " + name)
    dg = found[0]["digest"]
    if DIGEST.fullmatch(dg) is None:
        raise ValueError("invalid Ollama digest")
    return "sha256:" + dg


def build_plan(names):
    names = validate_models(",".join(names))
    installed = inventory()
    models = [{"name": name, "digest": model_sha(installed, name)} for name in names]
    if len({model["digest"] for model in models}) != len(models):
        raise ValueError("duplicate installed model digest: aliases are not independent teachers")
    return {"schema": SCHEMA, "source_closure": closure(),
            "models": models,
            "tasks": tasks(), "sampling": SETTINGS, "response_policy": RESPONSE_POLICY,
            "endpoint": "http://127.0.0.1:11434",
            "count": COUNT_ITEMS * len(models), "status": "QUARANTINED_NOT_ADMITTED",
            "admission": False}


def validate_plan(plan):
    if not isinstance(plan, dict) or set(plan) != {
        "schema", "source_closure", "models", "tasks", "sampling", "response_policy",
        "endpoint", "count", "status", "admission",
    } or plan["schema"] != SCHEMA or plan["source_closure"] != closure():
        raise ValueError("source or plan closure drift")
    if plan["tasks"] != tasks() or plan["sampling"] != SETTINGS or (
        plan["response_policy"] != RESPONSE_POLICY or
        plan["endpoint"] != "http://127.0.0.1:11434" or
        plan["status"] != "QUARANTINED_NOT_ADMITTED" or
        plan["admission"] is not False
    ):
        raise ValueError("frozen request changed")
    models = plan["models"]
    if type(models) is not list or not MIN_MODELS <= len(models) <= MAX_MODELS:
        raise ValueError("invalid model list")
    if any(type(entry) is not dict or set(entry) != {"name", "digest"} or
           type(entry["name"]) is not str or type(entry["digest"]) is not str
           for entry in models):
        raise ValueError("invalid frozen model identity")
    names = validate_models(",".join(entry["name"] for entry in models))
    for entry, name in zip(models, names):
        if (entry["name"] != name or
                re.fullmatch(r"sha256:[a-f0-9]{64}", entry["digest"]) is None):
            raise ValueError("invalid frozen model identity")
    if len({entry["digest"] for entry in models}) != len(models):
        raise ValueError("duplicate frozen model digest")
    if type(plan["count"]) is not int or plan["count"] != len(models) * COUNT_ITEMS:
        raise ValueError("frozen request count mismatch")


def expected_requests(plan):
    for entry in plan["models"]:
        for item in plan["tasks"]:
            yield entry, item


def chat_body(model, task):
    return {"model": model, "stream": False, **RESPONSE_POLICY,
            "messages": [{"role": "user", "content": task["prompt"]}],
            "options": SETTINGS}


def parse_answer(response):
    if (not isinstance(response, dict) or response.get("done") is not True
            or response.get("done_reason") == "length"):
        # A token-exhausted reply is never evidence of a final answer,
        # even if its content looks superficially parseable.
        return None
        return None
    message = response.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        return None
    try:
        answer = strict_json(message["content"])
    except ValueError:
        return None
    return answer["answer"] if type(answer) is dict and set(answer) == {"answer"} and (
        type(answer["answer"]) is str and len(answer["answer"]) <= 32) else None


def validate_attempts(plan, attempts):
    expected = list(expected_requests(plan))
    if len(attempts) > len(expected):
        raise ValueError("excess attempts")
    for index, row in enumerate(attempts):
        model, task = expected[index]
        if not isinstance(row, dict) or set(row) != {
            "index", "item", "split", "model", "model_digest", "request_sha256",
            "status", "http_status", "response_base64", "truncated", "error",
            "answer", "oracle_match",
        }:
            raise ValueError("invalid attempt fields")
        if (type(row["index"]) is not int or row["index"] != index or
                row["item"] != task["id"] or row["split"] != task["split"] or
                row["model"] != model["name"] or row["model_digest"] != model["digest"] or
                row["request_sha256"] != sha(canonical(chat_body(model["name"], task)))):
            raise ValueError("attempt binding mismatch")
        raw = row["response_base64"]
        if type(raw) is not str:
            raise ValueError("missing response envelope")
        try:
            decoded = base64.b64decode(raw, validate=True)
        except (ValueError, base64.binascii.Error):
            raise ValueError("invalid response encoding") from None
        if len(decoded) > MAX_HTTP or type(row["truncated"]) is not bool:
            raise ValueError("invalid response limit")
        if row["status"] == "completed":
            if (row["http_status"] != 200 or row["truncated"] or row["error"] is not None):
                raise ValueError("completed envelope invalid")
            response = strict_json(decoded)
            if not isinstance(response, dict) or response.get("model") != model["name"]:
                raise ValueError("recorded model identity mismatch")
            answer = parse_answer(response)
            if row["answer"] != answer or row["oracle_match"] != (
                    answer == oracle(task) if answer is not None else False):
                raise ValueError("oracle label mismatch")
        elif row["status"] == "failed":
            if row["answer"] is not None or row["oracle_match"] is not False or (
                    row["error"] not in ("response_truncated", "http_failure", "invalid_json",
                                         "deadline", "transport_failure", "model_digest_drift",
                                         "model_identity_mismatch")):
                raise ValueError("invalid failure envelope")
            if row["truncated"] and len(decoded) != MAX_HTTP:
                raise ValueError("incorrect truncated flag")
        else:
            raise ValueError("invalid attempt state")


def read_attempts(path):
    raw = checked_file(path)
    result = []
    if not raw:
        return result
    if not raw.endswith(b"\n"):
        raise ValueError("incomplete attempt journal; retain partial bytes")
    for line in raw.splitlines():
        result.append(strict_json(line))
    return result


def summary(attempts, plan):
    return {"planned": plan["count"],
            "attempted": len(attempts),
            "completed": sum(x["status"] == "completed" for x in attempts),
            "failed": sum(x["status"] == "failed" for x in attempts),
            "oracle_matches": sum(x["oracle_match"] for x in attempts),
            "rights_status": "QUARANTINED_NOT_ADMITTED",
            "admitted_rows": 0}


def plan_command(args):
    names = validate_models(args.models)
    plan = build_plan(names)
    validate_plan(plan)
    output = create_private(args.out)
    write_new(output / "plan.json", canonical(plan))
    print("request_identity=" + sha(canonical(plan)))
    print("planned_calls=" + str(plan["count"]) + "; no inference executed")


def capture_command(args):
    if not args.authorize_local_capture:
        raise ValueError("explicit owner authorization required")
    plan_raw = checked_file(args.plan)
    if sha(plan_raw) != args.request_identity:
        raise ValueError("request identity mismatch")
    plan = strict_json(plan_raw)
    validate_plan(plan)
    run_dir = Path(args.out)
    if args.resume:
        if not run_dir.is_dir() or (run_dir / "manifest.json").exists():
            raise ValueError("no resumable incomplete run")
        if checked_file(run_dir / "plan.json") != plan_raw:
            raise ValueError("resume plan mismatch")
        prior = read_attempts(run_dir / "attempts.jsonl")
        validate_attempts(plan, prior)
    else:
        create_private(run_dir)
        write_new(run_dir / "plan.json", plan_raw)
        write_new(run_dir / "attempts.jsonl", b"")
        prior = []
    expected = list(expected_requests(plan))
    for index in range(len(prior), len(expected)):
        model, item = expected[index]
        request = chat_body(model["name"], item)
        row = {"index": index, "item": item["id"], "split": item["split"],
               "model": model["name"], "model_digest": model["digest"],
               "request_sha256": sha(canonical(request)),
               "status": "failed", "http_status": None,
               "response_base64": "", "truncated": False, "error": None,
               "answer": None, "oracle_match": False}
        try:
            if model_sha(inventory(), model["name"]) != model["digest"]:
                raise TransportError("model_digest_drift")
            response, raw = api("/api/chat", request)
            if not isinstance(response, dict) or response.get("model") != model["name"]:
                raise TransportError("model_identity_mismatch", raw, 200)
            answer = parse_answer(response)
            row.update(status="completed", http_status=200,
                       response_base64=base64.b64encode(raw).decode("ascii"),
                       answer=answer, oracle_match=(answer == oracle(item)))
        except TransportError as exc:
            row.update(error=exc.code, http_status=exc.status,
                       response_base64=base64.b64encode(exc.response).decode("ascii"),
                       truncated=exc.truncated)
        append_line(run_dir / "attempts.jsonl", row)
        prior.append(row)
        # A model digest mismatch is terminal: do not continue into another
        # model revision, but preserve the failed attempt in the journal.
        if row["error"] == "model_digest_drift":
            raise ValueError("installed model digest drift; partial evidence retained")
    validate_attempts(plan, prior)
    write_new(run_dir / "summary.json", canonical(summary(prior, plan)))
    manifest = {"schema": MANIFEST_SCHEMA,
                "members": {name: sha(checked_file(run_dir / name)) for name in
                            ("plan.json", "attempts.jsonl", "summary.json")}}
    write_new(run_dir / "manifest.json", canonical(manifest))
    print("manifest_identity=" + sha(canonical(manifest)))
    print("attempts=" + str(len(prior)) + "; private quarantined capture; no student training")


def verify_command(args):
    run = Path(args.run)
    m_raw = checked_file(run / "manifest.json")
    if sha(m_raw) != args.manifest_identity:
        raise ValueError("external manifest identity mismatch")
    manifest = strict_json(m_raw)
    if (not isinstance(manifest, dict) or set(manifest) != {"schema", "members"}
            or manifest["schema"] != MANIFEST_SCHEMA):
        raise ValueError("invalid manifest")
    members = manifest["members"]
    if set(members) != {"plan.json", "attempts.jsonl", "summary.json"}:
        raise ValueError("unexpected manifest members")
    if {p.name for p in run.iterdir()} != {
        "manifest.json", "plan.json", "attempts.jsonl", "summary.json"
    }:
        raise ValueError("unexpected run membership")
    for name, digest in members.items():
        if sha(checked_file(run / name)) != digest:
            raise ValueError("member hash mismatch")
    plan = strict_json(checked_file(run / "plan.json"))
    validate_plan(plan)
    attempts = read_attempts(run / "attempts.jsonl")
    validate_attempts(plan, attempts)
    if len(attempts) != plan["count"]:
        raise ValueError("incomplete attempt inventory")
    if checked_file(run / "summary.json") != canonical(summary(attempts, plan)):
        raise ValueError("summary mismatch")
    print("verified_recorded_capture=" + args.manifest_identity)
    print("rights_status=QUARANTINED_NOT_ADMITTED; training_admission=0")


def main(argv=None):
    parser = argparse.ArgumentParser(description="MTC-LFM3-002 local one-to-five-teacher capture")
    cmds = parser.add_subparsers(dest="command", required=True)
    freeze = cmds.add_parser("plan")
    freeze.add_argument("--models", required=True, help="1-5 distinct comma-separated installed LFM3 names")
    freeze.add_argument("--out", type=Path, required=True)
    capture = cmds.add_parser("capture")
    capture.add_argument("--plan", type=Path, required=True)
    capture.add_argument("--request-identity", required=True)
    capture.add_argument("--out", type=Path, required=True)
    capture.add_argument("--authorize-local-capture", action="store_true")
    capture.add_argument("--resume", action="store_true")
    verify = cmds.add_parser("verify")
    verify.add_argument("--run", type=Path, required=True)
    verify.add_argument("--manifest-identity", required=True)
    args = parser.parse_args(argv)
    try:
        {"plan": plan_command, "capture": capture_command,
         "verify": verify_command}[args.command](args)
    except (ValueError, OSError, TypeError, KeyError, TransportError) as exc:
        parser.exit(2, "MTC-LFM3-001: " + type(exc).__name__ + ": " + str(exc)[:140] + "\n")


if __name__ == "__main__":
    main()
