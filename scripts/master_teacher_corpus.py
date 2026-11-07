#!/usr/bin/env python3
"""MTC-001: bounded owner-local Ollama collection; captures are never admitted data."""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request
import zlib

SCHEMA = "trinite.master-teacher-corpus.v1"
MAX_REQUESTS = 32
MAX_RESPONSE = 65536
MAX_LOG = 8 * 1024 * 1024
DEFAULT_URL = "http://127.0.0.1:11434"
CONTRACT_FILES = (
    "AGENTS.md",
    ".github/AGENTS.md",
    "docs/AGENTS.md",
    "scripts/AGENTS.md",
    "tests/AGENTS.md",
    "docs/AGENT-IBAE-GOVERNANCE.md",
    "docs/DATA.md",
    "docs/INVARIANTS.md",
    "docs/INSTRUCTIONS.md",
    "docs/ROADMAP.md",
    "scripts/master_teacher_corpus.py",
    "tests/master_teacher_corpus_v1/test_master_teacher_corpus.py",
    "docs/MASTER-TEACHER-CORPUS-V1.md",
    ".github/workflows/master-teacher-corpus.yml",
    ".github/workflows/master-teacher-corpus-contract.yml",
)


def contract_closure():
    root = Path(__file__).resolve().parents[1]
    return {name: digest(require_regular(root / name)) for name in CONTRACT_FILES}


def canonical(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def require_regular(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"expected regular, unlinked-only file: {path}")
    return path.read_bytes()


def private_directory(path):
    # Do not create world-traversable intermediate directories under a loose umask.
    path.mkdir(mode=0o700, parents=False, exist_ok=False)


def write_exclusive(path, data):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())


def append_sync(path, data):
    flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "ab") as file:
        info = os.fstat(file.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("attempt journal is not a regular single-link file")
        file.write(data)
        file.flush()
        os.fsync(file.fileno())


def local_url(url):
    p = urllib.parse.urlsplit(url)
    if p.scheme != "http" or p.hostname not in ("127.0.0.1", "localhost") or p.username or p.password or p.query or p.fragment or p.path not in ("", "/"):
        raise ValueError("only direct loopback HTTP Ollama is supported")
    if not (1 <= (p.port or 80) <= 65535):
        raise ValueError("invalid port")
    return url.rstrip("/")


class APIResponseError(ValueError):
    """An HTTP/JSON failure with a preserved bounded response byte prefix."""

    def __init__(self, reason, body=b"", status=None, truncated=False):
        super().__init__(reason)
        self.body = body
        self.status = status
        self.truncated = truncated


def request_json(url, path, payload=None, return_raw=False):
    data = canonical(payload) if payload is not None else None
    req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"}, method="POST" if data is not None else "GET")
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *unused):
            return None  # urllib raises HTTPError, carrying the redirect body.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect)
    try:
        with opener.open(req, timeout=90) as res:
            body = res.read(MAX_RESPONSE + 1)
            status = res.status
    except urllib.error.HTTPError as exc:
        raw = exc.read(MAX_RESPONSE + 1)
        raise APIResponseError(f"Ollama HTTP status {exc.code}", raw[:MAX_RESPONSE], exc.code, len(raw) > MAX_RESPONSE) from exc
    if len(body) > MAX_RESPONSE:
        raise APIResponseError("Ollama response exceeds maximum size", body[:MAX_RESPONSE], status, True)
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise APIResponseError("malformed Ollama API JSON envelope", body, status) from exc
    return (parsed, body) if return_raw else parsed


def model_digest(url, name):
    tags = request_json(url, "/api/tags")
    candidates = [m for m in tags.get("models", []) if m.get("name") == name or m.get("model") == name]
    if len(candidates) != 1 or not isinstance(candidates[0].get("digest"), str):
        raise ValueError(f"model not installed with unique digest: {name}")
    sha = candidates[0]["digest"]
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("invalid model digest")
    return "sha256:" + sha


def formal_items():
    items = []
    for modulus, split in [(5, "train"), (7, "train"), (11, "development"), (13, "locked_test")]:
        for index in range(2):
            a, b = (2 * index + 1) % modulus, (3 * index + 2) % modulus
            items.append(dict(id=f"mod-{modulus}-{index}", lane="formal", task="modular_add", family=f"mod-{modulus}", split=split,
                              prompt=f"Return ONLY JSON of the form {{\"answer\":\"N\"}} with N the integer value of ({a}+{b}) mod {modulus}.", answer=str((a+b)%modulus)))
    for operation, split in [("sum", "train"), ("swap", "development"), ("inverse_sum", "locked_test")]:
        for index in range(2):
            a, b = index, 2-index
            if operation == "sum":
                result = (a, (a+b)%3)
            elif operation == "swap":
                result = (b, a)
            else:
                result = (a, (b-a)%3)
            items.append(dict(id=f"gf3-{operation}-{index}", lane="formal", task="qutrit_gf3", family=f"gf3-{operation}", split=split,
                              prompt=f"In GF(3), apply {operation} to pair ({a},{b}). Return ONLY JSON {{\"answer\":\"x,y\"}} using digits 0..2.", answer=f"{result[0]},{result[1]}"))
    for context, split in [(0, "train"), (1, "train"), (2, "development"), (3, "locked_test")]:
        for index in range(2):
            x, y = index, (context + 1) % 2
            result = x ^ y ^ (context % 2)
            items.append(dict(id=f"context-{context}-{index}", lane="formal", task="context_symbol", family=f"context-{context}", split=split,
                              prompt=f"A context table defines f(x,y,c)=x XOR y XOR (c mod 2) for binary x,y. With x={x}, y={y}, c={context}, return ONLY JSON {{\"answer\":\"0 or 1\"}}.", answer=str(result)))
    return items


def vision_items():
    items = []
    for group, split, masks in [
        ("sparse", "train", [(0, 8), (1, 7)]),
        ("balanced", "development", [(0, 2, 4, 6), (1, 3, 5, 7)]),
        ("dense", "locked_test", [(0, 1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 6, 7, 8)]),
    ]:
        for i, cells in enumerate(masks):
            items.append(dict(id=f"grid-{group}-{i}", lane="vision", task="visual_grid_count", family=f"grid-{group}", split=split,
                              prompt='Count the BLACK filled squares in this 3 by 3 grid. Return ONLY JSON {"answer":"N"} with N a decimal integer.',
                              cells=list(cells), answer=str(len(cells))))
    return items


def png_grid(cells):
    # Exactly specified 3x3 black/white raster: no pillow or external encoder.
    width, height, tile = 192, 192, 64
    raw = bytearray()
    marked = set(cells)
    for y in range(height):
        raw.append(0)
        for x in range(width):
            col, row = x // tile, y // tile
            inner_x, inner_y = x % tile, y % tile
            edge = inner_x < 3 or inner_y < 3 or inner_x >= tile-3 or inner_y >= tile-3
            val = 175 if edge else (0 if (row * 3 + col) in marked else 255)
            raw += bytes((val, val, val))
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def independent_oracle(item):
    """Second calculation path; does not trust generator-stored answer."""
    if item["task"] == "modular_add":
        _, modulus, index = item["id"].split("-")
        n, i = int(modulus), int(index)
        total = ((2*i+1) % n) + ((3*i+2) % n)
        while total >= n:
            total -= n
        return str(total)
    if item["task"] == "qutrit_gf3":
        operation = item["family"][4:]
        i = int(item["id"].rsplit("-", 1)[1])
        a, b = i, 2-i
        plus = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
        minus = ((0, 2, 1), (1, 0, 2), (2, 1, 0))
        answer = {"sum": (a, plus[a][b]), "swap": (b, a), "inverse_sum": (a, minus[b][a])}[operation]
        return f"{answer[0]},{answer[1]}"
    if item["task"] == "context_symbol":
        c = int(item["id"].split("-")[1])
        x = int(item["id"].split("-")[2])
        y = (c+1) % 2
        return str((x + y + (c % 2)) % 2)
    if item["task"] == "visual_grid_count":
        img = png_grid(item["cells"])
        pos = 8
        pixel_data = bytearray()
        while pos < len(img):
            length = struct.unpack(">I", img[pos:pos+4])[0]
            kind = img[pos+4:pos+8]
            payload = img[pos+8:pos+8+length]
            pos += 12+length
            if kind == b"IDAT":
                pixel_data.extend(zlib.decompress(payload))
        stride = 1 + 192*3
        count = 0
        for row in range(3):
            for col in range(3):
                offset = (row*64+32)*stride + 1 + (col*64+32)*3
                count += int(pixel_data[offset:offset+3] == b"\x00\x00\x00")
        return str(count)
    raise ValueError("unrecognized oracle task")


def public_item(item):
    return {key: value for key, value in item.items() if key != "answer"}


def generate_items():
    return formal_items() + vision_items()


def check_request(req):
    if set(req) != {"schema", "source_sha256", "contract_closure", "ollama_url", "models", "items", "max_requests", "sampling", "rights_status"} or req["schema"] != SCHEMA:
        raise ValueError("invalid frozen request fields")
    if req["rights_status"] != "QUARANTINED_NOT_ADMITTED":
        raise ValueError("capture is never admitted data")
    if req["source_sha256"] != digest(require_regular(Path(__file__))):
        raise ValueError("collector source hash changed")
    if req["contract_closure"] != contract_closure():
        raise ValueError("source/protocol/workflow/test closure changed")
    local_url(req["ollama_url"])
    if req["max_requests"] != len(req["items"]) or not (1 <= req["max_requests"] <= MAX_REQUESTS):
        raise ValueError("bounded request count mismatch")
    if req["items"] != [public_item(i) for i in generate_items()]:
        raise ValueError("generator/public items drift")
    if req["sampling"] != {"temperature": 0, "top_p": 1, "seed": 0, "num_predict": 64, "num_ctx": 2048}:
        raise ValueError("sampling policy drift")
    if set(req["models"]) != {"formal", "vision"} or not all(isinstance(x, dict) and set(x) == {"name", "digest"} for x in req["models"].values()):
        raise ValueError("model manifest malformed")
    for model in req["models"].values():
        if not isinstance(model["name"], str) or not model["name"] or not isinstance(model["digest"], str) or not model["digest"].startswith("sha256:") or len(model["digest"]) != 71:
            raise ValueError("invalid model identity")


def plan(args):
    url = local_url(args.url)
    items = generate_items()
    if any(i["answer"] != independent_oracle(i) for i in items):
        raise ValueError("independent oracle mismatch")
    if len(items) > MAX_REQUESTS:
        raise ValueError("hard request ceiling exceeded")
    models = {"formal": {"name": args.formal_model, "digest": model_digest(url, args.formal_model)},
              "vision": {"name": args.vision_model, "digest": model_digest(url, args.vision_model)}}
    req = {"schema": SCHEMA, "source_sha256": digest(require_regular(Path(__file__))),
           "contract_closure": contract_closure(), "ollama_url": url,
           "models": models, "items": [public_item(i) for i in items], "max_requests": len(items),
           "sampling": {"temperature": 0, "top_p": 1, "seed": 0, "num_predict": 64, "num_ctx": 2048},
           "rights_status": "QUARANTINED_NOT_ADMITTED"}
    private_directory(args.out)
    write_exclusive(args.out / "request.json", canonical(req))
    print("request_identity=" + digest(canonical(req)))
    print("calls_planned=" + str(len(items)))


def strict_answer(raw):
    try:
        obj = json.loads(raw, object_pairs_hook=lambda pairs: no_duplicates(pairs))
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    if type(obj) is not dict or set(obj) != {"answer"} or type(obj["answer"]) is not str or len(obj["answer"]) > 32:
        return None
    return obj["answer"]


def no_duplicates(pairs):
    o = {}
    for key, val in pairs:
        if key in o:
            raise ValueError("duplicate JSON keys")
        o[key] = val
    return o


def capture(args):
    if not args.authorize_local_capture:
        raise ValueError("explicit --authorize-local-capture required")
    req_bytes = require_regular(args.request)
    if digest(req_bytes) != args.request_identity:
        raise ValueError("externally retained request identity differs")
    req = json.loads(req_bytes)
    check_request(req)
    url = req["ollama_url"]
    for model in req["models"].values():
        if model_digest(url, model["name"]) != model["digest"]:
            raise ValueError("installed Ollama model digest changed")
    private_directory(args.out)
    write_exclusive(args.out / "request.json", req_bytes)
    raw_path = args.out / "attempts.jsonl"
    write_exclusive(raw_path, b"")
    targets = []
    candidates = []
    for item in generate_items():
        if item["answer"] != independent_oracle(item):
            raise ValueError("independent oracle mismatch")
        model = req["models"][item["lane"]]
        if model_digest(url, model["name"]) != model["digest"]:
            raise ValueError("model digest drift during capture; partial evidence retained")
        body = {"model": model["name"], "stream": False, "messages": [{"role": "user", "content": item["prompt"]}], "options": req["sampling"]}
        if item["lane"] == "vision":
            body["messages"][0]["images"] = [base64.b64encode(png_grid(item["cells"])).decode("ascii")]
        result = {"item_id": item["id"], "model": model["name"], "model_digest": model["digest"],
                  "request_sha256": digest(canonical(body)), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "status": "runtime_error", "raw_content": None, "parsed_answer": None, "oracle_match": False,
                  "raw_response_base64": None, "response_meta": {}, "http_status": None,
                  "response_truncated": False, "error": "unknown"}
        try:
            response, raw_api = request_json(url, "/api/chat", body, return_raw=True)
            message = response.get("message") if isinstance(response, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str):
                content = None
            elif len(content.encode("utf-8")) > 4096:
                content = None
            value = strict_answer(content) if content is not None else None
            result.update(status="completed", raw_content=content, http_status=200,
                          raw_response_base64=base64.b64encode(raw_api).decode("ascii"),
                          parsed_answer=value, oracle_match=value == item["answer"], error=None,
                          response_meta={k: response.get(k) for k in ("done_reason", "total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration")})
        except APIResponseError as exc:
            result.update(raw_response_base64=base64.b64encode(exc.body).decode("ascii"),
                          http_status=exc.status, response_truncated=exc.truncated,
                          error=type(exc).__name__ + ": " + str(exc)[:300])
        except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError) as exc:
            result["error"] = type(exc).__name__ + ": " + str(exc)[:300]
        result["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        append_sync(raw_path, canonical(result))
        targets.append({"item_id": item["id"], "split": item["split"], "answer": item["answer"], "family": item["family"]})
        if item["split"] == "train" and result["status"] == "completed" and result["oracle_match"]:
            candidates.append({"item_id": item["id"], "split": "train", "task": item["task"],
                               "family": item["family"], "prompt": item["prompt"], "oracle_answer": item["answer"],
                               "teacher": model["name"], "teacher_digest": model["digest"],
                               "status": "CANDIDATE_UNADMITTED"})
    write_exclusive(args.out / "privileged-targets.jsonl", b"".join(canonical(x) for x in targets))
    write_exclusive(args.out / "train-candidates.jsonl", b"".join(canonical(x) for x in candidates))
    summary = summarize([json.loads(l) for l in raw_path.read_bytes().splitlines()], generate_items())
    write_exclusive(args.out / "summary.json", canonical(summary))
    if args.server_log is not None:
        data = require_regular(args.server_log)
        if len(data) > MAX_LOG:
            raise ValueError("server log over 8 MiB: partial capture retained; no manifest")
        write_exclusive(args.out / "ollama-serve.log", data)
    files = sorted(p.name for p in args.out.iterdir() if p.name != "manifest.json")
    manifest = {"schema": SCHEMA + ".manifest", "request_identity": args.request_identity,
                "files": {p: digest(require_regular(args.out / p)) for p in files},
                "status": "QUARANTINED_NOT_ADMITTED"}
    write_exclusive(args.out / "manifest.json", canonical(manifest))
    print("manifest_identity=" + digest(canonical(manifest)))
    print("out=" + str(args.out))


def summarize(attempts, items):
    table = {}
    for a, item in zip(attempts, items):
        key = item["lane"] + ":" + item["task"] + ":" + item["split"]
        entry = table.setdefault(key, {"calls": 0, "completed": 0, "schema_valid": 0, "oracle_match": 0, "runtime_errors": 0})
        entry["calls"] += 1
        entry["completed"] += int(a["status"] == "completed")
        entry["schema_valid"] += int(a["parsed_answer"] is not None)
        entry["oracle_match"] += int(a["oracle_match"])
        entry["runtime_errors"] += int(a["status"] != "completed")
    return {"schema": SCHEMA + ".summary", "status": "QUARANTINED_NOT_ADMITTED", "by_lane_task_split": table}


def verify(args):
    manifest_bytes = require_regular(args.run / "manifest.json")
    if digest(manifest_bytes) != args.manifest_identity:
        raise ValueError("independent manifest digest mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != SCHEMA + ".manifest" or manifest.get("status") != "QUARANTINED_NOT_ADMITTED":
        raise ValueError("manifest schema/status mismatch")
    members = {p.name for p in args.run.iterdir()}
    needed = {"request.json", "attempts.jsonl", "privileged-targets.jsonl", "train-candidates.jsonl", "summary.json"}
    if not isinstance(manifest.get("files"), dict) or not needed.issubset(manifest["files"]) or not set(manifest["files"]).issubset(needed | {"ollama-serve.log"}) or members != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("unexpected member set")
    for name, identity in manifest["files"].items():
        if digest(require_regular(args.run / name)) != identity:
            raise ValueError("member mismatch: " + name)
    req_bytes = require_regular(args.run / "request.json")
    if digest(req_bytes) != manifest["request_identity"]:
        raise ValueError("request digest mismatch")
    req = json.loads(req_bytes)
    check_request(req)
    items = generate_items()
    attempts = [json.loads(l) for l in require_regular(args.run / "attempts.jsonl").splitlines()]
    targets = [json.loads(l) for l in require_regular(args.run / "privileged-targets.jsonl").splitlines()]
    if len(attempts) != len(items) or len(targets) != len(items):
        raise ValueError("attempt or target membership mismatch")
    for item, a, t in zip(items, attempts, targets):
        if item["answer"] != independent_oracle(item):
            raise ValueError("independent oracle mismatch")
        if t != {"item_id": item["id"], "split": item["split"], "answer": item["answer"], "family": item["family"]}:
            raise ValueError("privileged independent oracle mismatch")
        if a.get("item_id") != item["id"] or a.get("model") != req["models"][item["lane"]]["name"] or a.get("model_digest") != req["models"][item["lane"]]["digest"]:
            raise ValueError("attempt identity mismatch")
        body = {"model": req["models"][item["lane"]]["name"], "stream": False,
                "messages": [{"role": "user", "content": item["prompt"]}], "options": req["sampling"]}
        if item["lane"] == "vision":
            body["messages"][0]["images"] = [base64.b64encode(png_grid(item["cells"])).decode("ascii")]
        if a.get("request_sha256") != digest(canonical(body)):
            raise ValueError("request payload identity mismatch")
        if a.get("status") == "completed":
            if a.get("http_status") != 200 or a.get("response_truncated") is not False:
                raise ValueError("completed attempt has invalid HTTP status or truncation")
            if type(a.get("raw_response_base64")) is not str:
                raise ValueError("completed attempt missing response bytes")
            response_raw = base64.b64decode(a.get("raw_response_base64", ""), validate=True)
            if len(response_raw) > MAX_RESPONSE:
                raise ValueError("response payload too large")
            response = json.loads(response_raw)
            message = response.get("message") if isinstance(response, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or len(content.encode("utf-8")) > 4096:
                content = None
            if content != a.get("raw_content"):
                raise ValueError("response payload mismatch")
            if a.get("parsed_answer") != (strict_answer(content) if content is not None else None) or a.get("oracle_match") is not (a.get("parsed_answer") == item["answer"]):
                raise ValueError("oracle replay mismatch")
            expected_meta = {k: response.get(k) for k in ("done_reason", "total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration")}
            if a.get("response_meta") != expected_meta or a.get("error") is not None:
                raise ValueError("response metadata mismatch")
        elif a.get("status") == "runtime_error":
            if a.get("oracle_match") is not False or a.get("parsed_answer") is not None or a.get("raw_content") is not None or a.get("response_meta") != {} or not isinstance(a.get("error"), str) or not a["error"]:
                raise ValueError("invalid failure metadata")
            encoded = a.get("raw_response_base64")
            status = a.get("http_status")
            truncated = a.get("response_truncated")
            if type(truncated) is not bool or (status is not None and (type(status) is not int or not 100 <= status <= 599)):
                raise ValueError("invalid failure HTTP metadata")
            if encoded is None:
                if status is not None or truncated:
                    raise ValueError("HTTP failure missing retained bytes")
            else:
                if type(encoded) is not str or status is None:
                    raise ValueError("failure bytes missing HTTP status")
                raw = base64.b64decode(encoded, validate=True)
                if len(raw) > MAX_RESPONSE or (truncated and len(raw) != MAX_RESPONSE):
                    raise ValueError("invalid retained failure byte bound")
                if status == 200 and not truncated:
                    try:
                        json.loads(raw)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
                    else:
                        raise ValueError("successful JSON incorrectly labelled runtime error")
        else:
            raise ValueError("invalid attempt status")
    expected_candidates = []
    for a, item in zip(attempts, items):
        if item["split"] == "train" and a["status"] == "completed" and a["oracle_match"]:
            model = req["models"][item["lane"]]
            expected_candidates.append({"item_id": item["id"], "split": "train", "task": item["task"],
                                        "family": item["family"], "prompt": item["prompt"], "oracle_answer": item["answer"],
                                        "teacher": model["name"], "teacher_digest": model["digest"],
                                        "status": "CANDIDATE_UNADMITTED"})
    if require_regular(args.run / "train-candidates.jsonl") != b"".join(canonical(i) for i in expected_candidates):
        raise ValueError("train candidate reconstruction mismatch")
    if require_regular(args.run / "summary.json") != canonical(summarize(attempts, items)):
        raise ValueError("summary mismatch")
    print("verified_capture_attempts=" + str(len(items)))
    print("matched_oracle=" + str(sum(a["oracle_match"] for a in attempts)))
    print("status=QUARANTINED_NOT_ADMITTED")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("plan", help="freeze public tasks and installed local model digests; no inference")
    a.add_argument("--url", default=DEFAULT_URL)
    a.add_argument("--formal-model", required=True)
    a.add_argument("--vision-model", required=True)
    a.add_argument("--out", type=Path, required=True)
    a.set_defaults(fn=plan)
    a = sub.add_parser("capture", help="bounded local inference; quarantined only")
    a.add_argument("--request", type=Path, required=True)
    a.add_argument("--request-identity", required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--server-log", type=Path)
    a.add_argument("--authorize-local-capture", action="store_true")
    a.set_defaults(fn=capture)
    a = sub.add_parser("verify", help="offline integrity and oracle-check recomputation")
    a.add_argument("--run", type=Path, required=True)
    a.add_argument("--manifest-identity", required=True)
    a.set_defaults(fn=verify)
    args = p.parse_args(argv)
    try:
        args.fn(args)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        p.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
