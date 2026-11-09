#!/usr/bin/env python3
"""TRI-CHAT-001: isolated, loopback-only local conversational prototype.

No native Trinite imports, training, model downloads, dynamic commands or d1 calls.
This module is an application experiment, not evidence of native model capability.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import io
import json
import operator
import os
from pathlib import Path
import re
import secrets
import socket
import stat
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

SCHEMA = "trinite.chat.request.v1"
RESULT = "trinite.chat.result.v1"
MAX_REQUEST = 16 * 1024
MAX_MESSAGE = 8 * 1024
MAX_HISTORY = 16
MAX_TEACHER = 64 * 1024
TEACHER_DEADLINE = 30.0
HEADER_DEADLINE = 2.0
BODY_DEADLINE = 5.0
MAX_GENERATED_TOKENS = 512
UI = Path(__file__).resolve().parent / "trinite_chat_v1_ui"
INT = r"-?(?:0|[1-9][0-9]{0,11})"
BASIC = re.compile(
    r"^(?:(?:what\s+is|calculate|compute)\s+)?(" + INT +
    r")\s*([+*x×])\s*(" + INT + r")\s*\??$", re.I
)
GF3 = re.compile(
    r"^(?:gf3|gf\(3\))\s*\(\s*(" + INT +
    r")\s*([+*])\s*(" + INT + r")\s*\)\s*\??$", re.I
)
FORMAL_PREFIX = re.compile(
    r"^\s*(?:calculate\b|compute\b|gf3\b|gf\(3\)|[-+]?[0-9]|what\s+is\s+[-+]?[0-9])", re.I
)


class DeadlineSocketReader(io.RawIOBase):
    """Buffered-read source with a single absolute deadline across all recv calls.

    Socket receive timeouts alone allow a peer to keep a blocking read alive
    indefinitely by trickling bytes. The deadline is rechecked per raw recv,
    including buffered HTTP header parsing and response bodies.
    """

    def __init__(self, sock, deadline):
        super().__init__()
        self.sock = sock
        self.deadline = deadline

    def readable(self):
        return True

    def readinto(self, b):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("stream_deadline_exceeded")
        self.sock.settimeout(remaining)
        return self.sock.recv_into(b)


class DeadlineHTTPResponse(http.client.HTTPResponse):
    """Use the same absolute deadline for status/header parsing and body bytes."""

    def __init__(self, sock, *, deadline, **kwargs):
        super().__init__(sock, **kwargs)
        # Retain the original socket.makefile reference until response close.
        # HTTPConnection closes its socket after HTTP/1.0 response headers;
        # closing the makefile here would drop the last IO reference and close
        # the descriptor before the buffered body could be read.
        self._socket_lifetime_fp = self.fp
        self.fp = io.BufferedReader(DeadlineSocketReader(sock, deadline))

    def close(self):
        try:
            super().close()
        finally:
            self._socket_lifetime_fp.close()


class Invalid(ValueError):
    """Strict request rejection."""


class TeacherFault(RuntimeError):
    """A bounded backend failure without exposing private response bytes."""

    def __init__(self, code):
        super().__init__(code)
        self.code = code


def strict_json(raw):
    def pairs(entries):
        result = {}
        for key, value in entries:
            if key in result:
                raise Invalid("duplicate_json_key")
            result[key] = value
        return result

    def reject_constant(_):
        raise Invalid("nonfinite_json")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, Invalid):
            raise
        raise Invalid("malformed_json") from None


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def valid_text(value, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise Invalid("invalid_text")
    try:
        raw = value.encode("utf-8", errors="strict")
    except UnicodeError:
        raise Invalid("invalid_unicode") from None
    if len(raw) > MAX_MESSAGE or "\x00" in value:
        raise Invalid("message_limit")
    return value


def parse_request(body):
    data = strict_json(body)
    if not isinstance(data, dict) or set(data) != {"schema", "message", "history"}:
        raise Invalid("invalid_request_fields")
    if data["schema"] != SCHEMA:
        raise Invalid("invalid_schema")
    message = valid_text(data["message"])
    history = data["history"]
    if not isinstance(history, list) or len(history) > MAX_HISTORY or len(history) % 2:
        raise Invalid("invalid_history")
    cleaned = []
    for i, entry in enumerate(history):
        role = "user" if i % 2 == 0 else "assistant"
        if not isinstance(entry, dict) or set(entry) != {"role", "content"}:
            raise Invalid("invalid_history_fields")
        if entry["role"] != role:
            raise Invalid("invalid_history_roles")
        cleaned.append({"role": role, "content": valid_text(entry["content"])})
    return {"schema": SCHEMA, "message": message, "history": cleaned}


def formal_task(text):
    stripped = text.strip()
    match = GF3.fullmatch(stripped)
    if match:
        a, sign, b = match.groups()
        return (int(a), sign, int(b), True)
    match = BASIC.fullmatch(stripped)
    if match:
        a, sign, b = match.groups()
        return (int(a), "*" if sign in ("x", "×") else sign, int(b), False)
    return None


def primary_exact(task):
    a, op, b, gf3 = task
    f = operator.add if op == "+" else operator.mul
    answer = f(a, b)
    return answer % 3 if gf3 else answer


def independent_oracle(task):
    """Separate bounded integer derivation; no model-provided labels."""
    a, op, b, gf3 = task
    if gf3:
        a, b = divmod(a, 3)[1], divmod(b, 3)[1]
    if op == "+":
        value = sum((a, b))
    else:
        neg = (a < 0) != (b < 0)
        x, y = abs(a), abs(b)
        value = 0
        while y:
            if y & 1:
                value += x
            x <<= 1
            y >>= 1
        if neg:
            value = -value
    return divmod(value, 3)[1] if gf3 else value


def choose_route(message):
    task = formal_task(message)
    if task is not None:
        return "formal", task
    if FORMAL_PREFIX.match(message):
        return "unsupported", None
    return "chat", None


def file_hash(path):
    """Reject symlink weight paths; verify a regular file before local access."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise Invalid("invalid_model_file")
        h = hashlib.sha256()
        with os.fdopen(os.dup(fd), "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    finally:
        os.close(fd)


class Config:
    def __init__(self, *, model_port=None, model_id=None, model_file=None,
                 model_sha256=None, runtime_version=None, chat_template=None,
                 authorize_local_teacher=False):
        fields = (model_port, model_id, model_file, model_sha256, runtime_version, chat_template)
        if any(x is not None for x in fields) and not all(x is not None for x in fields):
            raise Invalid("incomplete_model_configuration")
        self.active = all(x is not None for x in fields) and authorize_local_teacher
        if all(x is not None for x in fields) and not authorize_local_teacher:
            raise Invalid("local_teacher_authorization_required")
        if model_port is not None:
            if not isinstance(model_port, int) or isinstance(model_port, bool) or not 1 <= model_port <= 65535:
                raise Invalid("invalid_model_port")
            if not isinstance(model_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", model_sha256) is None:
                raise Invalid("invalid_model_sha256")
            if any(not isinstance(s, str) or not 0 < len(s) <= 256 for s in (model_id, runtime_version, chat_template)):
                raise Invalid("invalid_model_metadata")
            try:
                actual = file_hash(model_file)
            except (OSError, Invalid):
                raise Invalid("invalid_model_file") from None
            if actual != model_sha256:
                raise Invalid("model_file_hash_mismatch")
        self.port = model_port
        self.model_id = model_id
        self.model_sha256 = model_sha256
        self.runtime_version = runtime_version
        self.chat_template = chat_template
        self.policy_sha256 = digest(canonical({
            "schema": RESULT, "routes": ["formal", "chat"],
            "backend": "127.0.0.1:" + str(model_port) if self.active else None,
            "model_id": model_id if self.active else None,
            "model_sha256": model_sha256 if self.active else None,
            "runtime_version": runtime_version if self.active else None,
            "chat_template": chat_template if self.active else None,
            "limits": [MAX_REQUEST, MAX_MESSAGE, MAX_HISTORY, MAX_TEACHER, TEACHER_DEADLINE],
        }))


def teacher_chat(config, message, history):
    if not config.active:
        raise TeacherFault("teacher_not_configured")
    req = {
        "model": config.model_id,
        "messages": [{"role": "system", "content": (
            "You are a local conversational teacher. You cannot authorize tool calls. "
            "Exact arithmetic, if any, is not independently verified by your prose."
        )}] + history + [{"role": "user", "content": message}],
        "stream": False, "temperature": 0, "max_tokens": MAX_GENERATED_TOKENS,
    }
    payload = canonical(req)
    start = time.monotonic()
    conn = http.client.HTTPConnection("127.0.0.1", config.port, timeout=TEACHER_DEADLINE)
    deadline = start + TEACHER_DEADLINE
    conn.response_class = lambda sock, **kwargs: DeadlineHTTPResponse(
        sock, deadline=deadline, **kwargs
    )
    response = None
    try:
        conn.request("POST", "/v1/chat/completions", body=payload,
                     headers={"Content-Type": "application/json", "Accept": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            raise TeacherFault("teacher_http_failure")
        if response.getheader("Content-Encoding", "identity").lower() != "identity":
            raise TeacherFault("teacher_encoding_rejected")
        raw = bytearray()
        while True:
            if time.monotonic() >= deadline:
                raise TeacherFault("teacher_timeout")
            # The underlying raw socket reader independently reapplies the
            # absolute deadline to every recv, even if response.read waits for
            # more bytes or the peer sends many short chunks.
            chunk = response.read(min(4096, MAX_TEACHER + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > MAX_TEACHER:
                raise TeacherFault("teacher_output_limit")
        try:
            item = strict_json(bytes(raw))
        except Invalid:
            raise TeacherFault("teacher_malformed_response") from None
        if not isinstance(item, dict) or item.get("model") != config.model_id:
            raise TeacherFault("teacher_model_identity_mismatch")
        choices = item.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise TeacherFault("teacher_invalid_choices")
        choice = choices[0]
        if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
            raise TeacherFault("teacher_incomplete_response")
        message_obj = choice.get("message")
        if not isinstance(message_obj, dict) or message_obj.get("role") != "assistant":
            raise TeacherFault("teacher_invalid_message")
        result = message_obj.get("content")
        if not isinstance(result, str):
            raise TeacherFault("teacher_invalid_content")
        try:
            if not result.strip() or len(result.encode("utf-8")) > MAX_TEACHER:
                raise TeacherFault("teacher_invalid_content")
        except UnicodeError:
            raise TeacherFault("teacher_invalid_content") from None
        return result
    except (TimeoutError, socket.timeout) as exc:
        raise TeacherFault("teacher_timeout") from None
    except (ConnectionError, OSError, http.client.HTTPException) as exc:
        # io.BufferedReader may surface a socket timeout as a plain OSError
        # ('cannot read from timed out object') rather than TimeoutError.
        # The absolute monotonic deadline, not the exception wording, owns
        # the timeout classification.
        if time.monotonic() >= deadline:
            raise TeacherFault("teacher_timeout") from None
        raise TeacherFault("teacher_unavailable") from None
    finally:
        if response is not None:
            response.close()
        conn.close()


def result_for(request, config, adapter=teacher_chat, verifier=independent_oracle):
    route, task = choose_route(request["message"])
    receipt = {
        "schema": "trinite.chat.receipt.v1",
        "request_sha256": digest(canonical(request)),
        "policy_sha256": config.policy_sha256,
        "model_sha256": config.model_sha256 if config.active and route == "chat" else None,
        "route": route,
        "replay_scope": "controller_record_only; not fresh neural inference",
    }
    base = {"schema": RESULT, "route": route, "text": "", "exact": None,
            "status": "blocked", "reason": None, "receipt": receipt}
    if route == "unsupported":
        base["reason"] = "unsupported_formal_expression"
    elif route == "formal":
        value = primary_exact(task)
        if value != verifier(task):
            base["status"] = "failed"
            base["reason"] = "independent_oracle_mismatch"
        else:
            base["exact"] = {"value": str(value), "operation": "gf3" if task[3] else "integer",
                             "verified": True}
            base["text"] = "Verified exact result: " + str(value)
            base["status"] = "verified_exact"
    else:
        try:
            base["text"] = adapter(config, request["message"], request["history"])
            base["status"] = "model_generated_unverified"
        except TeacherFault as exc:
            base["status"] = "failed" if config.active else "blocked"
            base["reason"] = exc.code
    receipt["outcome"] = base["status"]
    return base



PERSIST_REQUEST = "trinite.chat.persist.request.v1"
SESSION_REQUEST = "trinite.chat.session.request.v1"
SESSION_RESULT = "trinite.chat.session.result.v1"
SESSION_ID = re.compile(r"[0-9a-f]{32}\Z")


def session_id(value):
    if not isinstance(value, str) or SESSION_ID.fullmatch(value) is None:
        raise Invalid("invalid_session_id")
    return value


def parse_persist_request(raw):
    data = strict_json(raw)
    if not isinstance(data, dict) or set(data) != {"schema", "message", "session_id"}:
        raise Invalid("invalid_persist_fields")
    if data["schema"] != PERSIST_REQUEST:
        raise Invalid("invalid_schema")
    message = valid_text(data["message"])
    return message, None if data["session_id"] is None else session_id(data["session_id"])


def parse_session_request(raw):
    data = strict_json(raw)
    if not isinstance(data, dict) or set(data) != {"schema", "action", "session_id"}:
        raise Invalid("invalid_session_fields")
    if data["schema"] != SESSION_REQUEST or data["action"] not in ("load", "delete"):
        raise Invalid("invalid_session_action")
    return data["action"], session_id(data["session_id"])


def selected_history(rows, message):
    """Keep only a contiguous, bounded suffix of complete stored exchanges.

    An oversized teacher response is displayable from disk but never passed
    back to a model as a history entry: terminate history selection there.
    """
    if len(rows) % 2:
        raise Invalid("invalid_stored_history")
    result = []
    for i in range(0, len(rows), 2):
        user, assistant = rows[i:i + 2]
        if user[0] != "user" or user[2] != "user" or assistant[0] != "assistant":
            raise Invalid("invalid_stored_history")
        if assistant[2] not in ("verified_exact", "model_generated_unverified"):
            raise Invalid("invalid_stored_history")
    for i in range(len(rows) - 2, -1, -2):
        user, assistant = rows[i:i + 2]
        if (len(user[1].encode("utf-8")) > MAX_MESSAGE or
                len(assistant[1].encode("utf-8")) > MAX_MESSAGE):
            break
        pair = [{"role": "user", "content": user[1]},
                {"role": "assistant", "content": assistant[1]}]
        candidate = pair + result
        if (len(candidate) > MAX_HISTORY or
                len(canonical({"schema": SCHEMA, "message": message,
                               "history": candidate})) > MAX_REQUEST):
            break
        result = candidate
    return result


class LocalSessions:
    """Opt-in local SQLite layer; no implicit initialization, model or cloud."""

    def __init__(self, path):
        import importlib.util
        module_path = Path(__file__).resolve().with_name("trinite_chat_sqlite_v1.py")
        spec = importlib.util.spec_from_file_location("trinite_chat_sqlite_v1", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.store = module
        self.path = Path(path)
        # Server startup validates an existing owner-authorized DB; no auto
        # creation or persistent schema migration is permitted here.
        connection = module.open_store(self.path, enable_persistence=True)
        connection.close()

    def tail(self, identifier):
        return self.store.load_session_tail(self.path, identifier, limit=MAX_HISTORY)

    def append(self, identifier, user, answer, classification):
        self.store.record_exchange(
            self.path, identifier, user, answer,
            classification=classification, enable_persistence=True
        )

    def delete(self, identifier):
        return self.store.delete_conversation(
            self.path, identifier, enable_persistence=True
        )


def persisted_result(message, identifier, config, sessions, adapter):
    if identifier is None:
        identifier = secrets.token_hex(16)
        rows = []
    else:
        rows = sessions.tail(identifier)
        if rows is None:
            raise Invalid("unknown_session")
    history = selected_history(rows, message)
    request = {"schema": SCHEMA, "message": message, "history": history}
    response = result_for(request, config, adapter)
    # Never save blocked/failed outcomes as completed exchange pairs.
    response["persisted"] = False
    response["session_id"] = identifier if rows else None
    if response["status"] in ("verified_exact", "model_generated_unverified"):
        sessions.append(identifier, message, response["text"], response["status"])
        response["persisted"] = True
        response["session_id"] = identifier
    return response


def http_handler(config, adapter=teacher_chat, sessions=None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def setup(self):
            super().setup()
            self.rfile.close()
            self._deadline_reader = DeadlineSocketReader(
                self.connection, time.monotonic() + HEADER_DEADLINE
            )
            self.rfile = io.BufferedReader(self._deadline_reader)

        def log_message(self, *_):
            """Never log user text, query strings or model output."""

        def secure_headers(self, mime, length):
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", (
                "default-src 'none'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
            ))

        def reply(self, code, obj):
            raw = canonical(obj)
            self.send_response(code)
            self.secure_headers("application/json; charset=utf-8", len(raw))
            self.end_headers()
            self.wfile.write(raw)

        def fail(self, status, reason):
            self.reply(status, {"schema": RESULT, "status": "blocked", "reason": reason})

        def validate_browser(self):
            port = self.server.server_port
            expected_host = "127.0.0.1:" + str(port)
            if self.headers.get_all("Host", []) != [expected_host]:
                raise Invalid("host_denied")
            origin = self.headers.get_all("Origin", [])
            if origin and origin != ["http://" + expected_host]:
                raise Invalid("origin_denied")
            if self.headers.get_all("Transfer-Encoding") or self.headers.get_all("Content-Encoding"):
                raise Invalid("unsupported_encoding")

        def do_GET(self):
            try:
                self.validate_browser()
            except Invalid as exc:
                self.fail(403, str(exc))
                return
            assets = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/style.css": ("style.css", "text/css; charset=utf-8"),
            }
            if self.path not in assets:
                self.fail(404, "not_found")
                return
            name, mime = assets[self.path]
            data = (UI / name).read_bytes()
            self.send_response(200)
            self.secure_headers(mime, len(data))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            try:
                self.validate_browser()
            except Invalid as exc:
                self.fail(403, str(exc))
                return
            if self.path not in ("/api/chat", "/api/session"):
                self.fail(404, "not_found")
                return
            if self.headers.get_all("Content-Type", []) != ["application/json"]:
                self.fail(415, "content_type_denied")
                return
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or not re.fullmatch(r"[0-9]+", lengths[0]):
                self.fail(400, "content_length_required")
                return
            length = int(lengths[0])
            if length > MAX_REQUEST:
                self.fail(413, "request_limit")
                return
            if length == 0:
                self.fail(400, "empty_request")
                return
            # The same reader that bounded request lines and headers now
            # bounds the complete body by an independent absolute deadline.
            self._deadline_reader.deadline = time.monotonic() + BODY_DEADLINE
            try:
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise Invalid("incomplete_body")
                data = strict_json(raw)
                if not isinstance(data, dict):
                    raise Invalid("invalid_request_fields")
                schema = data.get("schema")
                if self.path == "/api/session":
                    action, identifier = parse_session_request(raw)
                    if sessions is None:
                        self.fail(403, "persistence_disabled")
                        return
                    try:
                        if action == "delete":
                            removed = bool(sessions.delete(identifier))
                            self.reply(200, {"schema": SESSION_RESULT, "status": "ok",
                                             "deleted": removed})
                        else:
                            rows = sessions.tail(identifier)
                            if rows is None:
                                self.fail(404, "unknown_session")
                                return
                            self.reply(200, {"schema": SESSION_RESULT, "status": "ok",
                                             "session_id": identifier,
                                             "messages": [{"role": row[0], "content": row[1],
                                                           "classification": row[2]}
                                                          for row in rows]})
                    except (sessions.store.StoreError, sessions.store.sqlite3.Error, OSError):
                        self.fail(503, "storage_unavailable")
                    return
                if schema == PERSIST_REQUEST:
                    if sessions is None:
                        self.fail(403, "persistence_disabled")
                        return
                    message, identifier = parse_persist_request(raw)
                    try:
                        response = persisted_result(message, identifier, config,
                                                    sessions, adapter)
                    except (sessions.store.StoreError, sessions.store.sqlite3.Error, OSError):
                        self.fail(503, "storage_unavailable")
                        return
                    self.reply(200 if response["status"] != "failed" else 502, response)
                    return
                req = parse_request(raw)
            except (TimeoutError, socket.timeout):
                self.fail(408, "request_timeout")
                return
            except (Invalid, UnicodeError) as exc:
                self.fail(400, str(exc))
                return
            response = result_for(req, config, adapter)
            self.reply(200 if response["status"] != "failed" else 502, response)

        def do_HEAD(self):
            self.fail(405, "method_denied")

        def do_OPTIONS(self):
            self.fail(405, "method_denied")

        def do_PUT(self):
            self.fail(405, "method_denied")

        def do_DELETE(self):
            self.fail(405, "method_denied")

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description="TRI-CHAT-001: loopback chat prototype")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-port", type=int)
    parser.add_argument("--model-id")
    parser.add_argument("--model-file")
    parser.add_argument("--model-sha256")
    parser.add_argument("--runtime-version")
    parser.add_argument("--chat-template")
    parser.add_argument("--authorize-local-teacher", action="store_true")
    parser.add_argument("--enable-persistence", action="store_true",
                        help="allow explicit browser opt-in to existing local SQLite DB")
    parser.add_argument("--db-path", help="path to preinitialized local SQLite DB")
    args = parser.parse_args(argv)
    if args.db_path is not None and not args.enable_persistence:
        parser.error("db_path_requires_enable_persistence")
    if not 1 <= args.port <= 65535 or args.model_port == args.port:
        parser.error("invalid_listening_port")
    try:
        config = Config(model_port=args.model_port, model_id=args.model_id,
                        model_file=args.model_file, model_sha256=args.model_sha256,
                        runtime_version=args.runtime_version, chat_template=args.chat_template,
                        authorize_local_teacher=args.authorize_local_teacher)
    except Invalid as exc:
        parser.error(str(exc))
    sessions = None
    if args.enable_persistence:
        try:
            sessions = LocalSessions(args.db_path or (
                Path(__file__).resolve().parent.parent / "data" / "trinite.db"
            ))
        except Exception as exc:
            # Never fall back to in-tab mode without an explicit owner choice.
            parser.error("persistence_preflight_failed: " + type(exc).__name__)
    server = HTTPServer(("127.0.0.1", args.port),
                        http_handler(config, sessions=sessions))
    print("TRI-CHAT-001 listening at http://127.0.0.1:" + str(args.port),
          "| local teacher:", "enabled" if config.active else "disabled",
          "| SQLite opt-in:", "available" if sessions is not None else "disabled",
          flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
