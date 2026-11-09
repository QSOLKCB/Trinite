#!/usr/bin/env python3
"""Owner-only execution wrapper for MTC-LFM3-001 GitHub self-hosted dispatch.

Never checkout branch-selected code or upload private artifacts from this job.
The caller must configure a protected Actions environment and restrict the runner.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import http.client
import importlib.util
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import time

RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}\Z")
SHA = re.compile(r"[0-9a-f]{40}\Z")
CHECKSUM = re.compile(r"sha256:[0-9a-f]{64}\Z")


def run(command):
    subprocess.run(command, check=True)


def stdout(command):
    return subprocess.check_output(command, text=True).strip()


def verify_checkout(root, expected):
    if SHA.fullmatch(expected) is None:
        raise ValueError("invalid reviewed SHA")
    if Path.cwd().resolve() != root.resolve() or (
            Path(__file__).resolve().parents[1] != root.resolve()):
        raise ValueError("owner launcher not in reviewed checkout")
    if stdout(["git", "rev-parse", "HEAD"]) != expected:
        raise ValueError("local checkout SHA does not match dispatch")
    if stdout(["git", "status", "--porcelain", "--untracked-files=normal"]):
        raise ValueError("local checkout has unreviewed edits or untracked files")


def private_root():
    base = Path.home() / "trinite-mtc-lfm3"
    if base.is_symlink():
        raise ValueError("private capture root must not be symlink")
    if not base.exists():
        base.mkdir(mode=0o700)
    info = base.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("capture root is not directory")
    if os.name == "posix" and (info.st_uid != os.geteuid() or
                               stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError("private capture root requires owner mode 0700")
    return base


def bool_arg(text):
    if text not in ("true", "false"):
        raise ValueError("expected explicit true or false")
    return text == "true"


def ollama_available():
    conn = http.client.HTTPConnection("127.0.0.1", 11434, timeout=2)
    try:
        conn.request("GET", "/api/version")
        response = conn.getresponse()
        return response.status == 200 and len(response.read(4096)) > 0
    except (OSError, TimeoutError, http.client.HTTPException):
        return False
    finally:
        conn.close()


def stop_group(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=4)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=4)


def ensure_ollama(root, run_id):
    if ollama_available():
        return None
    name = root / ("ollama-serve-" + run_id + ".log")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(name, flags, 0o600)
    log = os.fdopen(fd, "wb")
    env = os.environ.copy()
    env["OLLAMA_HOST"] = "127.0.0.1:11434"
    env["OLLAMA_NO_CLOUD"] = "1"
    try:
        process = subprocess.Popen(
            ["ollama", "serve"], stdout=log, stderr=subprocess.STDOUT,
            env=env, start_new_session=True
        )
    finally:
        log.close()
    for _ in range(35):
        if ollama_available():
            return process
        if process.poll() is not None:
            break
        time.sleep(1)
    stop_group(process)
    raise ValueError("local Ollama server did not become ready")


def store_preflight():
    # Use the existing reviewed storage API, without creating or replacing DB.
    script = Path("scripts/trinite_chat_sqlite_v1.py")
    spec = importlib.util.spec_from_file_location("trinite_sqlite_owner", script)
    store = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(store)
    connection = store.open_store(Path("data/trinite.db"), enable_persistence=True)
    connection.close()


def ensure_db():
    db = Path("data/trinite.db")
    if db.exists() or db.is_symlink():
        store_preflight()
        print("existing_sqlite=validated_no_replacement", flush=True)
        return
    # Exactly the explicitly requested command, only for a missing DB.
    run(["uv", "run", "--no-project", "python", "-I", "-B",
         "scripts/trinite_chat_sqlite_v1.py", "init", "--enable-persistence"])
    store_preflight()
    print("sqlite=initialized_and_validated", flush=True)


def port_is_busy():
    try:
        with socket.create_connection(("127.0.0.1", 8765), timeout=1):
            return True
    except (ConnectionRefusedError, TimeoutError, OSError):
        return False


def chat_smoke():
    store_preflight()
    if port_is_busy():
        raise ValueError("port 8765 already occupied; refusing false positive smoke")
    # Start only for a bounded health check; never leave an unmanaged server
    # behind on the persistent owner workstation after the Actions job exits.
    process = subprocess.Popen(
        ["uv", "run", "--no-project", "python", "-I", "-B",
         "scripts/run_trinite_chat_v1.py", "--port", "8765",
         "--enable-persistence", "--db-path", "data/trinite.db"],
        start_new_session=True, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(22):
            if process.poll() is not None:
                raise ValueError("chat process exited before health check")
            conn = http.client.HTTPConnection("127.0.0.1", 8765, timeout=2)
            try:
                conn.request("GET", "/", headers={"Host": "127.0.0.1:8765"})
                response = conn.getresponse()
                data = response.read(8192)
                if response.status == 200 and b"TRINITE" in data:
                    print("chat_sqlite_health=PASS_temporary_local_server", flush=True)
                    return
            except (OSError, TimeoutError, http.client.HTTPException):
                pass
            finally:
                conn.close()
            time.sleep(1)
        raise ValueError("chat did not pass bounded local GET smoke")
    finally:
        stop_group(process)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Owner-reviewed local capture launcher")
    parser.add_argument("--authorize-owner-local", action="store_true")
    parser.add_argument("--reviewed-sha", required=True)
    parser.add_argument("--operation", choices=("plan", "capture", "verify"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--models", default="")
    parser.add_argument("--request-identity", default="")
    parser.add_argument("--manifest-identity", default="")
    parser.add_argument("--resume", default="false")
    parser.add_argument("--initialize-db", default="false")
    parser.add_argument("--chat-smoke", default="false")
    args = parser.parse_args(argv)
    if not args.authorize_owner_local:
        parser.error("owner authorization required")
    if RUN_ID.fullmatch(args.run_id) is None:
        parser.error("invalid local run_id")
    root = Path.cwd()
    try:
        resume = bool_arg(args.resume)
        initialize = bool_arg(args.initialize_db)
        smoke = bool_arg(args.chat_smoke)
        verify_checkout(root, args.reviewed_sha)
        if os.environ.get("UV_OFFLINE") != "1":
            raise ValueError("UV_OFFLINE=1 required")
        base = private_root() / args.run_id
        server = None
        try:
            if args.operation != "verify":
                server = ensure_ollama(base.parent, args.run_id)
            cli = ["uv", "run", "--no-project", "python", "-I", "-B",
                   "scripts/mtc_lfm3_001.py"]
            if args.operation == "plan":
                if not args.models:
                    raise ValueError("five explicit LFM3 model tags required")
                base.mkdir(mode=0o700, exist_ok=False)
                run(cli + ["plan", "--models", args.models,
                           "--out", str(base / "plan")])
            elif args.operation == "capture":
                if CHECKSUM.fullmatch(args.request_identity) is None:
                    raise ValueError("frozen plan identity required")
                command = cli + ["capture", "--plan", str(base / "plan" / "plan.json"),
                                 "--request-identity", args.request_identity,
                                 "--out", str(base / "capture"),
                                 "--authorize-local-capture"]
                if resume:
                    command.append("--resume")
                run(command)
            else:
                if CHECKSUM.fullmatch(args.manifest_identity) is None:
                    raise ValueError("external manifest identity required")
                run(cli + ["verify", "--run", str(base / "capture"),
                           "--manifest-identity", args.manifest_identity])
            if initialize:
                ensure_db()
            if smoke:
                chat_smoke()
            print("private_run_directory=" + str(base), flush=True)
            print("no_cloud_upload_no_training_admission", flush=True)
        finally:
            stop_group(server)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(2, "MTC-LFM3-001 owner job failed: " +
                    type(exc).__name__ + ": " + str(exc)[:160] + "\n")


if __name__ == "__main__":
    main()
