"""Additive, exact-commit full-source supplement for the v2 negative result.

Does not touch or reseal the original Actions ZIP, PR #35 inner ZIP, or
historical source commitments. No network, Torch, or model weights required.
"""
import argparse
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile

REVIEWED_COMMIT = "93522bf8f18fe38970ceafdd2ace792f8922c382"
RESEARCH_RUN = 38355170260
INNER_ARCHIVE = "ac3b3627a27bcf6d43f89b31eeabaf1859949ae939784da06ba49916b495fdca"
ORIGINAL_SUMMARY = "a6d53d4adff35f806226333891a38dd472506752c3d22d70cc96c76e8e2e0a68"
ORIGINAL_REQUEST = "35dfbaafd2f85e5c6836bab1837bdc99c18a8b003ca7c46273f2f4d265ce37b1"
DATASET_DIGEST = "120b95ac68825b50b92dc0210d1f142a30980bfe5acedcfafe6d8f3cdadf83c4"
SCHEMA = "trinite.procedural-qec-full-source-supplement.v1"
ROOT = Path(__file__).resolve().parents[1]
LIMIT = 512 * 1024 * 1024
MAX_MEMBER = 64 * 1024 * 1024
MAX_FILES = 5000


class SourceError(ValueError):
    """Failed to preserve or independently verify the historical Git source."""


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def name_check(name):
    if (type(name) is not str or not name.startswith("source/")
            or "\\" in name or "\x00" in name or len(name) > 512
            or any(part in ("", ".", "..") for part in name.split("/"))
            or any(ord(c) < 32 or ord(c) == 127 for c in name)):
        raise SourceError("unsafe historical source member")
    return name


def sha_manifest(manifest):
    if type(manifest) is not dict or set(manifest) != {
        "schema", "git_commit", "git_tree", "run_id", "negative_summary_sha256",
        "original_request_sha256", "original_research_zip_sha256",
        "expected_dataset_sha256", "members", "scope"}:
        raise SourceError("wrong supplement schema/fields")
    if (manifest["schema"] != SCHEMA or manifest["git_commit"] != REVIEWED_COMMIT
            or manifest["run_id"] != RESEARCH_RUN
            or manifest["negative_summary_sha256"] != ORIGINAL_SUMMARY
            or manifest["original_request_sha256"] != ORIGINAL_REQUEST
            or manifest["original_research_zip_sha256"] != INNER_ARCHIVE
            or manifest["expected_dataset_sha256"] != DATASET_DIGEST
            or manifest["scope"] != "full Git tracked source tree; not a new trained model or independent numerical replication"):
        raise SourceError("supplement contradicts exact historical result")
    if type(manifest["git_tree"]) is not str or len(manifest["git_tree"]) != 40:
        raise SourceError("invalid historical Git tree identifier")
    return manifest


def git_output(*args):
    p = subprocess.run(("git", "-C", str(ROOT), *args), capture_output=True,
                       timeout=120, check=True)
    return p.stdout.strip().decode("ascii")


def members_at_exact_commit():
    if git_output("cat-file", "-t", REVIEWED_COMMIT) != "commit":
        raise SourceError("required historical commit is unavailable")
    tree = git_output("rev-parse", REVIEWED_COMMIT + "^{tree}")
    cmd = ("git", "-C", str(ROOT), "archive", "--format=tar",
           "--prefix=source/", REVIEWED_COMMIT)
    members, total = {}, 0
    with subprocess.Popen(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as process:
        with tarfile.open(fileobj=process.stdout, mode="r|") as tar:
            for file in tar:
                if file.isdir():
                    continue
                if not file.isfile() or file.issym() or file.islnk():
                    raise SourceError("historical Git tree contains unsupported member type")
                name = name_check(file.name)
                if name in members or file.size > MAX_MEMBER:
                    raise SourceError("duplicate or oversized tracked source")
                value = tar.extractfile(file).read(MAX_MEMBER + 1)
                if len(value) != file.size:
                    raise SourceError("incomplete original Git archive bytes")
                total += len(value)
                if total > LIMIT or len(members) >= MAX_FILES:
                    raise SourceError("historical source bundle exceeds bounds")
                members[name] = value
        if process.wait(timeout=60) != 0:
            raise SourceError("unable to archive exact historical commit")
    if "source/src/trinite/foundations_data.py" not in members:
        raise SourceError("missing transitive foundational dependency")
    if "source/src/provenance_core/__init__.py" not in members:
        raise SourceError("missing transitive provenance dependency")
    return tree, members


def isolated_data_regeneration(members):
    """Require import/data regeneration entirely from the retained source tree.

    -I -S means neither the checkout nor site-installed Torch is imported.
    """
    with tempfile.TemporaryDirectory(prefix="trinite-v2-source-only-") as directory:
        root = Path(directory)
        for name, raw in members.items():
            if name.startswith("source/src/"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(raw)
        program = (
            "import hashlib, pathlib, sys; "
            "sys.path.insert(0,str(pathlib.Path.cwd()/'source/src')); "
            "from trinite.procedural_qec import data; "
            "raw,manifest=data.dataset(); "
            "assert hashlib.sha256(raw).hexdigest() == " + repr(DATASET_DIGEST) + "; "
            "data.audit_bytes(raw,manifest); "
            "assert 'torch' not in sys.modules"
        )
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": ""}
        task = subprocess.run([sys.executable, "-I", "-S", "-c", program],
                              cwd=root, env=env, capture_output=True, text=True,
                              timeout=90)
        if task.returncode:
            raise SourceError("isolated historical dataset regeneration failed: "
                              + (task.stderr[-1500:] or task.stdout[-1500:]))


def serialize(items):
    out = BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, raw in sorted(items.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, raw, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    return out.getvalue()


def verify(path, external_digest):
    if type(external_digest) is not str or len(external_digest) != 64 or any(
            c not in "0123456789abcdef" for c in external_digest):
        raise SourceError("independent source supplement SHA-256 is required")
    data = Path(path).read_bytes()
    if sha(data) != external_digest or len(data) > LIMIT:
        raise SourceError("supplement ZIP differs from independently retained commitment")
    members = {}
    with zipfile.ZipFile(BytesIO(data)) as z:
        if len(z.infolist()) > MAX_FILES + 1:
            raise SourceError("oversized source supplement inventory")
        total = 0
        for info in z.infolist():
            if (info.filename in members or info.is_dir() or info.file_size > MAX_MEMBER
                    or (info.create_system == 3 and
                        stat.S_IFMT(info.external_attr >> 16) not in (0, stat.S_IFREG))):
                raise SourceError("invalid ZIP member type, size, or name")
            if info.filename != "MANIFEST.json":
                name_check(info.filename)
            total += info.file_size
            if total > LIMIT:
                raise SourceError("source supplement uncompressed size exceeded")
            members[info.filename] = z.read(info)
    manifest_raw = members.pop("MANIFEST.json")
    manifest = json.loads(manifest_raw)
    if canonical(manifest) != manifest_raw:
        raise SourceError("noncanonical supplement manifest")
    sha_manifest(manifest)
    if set(manifest["members"]) != set(members):
        raise SourceError("supplement missing or extra original tree objects")
    for name, commitment in manifest["members"].items():
        if (commitment != {"bytes": len(members[name]), "sha256": sha(members[name])}):
            raise SourceError("supplement entry SHA-256 or size mismatch")
    isolated_data_regeneration(members)
    return {"status": "passed", "archive_sha256": external_digest,
            "tree": manifest["git_tree"], "members": len(members),
            "dataset_regenerated_in_isolation": True,
            "original_evidence_mutated": False, "independent_numerical_replay": False}


def create(path):
    path = Path(path)
    if path.is_symlink() or path.exists() or any(p.is_symlink() for p in path.parents):
        raise SourceError("supplement must be a new regular output path")
    tree, members = members_at_exact_commit()
    isolated_data_regeneration(members)
    manifest = sha_manifest({
        "schema": SCHEMA, "git_commit": REVIEWED_COMMIT, "git_tree": tree,
        "run_id": RESEARCH_RUN, "negative_summary_sha256": ORIGINAL_SUMMARY,
        "original_request_sha256": ORIGINAL_REQUEST,
        "original_research_zip_sha256": INNER_ARCHIVE,
        "expected_dataset_sha256": DATASET_DIGEST,
        "scope": "full Git tracked source tree; not a new trained model or independent numerical replication",
        "members": {name: {"bytes": len(raw), "sha256": sha(raw)}
                    for name, raw in sorted(members.items())}})
    items = {**members, "MANIFEST.json": canonical(manifest)}
    packed = serialize(items)
    if len(packed) > LIMIT:
        raise SourceError("supplement ZIP size exceeded")
    with path.open("xb") as sink:
        sink.write(packed)
    return verify(path, sha(packed))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", type=Path)
    mode.add_argument("--verify", type=Path)
    p.add_argument("--expected-sha256")
    a = p.parse_args(argv)
    if a.create is not None and a.expected_sha256 is not None:
        p.error("creation computes a new independent digest; verify uses a supplied one")
    if a.verify is not None and a.expected_sha256 is None:
        p.error("verification requires externally supplied source ZIP digest")
    try:
        import json
        result = create(a.create) if a.create is not None else verify(a.verify, a.expected_sha256)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (SourceError, subprocess.CalledProcessError, subprocess.TimeoutExpired,
            OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
