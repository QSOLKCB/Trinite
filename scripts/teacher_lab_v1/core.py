"""TT-001 teacher preflight and source-bound, non-training artifacts.

No model construction, optimizer, dataset admission, or network access.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

from trinite.contracts import (
    ContractError, ModelConfig, exact_keys, identity, json_bytes, parse_json,
)
from model_size_policy_v1 import preflight_parameter_count

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = "configs/teacher/gf3-preflight-v1.json"
SCHEMA = "trinite.teacher-lab.v1"
PLAN_SCHEMA = "trinite.teacher-lab-plan.v1"
CATALOG_SCHEMA = "trinite.teacher-lab-candidates.v1"
MAX_INPUT = 1_048_576
SOURCE_CLOSURE = (
    "scripts/teacher_lab.py",
    "scripts/teacher_lab_v1/__init__.py",
    "scripts/teacher_lab_v1/core.py",
    "scripts/model_size_policy_v1.py",
    "src/trinite/__init__.py",
    "src/trinite/contracts.py",
    "src/trinite/model.py",
    "src/trinite/training.py",
    "docs/TEACHER-LAB-V1.md",
    "docs/DATA.md",
    "docs/MODEL-SIZE-POLICY.md",
    "docs/PROVENANCE.md",
    "AGENTS.md",
    "LICENSE",
)
SHA40 = re.compile(r"[0-9a-f]{40}\Z")
SOURCE_ID = re.compile(r"[a-z][a-z0-9-]{1,39}\Z")


def repo_file(relative: str) -> Path:
    """Read only a canonical, symlink-free repository-relative regular file."""
    if type(relative) is not str or not relative or "\\" in relative:
        raise ContractError("invalid repository path")
    parts = relative.split("/")
    if (relative.startswith("/") or any(p in ("", ".", "..") for p in parts)
            or PurePosixPath(relative).as_posix() != relative):
        raise ContractError("repository paths must be canonical and relative")
    path = ROOT
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ContractError("source path contains symlink")
    if not path.is_file() or path.stat().st_size > MAX_INPUT:
        raise ContractError("source file missing, nonregular, or too large")
    return path


def source_bytes(relative: str) -> bytes:
    path = repo_file(relative)
    with path.open("rb") as fh:
        raw = fh.read(MAX_INPUT + 1)
    if len(raw) > MAX_INPUT:
        raise ContractError("oversized source")
    return raw


def source_json(relative: str) -> dict:
    value = parse_json(source_bytes(relative))
    if type(value) is not dict:
        raise ContractError("expected JSON object")
    return value


def validate_catalog(value: object) -> dict:
    exact_keys(value, {"schema", "sources"}, "teacher source catalog")
    if value["schema"] != CATALOG_SCHEMA:
        raise ContractError("wrong source catalog schema")
    sources = value["sources"]
    if type(sources) is not list or not 1 <= len(sources) <= 16:
        raise ContractError("source candidate count must be 1..16")
    seen = set()
    for item in sources:
        exact_keys(item, {"id", "repository", "revision", "code_license",
                          "role", "rights_status", "scope", "limits"}, "candidate")
        sid = item["id"]
        if type(sid) is not str or SOURCE_ID.fullmatch(sid) is None or sid in seen:
            raise ContractError("invalid or duplicate candidate ID")
        seen.add(sid)
        if (type(item["repository"]) is not str or
                not item["repository"].startswith("QSOLKCB/") or
                not re.fullmatch(r"QSOLKCB/[A-Za-z0-9_.-]+", item["repository"])):
            raise ContractError("only explicit QSOLKCB source candidates")
        if type(item["revision"]) is not str or SHA40.fullmatch(item["revision"]) is None:
            raise ContractError("candidate requires immutable revision")
        if item["code_license"] not in ("MPL-2.0", "Apache-2.0", "MIT"):
            raise ContractError("candidate code license requires review")
        if item["role"] not in ("formal-task-candidate", "optimization-method-only"):
            raise ContractError("unexpected source role")
        if item["rights_status"] != "REVIEW_REQUIRED":
            raise ContractError("TT-001 never admits a source automatically")
        for field in ("scope", "limits"):
            if type(item[field]) is not str or not 8 <= len(item[field]) <= 512:
                raise ContractError("source scope and limits must be explicit")
    return value




def _method(tree: ast.Module, owner: str, name: str) -> ast.FunctionDef:
    """Locate one explicitly named native method without importing Torch."""
    cls = [node for node in tree.body
           if isinstance(node, ast.ClassDef) and node.name == owner]
    if len(cls) != 1:
        raise ContractError(f"native {owner} definition missing or ambiguous")
    methods = [node for node in cls[0].body
               if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(methods) != 1:
        raise ContractError(f"native {owner}.{name} not uniquely defined")
    return methods[0]


def native_limits() -> tuple[int, int]:
    """Conservative source-inspected native constructor and RunConfig caps.

    Parse the actual current native code; do not import model.py/training.py
    because TT-001 remains standard-library-only and allocation-free. Fail
    closed when the defining guard shape changes pending explicit review.
    This source inspection is not a proof of arbitrary Python equivalence.
    """
    try:
        model = ast.parse(source_bytes("src/trinite/model.py"))
        training = ast.parse(source_bytes("src/trinite/training.py"))
    except (SyntaxError, UnicodeError, ValueError) as exc:
        raise ContractError("invalid native model or training source") from exc

    maximums = [
        stmt.value.value
        for stmt in model.body
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id == "MAX_PARAMETERS"
        and isinstance(stmt.value, ast.Constant)
    ]
    if len(maximums) != 1 or type(maximums[0]) is not int or maximums[0] < 1:
        raise ContractError("native constructor parameter maximum changed shape")
    constructor = _method(model, "Decoder", "__init__")
    guards = [
        node for node in ast.walk(constructor)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "integer" and len(node.args) == 4
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "parameter_budget"
        and isinstance(node.args[3], ast.Name)
        and node.args[3].id == "MAX_PARAMETERS"
    ]
    if len(guards) != 1 or not any(
        isinstance(node, ast.Compare)
        and ast.unparse(node) == "config.parameter_count > parameter_budget"
        for node in ast.walk(constructor)
    ):
        raise ContractError("native constructor budget validation changed shape")

    run = _method(training, "RunConfig", "__post_init__")
    limits = [
        node.comparators[0].value
        for node in ast.walk(run)
        if isinstance(node, ast.Compare) and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Gt)
        and ast.unparse(node.left) == "self.model.parameter_count"
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
    ]
    if len(limits) != 1 or type(limits[0]) is not int or limits[0] < 1:
        raise ContractError("native training parameter limit changed shape")
    return maximums[0], limits[0]

def validate_plan(plan: object) -> dict:
    exact_keys(plan, {"schema", "role", "lane", "model_config", "parameter_budget",
                      "source_catalog", "objective", "training_authorized",
                      "data_admitted"}, "teacher plan")
    if plan["schema"] != PLAN_SCHEMA or plan["role"] != "teacher":
        raise ContractError("not a TT-001 teacher plan")
    if plan["lane"] != "dense":
        raise ContractError("TT-001 supports only dense preflight")
    if plan["training_authorized"] is not False or plan["data_admitted"] is not False:
        raise ContractError("TT-001 does not authorize training or data admission")
    if type(plan["parameter_budget"]) is not int or plan["parameter_budget"] < 1:
        raise ContractError("parameter budget must be a positive integer")
    constructor_limit, run_limit = native_limits()
    if plan["parameter_budget"] > min(constructor_limit, run_limit):
        raise ContractError("parameter budget exceeds current native run limit")
    if type(plan["objective"]) is not str or not 8 <= len(plan["objective"]) <= 128:
        raise ContractError("missing bounded objective")
    repo_file(plan["model_config"])
    repo_file(plan["source_catalog"])
    return plan


def inspect(plan_path: str = DEFAULT_PLAN) -> tuple[dict, dict, dict]:
    plan = validate_plan(source_json(plan_path))
    config = ModelConfig.from_dict(source_json(plan["model_config"]))
    catalog = validate_catalog(source_json(plan["source_catalog"]))
    count = preflight_parameter_count(config.parameter_count)
    constructor_limit, run_limit = native_limits()
    if count > constructor_limit or count > run_limit or count > plan["parameter_budget"]:
        raise ContractError("model exceeds existing constructor or requested run budget")
    return plan, config.to_dict(), catalog


def rights_records(plan: dict, catalog: dict) -> tuple[dict, dict]:
    data = {
        "schema": "trinite.teacher-lab-dataset-rights.v1",
        "status": "NOT_CREATED_NOT_ADMITTED",
        "scope": "source candidates only; no training examples, labels or corpora admitted",
        "candidate_ids": [item["id"] for item in catalog["sources"]],
        "rights_review": "REQUIRED_BEFORE_TRAINING",
        "generator_revision": "NOT_ASSIGNED",
        "dataset_identity": "NOT_CREATED",
    }
    weights = {
        "schema": "trinite.teacher-lab-weight-rights.v1",
        "status": "NOT_CREATED",
        "rights_review": "REQUIRED_BEFORE_DISTRIBUTION",
        "code_license": "MPL-2.0",
        "weights_license": "UNDECIDED",
        "model_checkpoint_identity": "NOT_CREATED",
        "data_lineage": "NOT_CREATED",
    }
    return data, weights


def request_and_rights(plan_path: str = DEFAULT_PLAN) -> tuple[dict, dict, dict]:
    plan, config, catalog = inspect(plan_path)
    count = ModelConfig.from_dict(config).parameter_count
    data, weights = rights_records(plan, catalog)
    paths = sorted(set(SOURCE_CLOSURE) |
                   {plan_path, plan["model_config"], plan["source_catalog"]})
    sources = [{"path": p, "identity": identity(source_bytes(p))} for p in paths]
    request = {
        "schema": SCHEMA,
        "status": "PREFLIGHT_ONLY",
        "plan_path": plan_path,
        "role": "teacher",
        "lane": "dense",
        "parameter_count": count,
        "parameter_budget": plan["parameter_budget"],
        "fp32_weight_bytes_lower_bound": count * 4,
        "fp32_adamw_state_bytes_lower_bound": count * 16,
        "memory_estimate_limitations": "excludes activations, temporary buffers and overhead",
        "model_identity": identity(json_bytes(config)),
        "catalog_identity": identity(json_bytes(catalog)),
        "source_closure": sources,
        "dataset_rights_identity": identity(json_bytes(data)),
        "weight_rights_identity": identity(json_bytes(weights)),
        "training_authorized": False,
        "data_admitted": False,
        "heldout_scored": False,
        "optimizer_updates": 0,
        "evidence_ready": False,
    }
    return request, data, weights


def freeze(destination: Path, plan_path: str = DEFAULT_PLAN) -> str:
    """Create a new, never-overwritten preflight packet; no model operations."""
    request, data, weights = request_and_rights(plan_path)
    destination = Path(destination)
    if destination.is_symlink() or destination.exists():
        raise ContractError("preflight destination already exists")
    # Do not expose a partially written packet at the requested destination.
    # Staging lives in the same parent filesystem for atomic directory rename.
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        for name, obj in (
            ("dataset-rights.json", data),
            ("weights-rights.json", weights),
            ("request.json", request),
        ):
            with (stage / name).open("xb") as fh:
                fh.write(json_bytes(obj))
                fh.flush()
                os.fsync(fh.fileno())
        if destination.is_symlink() or destination.exists():
            raise ContractError("preflight destination already exists")
        # No hostile concurrent parent-directory mutator is in TT-001's threat model.
        os.rename(stage, destination)
    finally:
        # On any write, fsync, or rename failure, only our private stage is removed.
        if stage.exists():
            shutil.rmtree(stage)
    return identity(json_bytes(request))


def verify(directory: Path, externally_retained_identity: str) -> dict:
    """Cold, read-only check against an identity supplied outside the packet."""
    from trinite.contracts import require_identity
    require_identity(externally_retained_identity)
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ContractError("invalid packet directory")
    expected_names = {"request.json", "dataset-rights.json", "weights-rights.json"}
    seen = set()
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.name not in expected_names or entry.name in seen:
                raise ContractError("unexpected packet member")
            seen.add(entry.name)
    if seen != expected_names:
        raise ContractError("missing packet member")
    observed = {}
    for name in ("dataset-rights.json", "weights-rights.json", "request.json"):
        p = directory / name
        if p.is_symlink() or not p.is_file() or p.stat().st_size > MAX_INPUT:
            raise ContractError("invalid packet member")
        raw = p.read_bytes()
        observed[name] = parse_json(raw, canonical=True)
    request = observed["request.json"]
    if identity(json_bytes(request)) != externally_retained_identity:
        raise ContractError("external request identity mismatch")
    expected_request, data, weights = request_and_rights(request["plan_path"])
    if (request != expected_request or
            observed["dataset-rights.json"] != data or
            observed["weights-rights.json"] != weights):
        raise ContractError("preflight source or rights mismatch")
    return {"verified": True, "scope": "preflight-only",
            "training_authorized": False, "heldout_scored": False,
            "evidence_ready": False}


def preview_formal(plan_path: str = DEFAULT_PLAN) -> dict:
    """Original finite GF(3) smoke facts, never an admitted training split."""
    _, _, catalog = inspect(plan_path)
    ids = {item["id"] for item in catalog["sources"]}
    if "qec-v1772" not in ids:
        raise ContractError("GF(3) reference source missing")
    examples = []
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            # Two separately spelled, exact finite-field calculations.
            answer = ((a + b + 1) % 3) - 1
            residue = ((a % 3) + (b % 3)) % 3
            independent = {0: 0, 1: 1, 2: -1}[residue]
            if answer != independent:
                raise ContractError("GF(3) independent arithmetic mismatch")
            examples.append({"family": f"gf3-pair-{a}-{b}", "a": a, "b": b,
                             "sum": answer, "source_id": "qec-v1772"})
    return {"schema": "trinite.teacher-lab-formal-preview.v1",
            "status": "NOT_ADMITTED_PUBLIC_ENGINEERING_ONLY",
            "rights": "REVIEW_REQUIRED",
            "split": "NONE",
            "examples": examples}
