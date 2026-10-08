"""Explicit run orchestration: computation, files, and optional pinned observer."""
from datetime import datetime, timezone
from pathlib import Path
import re
import resource
import time

from .contracts import ContractError, identity, json_bytes
from .data import audit_directory
from .checkpoint import load_checkpoint_bytes, read_checkpoint, save_checkpoint
from .observation import BundleObserver
from .inspection import inventory
from .training import (RunConfig, create_state, environment, evaluate, model_identity,
                       run_steps, source_receipt)


def train_run(output: Path, dataset: Path, config: RunConfig, *, lane: str = "ternary",
              resume: Path | None = None, stop_after: int | None = None,
              observer: bool = True, source_revision: str = "unreported",
              observer_factory=BundleObserver) -> dict:
    if type(observer) is not bool or (source_revision != "unreported" and
            (type(source_revision) is not str or not re.fullmatch("[0-9a-f]{40}", source_revision))):
        raise ContractError("invalid observer mode or declared source revision")
    audit_directory(dataset)
    data = (Path(dataset)/"dataset.json").read_bytes()
    manifest = (Path(dataset)/"manifest.json").read_bytes()
    candidate = read_checkpoint(resume) if resume is not None else None
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plan_bytes = json_bytes(config.to_dict())
    (output/"plan.json").write_bytes(plan_bytes)
    started, clock = datetime.now(timezone.utc).isoformat(), time.monotonic()
    errors, verification, collector, state = [], None, None, None
    observation_error = []
    inputs = {"dataset.json": data, "dataset-manifest.json": manifest, "plan.json": plan_bytes}
    if candidate:
        inputs.update({"parent-tensors.safetensors": candidate[0], "parent-metadata.json": candidate[1]})
    context = {"schema": "trinite.execution-context.v1", "source_revision": source_revision,
               "revision_assurance": "caller-declared; source files independently hashed",
               "dirty_state": "not inspected; exact installed source receipt retained",
               "started_at": started, "clock_assurance": "local", "lane": lane,
               "observer_enabled": observer, "environment": environment(), "source": source_receipt(),
               "dependency_lock_identity": source_receipt()["cpu-dependencies.lock"],
               "resume": resume is not None, "requested_stop_after": stop_after,
               "time_budget_scope": "per invocation; checked between optimizer steps"}
    (output/"context.json").write_bytes(json_bytes(context))
    inputs["context.json"] = json_bytes(context)
    observation_seconds = 0.
    def observed(call):
        nonlocal observation_seconds
        start = time.monotonic()
        try:
            return call()
        except Exception as error:
            observation_error.append(type(error).__name__+": "+str(error))
            return None
        finally:
            observation_seconds += time.monotonic()-start
    if observer:
        collector = observed(lambda: observer_factory(output/"provenance"))
    try:
        state = (load_checkpoint_bytes(*candidate, data, manifest, config, lane) if candidate
                 else create_state(config, lane, data, manifest))
        initial = json_bytes({"schema": "trinite.run-initialization.v1",
                              "initialization_identity": state.initialization_identity,
                              "entry_model_identity": model_identity(state.model), "entry_step": state.step})
        (output/"initialization.json").write_bytes(initial)
        if collector:
            observed(lambda: collector.record("resume" if candidate else "initialize",
                     inputs=inputs, outputs={"initialization.json": initial}))
        run_steps(state, stop_after=stop_after)
        checkpoint = save_checkpoint(output/"checkpoint", state)
        final_loss = evaluate(state.model, state.data["train"], config.batch_size)
        model_inventory = json_bytes(inventory(state.model))
        (output/"inventory.json").write_bytes(model_inventory)
        compute = {"schema": "trinite.training-result.v1", "lane": lane,
                   "outcome": "completed" if state.step == config.steps else "paused",
                   "plan_identity": config.content_identity, "dataset_identity": state.dataset_identity,
                   "manifest_identity": state.manifest_identity, "initialization_identity": state.initialization_identity,
                   "model_identity": model_identity(state.model), "step": state.step,
                   "inventory_identity": identity(model_inventory),
                   "cursor": state.cursor, "target_tokens": state.target_tokens,
                   "initial_train_loss_hex": state.initial_train_loss, "final_train_loss_hex": final_loss,
                   "validation": state.validation, "checkpoint": checkpoint,
                   "evidence_scope": "visible formal software/learning smoke; no fresh capability benchmark"}
        logs = json_bytes({"schema": "trinite.step-history.v1", "steps": state.history})
        result_bytes = json_bytes(compute)
        (output/"steps.json").write_bytes(logs)
        (output/"result.json").write_bytes(result_bytes)
        if collector:
            final_inputs = {**inputs, "initialization.json": initial}
            observed(lambda: collector.record("train-and-validate", inputs=final_inputs, outputs={
                "result.json": result_bytes, "steps.json": logs,
                "inventory.json": model_inventory,
                "tensors.safetensors": (output/"checkpoint/tensors.safetensors").read_bytes(),
                "metadata.json": (output/"checkpoint/metadata.json").read_bytes()}))
    except Exception as error:
        errors.append(type(error).__name__+": "+str(error))
        compute = {"schema": "trinite.training-result.v1", "outcome": "failed", "lane": lane,
                   "completed_steps": state.step if state else 0, "errors": errors,
                   "checkpoint": None, "checkpoint_recovery_errors": []}
        recovered = {}
        if state:
            try:
                state.optimizer.zero_grad(set_to_none=True)
                if not (output/'checkpoint').exists():
                    save_checkpoint(output/'checkpoint', state)
                payload, metadata = read_checkpoint(output/'checkpoint')
                compute['checkpoint'] = {'tensors_identity': identity(payload), 'metadata_identity': identity(metadata)}
                recovered = {'tensors.safetensors': payload, 'metadata.json': metadata}
            except Exception as recovery_error:
                compute['checkpoint_recovery_errors'].append(type(recovery_error).__name__+': '+str(recovery_error))
        (output/"result.json").write_bytes(json_bytes(compute))
        if state:
            (output/"steps.json").write_bytes(json_bytes({"schema": "trinite.step-history.v1", "steps": state.history}))
        if collector:
            observed(lambda: collector.record("training-failed", inputs=inputs,
                     outputs={"failure-result.json": json_bytes(compute), **recovered}))
    if collector:
        verification = observed(collector.finalize)
    if verification is not None:
        (output/"verification.json").write_bytes(json_bytes(verification))
    eligible = bool(observer and collector and verification and verification["integrity_verified"]
                    and verification["manifest_scope"] == "closed" and not verification["known_missing_artifacts"]
                    and not verification["errors"] and not observation_error and not errors)
    report = {"schema": "trinite.run-report.v1", "compute_outcome": compute["outcome"],
              "result_identity": identity((output/"result.json").read_bytes()),
              "observer_enabled": observer, "observer_evidence_eligible": eligible,
              "release_ready": False, "errors": errors, "observation_errors": observation_error,
              "verification": verification, "wall_seconds_hex": (time.monotonic()-clock).hex(),
              "observer_seconds_hex": observation_seconds.hex(), "started_at": started,
              "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "rss_scope": "Linux process lifetime peak; includes imports and observer",
              "finished_at": datetime.now(timezone.utc).isoformat()}
    (output/"report.json").write_bytes(json_bytes(report))
    return report
