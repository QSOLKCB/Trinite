"""Bounded native CPU training; no observer, checkpoint IO, or geometry imports."""
from dataclasses import dataclass, field
import importlib.metadata
import math
from pathlib import Path
import platform
import time

import torch
from torch.nn import functional as F

from .contracts import ContractError, ModelConfig, exact_keys, identity, integer, json_bytes, parse_json
from .data import audit_bytes
from .inspection import tensor_identity
from .model import Decoder
from .tokenizer import ByteTokenizer


def number(text: object, label: str, low: float, high: float) -> float:
    if type(text) is not str:
        raise ContractError(f"{label} requires a numeric string")
    try:
        value = float(text)
    except ValueError as error:
        raise ContractError(f"invalid {label}") from error
    if not math.isfinite(value) or not low <= value <= high:
        raise ContractError(f"{label} outside [{low}, {high}]")
    return value


@dataclass(frozen=True)
class RunConfig:
    schema: str = "trinite.training-plan.v1"
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        context_length=32, blocks=1, width=16, heads=2, feed_forward_width=32))
    seed: int = 0
    steps: int = 60
    batch_size: int = 4
    learning_rate: str = "0.003"
    beta1: str = "0.9"
    beta2: str = "0.999"
    optimizer_epsilon: str = "0.00000001"
    weight_decay: str = "0.01"
    gradient_clip: str = "1"
    accumulation: int = 1
    schedule: str = "constant"
    order: str = "seeded-hash-rank-cycle.v1"
    loss: str = "answer-and-eos-target-mean.v1"
    validation_every: int = 20
    selection: str = "final-step; no early stopping; no test evaluation"
    max_seconds: int = 60
    max_target_tokens: int = 4096

    def __post_init__(self):
        if not isinstance(self.model, ModelConfig) or self.model.parameter_count > 2_000_000:
            raise ContractError("training model must fit the 2,000,000-parameter CPU budget")
        for key, value in {"schema": "trinite.training-plan.v1", "accumulation": 1,
                           "schedule": "constant", "order": "seeded-hash-rank-cycle.v1",
                           "loss": "answer-and-eos-target-mean.v1",
                           "selection": "final-step; no early stopping; no test evaluation"}.items():
            if type(getattr(self, key)) is not type(value) or getattr(self, key) != value:
                raise ContractError(f"unsupported training {key}")
        integer(self.seed, "run seed", 0, 2**32-1)
        integer(self.steps, "training steps", 1, 256)
        integer(self.batch_size, "batch size", 1, 8)
        integer(self.validation_every, "validation interval", 1, 256)
        integer(self.max_seconds, "invocation seconds", 1, 600)
        integer(self.max_target_tokens, "target token budget", 1, 4096)
        if self.steps * self.batch_size * 2 > self.max_target_tokens:
            raise ContractError("formal answer/EOS targets exceed the frozen token budget")
        for name, low, high in (("learning_rate", 1e-6, .1), ("beta1", 0, .9999),
                                ("beta2", 0, .999999), ("optimizer_epsilon", 1e-12, .01),
                                ("weight_decay", 0, 1), ("gradient_clip", 1e-6, 100)):
            number(getattr(self, name), name, low, high)

    def to_dict(self):
        return {name: self.model.to_dict() if name == "model" else getattr(self, name)
                for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value):
        value = dict(exact_keys(value, set(cls.__dataclass_fields__), "training plan"))
        value["model"] = ModelConfig.from_dict(value["model"])
        return cls(**value)

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as stream: data = stream.read(16385)
        if len(data) > 16384:
            raise ContractError("training plan exceeds 16 KiB")
        return cls.from_dict(parse_json(data))

    @property
    def content_identity(self):
        return identity(json_bytes(self.to_dict()))


def cpu_identity() -> dict:
    with Path('/proc/cpuinfo').open('rb') as stream:
        first = stream.read(65536).decode('ascii').split('\n\n', 1)[0]
    fields = dict(line.split(':', 1) for line in first.splitlines() if ':' in line)
    fields = {k.strip(): v.strip() for k, v in fields.items()}
    return {key: fields.get(key, 'unreported') for key in ('vendor_id', 'model name', 'flags')}


def environment() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "torch": str(torch.__version__),
            "cpu": cpu_identity(), "cpu_kernel_capability": torch.backends.cpu.get_cpu_capability(),
            "default_dtype": str(torch.get_default_dtype()), "default_device": str(torch.get_default_device()),
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "dtype": "float32", "device": "cpu", "optimizer": "AdamW-single-tensor",
            "packages": {p: importlib.metadata.version(p) for p in (
                "torch", "numpy", "safetensors", "packaging", "filelock", "typing-extensions", "setuptools",
                "sympy", "networkx", "jinja2", "fsspec", "mpmath", "markupsafe")}}


def require_environment() -> dict:
    result = environment()
    packages = {"torch": "2.8.0+cpu", "numpy": "2.3.5", "safetensors": "0.6.2", "packaging": "25.0",
                "filelock": "3.32.3", "typing-extensions": "4.16.0", "setuptools": "78.1.0",
                "sympy": "1.14.0", "networkx": "3.6.1", "jinja2": "3.1.6", "fsspec": "2026.7.0",
                "mpmath": "1.3.0", "markupsafe": "3.0.3"}
    if (result["torch"] != "2.8.0+cpu" or torch.version.cuda is not None
            or result["packages"] != packages or result["threads"] != 1
            or result["machine"] != "x86_64" or not result["platform"].startswith("Linux")
            or result["default_dtype"] != "torch.float32" or result["default_device"] != "cpu"
            or not result["deterministic"] or torch.is_autocast_enabled("cpu")):
        raise ContractError("training requires the pinned CPU backend, one thread, and strict determinism")
    return result


def source_receipt() -> dict:
    package = Path(__file__).resolve().parent
    return {p: identity((package/p).read_bytes()) for p in (
        "contracts.py", "tokenizer.py", "data.py", "model.py", "quantizer.py", "inspection.py",
        "training.py", "checkpoint.py", "experiment.py", "cli.py", "observation.py",
        "provenance-pin.json", "cpu-dependencies.lock")}


def model_identity(model: Decoder) -> str:
    return identity(json_bytes({n: tensor_identity(p) for n, p in model.named_parameters()}))


def admitted_data(dataset_bytes: bytes, manifest_bytes: bytes, config: RunConfig) -> dict:
    audit_bytes(dataset_bytes, manifest_bytes)
    dataset = parse_json(dataset_bytes, canonical=True)
    tokenizer = ByteTokenizer(config.model.context_length)
    selected = {s: [] for s in ("train", "validation")}
    for example in dataset["examples"]:
        if example["split"] not in selected:
            continue  # Test labels never reach batches, validation, or selection.
        encoding = tokenizer.encode(example["text"], answer_start=len(example["prompt"].encode()))
        if sum(encoding.loss_mask) != 2:
            raise ContractError("Phase 3 formal source requires answer+EOS target pairs")
        selected[example["split"]].append({"identity": example["example_identity"],
                                          "ids": encoding.input_ids, "mask": encoding.attention_mask,
                                          "loss": encoding.loss_mask})
    selected["train"].sort(key=lambda e: identity(json_bytes({"seed": config.seed, "item": e["identity"]})))
    if len(selected["train"]) != 36 or len(selected["validation"]) != 6:
        raise ContractError("training requires the frozen 36/6 admitted family split")
    return selected


def batch(examples: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = max(len(e["ids"]) for e in examples)
    ids = torch.tensor([list(e["ids"])+[258]*(length-len(e["ids"])) for e in examples], dtype=torch.int64, device="cpu")
    mask = torch.tensor([list(e["mask"])+[0]*(length-len(e["ids"])) for e in examples], dtype=torch.bool, device="cpu")
    loss = torch.tensor([list(e["loss"])+[0]*(length-len(e["ids"])) for e in examples], dtype=torch.bool, device="cpu")
    return ids, mask, loss


def target_loss(model: Decoder, examples: list[dict]) -> tuple[torch.Tensor, int]:
    ids, mask, scored = batch(examples)
    logits = model(ids[:, :-1], mask[:, :-1]).logits
    weights = scored[:, 1:]
    tokens = int(weights.sum())
    if not tokens:
        raise ContractError("batch contains no scored targets")
    losses = F.cross_entropy(logits.reshape(-1, 259), ids[:, 1:].reshape(-1), reduction="none")
    result = (losses.reshape(weights.shape)*weights).sum()/tokens
    if not bool(torch.isfinite(result)):
        raise ContractError("nonfinite training loss")
    return result, tokens


def evaluate(model: Decoder, examples: list[dict], batch_size: int) -> str:
    total, tokens = 0., 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            loss, count = target_loss(model, examples[start:start+batch_size])
            total += float(loss)*count
            tokens += count
    return (total/tokens).hex()


@dataclass
class TrainingState:
    config: RunConfig
    lane: str
    model: Decoder
    optimizer: torch.optim.AdamW
    data: dict
    dataset_identity: str
    manifest_identity: str
    initialization_identity: str
    initial_train_loss: str
    step: int = 0
    cursor: int = 0
    target_tokens: int = 0
    history: list = field(default_factory=list)
    validation: list = field(default_factory=list)


def create_state(config: RunConfig, lane: str, dataset: bytes, manifest: bytes) -> TrainingState:
    require_environment()
    data = admitted_data(dataset, manifest, config)
    model = Decoder(config.model, lane=lane, seed=config.seed)
    optimizer = create_optimizer(model, config)
    return TrainingState(config, lane, model, optimizer, data, identity(dataset), identity(manifest),
                         model_identity(model), evaluate(model, data["train"], config.batch_size))


def create_optimizer(model: Decoder, config: RunConfig) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate),
                                 betas=(float(config.beta1), float(config.beta2)),
                                 eps=float(config.optimizer_epsilon), weight_decay=float(config.weight_decay),
                                 foreach=False, fused=False, capturable=False, amsgrad=False,
                                 maximize=False, differentiable=False)


def require_optimizer(state: TrainingState) -> None:
    if type(state.optimizer) is not torch.optim.AdamW:
        raise ContractError("optimizer must be the frozen AdamW implementation")
    group = state.optimizer.param_groups
    if len(group) != 1 or [id(p) for p in group[0]["params"]] != [id(p) for p in state.model.parameters()]:
        raise ContractError("optimizer parameter grouping differs from the plan")
    expected = {"lr": float(state.config.learning_rate), "betas": (float(state.config.beta1), float(state.config.beta2)),
                "eps": float(state.config.optimizer_epsilon), "weight_decay": float(state.config.weight_decay),
                "amsgrad": False, "maximize": False, "foreach": False, "capturable": False,
                "differentiable": False, "fused": False, "decoupled_weight_decay": True}
    exact_keys(group[0], set(expected) | {"params"}, "optimizer parameter group")
    if any(type(group[0][n]) is not type(v) or group[0][n] != v for n, v in expected.items()):
        raise ContractError("optimizer settings differ from the plan")

def run_steps(state: TrainingState, *, stop_after: int | None = None) -> None:
    require_environment()
    require_optimizer(state)
    end = state.config.steps if stop_after is None else integer(stop_after, "stop step", state.step, state.config.steps)
    deadline = time.monotonic() + state.config.max_seconds
    while state.step < end:
        if time.monotonic() > deadline:
            raise ContractError("training invocation exceeded its time budget")
        examples = [state.data["train"][(state.cursor+i) % len(state.data["train"])]
                    for i in range(state.config.batch_size)]
        state.optimizer.zero_grad(set_to_none=True)
        loss, targets = target_loss(state.model, examples)
        if state.target_tokens + targets > state.config.max_target_tokens:
            raise ContractError("training exceeds its scored-target budget")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(state.model.parameters(), float(state.config.gradient_clip),
                                             error_if_nonfinite=True, foreach=False)
        state.optimizer.step()
        state.step += 1
        state.cursor += state.config.batch_size
        state.target_tokens += targets
        state.history.append({"step": state.step, "loss_hex": float(loss.detach()).hex(),
                              "gradient_norm_hex": float(norm).hex(), "target_tokens": targets,
                              "examples": [e["identity"] for e in examples]})
        if state.step % state.config.validation_every == 0 or state.step == state.config.steps:
            state.validation.append({"step": state.step, "loss_hex": evaluate(
                state.model, state.data["validation"], state.config.batch_size)})
    state.optimizer.zero_grad(set_to_none=True)
    state.model.validate_state()
