"""Explicit bounded CPU float32 decoder; no observer, data, or geometry imports."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import ContractError, ModelConfig, integer
from .quantizer import quantize, quantize_four, require_float_tensor

MODEL_ID = "trinite.decoder.v0"
MAX_PARAMETERS = 5_000_000
MAX_BATCH = 8
MAX_ATTENTION_CELLS = 4_194_304
MAX_CAPTURE_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class CaptureSpec:
    layers: tuple[str, ...]
    positions: tuple[int, ...]
    max_bytes: int = 64 * 1024

    def validate(self, config: ModelConfig, batch: int, length: int) -> None:
        names = {"embedding", "final", *(f"block.{i}" for i in range(config.blocks))}
        if (type(self.layers) is not tuple or not self.layers
                or any(type(n) is not str or n not in names for n in self.layers)
                or len(set(self.layers)) != len(self.layers)):
            raise ContractError("capture requires distinct implemented layer names")
        if type(self.positions) is not tuple or not self.positions:
            raise ContractError("capture requires explicit token positions")
        for position in self.positions:
            integer(position, "capture position", 0, length-1)
        if len(set(self.positions)) != len(self.positions):
            raise ContractError("capture positions must be distinct")
        integer(self.max_bytes, "capture byte budget", 1, MAX_CAPTURE_BYTES)
        if len(self.layers) * batch * len(self.positions) * config.width * 4 > self.max_bytes:
            raise ContractError("selected hidden states exceed capture byte budget")


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    captures: dict[str, torch.Tensor]


def _weight(shape: tuple[int, ...], generator: torch.Generator) -> nn.Parameter:
    # Private generator; all allocations specify CPU/dtype rather than globals.
    value = torch.empty(shape, device="cpu", dtype=torch.float32)
    value.normal_(mean=0, std=0.02, generator=generator)
    return nn.Parameter(value)


class Linear(nn.Module):
    def __init__(self, inputs: int, outputs: int, lane: str, generator: torch.Generator):
        super().__init__()
        self.weight = _weight((outputs, inputs), generator)
        self.lane = lane

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weight = (quantize(self.weight).weight if self.lane == "ternary" else
                  quantize_four(self.weight).weight if self.lane == "four-state" else self.weight)
        return F.linear(value, weight)


class RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, device="cpu", dtype=torch.float32))
        self.epsilon = epsilon

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return (value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True)
                                   + self.epsilon)) * self.weight


class Block(nn.Module):
    def __init__(self, config: ModelConfig, lane: str, generator: torch.Generator):
        super().__init__()
        d = config.width
        self.heads, self.head_width = config.heads, d // config.heads
        self.attention_norm = RMSNorm(d, float(config.rms_epsilon))
        self.q = Linear(d, d, lane, generator)
        self.k = Linear(d, d, lane, generator)
        self.v = Linear(d, d, lane, generator)
        self.o = Linear(d, d, lane, generator)
        self.feed_forward_norm = RMSNorm(d, float(config.rms_epsilon))
        self.up = Linear(d, config.feed_forward_width, lane, generator)
        self.down = Linear(config.feed_forward_width, d, lane, generator)

    def forward(self, value: torch.Tensor, allowed: torch.Tensor,
                live: torch.Tensor) -> torch.Tensor:
        batch, length, width = value.shape
        normalized = self.attention_norm(value)
        def heads(projected):
            return projected.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        q, k, v = (heads(projection(normalized)) for projection in (self.q, self.k, self.v))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_width)
        probabilities = scores.masked_fill(~allowed, -torch.inf).softmax(dim=-1)
        attended = (probabilities @ v).transpose(1, 2).contiguous().reshape(batch, length, width)
        value = (value + self.o(attended)) * live
        feed_forward = self.down(F.gelu(self.up(self.feed_forward_norm(value)), approximate="none"))
        return (value + feed_forward) * live


class Decoder(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig(), *, lane: str = "ternary",
                 seed: int = 0, parameter_budget: int = 2_000_000):
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise ContractError("decoder requires a validated ModelConfig")
        if type(lane) is not str or lane not in ("dense", "ternary", "four-state"):
            raise ContractError("decoder lane must be dense, ternary or four-state")
        integer(seed, "initialization seed", 0, 2**32-1)
        integer(parameter_budget, "parameter budget", 1, MAX_PARAMETERS)
        if config.parameter_count > parameter_budget:
            raise ContractError("model exceeds parameter budget before allocation")
        self.config, self.lane, self.seed = config, lane, seed
        self._construction_config = config
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.token_embedding = _weight((config.vocabulary_size, config.width), generator)
        self.position_embedding = _weight((config.context_length, config.width), generator)
        self.blocks = nn.ModuleList([Block(config, lane, generator) for _ in range(config.blocks)])
        self.final_norm = RMSNorm(config.width, float(config.rms_epsilon))
        # A registered alias makes output sharing visible; named_parameters()
        # counts this storage once, while inspection can list both names.
        self.output_weight = self.token_embedding
        if sum(p.numel() for p in self.parameters()) != config.parameter_count:
            raise ContractError("constructed parameter inventory disagrees with config")
        self._parameter_shapes = {name: tuple(p.shape) for name, p in self.named_parameters()}

    def validate_state(self) -> None:
        if self.config != self._construction_config:
            raise ContractError("decoder config must match its constructed tensors/operations")
        if self.lane not in ("dense", "ternary", "four-state"):
            raise ContractError("unsupported decoder lane")
        parameters = dict(self.named_parameters())
        if set(parameters) != set(self._parameter_shapes):
            raise ContractError("decoder parameter names differ from the constructed inventory")
        for name, parameter in parameters.items():
            if tuple(parameter.shape) != self._parameter_shapes[name]:
                raise ContractError(f"decoder tensor {name} has an incompatible shape")
            require_float_tensor(parameter, name)
        if self.output_weight is not self.token_embedding:
            raise ContractError("decoder output must remain tied to the token embedding")
        if any(m.lane != self.lane for m in self.modules() if isinstance(m, Linear)):
            raise ContractError("linear layers must match the declared decoder lane")
        epsilon = float(self.config.rms_epsilon)
        if any(type(m.epsilon) is not float or m.epsilon != epsilon
               for m in self.modules() if isinstance(m, RMSNorm)):
            raise ContractError("normalization epsilon must match the declared config")
        if any(b.heads != self.config.heads or b.head_width != self.config.width//self.config.heads
               for b in self.blocks):
            raise ContractError("attention head layout must match the declared config")

    def _validate_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        if (not isinstance(input_ids, torch.Tensor) or input_ids.device.type != "cpu"
                or input_ids.dtype != torch.int64 or input_ids.layout != torch.strided
                or input_ids.ndim != 2):
            raise ContractError("input IDs require a CPU int64 [batch, sequence] tensor")
        batch, length = input_ids.shape
        integer(batch, "batch size", 1, MAX_BATCH)
        integer(length, "sequence length", 1, self.config.context_length)
        if batch * self.config.heads * length * length > MAX_ATTENTION_CELLS:
            raise ContractError("attention workspace exceeds the CPU cell budget")
        if (not isinstance(attention_mask, torch.Tensor) or attention_mask.device.type != "cpu"
                or attention_mask.dtype != torch.bool or attention_mask.layout != torch.strided
                or attention_mask.shape != input_ids.shape):
            raise ContractError("attention mask requires CPU bool with the input shape")
        if (bool((input_ids < 0).any()) or bool((input_ids >= self.config.vocabulary_size).any())
                or not bool(attention_mask[:, 0].all())
                or bool((~attention_mask[:, :-1] & attention_mask[:, 1:]).any())):
            raise ContractError("IDs must be in vocabulary; mask must have a nonempty right-padded prefix")
        self.validate_state()
        if torch.is_autocast_enabled("cpu"):
            raise ContractError("CPU autocast is outside the float32 reference contract")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                *, capture: CaptureSpec | None = None) -> ModelOutput:
        self._validate_inputs(input_ids, attention_mask)
        batch, length = input_ids.shape
        if capture is not None:
            if not isinstance(capture, CaptureSpec):
                raise ContractError("capture requires a CaptureSpec")
            capture.validate(self.config, batch, length)
        captures = {}
        def retain(name: str, value: torch.Tensor) -> None:
            if capture is not None and name in capture.layers:
                captures[name] = value[:, capture.positions, :].detach().clone()
        # Masked positions may contain any in-vocabulary ID; they cannot
        # influence attention, residuals, returned logits, or captures.
        live = attention_mask.unsqueeze(-1).to(torch.float32)
        positions = torch.arange(length, device="cpu")
        value = (F.embedding(input_ids, self.token_embedding)
                 + F.embedding(positions, self.position_embedding)) * live
        retain("embedding", value)
        causal = torch.ones((length, length), device="cpu", dtype=torch.bool).tril()
        allowed = causal[None, None, :, :] & attention_mask[:, None, None, :]
        for i, block in enumerate(self.blocks):
            value = block(value, allowed, live)
            retain(f"block.{i}", value)
        value = self.final_norm(value) * live
        retain("final", value)
        logits = F.linear(value, self.output_weight) * live
        require_float_tensor(logits, "decoder logits")
        return ModelOutput(logits, captures)
