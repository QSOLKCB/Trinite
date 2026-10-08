"""Bounded foundation and explicit optional CPU model inspection commands."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .contracts import ContractError, ModelConfig, json_bytes
from .data import audit_directory, write_dataset
from .tokenizer import ByteTokenizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trinite")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-config")
    validate.add_argument("config", type=Path)
    tokenize = commands.add_parser("tokenize")
    tokenize.add_argument("text")
    tokenize.add_argument("--answer-start", type=int)
    tokenize.add_argument("--pad-to", type=int)
    tokenize.add_argument("--config", type=Path)
    generate = commands.add_parser("generate-fixture")
    generate.add_argument("output", type=Path)
    generate.add_argument("--config", type=Path)
    generate.add_argument("--seed", type=int, default=0)
    audit = commands.add_parser("audit-fixture")
    audit.add_argument("directory", type=Path)
    inspect = commands.add_parser("inspect-model")
    inspect.add_argument("--config", type=Path)
    inspect.add_argument("--lane", choices=("dense", "ternary"), default="ternary")
    inspect.add_argument("--seed", type=int, default=0)
    inspect.add_argument("--text")
    inspect.add_argument("--capture-layer", action="append")
    inspect.add_argument("--capture-position", type=int, action="append")
    inspect.add_argument("--capture-bytes", type=int, default=65536)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-config":
            config = ModelConfig.load(args.config)
            result = {"valid": True, "config_identity": config.content_identity,
                      "planned_parameters": config.parameter_count}
        elif args.command == "tokenize":
            config = ModelConfig.load(args.config) if args.config else ModelConfig()
            result = ByteTokenizer(config.context_length).encode(
                args.text, answer_start=args.answer_start, pad_to=args.pad_to).to_dict()
        elif args.command == "generate-fixture":
            config = ModelConfig.load(args.config) if args.config else ModelConfig()
            result = write_dataset(args.output, config, seed=args.seed)
        elif args.command == "audit-fixture":
            result = audit_directory(args.directory)
        else:
            # Import only on explicit model selection; foundation stays stdlib.
            try:
                import torch
                from .model import CaptureSpec, Decoder
                from .inspection import capture_inventory, inventory, tensor_identity
            except ImportError as error:
                raise ContractError("model inspection requires the CPU dependencies; "
                                    "see docs/GETTING_STARTED.md") from error
            config = ModelConfig.load(args.config) if args.config else ModelConfig()
            if (bool(args.capture_layer) != bool(args.capture_position)
                    or (args.capture_layer and args.text is None)):
                raise ContractError("capture requires text, layers, and positions together")
            encoding = (ByteTokenizer(config.context_length).encode(args.text)
                        if args.text is not None else None)
            capture = (CaptureSpec(tuple(args.capture_layer), tuple(args.capture_position),
                                   args.capture_bytes) if args.capture_layer else None)
            if capture:
                capture.validate(config, 1, len(encoding.input_ids))
            # Explicit controls scoped to this CLI process, never hidden in API.
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            model = Decoder(config, lane=args.lane, seed=args.seed)
            result = inventory(model)
            if encoding:
                ids = torch.tensor([encoding.input_ids], dtype=torch.int64, device="cpu")
                mask = torch.tensor([encoding.attention_mask], dtype=torch.bool, device="cpu")
                with torch.no_grad():
                    output = model(ids, mask, capture=capture)
                result["forward_result"] = {"input_identity": tensor_identity(ids),
                                            "mask_identity": tensor_identity(mask),
                                            "input_shape": list(ids.shape),
                                            "capture_selection": (
                                                {"layers": list(capture.layers),
                                                 "positions": list(capture.positions),
                                                 "max_bytes": capture.max_bytes}
                                                if capture else None),
                                            "logits_identity": tensor_identity(output.logits),
                                            "logits_shape": list(output.logits.shape),
                                            "captures": capture_inventory(output.captures)}
        sys.stdout.write(json_bytes(result).decode("utf-8"))
        return 0
    except (ContractError, OSError) as exc:
        print(f"trinite: {exc}", file=sys.stderr)
        return 1
