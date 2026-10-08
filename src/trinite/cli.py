"""Bounded CPU-only foundation commands. No model or training commands yet."""
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
        else:
            result = audit_directory(args.directory)
        sys.stdout.write(json_bytes(result).decode("utf-8"))
        return 0
    except (ContractError, OSError) as exc:
        print(f"trinite: {exc}", file=sys.stderr)
        return 1
