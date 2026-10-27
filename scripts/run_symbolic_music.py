#!/usr/bin/env python3
"""Source-bound music preparation only; never loads or trains a model."""
import argparse
import json
from pathlib import Path

from trinite.music_v1.corpus import generate, write_bundle, verify_bundle
from trinite.music_v1.core import from_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("generate", help="write a fresh complete symbolic bundle")
    create.add_argument("destination", type=Path)
    verify = sub.add_parser("verify", help="read-only replay against an external commitment")
    verify.add_argument("destination", type=Path)
    verify.add_argument("--manifest-sha256", required=True)
    render = sub.add_parser("render", help="optional audition outside the symbolic bundle")
    render.add_argument("destination", type=Path)
    render.add_argument("--row", type=int, default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.command == "generate":
        print(write_bundle(root, args.destination))
    elif args.command == "verify":
        print(json.dumps(verify_bundle(root, args.destination, args.manifest_sha256), sort_keys=True))
    else:
        if not 0 <= args.row < 160:
            parser.error("row must be in 0..159")
        from trinite.music_v1.render import wav_bytes
        data = wav_bytes(from_dict(generate()[args.row]["input"]["score"]))
        with args.destination.open("xb") as stream:
            stream.write(data)
        print("Illustrative audio only; not symbolic evidence or a learning result.")


if __name__ == "__main__":
    main()
