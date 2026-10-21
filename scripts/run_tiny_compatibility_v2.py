"""Freeze, run one explicitly authorized local Needle compatibility probe, or verify."""
import sys
if not sys.flags.isolated or not sys.flags.no_site:
    raise SystemExit("Use python -I -S -B scripts/run_tiny_compatibility_v2.py ...")
import argparse
from pathlib import Path
import types
ROOT=Path(__file__).resolve().parents[1]
module=types.ModuleType("tiny_compatibility_v2")
module.__file__=str(ROOT/"scripts/tiny_compatibility_v2.py")
exec(compile(Path(module.__file__).read_bytes(),module.__file__,"exec"),module.__dict__)


def read(path):
    if path.is_symlink() or not path.is_file():
        raise module.ContractError("regular input file required")
    with path.open("rb") as stream:
        raw=stream.read(1024*1024+1)
    if len(raw)>1024*1024:
        raise module.ContractError("input exceeds 1 MiB")
    return raw


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    freeze=sub.add_parser("freeze")
    freeze.add_argument("--asset-root",type=Path,required=True)
    freeze.add_argument("--output",type=Path,required=True)
    probe=sub.add_parser("probe")
    probe.add_argument("--asset-root",type=Path,required=True)
    probe.add_argument("--request",type=Path,required=True)
    probe.add_argument("--request-identity",required=True)
    probe.add_argument("--output",type=Path,required=True)
    probe.add_argument("--allow-native-execution",action="store_true")
    verify=sub.add_parser("verify")
    verify.add_argument("--packet",type=Path,required=True)
    verify.add_argument("--packet-identity",required=True)
    args=parser.parse_args()
    try:
        if args.action=="freeze":
            raw=module.encoded(module.freeze(args.asset_root))
            module.publish(args.output,raw)
            print(module.identity(raw))
        elif args.action=="probe":
            raw=module.probe(read(args.request),args.request_identity,args.asset_root,args.output,args.allow_native_execution)
            print(module.identity(raw))
        else:
            raw=read(args.packet)
            if module.identity(raw)!=args.packet_identity:
                raise module.ContractError("packet identity mismatch")
            print(module.encoded(module.verify(module.parse(raw))).decode(),end="")
        return 0
    except (OSError,ValueError,TypeError) as error:
        print(str(error),file=sys.stderr)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
