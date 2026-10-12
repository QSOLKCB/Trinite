"""Preserve a complete budget run and exact producer after fresh full replay."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--request',type=Path,required=True)
    p.add_argument('--request-identity',required=True)
    p.add_argument('--verification-request',type=Path,required=True)
    p.add_argument('--verification-request-identity',required=True)
    a=p.parse_args(argv)
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    from trinite.retention_v2.task_balance.archive import archive
    from trinite.contracts import ContractError,json_bytes
    try:
        print(json_bytes(archive(a.output,a.run,a.request,a.request_identity,
                                a.verification_request,a.verification_request_identity)).decode(),end='')
        return 0
    except (ContractError,OSError,KeyError,TypeError,ValueError) as error:
        print('task-balance archive: '+str(error),file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
