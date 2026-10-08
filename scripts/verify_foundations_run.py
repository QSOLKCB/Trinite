"""Reverify a retained run without changing its files or rescoring model outputs."""
import argparse
from pathlib import Path
import shutil
import tempfile

import torch

from trinite.contracts import ContractError, identity, json_bytes, parse_json, require_identity
from trinite.foundations import read_bytes, read_request, summarize
from trinite.foundations_training import RUNNER_IDENTITY

ROOT=Path(__file__).resolve().parents[1]


def verify(root,request_identity,summary_identity):
    require_identity(request_identity);require_identity(summary_identity)
    root=Path(root).resolve()
    raw=read_bytes(root/'summary.json',4*1024*1024)
    if identity(raw)!=summary_identity:raise ContractError('retained summary identity mismatch')
    retained=parse_json(raw,canonical=True)
    read_request(root/'request.json',request_identity)
    if identity((ROOT/'scripts/run_foundations.py').read_bytes())!=RUNNER_IDENTITY:
        raise ContractError('local runner differs from retained request source')
    files=list(root.rglob('*'))
    if len(files)>4096 or any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in files):
        raise ContractError('unsafe or oversized retained run inventory')
    if sum(p.stat().st_size for p in files if p.is_file())>512*1024*1024:
        raise ContractError('retained run exceeds review budget')
    # Existing summarization performs the authoritative verification. Only a
    # disposable copy receives its exclusive summary write; originals stay intact.
    with tempfile.TemporaryDirectory() as directory:
        copy=Path(directory)/'run';shutil.copytree(root,copy)
        (copy/'summary.json').unlink()
        replay=summarize(copy,copy/'request.json',request_identity,retained['worker_errors'])
        if json_bytes(replay)!=raw:raise ContractError('retained summary differs from independently verified replay')
    return {'summary_identity':summary_identity,'request_identity':request_identity,
        'verified_training_cells':sum(c.get('outcome')=='completed' for c in replay['cells']),
        'outcome':replay['outcome'],'dense_learning_adequate':replay['dense_learning_adequate'],
        'scope':'integrity/score/checkpoint/gate replay; no new generation or independent experiment'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--request-identity',required=True)
    parser.add_argument('--summary-identity',required=True)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    print(json_bytes(verify(args.root,args.request_identity,args.summary_identity)).decode())
    return 0


if __name__=='__main__':raise SystemExit(main())
