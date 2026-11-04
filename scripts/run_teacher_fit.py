#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Explicit FIT-001 CLI; target execution always requires separate owner approvals."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trinite.contracts import json_bytes
from teacher_fit_v1 import custody as io, spec


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    env=sub.add_parser('environment');env.add_argument('--out',type=Path,required=True)
    freeze=sub.add_parser('freeze');freeze.add_argument('--dataset',type=Path,required=True)
    freeze.add_argument('--dataset-identity',required=True);freeze.add_argument('--experiment',type=Path,required=True)
    freeze.add_argument('--review',type=Path,required=True);freeze.add_argument('--review-identity',required=True)
    freeze.add_argument('--environment',type=Path,required=True);freeze.add_argument('--environment-identity',required=True)
    for name in ('verify-request','run','_worker'):
        p=sub.add_parser(name)
        p.add_argument('--request',type=Path,required=True);p.add_argument('--identity',required=True)
        p.add_argument('--dataset',type=Path,required=True)
        if name!='verify-request':
            p.add_argument('--approval',type=Path,required=True);p.add_argument('--approval-identity',required=True)
        if name=='run':
            p.add_argument('--approval-set',type=Path,required=True);p.add_argument('--approval-set-identity',required=True)
        if name=='_worker':p.add_argument('--capability-fd',type=int,required=True)
    for name in ('verify','replay'):
        p=sub.add_parser(name);p.add_argument('--run',type=Path,required=True)
        p.add_argument('--manifest-identity',required=True);p.add_argument('--request-identity',required=True)
        p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('synthesize');p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--request-anchors',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();code=0
    if args.command=='environment':
        from teacher_fit_v1.native import environment
        from trinite.contracts import identity
        value=environment();io.record(args.out,value);result={'environment_identity':identity(json_bytes(value))}
    elif args.command=='freeze':
        result=spec.freeze(args.dataset,args.dataset_identity,args.experiment,args.review,args.review_identity,
            io.anchored(args.environment,args.environment_identity))
    elif args.command=='verify-request':
        spec.verify_request(args.request,args.identity,args.dataset);result={'verified':True,'training_authorized':False}
    elif args.command=='run':
        from teacher_fit_v1.supervisor import dispatch
        result=dispatch(args.request,args.identity,args.dataset,args.approval,args.approval_identity,
            args.approval_set,args.approval_set_identity)
        code=result['supervisor_exit_status']
    elif args.command=='_worker':
        from teacher_fit_v1.supervisor import worker
        worker(args);result={'worker_completed':True}
    elif args.command in ('verify','replay'):
        from teacher_fit_v1 import evidence
        operation=evidence.verify if args.command=='verify' else evidence.replay
        io.outside(args.out,args.run,args.dataset,ROOT)
        if args.out.exists():raise ValueError('fresh verification receipt required')
        try:
            result=operation(args.run,args.manifest_identity,args.request_identity,args.dataset)
        except BaseException as exc:
            io.record(args.out,{'verified':False,'operation':args.command,'type':type(exc).__name__,
                'message':str(exc)[:2000],'manifest_identity':args.manifest_identity,
                'request_identity':args.request_identity,'optimizer_replayed':False})
            raise
        io.record(args.out,result)
    else:
        from teacher_fit_v1.synthesis import combine
        result=combine(args.experiment,args.dataset,io.load(args.request_anchors),args.out)
    sys.stdout.buffer.write(json_bytes(result));return code


if __name__=='__main__':
    raise SystemExit(main())
