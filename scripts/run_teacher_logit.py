#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""TT-003-LOGIT-001 v2: owner-local frozen full-vocabulary logit audit."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from trinite.contracts import ContractError,json_bytes
from teacher_logit_v1 import core,native

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='operation',required=True)
    for name in ('freeze','verify-request','audit','verify','replay-forward'):
        cmd=sub.add_parser(name)
        for opt in ('dataset','teacher-run','previous-result'):cmd.add_argument('--'+opt,type=Path,required=True)
        if name in ('freeze','audit'):cmd.add_argument('--out',type=Path,required=True)
        if name!='freeze':
            cmd.add_argument('--request',type=Path,required=True);cmd.add_argument('--identity',required=True)
        if name in ('verify','replay-forward'):
            cmd.add_argument('--result',type=Path,required=True);cmd.add_argument('--result-identity',required=True)
    a=p.parse_args();inputs=(a.dataset,a.teacher_run,a.previous_result)
    try:
        if a.operation=='freeze':result=core.freeze(a.out,*inputs)
        elif a.operation=='verify-request':
            req=core.verify_request(a.request,a.identity,*inputs)
            result={'verified':True,'positions':req['positions'],'optimizer_updates':0}
        elif a.operation=='audit':result=native.run(a.request,a.identity,*inputs,a.out)
        else:result=(native.verify if a.operation=='verify' else native.replay)(a.request,a.identity,*inputs,a.result,a.result_identity)
    except (ContractError,OSError,ValueError,TypeError,KeyError,ImportError) as exc:
        print('TT-003-LOGIT-001 v2 rejected: '+str(exc),file=sys.stderr);return 2
    sys.stdout.buffer.write(json_bytes(result));return 0
if __name__=='__main__':raise SystemExit(main())
