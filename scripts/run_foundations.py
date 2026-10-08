"""Fresh fixed CPU workers; preserve blocked gates and every worker failure."""
import argparse
import itertools
import os
from pathlib import Path
import subprocess
import sys

import torch

from trinite.contracts import ContractError, identity, json_bytes
from trinite.foundations import read_request, decide, summarize
from trinite.foundations_training import LANES, SEEDS, WORKLOADS, RUNNER_IDENTITY

ROOT=Path(__file__).resolve().parents[1]


def launch(stage,directory,workload,lane,seed,request,request_identity,decision_identity,out,env):
    if stage not in ('train','test') or workload not in WORKLOADS or lane not in LANES or type(seed) is not int or seed not in SEEDS:
        raise ValueError('unknown fixed foundations worker')
    # Trusted locked interpreter/module/selectors; values are separate argv and
    # never executable shell fragments. No arbitrary external command input.
    # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit, dangerous-subprocess-use-audit
    return subprocess.run([sys.executable,'-m','trinite','foundations-cell',str(directory),
        '--stage',stage,'--workload',workload,'--lane',lane,'--seed',str(seed),
        '--request',str(request),'--request-identity',request_identity,
        '--decision-identity',decision_identity],shell=False,cwd=ROOT,env=env,
        stdout=out,stderr=subprocess.STDOUT,timeout=240)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--request-identity',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    request=args.request.resolve();read_request(request,args.request_identity)
    runner_identity=identity(Path(__file__).read_bytes())
    if runner_identity!=RUNNER_IDENTITY:raise ContractError('foundations runner differs from source receipt')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    (output/'request.json').write_bytes(request.read_bytes())
    (output/'runner-source.json').write_bytes(json_bytes({'source_identity':runner_identity}))
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    errors={};decision=None
    matrix=[(w,s,lane) for w,s in itertools.product(WORKLOADS,SEEDS) for lane in LANES[s:]+LANES[:s]]
    for stage in ('train','test'):
        if stage=='test':
            if identity(Path(__file__).read_bytes())!=runner_identity:errors['runner']='runner source changed'
            decision=decide(output,request,args.request_identity,errors)
            if not decision['eligible']:
                print('held-out scoring blocked; verified training decision retained',flush=True)
                break
        for workload,seed,lane in matrix:
            name=f'{workload}-{seed}-{lane}'
            if name in errors:continue
            with (output/(name+'-'+stage+'.log')).open('x') as out:
                try:
                    result=launch(stage,output/name,workload,lane,seed,request,args.request_identity,
                        decision['decision_identity'] if decision else 'not-used-during-training',out,env)
                    if result.returncode:errors[name]=f'{stage}: worker exit {result.returncode}; see retained log'
                except subprocess.TimeoutExpired:errors[name]=stage+': worker exceeded 240 seconds'
                except OSError as error:errors[name]=stage+': '+type(error).__name__+': '+str(error)
            print(stage,name,'failed' if name in errors else 'completed',flush=True)
    if identity(Path(__file__).read_bytes())!=runner_identity:errors['runner']='runner source changed'
    result=summarize(output,request,args.request_identity,errors)
    print(result['outcome'],'dense learning adequate:',result['dense_learning_adequate'],flush=True)
    return int(result['outcome']!='completed' or not result['dense_learning_adequate'])


if __name__=='__main__':raise SystemExit(main())
