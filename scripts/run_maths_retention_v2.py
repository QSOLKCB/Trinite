"""Fixed fresh v2 retention workers with explicit inference/replay verification."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def worker_timeout(stage):
    from trinite.retention_v2 import spec
    # The matrix stages verify all fixed cells sequentially, including scoring
    # and custody IO. Give each cell the same complete-worker allowance as train.
    return spec.protocol()['worker_seconds'] * (1 if stage == 'train' else len(spec.cells()))


def launch(stage, request, request_identity, verification, verification_identity, output, tier, log, cell=None):
    if stage not in ('train', 'summarize', 'verify') or tier not in ('inference', 'replay'):
        raise ValueError('unknown fixed v2 worker/tier')
    selectors = []
    if stage == 'train':
        if (type(cell) is not tuple or len(cell)!=2 or type(cell[0]) is not int or cell[0] not in (0,1)
                or type(cell[1]) is not str or cell[1] not in ('pooled','rehearsal')):
            raise ValueError('unknown fixed v2 cell')
        selectors = ['--seed', str(cell[0]), '--profile', cell[1]]
    elif cell is not None: raise ValueError('v2 summary cannot select a cell')
    verification_args = [] if stage == 'train' else ['--verification-request', str(verification), '--verification-request-identity', verification_identity]
    # Fixed argv/interpreter/script, no shell or caller-supplied executable.
    # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
    return subprocess.run([sys.executable, str(ROOT/'scripts/run_maths_retention_v2.py'), '--worker', stage,
        '--request', str(request), '--request-identity', request_identity, '--output', str(output),
        '--tier', tier, *verification_args, *selectors], cwd=ROOT, shell=False,
        env=dict(os.environ, PYTHONPATH=str(ROOT/'src'), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'),
        stdout=log, stderr=subprocess.STDOUT, timeout=worker_timeout(stage))


def controller_failure(output, request_identity, verification_identity, errors, message):
    from trinite.retention_v2 import evidence, spec
    value = {'schema': 'trinite.retention-controller-failure.v2', 'outcome':'failed',
        'request_identity':request_identity, 'verification_request_identity':verification_identity,
        'worker_errors':errors, 'error':message,
        'cells':[{'seed':s,'profile':p,'outcome':'failed','error':'verification not completed: '+message}
                 for s,p in spec.cells()], 'learning_adequacy':False,'phase5_ready':False,'release_ready':False}
    evidence.write(output/'controller-failure.json',value)


def matrix(request, request_identity, verification, output, tier):
    from trinite.retention_v2 import evidence, experiment, spec
    output.mkdir(parents=True, exist_ok=False)
    (output/'request.json').write_bytes(evidence.read_bytes(request))
    errors = {}
    for cell in spec.cells():
        name = spec.cell_name(*cell)
        with (output/(name+'.log')).open('x') as log:
            try:
                result = launch('train', request, request_identity, None, None, output/name, tier, log, cell)
                if result.returncode: errors[name] = 'training worker exited '+str(result.returncode)
            except subprocess.TimeoutExpired as error: errors[name] = 'training worker exceeded '+str(error.timeout)+' seconds'
            except OSError as error: errors[name] = type(error).__name__+': '+str(error)
        print(name+': '+errors.get(name,'completed'), flush=True)
    evidence.write(output/'worker-errors.json', errors)
    try:
        receipt = experiment.freeze_verification_request(verification, output, request, request_identity)
        verification_identity = receipt['verification_request_identity']
        (output/'verification-request.json').write_bytes(evidence.read_bytes(verification))
        print(evidence.json_bytes(receipt).decode(), end='', flush=True)
    except (evidence.ContractError, OSError, ValueError) as error:
        controller_failure(output,request_identity,None,errors,'verification request freeze: '+str(error));return 1
    with (output/('summary-'+tier+'.log')).open('x') as log:
        try:
            result = launch('summarize',request,request_identity,verification,verification_identity,output,tier,log)
            if result.returncode or not (output/('summary-'+tier+'.json')).exists():
                controller_failure(output,request_identity,verification_identity,errors,'summary worker exited '+str(result.returncode));return 1
        except subprocess.TimeoutExpired as error:
            controller_failure(output,request_identity,verification_identity,errors,'summary worker exceeded '+str(error.timeout)+' seconds');return 1
        except OSError as error:
            controller_failure(output,request_identity,verification_identity,errors,'summary spawn: '+str(error));return 1
    summary = evidence.parse_json(evidence.read_bytes(output/('summary-'+tier+'.json')),canonical=True)
    print(evidence.json_bytes({'outcome':summary['outcome'],'tier':tier,'contrasts':summary['contrasts'],
        'summary_identity':evidence.identity(evidence.json_bytes(summary)),'learning_adequacy':False,'phase5_ready':False}).decode(),end='')
    return 0 if summary['outcome']=='completed' else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-request',type=Path)
    parser.add_argument('--run-profile',choices=('research','smoke'))
    parser.add_argument('--request',type=Path);parser.add_argument('--request-identity')
    parser.add_argument('--verification-request',type=Path);parser.add_argument('--verification-request-identity')
    parser.add_argument('--output',type=Path);parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--tier',choices=('inference','replay'),default='replay')
    parser.add_argument('--worker',choices=('train','summarize','verify'))
    parser.add_argument('--seed',type=int,choices=(0,1));parser.add_argument('--profile',choices=('pooled','rehearsal'))
    args=parser.parse_args(argv)
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    from trinite.retention_v2 import evidence,experiment
    try:
        selectors=(args.seed,args.profile)
        if args.freeze_request:
            if (args.request or args.request_identity or args.verification_request or args.verification_request_identity
                    or args.output or args.verify_only or args.worker or any(x is not None for x in selectors)):
                raise evidence.ContractError('v2 freeze cannot run workers')
            print(evidence.json_bytes(experiment.freeze_request(args.freeze_request,args.run_profile or 'research')).decode(),end='');return 0
        if args.run_profile:raise evidence.ContractError('v2 run profile comes only from immutable experiment request')
        if not args.request or not args.request_identity or not args.output:raise evidence.ContractError('v2 request/identity/output required')
        paths=[args.request,args.output]+([args.verification_request] if args.verification_request else [])
        for path in paths:evidence.safe_path(path)
        args.request=args.request.resolve();args.output=args.output.resolve()
        if args.verification_request:args.verification_request=args.verification_request.resolve()
        immutable=[args.request]+([args.verification_request] if args.verification_request else [])
        if (any(p==args.output or args.output in p.parents for p in immutable)
                or len(set(immutable))!=len(immutable)):
            raise evidence.ContractError('v2 output overlaps immutable requests')
        experiment.read_request(args.request,args.request_identity)
        if args.worker=='train':
            if (args.verify_only or args.verification_request or args.verification_request_identity or any(x is None for x in selectors)):
                raise evidence.ContractError('v2 train requires only complete fixed cell selectors')
            experiment.train_cell(args.output,*selectors,args.request,args.request_identity);return 0
        if any(x is not None for x in selectors):raise evidence.ContractError('v2 matrix/summary cannot select a cell')
        if not args.verification_request:raise evidence.ContractError('v2 external verification request path required')
        if args.worker:
            if args.verify_only or not args.verification_request_identity:raise evidence.ContractError('v2 worker needs verification identity')
            value=experiment.summarize(args.output,args.request,args.request_identity,args.verification_request,args.verification_request_identity,tier=args.tier)
            path=args.output/('summary-'+args.tier+'.json')
            if args.worker=='summarize':evidence.write(path,value)
            elif path.exists() and evidence.read_bytes(path)!=evidence.json_bytes(value):
                raise evidence.ContractError('v2 retained summary differs from fresh verification')
            print(evidence.json_bytes(value).decode(),end='')
            return 0 if value['outcome']=='completed' else 1
        if args.verify_only:
            if not args.verification_request_identity:raise evidence.ContractError('v2 read-only verification needs externally supplied identity')
            try:return launch('verify',args.request,args.request_identity,args.verification_request,args.verification_request_identity,args.output,args.tier,None).returncode
            except subprocess.TimeoutExpired as error:raise evidence.ContractError('v2 verifier exceeded '+str(error.timeout)+' seconds') from error
        if args.verification_request_identity:raise evidence.ContractError('new v2 matrix freezes a new exclusive verification request')
        if args.verification_request.exists():raise evidence.ContractError('v2 matrix cannot overwrite verification request')
        return matrix(args.request,args.request_identity,args.verification_request,args.output,args.tier)
    except (evidence.ContractError,OSError,KeyError,TypeError,ValueError) as error:
        print('retention-v2: '+str(error),file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
