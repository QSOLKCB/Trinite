"""Bounded fit-first execution; all predictions anchored before heldout scores."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

import queue_evaluation_v1 as gate
from queue_learning_v1.artifacts import anchored_json, inventory, read_json, retain_failure, write_json
from queue_learning_v2.resources import supervise
from trinite.contracts import json_bytes,parse_json
from . import custody,data,oracle,scoring
from .spec import ATTEMPTS,ROOT,TOKEN_CEILING

LAUNCHER=ROOT/'scripts/run_context_symbol_learning_v1.py'


def launch(args,seconds,progress=None,max_updates=640,max_targets=TOKEN_CEILING):
    return supervise([sys.executable,'-I','-B',str(LAUNCHER),*map(str,args)],seconds,
        progress=progress,max_updates=max_updates,max_targets=max_targets)


def public_packet(content,freeze_sha):
    heldout={r['id'] for r in content['rows'] if r['split']!='train'}
    # No heldout family, request provenance, generator index, mutation identity
    # or label enters the worker packet, much less its model prompt.
    return dict(freeze_sha256=freeze_sha,train=content['train'],
        public=[r for r in content['public'] if r['id'] in heldout])


def execute(args):
    destination=gate.fresh(args.out)
    started=time.monotonic()
    predictions=scoring.empty()
    attempts=[dict(name=n,seed=s,fit=f,planned_updates=u,planned_targets=t,status='not_attempted') for n,s,f,u,t in ATTEMPTS]
    workers={}
    stopped=None
    summary=scoring.summary(predictions)
    barrier_sha=None
    scoring_started=False
    try:
        manifest,content=custody.context(Path(args.frozen),args.freeze_sha,Path(args.anchors),args.anchors_sha)
        custody.authorize(manifest,args.freeze_sha,Path(args.approval))
        gate.require(sum(r['planned_updates'] for r in attempts)==640 and
            sum(r['planned_targets'] for r in attempts)==TOKEN_CEILING, 'static study budget changed')
        # The independently retained anchor is consumed even when execution
        # fails. Changing --out never creates another attempt authorization.
        write_json(Path(args.anchors).with_name(Path(args.anchors).name+'.execution-claim.json'),
            dict(freeze_sha256=args.freeze_sha,out=str(destination),attempts=attempts))
        gate.write(destination/'owner-approval.json',gate.read(args.approval,16384))
        packet=destination/'worker-input.json'
        packet_sha=write_json(packet,public_packet(content,args.freeze_sha))
        constants=dict(majority=scoring.majority(content['train']))
        write_json(destination/'baselines.json',constants)
        certificate_sha=fit_training_sha=None
        for record in attempts:
            name,seed,fit,updates,targets=(record[k] for k in ('name','seed','fit','planned_updates','planned_targets'))
            begin=time.monotonic()
            remaining=min(900.,2700.-(begin-started))
            gate.require(remaining>0, 'whole-study budget exhausted')
            folder=destination/name
            command=['_attempt','--packet',packet,'--packet-sha',packet_sha,'--freeze-sha',args.freeze_sha,
                '--seed',seed,'--out',folder]
            if fit:
                command+=['--fit']
            else:
                command+=['--fit-folder',destination/'fit-17','--fit-training-sha',fit_training_sha,
                    '--fit-certificate',destination/'fit-certificate.json','--fit-certificate-sha',certificate_sha]
            record['status']='attempted'
            receipt=launch(command,remaining,folder/'progress.json',updates,targets)
            workers[name]=dict(attempt=receipt)
            write_json(destination/(name+'-worker.json'),receipt)
            if not fit:
                journal=folder/'predictions.jsonl'
                raw=[] if not journal.exists() else [parse_json(line+b'\n',canonical=True)
                    for line in gate.read(journal,4<<20).splitlines()]
                ids=[r['id'] for r in packet_content(content)]
                gate.require([r['id'] for r in raw]==ids[:len(raw)], 'partial prediction journal order changed')
                found={r['id']:r['raw'] for r in raw}
                for row in predictions:
                    if row['method']==name and row['id'] in found:
                        row['raw']=found[row['id']]
                if custody.successful(receipt):
                    gate.require(len(raw)==len(ids), 'missing final heldout predictions')
            if not custody.successful(receipt):
                record['status']='timeout' if receipt['failure'] in ('timeout','update_timeout') else 'failure'
                for row in predictions:
                    if row['method']==name and row['raw']['status']=='not_attempted':
                        row['raw']=dict(status=record['status'])
                stopped=name+': failed; dependent stages stopped'
                break
            if fit:
                fit_training_sha=gate.digest(gate.read(folder/'training.json',4<<20))
                remaining=min(900.-(time.monotonic()-begin),2700.-(time.monotonic()-started))
                gate.require(remaining>0,'fit lifecycle deadline before independent restoration')
                validation=destination/'fit-validation'
                receipt=launch(['_validate-fit','--checkpoint',folder,'--training-sha',fit_training_sha,
                    '--packet',packet,'--freeze-sha',args.freeze_sha,'--out',validation],remaining)
                workers[name]['validation']=receipt
                write_json(destination/'fit-validation-worker.json',receipt)
                gate.require(custody.successful(receipt), 'independent final fit restoration failed')
                certificate_sha=write_json(destination/'fit-certificate.json',read_json(validation/'certificate.json'))
            gate.require(time.monotonic()-begin<900 and time.monotonic()-started<2700, 'attempt/study lifecycle deadline')
            record['status']='completed'
            record['whole_seconds_hex']=(time.monotonic()-begin).hex()
        if stopped is None:
            gate.require(all(r['status']=='completed' for r in attempts), 'every attempt required')
            public={r['id']:r['observation'] for r in content['public']}
            for row in predictions:
                if row['method']=='majority':
                    row['raw']=scoring.prediction(constants['majority'])
                elif row['method']=='context-ignoring':
                    row['raw']=scoring.prediction(scoring.shortcut(public[row['id']]))
            scoring.check_predictions(predictions,complete=True)
            barrier_sha=write_json(destination/'committed-predictions.json',predictions)
            # External, exclusive barrier binds every method/id/raw token list.
            write_json(Path(args.run_anchor).with_name(Path(args.run_anchor).name+'.predictions.json'),
                dict(freeze_sha256=args.freeze_sha,predictions_sha256=barrier_sha,
                    checkpoint_inventory={r['name']:gate.digest(gate.read(destination/r['name']/'training.json',4<<20)) for r in attempts}))
            gate.require(time.monotonic()-started<2700,'study budget before privileged evaluation')
            custody.context(Path(args.frozen),args.freeze_sha,Path(args.anchors),args.anchors_sha)
            scoring_started=True
            targets={}
            gate.write(destination/'target-journal.jsonl',b'')
            gate.write(destination/'oracle-processes.jsonl',b'')
            from queue_learning_v1.artifacts import append_json
            for row in content['rows']:
                if row['split']=='train':
                    continue
                targets[row['id']]=oracle.target(row['request'],lambda v:append_json(destination/'oracle-processes.jsonl',v))
                append_json(destination/'target-journal.jsonl',dict(id=row['id'],answer=targets[row['id']]))
            write_json(destination/'targets.json',targets)
            summary=scoring.summary(predictions,targets)
            write_json(destination/'scores.json',summary)
            custody.context(Path(args.frozen),args.freeze_sha,Path(args.anchors),args.anchors_sha)
            gate.require(time.monotonic()-started<2700,'study deadline after scoring')
    except Exception as error:
        stopped=type(error).__name__+': '+str(error)
        retain_failure(destination,error)
        if isinstance(error,gate.ProcessError):
            write_json(destination/'oracle-process-failure.json',error.evidence)
        for record in attempts:
            if record['status']=='attempted':
                record['status']='failure'
    finally:
        if stopped:
            summary=scoring.summary(predictions)
        write_json(destination/'predictions.json',predictions)
        write_json(destination/'attempts.json',attempts)
        write_json(destination/'run.json',dict(schema='trinite.context-symbol-run.v1',producer_sha=None if 'manifest' not in locals() else manifest['producer_sha'],
            freeze_sha256=args.freeze_sha,stopped=stopped,workers=workers,barrier_sha256=barrier_sha,
            scoring_started=scoring_started,summary=summary,optimizer_replay=False,
            wall_seconds_hex=(time.monotonic()-started).hex()))
    return dict(failed=stopped is not None,stopped=stopped)


def packet_content(content):
    heldout={r['id'] for r in content['rows'] if r['split']!='train'}
    return [r for r in content['public'] if r['id'] in heldout]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='operation',required=True)
    sub.add_parser('build')
    for name in ('feasibility','_feasibility'):
        q=sub.add_parser(name); q.add_argument('--out',required=True)
    q=sub.add_parser('freeze')
    for field in ('feasibility','feasibility-sha','out','anchors'):
        q.add_argument('--'+field,required=True)
    for name in ('run','_run','verify'):
        q=sub.add_parser(name)
        for field in ('frozen','freeze-sha','anchors','anchors-sha','run-anchor'):
            q.add_argument('--'+field,required=True)
        if name=='verify':
            q.add_argument('--run',required=True);q.add_argument('--run-sha',required=True)
            q.add_argument('--scores-only',action='store_true',help='portable custody/data/score replay; no native tensor/runtime validation')
        else:
            q.add_argument('--out',required=True);q.add_argument('--approval',required=True)
    q=sub.add_parser('_attempt')
    for field in ('packet','packet-sha','freeze-sha','out'):
        q.add_argument('--'+field,required=True)
    q.add_argument('--seed',type=int,choices=(17,29),required=True)
    q.add_argument('--fit',action='store_true')
    for field in ('fit-folder','fit-training-sha','fit-certificate','fit-certificate-sha'):
        q.add_argument('--'+field)
    q=sub.add_parser('_validate-fit')
    for field in ('checkpoint','training-sha','packet','freeze-sha','out'):
        q.add_argument('--'+field,required=True)
    args=p.parse_args()
    try:
        if args.operation in ('run','feasibility'):
            out=gate.safe_path(args.out)
            gate.require(not out.exists(),'fresh output required; retained attempts cannot be rerun')
            if args.operation=='run':
                gate.disjoint(args.frozen,args.anchors,args.approval,args.out,args.run_anchor)
                gate.require(not Path(args.run_anchor).exists(), 'fresh independent run anchor required')
            receipt=launch(['_'+args.operation,*sys.argv[2:]],2700 if args.operation=='run' else 900)
            if not out.exists():
                gate.fresh(out)
            write_json(out/'supervisor.json',receipt)
            result=dict(failed=not custody.successful(receipt),supervisor=receipt)
            if args.operation=='feasibility' and custody.successful(receipt):
                proof=read_json(out/'feasibility.json')
                proof['supervisor']=receipt
                result['feasibility_sha256']=write_json(out/'supervised-feasibility.json',proof)
            elif args.operation=='run':
                result['run_sha256']=write_json(Path(args.run_anchor),dict(schema='trinite.context-symbol-evidence-anchor.v1',
                    freeze_sha256=args.freeze_sha,files=inventory(out)))
                if (out/'run.json').exists():
                    result['failed'] |= read_json(out/'run.json')['stopped'] is not None
        elif args.operation=='_feasibility':
            custody.feasibility(args.out);result=dict(optimizer_updates=0)
        elif args.operation=='freeze':
            result=custody.freeze(args.feasibility,args.feasibility_sha,args.out,args.anchors)
        elif args.operation=='_run':
            result=execute(args)
        elif args.operation=='_attempt':
            from .workers import attempt
            attempt(args);result=dict(completed=True)
        elif args.operation=='_validate-fit':
            from .workers import validate_fit
            out=gate.fresh(args.out)
            write_json(out/'certificate.json',validate_fit(args.checkpoint,args.training_sha,args.packet,args.freeze_sha))
            result=dict(completed=True)
        elif args.operation=='verify':
            from .verification import verify
            result=verify(args)
        else:
            result=dict(oracle_sha256=oracle.build())
        print(json_bytes(result).decode('ascii'),end='')
        return 2 if result.get('failed') else 0
    except (ValueError,OSError,RuntimeError,KeyError,TypeError,subprocess.SubprocessError) as error:
        print('context-symbol-learning-v1 rejected: '+type(error).__name__+': '+str(error),file=sys.stderr)
        return 2
