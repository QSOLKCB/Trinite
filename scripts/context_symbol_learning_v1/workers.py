"""Cooperative fixed workers: numerical state sees training labels/public facts only."""
from pathlib import Path
import time

import queue_evaluation_v1 as gate
from queue_learning_v1.artifacts import anchored_json, append_json, read_json, retain_failure, write_json
from queue_learning_v2.workers import progress, usage
from trinite.contracts import json_bytes
from . import data, numerical
from .spec import plan


def load_checkpoint(folder, info, train, seed, binding, fit=False):
    folder=Path(folder)
    state=numerical.restore(gate.read(folder/'tensors.safetensors',32<<20),
        gate.read(folder/'metadata.json',1<<20),train,seed,binding,info['checkpoint_sha256'],fit)
    schedule=numerical.schedule(train,seed,fit)
    gate.require(state.step==plan(seed,fit).steps and state.target_tokens==schedule['targets']
        and type(info['step']) is int and info['step']==state.step
        and type(info['target_tokens']) is int and info['target_tokens']==state.target_tokens
        and info['schedule']==schedule and info['seed']==seed and info['fit'] is fit
        and info['train_sha256']==gate.digest(json_bytes(train))
        and 0<float.fromhex(info['update_seconds_hex'])<120, 'incomplete checkpoint/attempt budget')
    return state


def validate_fit(folder, checkpoint_sha, packet, binding):
    info=anchored_json(Path(folder)/'training.json',checkpoint_sha)
    train=read_json(packet)['train']
    state=load_checkpoint(folder,info,train,17,binding,True)
    report=numerical.require_fit(state,train)
    return dict(schema='trinite.context-symbol-fit-gate.v1',freeze_sha256=binding,
        training_sha256=checkpoint_sha,checkpoint_sha256=info['checkpoint_sha256'],report=report,
        packet_sha256=gate.digest(gate.read(packet,4<<20)))


def attempt(args):
    import resource
    numerical.configure()
    packet=anchored_json(Path(args.packet),args.packet_sha)
    train=packet['train']
    numerical.check_rows(train)
    gate.require(packet['freeze_sha256']==args.freeze_sha, 'worker packet freeze mismatch')
    if not args.fit:
        expected=validate_fit(args.fit_folder,args.fit_training_sha,args.packet,args.freeze_sha)
        gate.require(anchored_json(Path(args.fit_certificate),args.fit_certificate_sha)==expected,
                     'independently restored final fit certificate required')
    destination=gate.fresh(args.out)
    gate.write(destination/'progress.jsonl',b'')
    gate.write(destination/'predictions.jsonl',b'')
    start,before=time.monotonic(),resource.getrusage(resource.RUSAGE_SELF)
    state=None
    try:
        progress(destination,'preparing')
        state=numerical.state_for(train,args.seed,args.freeze_sha,args.fit)
        selected=data.fit_rows(train) if args.fit else train
        write_json(destination/'initial.json',dict(step=0,unique_train=len(selected),
            train_loss_hex=state.initial_train_loss,fit_report=numerical.fit_report(state.model,train) if args.fit else None))
        updates=time.monotonic()
        progress(destination,'updates',state,updates)
        from trinite.training import run_steps
        for step in range(1,state.config.steps+1):
            gate.require(time.monotonic()-updates<120, 'cumulative native update deadline')
            run_steps(state,stop_after=step)
            gate.require(state.step==step and time.monotonic()-updates<120, 'incomplete/overdue native update')
            progress(destination,'updates',state,updates)
        elapsed=time.monotonic()-updates
        schedule=numerical.schedule(train,args.seed,args.fit)
        gate.require(state.target_tokens==schedule['targets'], 'exact final supervised-token count required')
        progress(destination,'checkpoint',state,updates)
        payload,metadata=numerical.snapshot(state,args.freeze_sha)
        gate.write(destination/'tensors.safetensors',payload)
        gate.write(destination/'metadata.json',metadata)
        info=dict(seed=args.seed,fit=args.fit,step=state.step,target_tokens=state.target_tokens,
            checkpoint_sha256=[gate.digest(payload),gate.digest(metadata)],schedule=schedule,
            train_sha256=gate.digest(json_bytes(train)),update_seconds_hex=elapsed.hex())
        write_json(destination/'training.json',info)
        # Reload actual serialized final tensors with closed inventory/finite and
        # schedule checks. Never use a worker-supplied pass boolean as the gate.
        del state
        state=load_checkpoint(destination,info,train,args.seed,args.freeze_sha,args.fit)
        from trinite.training import evaluate,model_identity
        write_json(destination/'final.json',dict(step=state.step,unique_train=len(selected),
            train_loss_hex=evaluate(state.model,selected,8),
            fit_report=numerical.fit_report(state.model,train) if args.fit else None))
        if args.fit:
            numerical.require_fit(state,train)
        else:
            before_identity=model_identity(state.model)
            for row in packet['public']:
                append_json(destination/'predictions.jsonl',dict(id=row['id'],raw=numerical.infer(state.model,row['observation'])))
            gate.require(model_identity(state.model)==before_identity, 'inference mutated checkpoint')
        progress(destination,'completed',state,updates)
        write_json(destination/'completed.json',dict(step=state.step,freeze_sha256=args.freeze_sha))
    except Exception as error:
        if state is not None and not (destination/'tensors.safetensors').exists():
            try:
                payload,metadata=numerical.snapshot(state,args.freeze_sha)
                gate.write(destination/'partial.safetensors',payload)
                gate.write(destination/'partial-metadata.json',metadata)
            except Exception as snapshot_error:
                write_json(destination/'partial-snapshot-failure.json',dict(type=type(snapshot_error).__name__))
        retain_failure(destination,error)
        raise
    finally:
        write_json(destination/'resources.json',usage(start,before))
