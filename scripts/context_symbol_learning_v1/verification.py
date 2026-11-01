"""Read-only source/custody/score replay. NEVER calls the numerical updater."""
from pathlib import Path

import queue_evaluation_v1 as gate
from queue_learning_v1.artifacts import anchored_json,inventory,read_json
from trinite.contracts import json_bytes,parse_json
from . import custody,data,numerical,oracle,scoring
from .spec import ATTEMPTS


def verify(args):
    run=gate.safe_path(args.run)
    scores_only=getattr(args,'scores_only',False)
    before=inventory(run)
    anchor=anchored_json(Path(args.run_anchor),args.run_sha)
    gate.require(anchor==dict(schema='trinite.context-symbol-evidence-anchor.v1',freeze_sha256=args.freeze_sha,
        files=before), 'external final inventory anchor mismatch')
    manifest,content=custody.context(Path(args.frozen),args.freeze_sha,Path(args.anchors),args.anchors_sha,live=False,scores_only=scores_only)
    # Verification can also deposit an interrupted outer-supervisor run. Its
    # missing run record is explicitly incomplete, never inferred successful.
    gate.require((run/'supervisor.json').exists(), 'outer resource receipt missing')
    outer=read_json(run/'supervisor.json')
    if not (run/'run.json').exists():
        gate.require(not custody.successful(outer), 'successful supervisor without run record')
        gate.require(inventory(run)==before,'verification changed interrupted evidence')
        return dict(verified=True,complete=False,optimizer_replay=False,scope='anchored interrupted custody only; no scores')
    record=read_json(run/'run.json')
    gate.require(record['producer_sha']==manifest['producer_sha'] and record['freeze_sha256']==args.freeze_sha,
        'producer/evidence context differs')
    custody.authorize(manifest,args.freeze_sha,run/'owner-approval.json')
    from .controller import public_packet
    expected_packet=public_packet(content,args.freeze_sha)
    gate.require(read_json(run/'worker-input.json')==expected_packet, 'worker public/private separation changed')
    attempts=read_json(run/'attempts.json')
    gate.require(len(attempts)==3,'attempt inventory changed')
    for actual,(name,seed,fit,updates,targets) in zip(attempts,ATTEMPTS):
        gate.require({k:actual[k] for k in ('name','seed','fit','planned_updates','planned_targets')}==
            dict(name=name,seed=seed,fit=fit,planned_updates=updates,planned_targets=targets), 'frozen attempt plan changed')
        gate.require(actual['status'] in ('completed','failure','timeout','not_attempted'),'attempt status changed')
        folder=run/name
        if actual['status']=='completed' and not scores_only:
            from .workers import load_checkpoint
            info=read_json(folder/'training.json')
            state=load_checkpoint(folder,info,content['train'],seed,args.freeze_sha,fit)
            gate.require(custody.successful(record['workers'][name]['attempt']), 'completed attempt lacks successful resource receipt')
            if fit:
                from .workers import validate_fit
                certificate=validate_fit(folder,gate.digest(gate.read(folder/'training.json',4<<20)),run/'worker-input.json',args.freeze_sha)
                gate.require(read_json(run/'fit-certificate.json')==certificate, 'independent fit certificate changed')
                numerical.require_fit(state,content['train'])
    predictions=read_json(run/'predictions.json')
    scoring.check_predictions(predictions)
    by_key={(r['method'],r['id']):r['raw'] for r in predictions}
    for name,seed,fit,_,_ in ATTEMPTS:
        if fit:
            continue
        journal=run/name/'predictions.jsonl'
        if journal.exists():
            raw=[parse_json(line+b'\n',canonical=True) for line in gate.read(journal,4<<20).splitlines()]
            gate.require([r['id'] for r in raw]==[r['id'] for r in expected_packet['public'][:len(raw)]],
                'raw prediction journal differs from declared public order')
            gate.require(all(by_key[name,r['id']]==r['raw'] for r in raw),'prediction inventory differs from raw journal')
            if next(r for r in attempts if r['name']==name)['status']=='completed':
                gate.require(len(raw)==len(expected_packet['public']), 'completed seed missing raw predictions')
        else:
            gate.require(next(r for r in attempts if r['name']==name)['status']!='completed', 'completed seed missing raw journal')
    if (run/'baselines.json').exists():
        constant=scoring.majority(content['train'])
        gate.require(read_json(run/'baselines.json')==dict(majority=constant),'baseline used non-training facts')
        if record['barrier_sha256'] is not None:
            for row in expected_packet['public']:
                gate.require(by_key['majority',row['id']]==scoring.prediction(constant) and
                    by_key['context-ignoring',row['id']]==scoring.prediction(scoring.shortcut(row['observation'])),
                    'baseline prediction replay differs')
    targets=None
    if record['barrier_sha256'] is not None:
        committed=anchored_json(run/'committed-predictions.json',record['barrier_sha256'])
        gate.require(committed==predictions,'committed raw predictions changed')
        scoring.check_predictions(committed,complete=True)
        external=read_json(Path(args.run_anchor).with_name(Path(args.run_anchor).name+'.predictions.json'))
        expected=dict(freeze_sha256=args.freeze_sha,predictions_sha256=record['barrier_sha256'],
            checkpoint_inventory={n:gate.digest(gate.read(run/n/'training.json',4<<20)) for n,_,_,_,_ in ATTEMPTS})
        gate.require(external==expected,'external prediction/checkpoint commitment changed')
        if (run/'target-journal.jsonl').exists():
            lines=gate.read(run/'target-journal.jsonl',4<<20).splitlines()
            journal=[parse_json(line+b'\n',canonical=True) for line in lines]
            heldout=[r for r in content['rows'] if r['split']!='train']
            gate.require([r['id'] for r in journal]==[r['id'] for r in heldout[:len(journal)]], 'partial target order changed')
            for row,truth in zip(heldout,journal):
                gate.require(truth['answer']==target(row['request'],scores_only),'partial target oracle mismatch')
        if (run/'targets.json').exists():
            targets={r['id']:target(r['request'],scores_only) for r in content['rows'] if r['split']!='train'}
            gate.require(read_json(run/'targets.json')==targets,'native and independent target replay differs')
            gate.require(read_json(run/'scores.json')==scoring.summary(predictions,targets),'score recomputation differs')
    else:
        gate.require(not record['scoring_started'] and not (run/'targets.json').exists()
            and not (run/'target-journal.jsonl').exists(), 'heldout truth without full prediction barrier')
    expected_summary=scoring.summary(predictions,targets if record['stopped'] is None else None)
    gate.require(json_bytes(record['summary'])==json_bytes(expected_summary),'run summary changed')
    if record['stopped'] is None:
        gate.require(targets is not None
            and all(r['status']=='completed' for r in attempts), 'complete result missing successful stages')
    gate.require(inventory(run)==before,'read-only verification changed run bytes')
    return dict(verified=True,complete=record['stopped'] is None and custody.successful(outer),
        outer_failure=outer['failure'],producer_sha=manifest['producer_sha'],
        scores_recomputed=targets is not None,optimizer_replay=False,scores_only=scores_only,
        scope=('portable source/custody/data/score replay; recorded runtime and tensor structure not independently validated; '
               if scores_only else 'source-bound custody, safe final checkpoint structure, fit regeneration and score replay; ')+
              'not independent optimizer replay, host attestation or proof of operator blindness')


def target(request,scores_only):
    if scores_only:
        from procedural_dataset_v1.cases import expected_target
        return expected_target('context',request)
    return oracle.target(request)
