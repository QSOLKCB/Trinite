# SPDX-License-Identifier: MPL-2.0
"""All-attempt synthesis and private combined archive; no uploads or seed selection."""
import hashlib
from pathlib import Path
import zipfile

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from . import custody as io, evidence, spec


def dataset_members(dataset, anchor):
    # Full-packet admission is deliberately outside the TRAIN-only worker.
    from teacher_dataset_v1 import core as data
    data.verify(Path(dataset),anchor)
    manifest=io.anchored(Path(dataset)/'manifest.json',anchor)
    expected={'dataset/'+m['name']:m['identity'] for m in manifest['files']}
    expected['dataset/manifest.json']=anchor
    return [('dataset/'+name,Path(dataset)/name) for name in data.MEMBERS],expected


def render(results):
    lines=['# TT-003-FIT-001-TRIPLE-SYNTHESIS', '',
        'DERIVED from the three source-bound attempt inventories and independently recalculated recorded evidence.',
        'These are exposed TRAIN fit observations. They cannot establish held-out generalisation, causal identification, or readiness for distillation.', '',
        'The primary endpoint is 72/72 exact complete answers at update 256 for each seed; every lesser result is not an exact fit.', '',
        '| Seed | Attempt | Update | Exact /72 | Apply /18 | Inverse /18 | Check 1 /18 | Check 0 /18 | Δ zero | Δ one | Native loss | Gold NLL | Premature EOS | Distinct outputs |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for result in results:
        by_step={c['step']:c for c in result['checkpoints']}
        for step in spec.STEPS:
            c=by_step.get(step)
            if c is None:
                lines.append(f"| {result['seed']} | {result['status']} | {step} | unavailable | — | — | — | — | — | — | — | — | — | — |")
                continue
            s=c['summary'];g=s['groups'];base=s['improvement_over_baseline']
            vals=[result['seed'],result['status'],step,g['all']['exact'],g['apply']['exact'],g['inverse']['exact'],g['check_1']['exact'],g['check_0']['exact'],
                base['0'],base['1'],format(float.fromhex(c['evaluation']['train_loss_hex']),'.9g'),
                format(float.fromhex(s['gold_prefix']['all']['mean_target_nll_hex']),'.9g'),
                g['all']['premature_eos'],g['all']['distinct_outputs']]
            lines.append('| '+' | '.join(map(str,vals))+' |')
    lines.extend(['','## Residue and EOS observations','',
        '| Seed | Update | Correct nonzero greedy residues / targets | Correct nonzero best residues at gold prefixes / targets | Gold-prefix zero preference /144 | Mean EOS/rest log-odds |',
        '| --- | --- | --- | --- | --- | --- |'])
    for r in results:
        for c in r['checkpoints']:
            s=c['summary'];v=s['vector_positions'];gold=s['gold_vector_positions']
            lines.append('| '+' | '.join(map(str,[r['seed'],c['step'],
                f"{sum(p['correct_nonzero'] for p in v)}/{sum(p['nonzero_targets'] for p in v)}",
                f"{sum(p['correct_nonzero_best_residue'] for p in gold)}/{sum(p['nonzero_targets'] for p in gold)}",
                sum(p['zero_best_residue'] for p in gold),
                format(float.fromhex(s['gold_prefix']['all']['mean_eos_rest_log_odds_hex']),'.9g')]))+' |')
    lines.extend(['','## Complete attempt inventory',''])
    for r in results:
        final=next((c for c in r['checkpoints'] if c['step']==256),None)
        endpoint=('endpoint unavailable' if final is None else
                  '72/72 endpoint observed' if final['summary']['exact_fit'] else 'not an exact fit')
        lines.append(f"- Seed {r['seed']}: {r['status']}; {endpoint}. Child exit status {r['supervisor']['child_returncode']}; recorded verification passed for the stated scope. Journal records {r['journal_updates']} updates and {r['journal_target_tokens']} supervised targets. Manifest `{r['manifest_identity']}`.")
        lines.append(f"  Reported completed updates: {r['reported_completed_updates']}; supervised targets: {r['reported_target_tokens']}. A null count is unknown, not zero.")
        lines.append(f"  Native forward replay: `{r.get('forward_replay', {'status':'NOT_PERFORMED','native_forward_replayed':False})}`. Optimizer update replay: not performed.")
        if r['failure'] is not None:
            lines.append(f"  Failure receipt: `{r['failure']}`. A hard kill can leave later consumption unknown; journal counts are not an invented final total.")
    lines.extend(['','Full output histograms, circuit scores, all four residue positions, per-update loss/gradient history, all token scores and exact hexadecimal aggregates are retained in the adjacent synthesis JSON and original evidence.',
        'The tables expose differences in progression and repeated zero preference across initializations without selecting a preferred seed or averaging away failures.',
        'Output diversity alone does not establish correct input-sensitive computation; compare it with exact answers and nonzero residue correctness.',
        'Recorded verification recalculates saved evidence. Native forward replay receipts are separate. Complete optimizer update replay was not performed by this workflow.',
        'All replay is same-host evidence, not independent-host reproducibility. Loss reduction demonstrates optimization and does not by itself demonstrate the intended GF(3) computation.', ''])
    return '\n'.join(lines).encode()


def archive_verify(path, anchor, expected):
    path=Path(path)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    if 'sha256:'+h.hexdigest()!=anchor:
        raise ContractError('external archive identity mismatch')
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or sorted(names)!=sorted(expected):
            raise ContractError('archive missing/duplicate/extra members')
        for name in names:
            if name.startswith('/') or '..' in Path(name).parts or identity(z.read(name))!=expected[name]:
                raise ContractError('archive path/member identity mismatch')
    return {'verified':True,'archive_identity':anchor,'members':len(expected)}


def combine(experiment, dataset, request_anchors, out):
    root=Path(experiment);results=[];inputs=[];requests=[]
    io.outside(out,dataset,spec.ROOT,*(root/f'seed-{s}' for s in (17,29,43)))
    for seed in (17,29,43):
        folder=root/f'seed-{seed}'
        anchor=io.load(root/f'seed-{seed}-anchor.json')['manifest_identity']
        result=evidence.verify(folder,anchor,request_anchors[f'seed-{seed}.json'],dataset)
        results.append(result);requests.append(io.load(folder/'request.json'))
        inputs.extend((f'attempts/seed-{seed}/'+p.relative_to(folder).as_posix(),p)
                      for p in sorted(folder.rglob('*')) if p.is_file())
        inputs.append((f'anchors/seed-{seed}.json',root/f'seed-{seed}-anchor.json'))
    if ([r['seed'] for r in results]!=[17,29,43]
            or any(r['source']!=requests[0]['source'] or r['selection']!=requests[0]['selection']
                   or r['environment']!=requests[0]['environment'] for r in requests)):
        raise ContractError('three-run source/environment/panel mismatch')
    initials=[r['checkpoints'][0]['metadata']['initialization_identity'] for r in results if r['checkpoints']]
    if len(set(initials))!=len(initials):
        raise ContractError('distinct seeds inherited identical initialization')
    # Retain failed replay receipts too; missing replay is explicitly unperformed.
    for r in results:
        p=root/f"seed-{r['seed']}-forward-replay.json"
        if p.exists():
            receipt=io.load(p)
            if (receipt.get('manifest_identity')!=r['manifest_identity']
                    or receipt.get('request_identity')!=r['request_identity']
                    or (receipt.get('verified') is True and receipt.get('steps_replayed')!=[c['step'] for c in r['checkpoints']])):
                raise ContractError('forward receipt does not bind this attempt')
            r['forward_replay']=receipt
            inputs.append(('replay/'+p.name,p))
        else:
            r['forward_replay']={'status':'NOT_PERFORMED','native_forward_replayed':False}
    out=io.fresh(out)
    io.record(out/'synthesis.json',{'schema':spec.SCHEMA+'.triple-synthesis','attempts':results,
        'optimizer_replayed':False,'heldout_scored':False})
    io.write(out/'TT-003-FIT-001-TRIPLE-SYNTHESIS.md',render(results))
    io.record(out/'request-anchors.json',request_anchors)
    for name in ('synthesis.json','TT-003-FIT-001-TRIPLE-SYNTHESIS.md','request-anchors.json'):
        inputs.append((name,out/name))
    dataset_inputs,dataset_identities=dataset_members(dataset,requests[0]['dataset_manifest_identity'])
    inputs.extend(dataset_inputs)
    for member in requests[0]['source']['members']:
        p=spec.ROOT/member['path']
        if identity(io.read(p))!=member['identity']:
            raise ContractError('source changed before packaging')
        inputs.append(('source/'+member['path'],p))
    expected={name:identity(io.read(path)) for name,path in inputs}
    if any(expected[name]!=anchor for name,anchor in dataset_identities.items()):
        raise ContractError('dataset changed after full packet verification')
    if len(expected)!=len(inputs):
        raise ContractError('duplicate combined evidence destination')
    archive=out/'TT-003-FIT-001-COMBINED.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,path in inputs:
            raw=io.read(path)
            if identity(raw)!=expected[name]:raise ContractError('input changed during packaging')
            z.writestr(name,raw)
    digest='sha256:'+hashlib.sha256(archive.read_bytes()).hexdigest()
    verified=archive_verify(archive,digest,expected)
    io.record(out/'archive-members.json',expected)
    io.write(out/'TT-003-FIT-001-COMBINED.zip.sha256',(digest[7:]+'  '+archive.name+'\n').encode())
    io.record(out/'TT-003-FIT-001-COMBINED.zip.verification.json',verified)
    return {'archive':str(archive.resolve()),'identity':digest,'verified':True,'attempts':[r['status'] for r in results]}
