#!/usr/bin/env python3
"""Read-only replay of this stopped deposit; no training entrypoint is invoked."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile

PRODUCER='578d4793a02585a0bf8b86660c4fd99720db118c'
FREEZE='f6cb7b0e254291bfdf7c2539ab9db5a3bd54ef5ae0fe696d125a132e9b2ee069'
PREPARATION='a1224ca2dcce2695ba6b1f89a1c1a5808db332289533bd1d0336d9e1b42fd1ef'
SOURCE='df817cadfc41efb7aa538cde8897f5327065645c4638321005353aaa5605e6ac'
RUN='c4ded51e926fd8857ee9c1ae8a9ab7fe7a802dcf23cd03fc509e86b1abb256ed'


def require(condition,message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path,expected=None):
    raw=path.read_bytes()
    require(len(raw)<=4<<20,'JSON size limit')
    if expected is not None:
        require(digest(raw)==expected,'external anchor mismatch: '+str(path))
    def pairs(values):
        result={}
        for key,value in values:
            require(key not in result,'duplicate JSON key')
            result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=pairs)


def files(root):
    return {p.relative_to(root).as_posix():digest(p.read_bytes()) for p in sorted(root.rglob('*')) if p.is_file()}


def unpack(archive,destination):
    """No paths, links, duplicates, oversized members or extraction overwrite."""
    with zipfile.ZipFile(archive) as z:
        entries=z.infolist()
        require(len(entries)<=4096 and len({i.filename for i in entries})==len(entries),'ZIP member inventory')
        require(sum(i.file_size for i in entries)<=128<<20,'ZIP total size')
        for item in entries:
            name=PurePosixPath(item.filename)
            require(not name.is_absolute() and '..' not in name.parts and
                name.as_posix()==item.filename.rstrip('/') and '\\' not in item.filename,'unsafe ZIP path')
            require(stat.S_IFMT(item.external_attr>>16) in (0,stat.S_IFREG,stat.S_IFDIR),'ZIP link/special file')
            require(item.file_size<=32<<20,'ZIP member size')
            target=destination.joinpath(*name.parts)
            if item.is_dir():
                target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as out:
                    out.write(z.read(item))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native',action='store_true',help='require exact producer CPU environment; safe-load and audit final fit, zero updates')
    args=parser.parse_args()
    require(sys.platform=='linux','frozen producer launcher requires Linux')
    here=Path(__file__).resolve().parent
    before=files(here)
    deposit=load(here/'manifest.json')
    require(deposit['producer_sha']==PRODUCER and deposit['freeze_sha256']==FREEZE and
        deposit['run_anchor_sha256']==RUN and deposit['source_archive_sha256']==SOURCE,'deposit identity mismatch')
    for name,expected in deposit['files'].items():
        require(PurePosixPath(name).name==name and digest((here/name).read_bytes())==expected,'deposit file hash: '+name)
    require(set(before)==set(deposit['files'])|{'manifest.json'},'extra/missing deposit files')
    with tempfile.TemporaryDirectory(prefix='trinite-context-evidence-') as folder:
        temporary=Path(folder)
        payload=temporary/'payload';payload.mkdir()
        unpack(here/'payload.zip',payload)
        expected_files=deposit['payload_files']
        require(files(payload)==expected_files,'payload file inventory mismatch')
        freeze=load(payload/'frozen/manifest.json',FREEZE)
        load(payload/'anchors/preparation.json',PREPARATION)
        run_anchor=load(payload/'anchors/run-anchor.json',RUN)
        require(freeze['producer_sha']==PRODUCER and digest((payload/'frozen/source.zip').read_bytes())==SOURCE,'producer source mismatch')
        for name,expected in run_anchor['files'].items():
            path=payload/'run'/name
            if expected=='directory':
                require(path.is_dir(),'missing retained run directory')
            else:
                require(digest(path.read_bytes())==expected,'retained run file changed: '+name)
        record=load(payload/'run/run.json')
        result_note=load(here/'result.json')
        info=load(payload/'run/fit-17/training.json')
        metadata=load(payload/'run/fit-17/metadata.json')
        fit_final=load(payload/'run/fit-17/final.json')
        train={r['identity']:r['answer'] for r in load(payload/'frozen/data.json')['train']}
        exact=0
        for row in fit_final['fit_report']:
            require(row['id'] in train and row['expected']==train[row['id']],'fit answer outside training admission')
            correct=row['raw']['tokens']==[*train[row['id']].encode('ascii'),257]
            require(row['exact'] is correct,'stored fit exact flag differs from canonical answer/EOS')
            exact+=correct
        require(info['step']==metadata['step']==128 and metadata['cursor']==1024 and
            info['target_tokens']==metadata['target_tokens']==2432 and len(metadata['history'])==128 and
            sum(r['target_tokens'] for r in metadata['history'])==2432,'native progress/target totals differ')
        require(exact==7 and len(fit_final['fit_report'])==8 and
            result_note['actual_optimizer_updates']==128 and result_note['actual_supervised_tokens']==2432 and
            result_note['fit']['final_exact']==exact and result_note['summary']==record['summary'],'report/counter consistency')
        require(record['stopped']=='fit-17: failed; dependent stages stopped' and
            not record['scoring_started'] and record['barrier_sha256'] is None,'mandatory lifecycle stop changed')
        require([r['status'] for r in load(payload/'run/attempts.json')]==['failure','not_attempted','not_attempted'],
            'stopped dependent-stage inventory differs')
        require(load(payload/'run/fit-17/failure-detail.json')['message']==
            'final fit below 8/8; stop dependent stages','fit stop reason changed')
        require(load(payload/'run/fit-17-worker.json')['observed_updates']==128 and
            load(payload/'run/fit-17-worker.json')['observed_target_tokens']==2432,'inner supervisor totals changed')
        require(not (payload/'run/fit-validation').exists() and
            not (payload/'run/fit-certificate.json').exists(),'unreached validation invented')
        for forbidden in ('targets.json','target-journal.jsonl','committed-predictions.json','scores.json'):
            require(not (payload/'run'/forbidden).exists(),'heldout artifact in stopped run')
        require(not (payload/'anchors/run-anchor.json.predictions.json').exists(),'heldout barrier invented')
        producer=temporary/'producer';producer.mkdir()
        unpack(payload/'frozen/source.zip',producer)
        require(files(producer)==freeze['sources'],'source closure regeneration mismatch')
        if args.native:
            binary=payload/'producer-artifacts/context-oracle.bin'
            require(digest(binary.read_bytes())==freeze['runtime']['oracle_sha256'],'oracle executable pin')
            target=producer/'external/context-symbol-relation-v1/target/release/trinite-context-symbol-relation-v1'
            target.parent.mkdir(parents=True)
            shutil.copyfile(binary,target);target.chmod(0o700)
        common=['--frozen',str(payload/'frozen'),'--freeze-sha',FREEZE,
            '--anchors',str(payload/'anchors/preparation.json'),'--anchors-sha',PREPARATION,
            '--run',str(payload/'run'),'--run-anchor',str(payload/'anchors/run-anchor.json'),'--run-sha',RUN]
        command=[sys.executable,'-I','-B',str(producer/'scripts/run_context_symbol_learning_v1.py'),'verify',*common]
        if not args.native:
            command+=['--scores-only']
        checked=subprocess.run(command,capture_output=True,text=True,check=True,timeout=120)
        result=json.loads(checked.stdout)
        require(result['verified'] and not result['complete'] and not result['scores_recomputed'],'stopped verification outcome changed')
        supplemental=None
        if args.native:
            audit=[sys.executable,'-I','-B',str(payload/'postrun-verification/audit_fit.py'),'--producer-root',str(producer),*common]
            # Audit API names its preparation anchor flags identically; all
            # args point at temporary copies of the externally anchored bytes.
            checked=subprocess.run(audit,capture_output=True,text=True,check=True,timeout=120)
            supplemental=json.loads(checked.stdout)
            require(supplemental==load(payload/'postrun-verification/fit-audit.json'),'post-run fit inference differs')
            require(supplemental['exact']==7 and supplemental['step']==128 and
                supplemental['optimizer_updates_in_this_audit']==0 and
                supplemental['replaces_failed_mandatory_gate'] is False,'fit audit scope changed')
        require(files(payload)==expected_files,'verification changed retained payload bytes')
    require(files(here)==before,'read-only replay changed deposit bytes')
    print(json.dumps(dict(verified=True,complete=False,producer_sha=PRODUCER,freeze_sha256=FREEZE,
        run_anchor_sha256=RUN,mode='native plus post-run fit inference' if args.native else 'portable stopped-custody/data/summary replay',
        fit_exact=None if supplemental is None else supplemental['exact'],fit_denominator=8,
        heldout_scored=False,optimizer_updates_in_this_verification=0,optimizer_replay=False,
        mandatory_lifecycle_gate='failed final 7/8; independent validation never reached',source_and_payload_unchanged=True),sort_keys=True))


if __name__=='__main__':
    main()
