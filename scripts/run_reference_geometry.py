"""Fixed fresh CPU workers; explicit freeze, run, read-only verify and replay."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def worker(directory, condition):
    if condition == 'native':
        # Preflight source bytes BEFORE importing the original numerical producer.
        from trinite.reference_inputs import validate
        _,locations = validate(directory)
        for name in tuple(sys.modules):
            if name == 'trinite' or name.startswith('trinite.'): del sys.modules[name]
        sys.path.insert(0,str(Path(locations['producer'])/'src'))
        import trinite
        trinite.__path__.append(str(ROOT/'src/trinite'))
    from trinite.reference_capture import capture
    from trinite.contracts import json_bytes
    sys.stdout.buffer.write(json_bytes(capture(directory,condition)))


def launch(directory, condition, logname):
    environment = dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                       HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    environment['PYTHONPATH'] = str(ROOT/'src')
    with (directory/condition/logname).open('xb') as log:
        result = subprocess.run([sys.executable,str(Path(__file__).resolve()),str(directory),'--worker',condition],
                 env=environment,stdout=subprocess.PIPE,stderr=log,timeout=1200,check=False)
    if result.returncode: raise RuntimeError('worker failed; retained '+condition+'/'+logname)
    if len(result.stdout) > 32*1024*1024: raise RuntimeError('worker output byte budget')
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--worker',choices=('native','stock','modelfile'),help=argparse.SUPPRESS)
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--model',type=Path); parser.add_argument('--producer',type=Path)
    parser.add_argument('--native-run',type=Path)
    parser.add_argument('--verify-only',action='store_true'); parser.add_argument('--replay',action='store_true')
    args = parser.parse_args(); directory = args.directory.resolve()
    if args.worker: worker(directory,args.worker); return
    from trinite.reference_inputs import freeze, validate, CONDITIONS, read
    from trinite.reference_evidence import publish, verify, audit
    from trinite.contracts import identity, json_bytes
    import reference_math_oracle as oracle
    if args.freeze:
        if not all((args.model,args.producer,args.native_run)): parser.error('freeze requires all local input locations')
        print(freeze(directory,args.model,args.producer,args.native_run)); return
    if args.verify_only or args.replay:
        result = verify(directory,oracle)
        if args.replay:
            request,_ = validate(directory)
            for c in CONDITIONS:
                # Disposable log outside the retained directory; no publication mutation.
                import tempfile
                with tempfile.TemporaryDirectory() as temp:
                    p = Path(temp); (p/c).mkdir()
                    # launch needs request inputs at the authoritative directory; log path is separate.
                    environment = dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
                    run = subprocess.run([sys.executable,str(Path(__file__).resolve()),str(directory),'--worker',c],
                        env=environment,capture_output=True,timeout=1200)
                    if run.returncode or run.stdout != read(directory/c/'capture.json'):
                        raise RuntimeError('fresh replay differs: '+c+' '+run.stderr.decode(errors='replace')[-1000:])
            result['fresh_replay'] = 'executed; three conditions exact match'
        print(json.dumps(result,sort_keys=True)); return
    request,_ = validate(directory)
    for c in CONDITIONS:
        (directory/c).mkdir()
        raw = launch(directory,c,'worker.log'); audit(raw,request,c)
        (directory/c/'capture.json').write_bytes(raw)
        repeated = launch(directory,c,'replay.log')
        if repeated != raw: raise RuntimeError('fresh process capture differs '+c)
        (directory/c/'replay.json').write_bytes(json_bytes({'capture_identity':identity(raw),'fresh_process_exact_match':True}))
        print(c,'captured and replayed',identity(raw),flush=True)
    publish(directory,oracle)
    print(json.dumps(verify(directory,oracle),sort_keys=True))


if __name__ == '__main__': main()
