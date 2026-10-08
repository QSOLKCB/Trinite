"""Paired, bounded characterization against exact frozen pre-detour sources.

No network or cached verification. Supply the two source files from the declared
baseline checkout; their Git blob identities are checked before local execution.
"""
import argparse
import hashlib
from pathlib import Path
import platform
import statistics
import tempfile
import time
import types

from trinite.contracts import identity, json_bytes, parse_json
from trinite.observation import BundleObserver

BASELINE_REVISION = 'bf4ff38ef2f0e08259b29a49806b4039d837d08f'
BASELINE_BLOBS = {'observation.py': '79962716a3b4f562e191390ada4cecb8cdf9b4e3',
                  'geometry_run.py': 'ae3755f518305efac199f529e4e826cfd672ab92'}
BASELINE_IDENTITIES = {
    'observation.py': 'sha256:f04bb843bcfba66ce0160dd1211a17d468d829b590f9bab7b1692cfad59b82ad',
    'geometry_run.py': 'sha256:8b7ef8f5889e6eaaa1538e00096b642225e05ebca98140fb4afa60b3cfb13649'}


def reference(directory, name):
    raw=(directory/name).read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if blob!=BASELINE_BLOBS[name] or identity(raw)!=BASELINE_IDENTITIES[name]:
        raise ValueError('reference source differs from frozen baseline: '+name)
    import trinite.observation
    module=types.ModuleType('trinite._benchmark_reference_'+name[:-3])
    module.__package__='trinite'
    # Resolve frozen dependency receipts from the installed target package.
    module.__file__=str(Path(trinite.observation.__file__).with_name(name))
    exec(compile(raw,module.__file__,'exec'),module.__dict__)
    return module


def tree_bytes(root):
    return {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}


def series(samples):
    a,b=samples['reference'],samples['optimized']
    return {'reference_seconds_hex':[x.hex() for x in a], 'optimized_seconds_hex':[x.hex() for x in b],
            'reference_median_seconds_hex':statistics.median(a).hex(),
            'optimized_median_seconds_hex':statistics.median(b).hex(),
            'median_speedup_hex':(statistics.median(a)/statistics.median(b)).hex()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-sources',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--geometry',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise ValueError('benchmark output already exists')
    ref_observer=reference(args.baseline_sources,'observation.py').BundleObserver
    optimized={n:identity((Path(__import__('trinite.observation',fromlist=['x']).__file__).parent/n).read_bytes())
               for n in BASELINE_BLOBS}
    result={'schema':'trinite.optimization-benchmark.v1','baseline_revision':BASELINE_REVISION,
            'baseline_blobs':BASELINE_BLOBS,'optimized_source':optimized,
            'python':platform.python_version(),'platform':platform.platform(),
            'processor':platform.processor(),'warmup_pairs':1,'measured_pairs':7,
            'order':'reference/optimized alternates each pair; includes construction, collection and actual verification',
            'timing_scope':'same process/host; temporary filesystem; no acquisition time; no performance threshold',
            'benchmarks':{}}
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        cases={'small-stage':(16384,32768,65536,131072),
               'payload-heavy-stage':(16384,512*1024,2*1024*1024,8*1024*1024)}
        for case,sizes in cases.items():
            inputs={str(i):bytes([i])*n for i,n in enumerate(sizes)}
            samples={'reference':[],'optimized':[]}
            for repeat in range(8):
                snapshots={};reports={}
                for label in (('reference','optimized') if repeat%2==0 else ('optimized','reference')):
                    path=root/(case+'-'+str(repeat)+'-'+label)
                    factory=ref_observer if label=='reference' else BundleObserver
                    start=time.perf_counter();observer=factory(path)
                    for step in range(3):observer.record('stage-'+str(step),inputs=inputs,outputs={str(step)+'-result':bytes([step])+b'result'})
                    reports[label]=observer.finalize();elapsed=time.perf_counter()-start
                    if repeat:samples[label].append(elapsed)
                    snapshots[label]=tree_bytes(path)
                if reports['reference']!=reports['optimized'] or snapshots['reference']!=snapshots['optimized']:
                    raise AssertionError('collector bundle/report parity failed')
            result['benchmarks'][case]={**series(samples),'unique_input_bytes':sum(sizes),'stages':3,
                                         'correctness':'all artifact/record/event/index/manifest bytes and full verifier reports identical'}
        if args.geometry:
            import torch
            from unittest.mock import patch
            from trinite.experiment import train_run
            from trinite.training import RunConfig,environment
            import trinite.geometry_run as run
            torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
            repo=Path(__file__).resolve().parents[1]
            for lane in ('dense','ternary'):
                report=train_run(root/lane,repo/'fixtures/formal-v1',RunConfig(),lane=lane,observer=False)
                if report['compute_outcome']!='completed':raise AssertionError(report)
            options=dict(dataset=repo/'fixtures/formal-v1',training_config=repo/'configs/tiny-training.json',
                         dense_checkpoint=root/'dense/checkpoint',ternary_checkpoint=root/'ternary/checkpoint')
            frozen=run.freeze_observation(root/'frozen',**options)
            ref_summary=reference(args.baseline_sources,'geometry_run.py').summarize
            optimized_summary=run.summarize;samples={'reference':[],'optimized':[]}
            for repeat in range(8):
                snapshots={};verification={}
                for label in (('reference','optimized') if repeat%2==0 else ('optimized','reference')):
                    path=root/('geometry-'+str(repeat)+'-'+label)
                    factory=ref_observer if label=='reference' else BundleObserver
                    with patch.object(run,'summarize',ref_summary if label=='reference' else optimized_summary):
                        start=time.perf_counter()
                        report=run.measure_observation(path,request=frozen['request_file'],request_identity=frozen['request_identity'],
                                                       observer_factory=factory,**options)
                        elapsed=time.perf_counter()-start
                    if not report['observer_evidence_eligible']:raise AssertionError(report)
                    if repeat:samples[label].append(elapsed)
                    snapshots[label]={n:b for n,b in tree_bytes(path).items() if n!='report.json'}
                    verification[label]=report['verification']
                if snapshots['reference']!=snapshots['optimized'] or verification['reference']!=verification['optimized']:
                    raise AssertionError('full observation/control/bundle parity failed')
            result['benchmarks']['complete-geometry-observation']={**series(samples),
                'correctness':'same immutable request/checkpoints; all computational and bundle bytes/full verifier reports identical; timing/RSS report excluded'}
            result['environment']=environment()
    # Write only after every parity check and trial succeeds.
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:stream.write(json_bytes(result))
    print(json_bytes(result).decode())


if __name__=='__main__':main()
