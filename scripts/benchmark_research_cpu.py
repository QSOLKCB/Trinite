"""Three alternating complete-suite pairs; exact executed case identity required."""
import argparse
import importlib.util
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from check_research_cpu import RESEARCH_SUITES, load_suites
from trinite.contracts import ContractError, identity, json_bytes


def ids(suite):
    import unittest
    for item in suite:
        if isinstance(item, unittest.TestSuite): yield from ids(item)
        else: yield item.id()


def commands(selector):
    if selector == 'grouped': return [[sys.executable, 'scripts/check_research_cpu.py']]
    if selector == 'standalone':
        return [[sys.executable, 'scripts/check_cpu.py'], *[
            [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests/'+name, '-v'] for name in RESEARCH_SUITES]]
    raise ValueError('unknown fixed benchmark selector')


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    from trinite.training import require_environment
    args.output.mkdir(parents=True,exist_ok=False)
    expected = list(ids(load_suites()))
    paths = [ROOT/'requirements.lock', ROOT/'scripts/check_cpu.py', ROOT/'scripts/check_research_cpu.py',
             ROOT/'scripts/benchmark_research_cpu.py', *sorted((ROOT/'tests').rglob('*.py')),
             *sorted((ROOT/'src').rglob('*.py')), *sorted((ROOT/'src/trinite').glob('*.json'))]
    def sources(): return {p.relative_to(ROOT).as_posix():identity(p.read_bytes()) for p in paths}
    initial = sources(); samples = {'standalone':[], 'grouped':[]}
    env = dict(os.environ,PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for trial in range(3):
        for selector in (('standalone','grouped') if trial % 2 == 0 else ('grouped','standalone')):
            log = args.output/f'{trial}-{selector}.log'; started = time.perf_counter()
            with log.open('wb') as stream:
                for command in commands(selector):
                    # Closed suite selectors and fixed argv; no external executable or shell.
                    # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit, dangerous-subprocess-use-audit
                    result = subprocess.run(command,shell=False,cwd=ROOT,env=env,
                        stdout=stream,stderr=subprocess.STDOUT,timeout=600)
                    if result.returncode: raise ContractError('research benchmark test process failed: '+selector)
            seconds = time.perf_counter()-started
            text = log.read_text(); actual = re.findall(r'^test\S+ \(([^\n]+)\) \.\.\. ok$',text,re.M)
            if actual != expected: raise ContractError('benchmark executed test identity/order/outcome differs')
            if sources() != initial: raise ContractError('benchmark source changed')
            samples[selector].append(seconds.hex()); print(trial,selector,seconds,flush=True)
    medians = {name:statistics.median(float.fromhex(v) for v in values) for name,values in samples.items()}
    record = {'schema':'trinite.research-cpu-characterization.v1','environment':require_environment(),
        'source':initial,'test_ids':expected,'required_cases':len(expected),'samples_seconds_hex':samples,
        'medians_seconds_hex':{k:v.hex() for k,v in medians.items()},
        'grouped_over_standalone_hex':(medians['grouped']/medians['standalone']).hex(),
        'grouped_lower_median':medians['grouped']<medians['standalone'],
        'scope':'Same-host three alternating paired complete required CPU research suites; all executed IDs/outcomes checked; excludes acquisition and inspection commands; no hosted or universal speedup claim'}
    (args.output/'measurement.json').write_bytes(json_bytes(record))
    return 0


if __name__ == '__main__': raise SystemExit(main())
