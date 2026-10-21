#!/usr/bin/env python3
"""Run with python -I -S: bind captured project source bytes, not bytecode caches."""
import sys
if not sys.flags.isolated or not sys.flags.no_site:
    raise SystemExit('require python -I -S scripts/check_cyclic_conformance.py')
import argparse
import importlib.abc
import importlib.util
import pathlib
import hashlib
import json

ROOT=pathlib.Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'scripts/cyclic_conformance_v1'
CAPTURE={p.stem:p.read_bytes() for p in PACKAGE.glob('*.py')}
class SourceLoader(importlib.abc.MetaPathFinder,importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname=='cyclic_conformance_v1':
            return importlib.util.spec_from_loader(fullname,self,is_package=True)
        if fullname.startswith('cyclic_conformance_v1.') and fullname.rsplit('.',1)[-1] in CAPTURE:
            return importlib.util.spec_from_loader(fullname,self)
        return None
    def create_module(self,spec):
        return None
    def exec_module(self,module):
        name='__init__' if module.__name__=='cyclic_conformance_v1' else module.__name__.rsplit('.',1)[-1]
        module.__file__=str(PACKAGE/(name+'.py'))
        exec(compile(CAPTURE[name],module.__file__,'exec'),module.__dict__)
sys.meta_path.insert(0,SourceLoader())
from cyclic_conformance_v1 import spec,runner

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','check'))
    parser.add_argument('--request',type=pathlib.Path,required=True)
    parser.add_argument('--request-identity')
    parser.add_argument('--output-dir',type=pathlib.Path)
    args=parser.parse_args()
    try:
        req=spec.request()
        for name,raw in CAPTURE.items():
            path='scripts/cyclic_conformance_v1/'+name+'.py'
            if req['sources'][path]!=hashlib.sha256(raw).hexdigest():
                raise ValueError('captured module/source binding mismatch')
        raw=spec.canonical(req)
        if args.action=='freeze':
            runner.publish(args.request.resolve(),raw)
            print(spec.identity(raw));return 0
        if not args.output_dir or not args.request_identity:
            raise ValueError('check requires output directory and independent request identity')
        if args.request.stat().st_size > spec.LIMIT:
            raise ValueError("request exceeds size bound")
        supplied=args.request.read_bytes()
        if spec.identity(supplied)!=args.request_identity or supplied!=raw:
            raise ValueError('request identity/content mismatch')
        result=runner.execute(req,args.output_dir.resolve())
        print(json.dumps({'status':result['status'],'result':str(args.output_dir/'result.json')}))
        return 0 if result['status']=='passed' else 1
    except (OSError,ValueError) as error:
        print(str(error),file=sys.stderr);return 1
if __name__=='__main__':
    raise SystemExit(main())
