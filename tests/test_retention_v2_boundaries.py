import ast
from pathlib import Path
import sys
import unittest
import hashlib
from trinite.contracts import ContractError, parse_json
from trinite.retention_v2.criteria import criterion_failures, training_failures

ROOT=Path(__file__).resolve().parents[1]
ALLOWED={
 '__init__':set(), 'criteria':{(2,'contracts')},
 'spec':{(2,'contracts'),(2,'maths_learning'),(2,'maths_retention'),(2,'maths_plan'),(1,'evidence')},
 'evidence':{(2,'contracts'),(2,'observation')},
 'experiment':{(2,'contracts'),(2,'comparison_checkpoint'),(2,'foundations'),(2,'model'),(2,'training'),
               (1,'spec'),(1,'evidence'),(1,'criteria')},
}


class V2Boundaries(unittest.TestCase):
    def test_package_dependencies_are_explicit_and_no_network_or_numerical_kernel_is_added(self):
        for path in (ROOT/'src/trinite/retention_v2').glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node,ast.ImportFrom):
                    if node.level:
                        for name in ([node.module] if node.module else [a.name for a in node.names]):
                            self.assertIn((node.level,name),ALLOWED[path.stem],str(path))
                        continue
                    names=[node.module.split('.')[0]]
                elif isinstance(node,ast.Import):names=[a.name.split('.')[0] for a in node.names]
                else:continue
                for name in names:
                    self.assertIn(name,sys.stdlib_module_names)
                    self.assertNotIn(name,{'socket','urllib','http','ssl','subprocess'})

    def test_v1_runtime_source_closure_keeps_its_published_bytes_and_names(self):
        receipt=parse_json((ROOT/'fixtures/maths-retention-v1/request.json').read_bytes())['source']
        package=ROOT/'src/trinite'
        paths=[*package.glob('*.py'),*package.glob('*.json'),package/'cpu-dependencies.lock',
               ROOT/'requirements.lock',ROOT/'scripts/run_maths.py',ROOT/'scripts/run_maths_retention.py']
        current={str(p.relative_to(ROOT)):'sha256:'+hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        self.assertEqual(current,receipt)
        self.assertNotIn('AGENTS.md',current)

    def test_exact_threshold_and_admitted_task_counts(self):
        counts={'add':66,'divide':66,'multiply':66,'subtract':66}
        tasks={k:{'examples':66,'correct':60,'baseline_correct':14} for k in counts}
        self.assertEqual(criterion_failures(tasks,counts),[])
        tasks['divide']['correct']=59
        self.assertEqual(criterion_failures(tasks,counts),['divide'])
        for name in ('add','unknown'):
            changed=dict(tasks)
            if name=='add':del changed[name]
            else:changed[name]=tasks['add']
            with self.assertRaises(ContractError):criterion_failures(changed,counts)
        with self.assertRaises(ContractError):criterion_failures(tasks,counts,numerator=True)
        with self.assertRaises(ContractError):criterion_failures(tasks,counts,denominator=0)
        for field,bad in [('examples',0),('examples',True),('examples',65),('correct',-1),('correct',67),('correct',True),('baseline_correct',67)]:
            changed={k:dict(v) for k,v in tasks.items()};changed['add'][field]=bad
            with self.assertRaises(ContractError):criterion_failures(changed,counts)
        with self.assertRaises(ContractError):training_failures({'examples':265,'correct':239,'tasks':tasks},counts)

    def test_all_published_gate_results_still_agree_with_strict_criteria(self):
        from trinite.maths_data import dataset as maths
        from trinite.learning_data import dataset as legacy
        from collections import Counter
        expected={k:dict(Counter(e['task'] for e in parse_json(fn()[0])['examples'] if e['split']=='train'))
                  for k,fn in [('maths',maths),('legacy',lambda:legacy('arithmetic'))]}
        summary=parse_json((ROOT/'fixtures/maths-retention-v1/summary.json').read_bytes())
        for cell in summary['cells']:
            for kind in ('maths','legacy'):
                report=cell['report']
                self.assertEqual(training_failures(report['scores'][kind+'-final-train'],expected[kind]),report['training_gate_failures'][kind])
