from collections import Counter
import ast
import copy
from pathlib import Path
import sys
import unittest

from trinite.contracts import ContractError, identity, parse_json
from trinite.retention_v2.task_balance.schedule import MATHS_TASKS, schedule_rows, task_visits
from retention_source_history import assert_preserved_sources

ROOT = Path(__file__).resolve().parents[1]


class TaskBalanceBoundaries(unittest.TestCase):
    def rows(self):
        legacy = [{'identity': 'a' + str(i)} for i in range(3)]
        maths = [{'identity': task + ':' + str(i)} for size, task in enumerate(MATHS_TASKS, 1)
                 for i in range(size)]
        mapping = {r['identity']: r['identity'].split(':')[0] for r in maths}
        return legacy, maths, mapping

    def test_control_preserves_row_cycle_and_candidate_arithmetic_slots(self):
        legacy, maths, mapping = self.rows()
        before = copy.deepcopy((legacy, maths, mapping))
        control = schedule_rows(legacy, maths, mapping, 8, 'control')
        candidate = schedule_rows(legacy, maths, mapping, 8, 'candidate')
        self.assertEqual(len(control), 64)
        self.assertEqual([r for i,r in enumerate(control) if i%8<4], (legacy*11)[:32])
        self.assertEqual([r for i,r in enumerate(control) if i%8>=4], maths[:32])
        self.assertEqual([r for i,r in enumerate(control) if i%8<4],
                         [r for i,r in enumerate(candidate) if i%8<4])
        self.assertEqual((legacy, maths, mapping), before)

    def test_exact_task_counts_and_within_task_order_at_all_budget_boundaries(self):
        legacy, maths, mapping = self.rows()
        for steps in (1, 2, 3, 8, 8192):
            rows = schedule_rows(legacy, maths, mapping, steps, 'candidate')
            selected = [r for r in rows if r['identity'] in mapping]
            counts = Counter(mapping[r['identity']] for r in selected)
            self.assertEqual({task: counts[task] for task in MATHS_TASKS}, task_visits(steps))
            self.assertEqual(sum(counts.values()), steps*4)
            for task in MATHS_TASKS:
                group = [r for r in maths if mapping[r['identity']]==task]
                visits = [r for r in selected if mapping[r['identity']]==task]
                self.assertEqual(visits, (group*(len(visits)//len(group)+1))[:len(visits)])
        self.assertEqual(list(task_visits(8192).values()), [2731]*8+[2730]*4)

    def test_invalid_bounds_task_maps_and_duplicate_training_rows_reject(self):
        legacy, maths, mapping = self.rows()
        for bad in (0, -1, 8193, True, 1.0, '8'):
            with self.assertRaises(ContractError): schedule_rows(legacy, maths, mapping, bad, 'candidate')
        for bad in ('pooled', True, None):
            with self.assertRaises(ContractError): schedule_rows(legacy, maths, mapping, 8, bad)
        for left, right, labels in (([],maths,mapping),(legacy,[],{}),
                                   (legacy+legacy,maths,mapping),(legacy,maths+maths,mapping),
                                   (legacy,maths,{}),(legacy,maths,{**mapping,'foreign':'choose'})):
            with self.assertRaises(ContractError): schedule_rows(left,right,labels,8,'candidate')

    def test_frozen_protocol_and_prior_budget_producer_are_preserved(self):
        raw=(ROOT/'src/trinite/retention_v2/task_balance/protocol.json').read_bytes()
        protocol=parse_json(raw,canonical=True)
        self.assertEqual(protocol['research_updates'],{'control':8192,'candidate':8192})
        self.assertEqual(protocol['smoke_updates'],{'control':8,'candidate':8})
        self.assertEqual(protocol['maths_task_order'],list(MATHS_TASKS))
        source=(ROOT/'src/trinite/retention_v2/task_balance/spec.py').read_text()
        self.assertIn(repr(identity(raw)), source)
        prior=parse_json((ROOT/'fixtures/retention-task-balance-v1/prior-budget-source.json').read_bytes(),canonical=True)
        self.assertIn('src/trinite/retention_v2/budget/spec.py',prior)
        # This is historical documentation preservation, not a claim that an
        # old request matches current AGENTS bytes. Runtime binding stays strict.
        assert_preserved_sources(self, ROOT, prior)

    def test_dependency_boundary_reuses_existing_numerical_and_evidence_owners(self):
        allowed={
            '__init__':set(), 'schedule':{(3,'contracts')},
            'spec':{(3,'contracts'),(2,'budget'),(2,'budget.plan'),(2,'spec'),(2,'evidence'),(1,'schedule')},
            'archive':{(3,'contracts'),(2,'evidence'),(1,'experiment'),(1,'spec')},
            'experiment':{(3,'contracts'),(3,'comparison_checkpoint'),(3,'foundations'),(3,'model'),
                          (3,'training'),(1,'spec'),(2,'evidence'),(2,'criteria'),(2,'experiment')},
        }
        for path in (ROOT/'src/trinite/retention_v2/task_balance').glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node,ast.ImportFrom) and node.level:
                    for name in ([node.module] if node.module else [a.name for a in node.names]):
                        self.assertIn((node.level,name),allowed[path.stem])
                elif isinstance(node,(ast.Import,ast.ImportFrom)):
                    names=[a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else [node.module.split('.')[0]]
                    for name in names:
                        self.assertIn(name,sys.stdlib_module_names)
                        self.assertNotIn(name,{'socket','urllib','http','ssl','subprocess'})


if __name__ == '__main__': unittest.main()
