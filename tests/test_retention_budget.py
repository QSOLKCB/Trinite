from dataclasses import replace
from pathlib import Path
import ast
import sys
import unittest
from trinite.contracts import ContractError
from trinite.maths_plan import MathsPlan
from trinite.retention_v2.budget.plan import BudgetRetentionPlan

ROOT=Path(__file__).resolve().parents[1]

class BudgetBoundaries(unittest.TestCase):
    def test_new_resource_bounds_do_not_relax_v1(self):
        self.assertEqual(BudgetRetentionPlan().steps,8192)
        with self.assertRaises(ContractError):MathsPlan(steps=8192)
        for field,value in [('steps',8193),('steps',True),('max_seconds',961),('max_target_tokens',1048577),
                            ('validation_every',8193),('model',None),('batch_size',9),('geometry_loss',1)]:
            with self.assertRaises((ContractError,TypeError)):replace(BudgetRetentionPlan(),**{field:value})
        with self.assertRaises(ContractError):BudgetRetentionPlan(max_target_tokens=1)
        value=BudgetRetentionPlan().to_dict()
        self.assertEqual(BudgetRetentionPlan.from_dict(value),BudgetRetentionPlan())
        value['extra']=0
        with self.assertRaises(ContractError):BudgetRetentionPlan.from_dict(value)

    def test_nested_dependencies_no_new_kernel_network_or_backend(self):
        allowed={
          'archive':{(3,'contracts'),(2,'evidence'),(1,'experiment'),(1,'spec')},
          '__init__':set(), 'plan':{(3,'contracts'),(3,'budget_plan'),(3,'foundations_plan')},
          'spec':{(3,'contracts'),(3,'maths_learning'),(3,'maths_plan'),(2,'spec'),(2,'evidence'),(1,'plan')},
          'experiment':{(3,'contracts'),(3,'comparison_checkpoint'),(3,'foundations'),(3,'model'),(3,'training'),
                        (1,'spec'),(2,'evidence'),(2,'criteria'),(2,'experiment')},
        }
        for path in (ROOT/'src/trinite/retention_v2/budget').glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node,ast.ImportFrom) and node.level:
                    for name in ([node.module] if node.module else [a.name for a in node.names]):
                        self.assertIn((node.level,name),allowed[path.stem])
                elif isinstance(node,(ast.Import,ast.ImportFrom)):
                    names=[a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else [node.module.split('.')[0]]
                    for name in names:
                        self.assertIn(name,sys.stdlib_module_names)
                        self.assertNotIn(name,{'socket','urllib','http','ssl','subprocess'})

    def test_original_assurance_source_bytes_remain_intact(self):
        from trinite.contracts import parse_json,identity
        expected=parse_json((ROOT/'fixtures/retention-budget-v2/prior-assurance-source.json').read_bytes(),canonical=True)
        self.assertTrue(expected)
        for name,key in expected.items():self.assertEqual(identity((ROOT/name).read_bytes()),key,name)
        self.assertTrue(any('retention_v2/spec.py' in name for name in expected))
