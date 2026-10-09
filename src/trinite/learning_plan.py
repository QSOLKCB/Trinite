"""Matched larger budget with explicit parent-content ordering semantics."""
from dataclasses import dataclass

from .contracts import ContractError
from .budget_plan import BudgetPlan


@dataclass(frozen=True)
class LearningPlan(BudgetPlan):
    schema: str = 'trinite.learning-training-plan.v1'
    order: str = 'foundations-content-order.v1'

    def __post_init__(self):
        if self.schema != 'trinite.learning-training-plan.v1' or type(self.schema) is not str:
            raise ContractError('unsupported learning plan schema')
        if self.order != 'foundations-content-order.v1' or type(self.order) is not str:
            raise ContractError('unsupported learning exposure order')
        # Resource/numeric semantics are unchanged; only the explicit record and
        # ordering-key profiles differ. Delegate every shared bound to BudgetPlan.
        common = self.to_dict()
        common.update(schema='trinite.budget-training-plan.v1', order='seeded-hash-rank-cycle.v1')
        BudgetPlan.from_dict(common)
