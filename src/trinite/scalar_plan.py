"""Separate scalar exposure profile; shared budget/numeric bounds are unchanged."""
from dataclasses import dataclass

from .contracts import ContractError
from .budget_plan import BudgetPlan


@dataclass(frozen=True)
class ScalarPlan(BudgetPlan):
    schema: str = 'trinite.scalar-learning-training-plan.v1'
    order: str = 'scalar-parent-projection-cycle.v1'

    def __post_init__(self):
        if (type(self.schema) is not str or self.schema != 'trinite.scalar-learning-training-plan.v1'
                or type(self.order) is not str or self.order != 'scalar-parent-projection-cycle.v1'):
            raise ContractError('unsupported scalar plan/exposure profile')
        common = self.to_dict()
        common.update(schema='trinite.budget-training-plan.v1', order='seeded-hash-rank-cycle.v1')
        BudgetPlan.from_dict(common)
