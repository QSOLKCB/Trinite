"""Separate bounded exposure profile; native numerical operations are unchanged."""
from dataclasses import dataclass

from .budget_plan import BudgetPlan
from .contracts import ContractError


@dataclass(frozen=True)
class FoldExposurePlan(BudgetPlan):
    schema: str = 'trinite.fold-exposure-plan.v1'
    order: str = 'fold-sum-multiplicity-cycle.v1'
    sum_repeats: int = 1

    def __post_init__(self):
        if (self.schema != 'trinite.fold-exposure-plan.v1'
                or self.order != 'fold-sum-multiplicity-cycle.v1'
                or type(self.sum_repeats) is not int or self.sum_repeats not in (1, 2, 4)):
            raise ContractError('unsupported fold exposure profile')
        common = self.to_dict(); common.pop('sum_repeats')
        common.update(schema='trinite.budget-training-plan.v1', order='seeded-hash-rank-cycle.v1')
        BudgetPlan.from_dict(common)
