"""Fixed fold-order profiles within the existing CPU budget."""
from dataclasses import dataclass

from .budget_plan import BudgetPlan
from .contracts import ContractError


@dataclass(frozen=True)
class FoldOrderPlan(BudgetPlan):
    schema: str = 'trinite.fold-order-plan.v1'
    order: str = 'fold-order-cycle.v1'
    ordering: str = 'parent'
    steps: int = 3450
    validation_every: int = 1150

    def __post_init__(self):
        if (self.schema != 'trinite.fold-order-plan.v1' or self.order != 'fold-order-cycle.v1'
                or type(self.ordering) is not str or self.ordering not in ('parent', 'mixed')):
            raise ContractError('unsupported fold order profile')
        common = self.to_dict(); common.pop('ordering')
        common.update(schema='trinite.budget-training-plan.v1', order='seeded-hash-rank-cycle.v1')
        BudgetPlan.from_dict(common)
