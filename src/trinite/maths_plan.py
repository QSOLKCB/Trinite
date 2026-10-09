"""Separate maths pilot profile; numerical and resource bounds delegate unchanged."""
from dataclasses import dataclass
from .contracts import ContractError
from .budget_plan import BudgetPlan


@dataclass(frozen=True)
class MathsPlan(BudgetPlan):
    schema: str = 'trinite.maths-plan.v1'

    def __post_init__(self):
        if type(self.schema) is not str or self.schema != 'trinite.maths-plan.v1':
            raise ContractError('unsupported maths plan')
        common = self.to_dict(); common['schema'] = 'trinite.budget-training-plan.v1'
        BudgetPlan.from_dict(common)
