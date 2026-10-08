"""Explicit larger development budget; prior plan limits remain unchanged."""
from dataclasses import dataclass

from .contracts import ContractError, integer
from .foundations_plan import FoundationsPlan


@dataclass(frozen=True)
class BudgetPlan(FoundationsPlan):
    schema: str = 'trinite.budget-training-plan.v1'
    steps: int = 4096
    learning_rate: str = '0.001'
    validation_every: int = 1024
    max_seconds: int = 480
    max_target_tokens: int = 524288

    def __post_init__(self):
        if self.schema != 'trinite.budget-training-plan.v1' or type(self.schema) is not str:
            raise ContractError('unsupported budget schema')
        for name, high in (('steps',4096), ('validation_every',4096),
                           ('max_seconds',480), ('max_target_tokens',524288)):
            integer(getattr(self,name), name, 1, high)
        if self.steps*self.batch_size*2 > self.max_target_tokens:
            raise ContractError('budget cannot cover minimum answer/EOS targets')
        # Validate all unchanged semantics through the authoritative small plan.
        # Only the four explicitly validated resource fields have larger bounds.
        common = self.to_dict()
        common.update(schema='trinite.foundations-training-plan.v1',
                      steps=min(self.steps,1024), validation_every=min(self.validation_every,1024),
                      max_seconds=min(self.max_seconds,120),
                      max_target_tokens=min(self.max_target_tokens,131072))
        FoundationsPlan.from_dict(common)
