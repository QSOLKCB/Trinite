"""Versioned larger bounds; frozen v1 limits remain unchanged."""
from dataclasses import dataclass
from ...contracts import ContractError, ModelConfig, integer
from ...budget_plan import BudgetPlan
from ...foundations_plan import FoundationsPlan


@dataclass(frozen=True)
class BudgetRetentionPlan(FoundationsPlan):
    schema: str = 'trinite.maths-retention-budget-plan.v2'
    steps: int = 8192
    learning_rate: str = '0.001'
    validation_every: int = 1024
    max_seconds: int = 960
    max_target_tokens: int = 1048576

    def __post_init__(self):
        if type(self.schema) is not str or self.schema != 'trinite.maths-retention-budget-plan.v2':
            raise ContractError('unsupported retention budget plan')
        for name, high in (('steps',8192), ('validation_every',8192),
                           ('max_seconds',960), ('max_target_tokens',1048576)):
            integer(getattr(self,name), name, 1, high)
        if self.steps*self.batch_size*2 > self.max_target_tokens:
            raise ContractError('retention budget cannot cover answer/EOS targets')
        if type(self.model) is not ModelConfig:
            raise ContractError('retention budget requires ModelConfig')
        common = self.to_dict()
        common.update(schema='trinite.budget-training-plan.v1',
                      steps=min(self.steps,4096), validation_every=min(self.validation_every,4096),
                      max_seconds=min(self.max_seconds,480),
                      max_target_tokens=min(self.max_target_tokens,524288))
        BudgetPlan.from_dict(common)
