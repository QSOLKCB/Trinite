"""Separate bounded budget; original native RunConfig limits stay unchanged."""
from dataclasses import dataclass, field
import math

from .contracts import ContractError, ModelConfig, exact_keys, integer, identity, json_bytes


@dataclass(frozen=True)
class FoundationsPlan:
    schema: str = 'trinite.foundations-training-plan.v1'
    model: ModelConfig = field(default_factory=lambda:ModelConfig(context_length=64,blocks=2,
        width=64,heads=4,feed_forward_width=128))
    seed: int = 0
    steps: int = 1024
    batch_size: int = 8
    learning_rate: str = '0.003'
    beta1: str = '0.9'
    beta2: str = '0.999'
    optimizer_epsilon: str = '0.00000001'
    weight_decay: str = '0.01'
    gradient_clip: str = '1'
    accumulation: int = 1
    schedule: str = 'constant'
    order: str = 'seeded-hash-rank-cycle.v1'
    loss: str = 'answer-and-eos-target-mean.v1'
    validation_every: int = 256
    selection: str = 'final-step; no early stopping; no test evaluation'
    max_seconds: int = 120
    max_target_tokens: int = 131072

    def __post_init__(self):
        if not isinstance(self.model,ModelConfig) or self.model.parameter_count>100000:
            raise ContractError('foundations model exceeds 100,000 parameters')
        for name,expected in {'schema':'trinite.foundations-training-plan.v1','accumulation':1,
            'schedule':'constant','order':'seeded-hash-rank-cycle.v1','loss':'answer-and-eos-target-mean.v1',
            'selection':'final-step; no early stopping; no test evaluation'}.items():
            if type(getattr(self,name)) is not type(expected) or getattr(self,name)!=expected:
                raise ContractError('unsupported foundations '+name)
        for name,low,high in (('seed',0,2**32-1),('steps',1,1024),('batch_size',1,8),
            ('validation_every',1,1024),('max_seconds',1,120),('max_target_tokens',1,131072)):
            integer(getattr(self,name),name,low,high)
        if self.steps*self.batch_size*2>self.max_target_tokens:
            raise ContractError('foundations budget cannot cover minimum answer/EOS targets')
        for name,low,high in (('learning_rate',1e-6,.1),('beta1',0,.9999),('beta2',0,.999999),
            ('optimizer_epsilon',1e-12,.01),('weight_decay',0,1),('gradient_clip',1e-6,100)):
            value=getattr(self,name)
            if type(value) is not str:raise ContractError('numeric string required')
            try:number=float(value)
            except ValueError as error:raise ContractError('invalid plan number') from error
            if not math.isfinite(number) or not low<=number<=high:raise ContractError('plan number outside bounds')

    def to_dict(self):
        return {name:self.model.to_dict() if name=='model' else getattr(self,name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls,value):
        fields=dict(exact_keys(value,set(cls.__dataclass_fields__),'foundations plan'))
        fields['model']=ModelConfig.from_dict(fields['model'])
        return cls(**fields)

    @property
    def content_identity(self):return identity(json_bytes(self.to_dict()))
