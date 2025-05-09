import dataclasses
import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.engine.optimizer.optimizer import Optimizer


class LearningRateStageType:
    constant = "constant"
    linear = "linear"
    power = "power"
    cosine = "cosine"


@config_class()
class LearningRateScheduleConfig(Config):
    base: float = Field(default=0.0001, desc="Base learning rate for the optimizer.", hint=FieldHint.core)
    decay_style: str = Field(default="constant", desc="The learning rate decay formula.", hint=FieldHint.feature)
    decay_iterations: int | None = Field(
        default=None, desc="Duration of the learning rate decay, in iterations.", hint=FieldHint.feature
    )
    decay_power: float = Field(
        default=1.0, desc="Exponent for learning rate decay, applied on the decay step..", hint=FieldHint.feature
    )
    warmup_iterations: int = Field(
        default=0, desc="Number of iteration for the learning rate warmup.", hint=FieldHint.feature
    )
    minimum: float = Field(default=0.0, desc="Learning rate at the end of decay.", hint=FieldHint.feature)
    schedule: str | None = Field(
        default=None,
        desc="Complex learning rate schedule encoded in a string (untested, replaces the other arguments.",
        hint=FieldHint.wip,
    )


@config_class()
class GradientScalerConfig(Config):
    constant: float | None = Field(
        default=None,
        desc="Constant multiplier applied to the loss. Setting this disables dynamic scaling.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.geq, 0)),
    )
    initial: float = Field(
        default=2**16,
        desc="Initial loss scale for dynamic scaling (fp16).",
        hint=FieldHint.feature,
        valid=check_field(Assert.gt, 0),
    )
    minimum: float = Field(
        default=1.0,
        desc="Minimum loss scale for dynamic scaling (fp16).",
        hint=FieldHint.feature,
        valid=check_field(Assert.gt, 0),
    )
    window: int = Field(
        default=1000,
        desc="Interval between dynamic scaling growth (fp16).",
        hint=FieldHint.feature,
        valid=check_field(Assert.gt, 0),
    )
    hysteresis: int = Field(
        default=2,
        desc="Number of failed updates to tolerate before lowering the learning rate in dynamic scaling (fp16).",
        hint=FieldHint.feature,
        valid=check_field(Assert.gt, 0),
    )


@config_class()
class OptimizerConfig(Config):

    learning_rate: LearningRateScheduleConfig = Field(
        desc="A schedule for the learning rate.",
        hint=FieldHint.core,
    )
    gradient_scaler: GradientScalerConfig = Field(
        desc="Configuration for the fixed or dynamic gradient scaling.",
        hint=FieldHint.feature,
    )
    weight_decay: float = Field(
        default=0.01,
        desc="Weight decay (Adamw).",
        hint=FieldHint.core,
        valid=check_field(Assert.geq, 0),
    )
    beta_1: float = Field(
        default=0.9,
        desc="First Adam momentum.",
        hint=FieldHint.optional,
        valid=check_field(Assert.in_range_incl, 0, 1),
    )
    beta_2: float = Field(
        default=0.999,
        desc="Second Adam Momentum.",
        hint=FieldHint.optional,
        valid=check_field(Assert.in_range_incl, 0, 1),
    )
    epsilon: float = Field(
        default=1e-8, desc="Regularizer for Adam.", hint=FieldHint.optional, valid=check_field(Assert.gt, 0)
    )
    gradient_norm_clipping: float = Field(
        default=1.0,
        desc="Clip the gradient norm to this value.",
        hint=FieldHint.feature,
        valid=check_field(Assert.gt, 0),
    )
    default_learning_rate_scale: float = Field(
        default=1.0,
        desc="Default multiplier to apply to the learning rate schedule, for parameters that do not define a scale.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 0),
    )

    @property
    def optimizer_cls(self) -> type["Optimizer"]:
        # Placeholder to enable other optimizers in the future.
        from fast_llm.engine.optimizer.optimizer import Optimizer

        return Optimizer

    @property
    def param_group_cls(self) -> type["ParamGroup"]:
        return ParamGroup

    @classmethod
    def state_names(cls) -> tuple[str, ...]:
        return "exp_avgs", "exp_avgs_sq"


@dataclasses.dataclass
class ParamGroup:
    # TODO: Validate list lengths?
    # TODO: Name is only used to combine matching groups. Use more robust comparison instead?
    # TODO: Use list[torch.Tensor] type hints (needs refactoring so torch isn't imported in this file)
    name: str
    params: list = dataclasses.field(default_factory=list)
    grads: list = dataclasses.field(default_factory=list)
    exp_avgs: list = dataclasses.field(default_factory=list)
    exp_avgs_sq: list = dataclasses.field(default_factory=list)
    weight_decay: float | None = None
    lr: float | None = None
    beta1: float | None = None
    beta2: float | None = None
    eps: float | None = None
    lr_scale: float | None = None

    def __len__(self):
        out = len(self.params)
        Assert.eq(out, len(self.grads))
        Assert.eq(out, len(self.exp_avgs))
        Assert.eq(out, len(self.exp_avgs_sq))
        return out
