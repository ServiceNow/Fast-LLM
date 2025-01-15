import abc
import dataclasses
import math

import numpy as np

from fast_llm.engine.optimizer.config import LearningRateScheduleConfig, LearningRateStageType
from fast_llm.utils import Assert


@dataclasses.dataclass()
class LRStage:
    begin_step: int
    end_step: int | None

    def __post_init__(self) -> None:
        Assert.in_range(self.begin_step, 0, math.inf if self.end_step is None else self.end_step)

    def __call__(self, step: int) -> float:
        Assert.in_range_incl(step, self.begin_step, math.inf if self.end_step is None else self.end_step)
        return self._get_lr(step)

    @abc.abstractmethod
    def _get_lr(self, step: int) -> float:
        pass


@dataclasses.dataclass()
class ConstantLRStage(LRStage):
    lr: float

    def _get_lr(self, step: int) -> float:
        return self.lr


@dataclasses.dataclass()
class InterpolateLRStage(LRStage):
    lr: float
    end_lr: float

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.end_step is not None

    def _get_lr(self, step: int) -> float:
        coeff = (step - self.begin_step) / (self.end_step - self.begin_step)
        return self.lr + self._interpolate(coeff) * (self.end_lr - self.lr)

    @abc.abstractmethod
    def _interpolate(self, coeff: float) -> float:
        pass


@dataclasses.dataclass()
class PowerLRStage(InterpolateLRStage):
    power: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        Assert.gt(self.power, 0)

    @abc.abstractmethod
    def _interpolate(self, coeff: float) -> float:
        return coeff**self.power


@dataclasses.dataclass()
class CosineLRStage(InterpolateLRStage):
    lr: int
    end_lr: int
    power: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        Assert.gt(self.power, 0)

    @abc.abstractmethod
    def _interpolate(self, coeff: float) -> float:
        return 0.5 * (1.0 - math.cos(math.pi * coeff**self.power))


class LearningRateSchedule:
    def __init__(self, stages: list[LRStage]):
        self.stages = stages
        for i in range(1, len(self.stages)):
            Assert.eq(self.stages[i].begin_step, self.stages[i - 1].end_step)
        Assert.eq(self.stages[0].begin_step, 0)
        end_step = self.stages[-1].end_step
        if self.stages[-1].end_step is not None:
            # Add an implicit constant stage at the end
            self.stages.append(ConstantLRStage(begin_step=end_step, end_step=None, lr=self.stages[-1](end_step)))
        self.stage_ends = np.array([stage.end_step for stage in self.stages[:-1]])

    def __call__(self, step: int) -> float:
        return self.stages[np.searchsorted(self.stage_ends, step, side="right").item()](step)


_STAGE_TYPE_MAP = {
    LearningRateStageType.constant: ConstantLRStage,
    LearningRateStageType.linear: PowerLRStage,
    LearningRateStageType.power: PowerLRStage,
    LearningRateStageType.cosine: CosineLRStage,
}


def create_schedule_from_config(config: LearningRateScheduleConfig) -> LearningRateSchedule:
    stages = []
    if config.schedule is None:
        if config.warmup_iterations > 0:
            stages.append(PowerLRStage(begin_step=0, end_step=config.warmup_iterations, lr=0, end_lr=config.base))
        kwargs = {
            "begin_step": config.warmup_iterations,
            "end_step": config.decay_iterations,
            "lr": float(config.base),
        }
        if config.decay_style != "constant":
            kwargs.update(end_lr=config.minimum, power=config.decay_power)
        stages.append(_STAGE_TYPE_MAP[config.decay_style](**kwargs))
    else:
        begin_step = 0
        for stage_arg_str in config.schedule.split(";"):
            try:
                for stage_type, num_steps, lr, *stage_args in stage_arg_str.split(","):
                    assert begin_step is not None
                    num_steps = int(num_steps)
                    end_step = None if num_steps < 0 else begin_step + num_steps
                    kwargs = {"begin_step": begin_step, "end_step": end_step, "lr": float(lr)}
                    if len(stage_args) > 0:
                        kwargs["end_lr"] = float(stage_args[0])
                    if len(stage_args) > 1:
                        kwargs["power"] = float(stage_args[1])
                    if len(stage_args) > 2:
                        raise ValueError(stage_args[2:])
                    stages.append(_STAGE_TYPE_MAP[stage_type](**kwargs))
                    begin_step = end_step
            except Exception:
                raise ValueError(f'Cannot parse optimizer stage definition "{stage_arg_str}"')
    return LearningRateSchedule(stages)
