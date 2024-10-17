import abc

import torch

from fast_llm.core.distributed import ReduceOp, all_reduce
from fast_llm.core.kernels import fused_adam, l2_norm, scale_
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.run import log_main_rank
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.optimizer.config import GradientScalerConfig, OptimizerConfig, ParamGroup
from fast_llm.engine.optimizer.learning_rate import create_schedule_from_config
from fast_llm.utils import Assert


def get_grad_scaler(config: GradientScalerConfig, distributed: Distributed) -> "GradScaler":
    if config.constant:
        return ConstantGradScaler(
            initial_scale=config.constant,
            distributed=distributed,
        )
    elif distributed.config.training_dtype == DataType.float16:
        return DynamicGradScaler(
            initial_scale=config.initial,
            min_scale=config.minimum,
            growth_interval=config.window,
            hysteresis=config.hysteresis,
            distributed=distributed,
        )
    else:
        return NoopGradScaler(distributed.device)


def _merge_and_filter_groups(groups: list["ParamGroup"]):
    # Merge groups with the same name
    merged_groups = {}
    for group in groups:
        # TODO: Compare parameters instead?
        if group.name in merged_groups:
            g = merged_groups[group.name]
            g.params.extend(group.params)
            g.grads.extend(group.grads)
            g.exp_avgs.extend(group.exp_avgs)
            g.exp_avgs_sq.extend(group.exp_avgs_sq)
        else:
            merged_groups[group.name] = group
    return list(merged_groups.values())


class Optimizer:
    _optimizer_step: int

    def __init__(
        self,
        config: OptimizerConfig,
        param_groups: list[ParamGroup],
        grads_for_norm: list[torch.Tensor],
        distributed: Distributed,
    ):
        self._config = config
        self._param_groups = _merge_and_filter_groups(param_groups)
        self._grads_for_norm = [g for g in grads_for_norm if g.device.type != "meta" and g.numel() > 0]
        self._grad_norm = None if self._grads_for_norm else torch.zeros([1], device=distributed.device)
        self._grads = [g for group in self._param_groups for g in group.grads]
        self._grad_scaler = get_grad_scaler(self._config.gradient_scaler, distributed)
        self._noop_flag = self._grad_scaler.noop_flag
        self._reduce_group = distributed.world_group
        self._lr_schedule = create_schedule_from_config(self._config.learning_rate)

    def _clip_grad_norm(self):
        # TODO: Optimize this.
        grad_norm = l2_norm(self._grads_for_norm, self._noop_flag) if self._grads_for_norm else self._grad_norm.zero_()
        Assert.eq(grad_norm.dtype, torch.float32)
        if self._reduce_group:
            # Sum across GPUs.
            all_reduce(
                grad_norm.pow_(2),
                op=ReduceOp.SUM,
                group=self._reduce_group,
            )
            grad_norm.pow_(0.5)
        if self._config.gradient_norm_clipping > 0.0:
            # TODO: Use noop flag instead of clamp.
            clip_coeff = torch.clamp_max_(self._config.gradient_norm_clipping / (grad_norm + 1.0e-6), 1.0)
            # if clip_coeff < 1.0:
            scale_(self._grads, self._noop_flag, clip_coeff)
        return grad_norm

    @torch.no_grad()
    def step(self, metrics: dict | None = None):
        for group in self._param_groups:
            for grad in group.grads:
                self._grad_scaler.unscale_and_check_nans(grad)
        # Update weights
        # Noop flag still used beyond this point, but no longer expecting nans.
        update_successful = self._grad_scaler.update_successful()
        if update_successful:
            self._optimizer_step += 1
            lr = self._lr_schedule(self._optimizer_step)
            grad_norm = (
                self._clip_grad_norm().item()
                if self._config.gradient_norm_clipping > 0.0 or metrics is not None
                else None
            )
            for group in self._param_groups:
                fused_adam(
                    params=group.params,
                    grads=group.grads,
                    exp_avgs=group.exp_avgs,
                    exp_avg_sqs=group.exp_avgs_sq,
                    noop_flag=self._noop_flag,
                    lr=lr * (self._config.default_learning_rate_scale if group.lr_scale is None else group.lr_scale),
                    beta1=self._config.beta_1 if group.beta1 is None else group.beta1,
                    beta2=self._config.beta_2 if group.beta2 is None else group.beta2,
                    wd=self._config.weight_decay if group.weight_decay is None else group.weight_decay,
                    eps=self._config.epsilon if group.eps is None else group.eps,
                    step=self._optimizer_step,
                )

            if metrics is not None:
                metrics.update({"grad_norm": grad_norm, "learning_rate": lr})

        self._grad_scaler.update(update_successful)
        return update_successful

    @property
    def grad_scale(self):
        return self._grad_scaler.scale

    @property
    def optimizer_step(self):
        return self._optimizer_step

    def reset_state(self):
        self._optimizer_step = 0
        self._grad_scaler.reset_state()

    def save(self):
        return {
            "current_step": self._optimizer_step,
            "grad_scaler": self._grad_scaler.save(),
        }

    def load(self, state, validate=True):
        self._optimizer_step = state["current_step"]
        self._grad_scaler.load(state["grad_scaler"], validate=validate)


class GradScaler(abc.ABC):
    def __init__(self, device: torch.device):
        # Flag to keep track of nan/inf, checked and reused in the optimizer(s)
        # Also determines the device
        # The optimizer needs a noop flag even when the grad scaler doesn't.
        self._noop_flag = torch.zeros([], dtype=torch.int32, device=device)

    @property
    @abc.abstractmethod
    def scale(self) -> float:
        pass

    def save(self):
        return {"type": self.__class__.__name__}

    def reset_state(self):
        pass

    def load(self, state, validate=True):
        if validate:
            Assert.eq(state["type"], self.__class__.__name__)
        self._noop_flag.zero_()

    @property
    def noop_flag(self) -> torch.Tensor:
        return self._noop_flag

    @abc.abstractmethod
    def update_successful(self) -> bool:
        pass

    def unscale_and_check_nans(self, tensor: torch.Tensor):
        pass

    def update(self, update_successful: bool) -> None:
        # TODO: Reset the flag for extra safety?
        pass

    def state_dict(self) -> dict:  # noqa
        return dict()


class NoopGradScaler(GradScaler):
    @property
    def scale(self) -> float:
        return 1.0

    def update_successful(self) -> bool:
        # We don't check for nans.
        return True


class VariableGradScaler(GradScaler):
    def __init__(self, *, initial_scale: float, distributed: Distributed):
        # Flag to keep track of nan/inf, checked and reused in the optimizer(s)
        # Also determines the device
        super().__init__(distributed.device)
        self._distributed = distributed
        self._inv_scale = torch.empty([], dtype=torch.float32, device=distributed.device)
        self._initial_scale = initial_scale
        # TODO: This assumes fsdp
        self._reduce_group = distributed.world_group

    @property
    def scale(self) -> float:
        return self._scale

    def reset_state(self):
        self._set_scale(self._initial_scale)

    def save(self):
        state = super().save()
        state["scale"] = self._scale
        return state

    def update(self, update_successful: bool) -> None:
        self._noop_flag.zero_()

    def update_successful(self) -> bool:
        if self._reduce_group:
            all_reduce(
                self._noop_flag,
                op=ReduceOp.SUM,
                group=self._reduce_group,
            )
        return not self._noop_flag.item()

    def _set_scale(self, value):
        log_main_rank(lambda: f"Setting loss scale to {value}")
        self._scale = value
        self._inv_scale.fill_(value**-1)

    def unscale_and_check_nans(self, tensor: torch.Tensor):
        # This is better than the torch._amp version which needs a float noop_flag and a tensor scale.
        scale_([tensor], self._noop_flag, self._scale**-1)


class ConstantGradScaler(VariableGradScaler):
    def load(self, state, validate=True):
        if validate:
            Assert.eq(self._scale, state["scale"])
        super().load(state, validate=validate)

    def _set_scale(self, value):
        assert not hasattr(self, "_scale")
        super()._set_scale(value)


class DynamicGradScaler(VariableGradScaler):
    def __init__(
        self,
        *,
        initial_scale: float,
        min_scale: float,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int,
        hysteresis: int,
        distributed: Distributed,
    ):
        """Grad scaler with dynamic scale that gets adjusted
        during training."""
        super().__init__(initial_scale=initial_scale, distributed=distributed)
        self.min_scale = min_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.hysteresis = hysteresis
        self._growth_tracker = 0
        self._hysteresis_tracker = self.hysteresis

    def save(self):
        state = super().save()
        state["growth"] = self._growth_tracker
        state["hysteresis"] = self._hysteresis_tracker
        return state

    def load(self, state, validate=True):
        super().load(state, validate=validate)
        self._growth_tracker = state["growth"]
        self._hysteresis_tracker = state["hysteresis"]

    def update(self, update_successful: bool) -> None:
        # If we have an inf/nan, growth tracker is set to 0
        # and hysteresis tracker is reduced by 1.
        if update_successful:
            # If there is no nan/inf, increment the growth tracker.
            self._growth_tracker += 1
            # If we have had enough consecutive intervals with no nan/inf:
            if self._growth_tracker == self.growth_interval:
                # Reset the tracker and hysteresis trackers,
                self._growth_tracker = 0
                self._hysteresis_tracker = self.hysteresis
                # and scale up the loss scale.
                self._set_scale(self._scale * self.growth_factor)
        else:
            self._growth_tracker = 0
            self._hysteresis_tracker -= 1
            # Now if we are out of hysteresis count, scale down the loss.
            if self._hysteresis_tracker <= 0:
                self._set_scale(max(self._scale * self.backoff_factor, self.min_scale))
        super().update(update_successful)
