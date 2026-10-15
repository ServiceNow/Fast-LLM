import dataclasses
import types

import pytest
import torch

from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.optimizer.config import OptimizerConfig, OptimizerStepResult, ParamGroup
from fast_llm.engine.optimizer.optimizer import Optimizer
from fast_llm.layers.decoder.mlp.config import MLPConfig
from fast_llm.utils import Assert
from tests.utils.utils import get_stage


@dataclasses.dataclass
class OptimizerStepTestConfig:
    name: str
    policy: str
    # Value written into the first gradient entry, `None` for a finite gradient.
    bad_value: float | None
    gradient_scaler: dict = dataclasses.field(default_factory=dict)

    @property
    def expected_result(self) -> OptimizerStepResult:
        if self.bad_value is None:
            return OptimizerStepResult.applied
        if self.gradient_scaler:
            # The constant scaler detects nan/inf itself, before the gradient norm is checked.
            return OptimizerStepResult.skipped
        if self.policy == "ignore":
            return OptimizerStepResult.applied
        return (
            OptimizerStepResult.stopped_nonfinite_gradient
            if self.policy == "stop"
            else OptimizerStepResult.skipped_nonfinite_gradient
        )

    @property
    def expect_update(self) -> bool:
        return self.expected_result == OptimizerStepResult.applied


_OPTIMIZER_STEP_CASES = [
    OptimizerStepTestConfig(name=f"{policy}_{bad_name}{'_constant_scaler' if scaler else ''}", **kwargs)
    for policy in ("ignore", "skip", "stop")
    for bad_name, bad_value in (("finite", None), ("nan", float("nan")), ("inf", float("inf")))
    for scaler in (False, True)
    for kwargs in [
        dict(policy=policy, bad_value=bad_value, gradient_scaler={"constant": 4.0} if scaler else {}),
    ]
    # The constant scaler is only relevant with a non-finite gradient.
    if not (scaler and bad_value is None)
]


def _get_optimizer(config: OptimizerStepTestConfig) -> tuple[Optimizer, list[torch.Tensor], list[torch.Tensor]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distributed = types.SimpleNamespace(
        device=device, world_group=None, config=types.SimpleNamespace(compute_dtype=DataType.bfloat16)
    )
    params = [torch.ones(16, device=device), torch.ones(8, device=device)]
    grads = [torch.full_like(param, 0.5) for param in params]
    if config.bad_value is not None:
        grads[0][0] = config.bad_value
    param_group = ParamGroup(
        name="group",
        params=params,
        grads=grads,
        exp_avgs=[torch.zeros_like(param) for param in params],
        exp_avgs_sq=[torch.zeros_like(param) for param in params],
    )
    optimizer = Optimizer(
        OptimizerConfig.from_dict(
            {
                "learning_rate": {"base": 0.1},
                "on_nonfinite_gradient": config.policy,
                "gradient_scaler": config.gradient_scaler,
            }
        ),
        param_groups=[param_group],
        grads_for_norm=grads,
        distributed=distributed,
    )
    optimizer.reset_state()
    return optimizer, params, grads


@pytest.mark.parametrize("config", _OPTIMIZER_STEP_CASES, ids=lambda config: config.name)
def test_optimizer_step_nonfinite_gradient(config: OptimizerStepTestConfig):
    optimizer, params, grads = _get_optimizer(config)
    param_group = optimizer._param_groups[0]
    grads_before = [grad.clone() for grad in grads]
    metrics = {}

    Assert.eq(optimizer.step(metrics), config.expected_result)

    Assert.eq(optimizer.optimizer_step, int(config.expect_update))
    for param, exp_avg, exp_avg_sq in zip(params, param_group.exp_avgs, param_group.exp_avgs_sq, strict=True):
        if config.expect_update:
            assert (param != 1).any()
        else:
            Assert.all_equal(param, torch.ones_like(param))
            Assert.all_equal(exp_avg, torch.zeros_like(exp_avg))
            Assert.all_equal(exp_avg_sq, torch.zeros_like(exp_avg_sq))
    if config.expected_result in (
        OptimizerStepResult.skipped_nonfinite_gradient,
        OptimizerStepResult.stopped_nonfinite_gradient,
    ):
        # Not clipped, so the dump can tell nan from inf.
        for grad, grad_before in zip(grads, grads_before, strict=True):
            torch.testing.assert_close(grad, grad_before, rtol=0, atol=0, equal_nan=True)
        assert optimizer.grad_norm == config.bad_value or (optimizer.grad_norm != optimizer.grad_norm)
        Assert.eq(metrics.keys(), {"grad_norm"})
    if config.policy != "ignore":
        # Cleared after a rejected update, so the next good step is not silently dropped.
        Assert.eq(optimizer._noop_flag.item(), 0)


@pytest.mark.parametrize(
    ("gradient_scaler", "first_result"),
    [
        # Rejected by the noop scaler's caller (gradient norm check), which must clear the noop flag itself.
        ({}, OptimizerStepResult.skipped_nonfinite_gradient),
        ({"constant": 4.0}, OptimizerStepResult.skipped),
    ],
    ids=["noop_scaler", "constant_scaler"],
)
def test_optimizer_step_after_nonfinite_skip(gradient_scaler: dict, first_result: OptimizerStepResult):
    # A skipped update must not prevent the next one; the learning rate schedule lags the skipped step.
    optimizer, params, grads = _get_optimizer(
        OptimizerStepTestConfig(name="", policy="skip", bad_value=float("nan"), gradient_scaler=gradient_scaler)
    )
    Assert.eq(optimizer.step(), first_result)
    for grad in grads:
        grad.fill_(0.5 * gradient_scaler.get("constant", 1.0))
    Assert.eq(optimizer.step(), OptimizerStepResult.applied)
    Assert.eq(optimizer.optimizer_step, 1)
    assert all((param != 1).all() for param in params)


def test_optimizer_stop_keeps_gradient_scaler_state():
    # The saved scaler state must be the pre-step one, so a replay uses the same loss scale.
    optimizer, _, _ = _get_optimizer(OptimizerStepTestConfig(name="", policy="stop", bad_value=float("nan")))
    scaler_state = optimizer._grad_scaler.save()
    updates = []
    optimizer._grad_scaler.update = updates.append
    Assert.eq(optimizer.step(), OptimizerStepResult.stopped_nonfinite_gradient)
    Assert.eq(updates, [])
    Assert.eq(optimizer._grad_scaler.save(), scaler_state)


def _get_mlp_stage():
    distributed = Distributed(DistributedConfig(use_cuda=torch.cuda.is_available()))
    hidden_dim = TensorDim("hidden", 16)
    mlp = MLPConfig.from_dict({"intermediate_size": 32}).get_layer(
        distributed.config, hidden_dim, lr_scale=None, peft=None, return_bias=False
    )
    stage = get_stage([mlp], distributed)
    (fsdp,) = stage.fsdps
    fsdp.track_nonfinite_local_grads()
    stage.reset_gradients()
    return distributed, stage, fsdp


def _write_local_grad(fsdp, parameter_name: str, value: float, mark_written: bool = True) -> None:
    buffer = fsdp._parameter_buffers[parameter_name]
    buffer.grad_buffer.fill_(value)
    if mark_written:
        buffer.param_grad_is_zero = False


def _parameter_names(fsdp) -> list[str]:
    return list(fsdp._parameter_buffers)


@pytest.mark.parametrize(
    ("name", "bad_value", "mark_written", "expected"),
    [
        ("finite", None, True, False),
        ("nan", float("nan"), True, True),
        ("inf", float("inf"), True, True),
        # Stale values in a buffer that received no gradient are zero-filled before the check.
        ("stale_nan", float("nan"), False, False),
    ],
)
def test_nonfinite_local_grads(name: str, bad_value: float | None, mark_written: bool, expected: bool):
    distributed, stage, fsdp = _get_mlp_stage()
    first, *others = _parameter_names(fsdp)
    for parameter_name in others:
        _write_local_grad(fsdp, parameter_name, 0.5)
    _write_local_grad(fsdp, first, 0.5 if bad_value is None else bad_value, mark_written)
    fsdp.reduce_gradients(distributed, allow_no_grad=True)
    Assert.eq(fsdp.nonfinite_local_grads.item(), expected)


def test_nonfinite_local_grads_accumulate_and_reset():
    distributed, stage, fsdp = _get_mlp_stage()
    first, *_ = _parameter_names(fsdp)
    # Mirror `ScheduleRunner._reduce`, which resets the gradients after every reduce:
    # a non-finite first reduce is kept through the reset and a finite second (accumulated) reduce.
    _write_local_grad(fsdp, first, float("nan"))
    fsdp.reduce_gradients(distributed, allow_no_grad=True)
    stage.reset_gradients()
    _write_local_grad(fsdp, first, 0.5)
    fsdp.reduce_gradients(distributed, accumulate=True, allow_no_grad=True)
    stage.reset_gradients()
    assert fsdp.nonfinite_local_grads.item()
    # Cleared only by the once-per-step reset.
    fsdp.reset_nonfinite_local_grads()
    assert not fsdp.nonfinite_local_grads.item()
