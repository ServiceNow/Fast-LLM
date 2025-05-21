import math

import pytest
import torch

from fast_llm.config import NoAutoValidate
from fast_llm.data.data.gpt.data import GPTBatch
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.optimizer.config import OptimizerConfig, ParamGroup
from fast_llm.engine.optimizer.optimizer import Optimizer
from fast_llm.engine.schedule.config import ScheduleConfig
from fast_llm.engine.schedule.runner import ScheduleRunner
from fast_llm.engine.schedule.schedule import Schedule
from fast_llm.models.gpt.config import GPTBatchConfig, LlamaGPTHuggingfaceCheckpointFormat, PretrainedGPTModelConfig
from tests.common import requires_cuda
from tests.test_gpt_generate_and_forward import model_and_tokenizer  # noqa: F401


def _get_model_runner_schedule(
    model_path: str,
    use_flash_attention: bool,
    use_bf16: bool,
    checkpoint_format=LlamaGPTHuggingfaceCheckpointFormat,
    phase=PhaseType.inference,
):
    assert phase == PhaseType.inference or phase == PhaseType.validation
    updates = {
        ("pretrained", "path"): model_path,
        ("pretrained", "model_weights"): True,
        ("pretrained", "format"): checkpoint_format.name,
        ("model", "base_model", "cross_entropy_impl"): "fused",
        ("model", "multi_stage", "zero_stage"): 2,
    }

    if use_flash_attention:
        updates[("model", "base_model", "transformer", "use_flash_attention")] = True
        updates[("model", "distributed", "training_dtype")] = "bf16"
    else:
        updates[("model", "base_model", "transformer", "use_flash_attention")] = False
        if use_bf16:
            updates[("model", "distributed", "training_dtype")] = "bf16"

    config = PretrainedGPTModelConfig.from_dict({}, updates)
    multi_stage = config.model.get_model_class()(
        config.model, optimizer_state_names=OptimizerConfig.state_names() if phase == PhaseType.validation else ()
    )
    schedule_config = ScheduleConfig()
    with NoAutoValidate():
        batch_config = GPTBatchConfig(micro_batch_size=2, sequence_length=2048, batch_size=2)
    batch_config.setup(config.model.distributed)
    batch_config.validate()

    schedule = Schedule(
        multi_stage=multi_stage,
        batch_config=batch_config,
        schedule_config=schedule_config,
        distributed_config=config.model.distributed,
        phase=phase,
    )

    if phase == PhaseType.validation:
        schedule_trainer = Schedule(
            multi_stage=multi_stage,
            batch_config=batch_config,
            schedule_config=schedule_config,
            distributed_config=config.model.distributed,
            phase=PhaseType.training,
        )
    else:
        schedule_trainer = None

    runner = ScheduleRunner(
        config=schedule_config,
        multi_stage=multi_stage,
        distributed_config=config.model.distributed,
    )

    distributed = Distributed(config.model.distributed)

    with torch.no_grad():
        multi_stage.setup(distributed)

    # Setup the optimizer.
    if phase == PhaseType.inference:
        optimizer = None
    else:
        zero_effect_optimizer_updates = {
            ("learning_rate", "base"): 0.0,
            ("learning_rate", "minimum"): 0.0,
            ("learning_rate", "decay_style"): "constant",
            ("learning_rate", "decay_iterations"): 100,
            ("learning_rate", "warmup_iterations"): 0,
            ("gradient_scaler", "initial"): 1.0,  # small value so it does not explode
            ("gradient_scaler", "constant"): 1.0,  # disables dynamic scaling if your code uses this
            ("weight_decay",): 0.0,  # disables weight decay
            ("beta_1",): 0.0,  # disables momentum
            ("beta_2",): 0.0,  # disables second moment
            ("epsilon",): 1.0,  # high epsilon can suppress tiny updates
            ("gradient_norm_clipping",): 1e-12,  # essentially no clipping
            ("default_learning_rate_scale",): 0.0,  # scaling factor to zero everything
        }
        optimizer_config = OptimizerConfig.from_dict({}, zero_effect_optimizer_updates)
        param_groups, grads_for_norm = multi_stage.get_param_groups(ParamGroup)
        optimizer = Optimizer(
            optimizer_config,
            param_groups=param_groups,
            grads_for_norm=grads_for_norm,
            distributed=distributed,
        )

    with torch.no_grad():
        runner.setup(distributed, optimizer)

    multi_stage.load_checkpoint(config.pretrained)
    if phase == PhaseType.validation:
        optimizer.reset_state()

    return multi_stage, runner, schedule, schedule_trainer, batch_config


def _test_for_phase(model_path, fast_llm_checkpoint_format, phase):
    model, runner, schedule, schedule_training, batch_config = _get_model_runner_schedule(
        model_path, True, True, fast_llm_checkpoint_format, phase
    )

    inputs = GPTBatch(
        torch.randint(
            1,
            model.config.base_model.vocab_size,
            [2, batch_config.sequence_length + 1],
            dtype=torch.int64,
            generator=torch.Generator().manual_seed(42),
        )
    )

    if phase == PhaseType.validation:
        # Needs to run at least once as otherwise getting exception about non existent param
        # AttributeError: 'Parameter' object has no attribute 'param_grad_is_zero'
        iter_losses, _, _ = runner.run_step(iter((inputs,)), schedule_training, iteration=1)

    iter_losses, _, _ = runner.run_step(iter((inputs,)), schedule, iteration=1)

    return iter_losses


@pytest.mark.extra_slow
@requires_cuda
def test_loss_validation_vs_inference(model_and_tokenizer):
    model_path, _, fast_llm_checkpoint_format = model_and_tokenizer

    iter_losses_validation = _test_for_phase(model_path, fast_llm_checkpoint_format, PhaseType.validation)

    iter_losses_inference = _test_for_phase(model_path, fast_llm_checkpoint_format, PhaseType.inference)

    assert len(iter_losses_validation) == len(iter_losses_inference)
    for key in iter_losses_validation.keys():
        assert math.isclose(iter_losses_validation[key], iter_losses_inference[key], rel_tol=1e-5)
