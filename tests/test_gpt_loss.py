import math

import torch

from fast_llm.config import NoAutoValidate
from fast_llm.data.data.gpt.data import GPTBatch
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.optimizer.config import OptimizerConfig
from fast_llm.engine.schedule.config import ScheduleConfig
from fast_llm.engine.schedule.runner import ScheduleRunner
from fast_llm.engine.schedule.schedule import Schedule
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.models.gpt.config import GPTBatchConfig, LlamaGPTHuggingfaceCheckpointFormat, PretrainedGPTModelConfig
from tests.test_gpt_generate_and_forward import model_and_tokenizer  # noqa: F401
from tests.utils.utils import requires_cuda


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

    runner = ScheduleRunner(
        config=schedule_config,
        multi_stage=multi_stage,
        distributed_config=config.model.distributed,
    )

    distributed = Distributed(config.model.distributed)

    with torch.no_grad():
        multi_stage.setup(distributed)

    with torch.no_grad():
        runner.setup(distributed)

    multi_stage.load_checkpoint(config.pretrained)

    return multi_stage, runner, schedule, batch_config


def _test_for_phase(model_path, fast_llm_checkpoint_format, phase):
    model, runner, schedule, batch_config = _get_model_runner_schedule(
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

    iteration = 1

    # we need to set phase to validation here so preprocess would crate labels from input
    # so it is the same process for validation and inference phases
    # otherwise we can add labels manually after preprocess for inference phase
    batch = model.base_model.preprocess(inputs, phase=PhaseType.validation, iteration=iteration)
    ((inputs_, kwargs),) = batch
    kwargs[LanguageModelKwargs.phase] = phase
    iter_losses, _, _ = runner.run_step(
        iter((((inputs_, kwargs),),)), schedule, iteration=iteration, preprocessed=True
    )

    return iter_losses


# @pytest.mark.extra_slow
@requires_cuda
def test_loss_validation_vs_inference(model_and_tokenizer):
    model_path, _, fast_llm_checkpoint_format = model_and_tokenizer

    iter_losses_validation = _test_for_phase(model_path, fast_llm_checkpoint_format, PhaseType.validation)

    iter_losses_inference = _test_for_phase(model_path, fast_llm_checkpoint_format, PhaseType.inference)

    assert len(iter_losses_validation) == len(iter_losses_inference)
    for key in iter_losses_validation.keys():
        assert math.isclose(iter_losses_validation[key], iter_losses_inference[key], rel_tol=1e-5)
