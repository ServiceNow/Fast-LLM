import pytest

from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.engine.training.trainer import Trainer
from fast_llm.layers.ssm.llamba_block import LlambaBlock
from fast_llm.layers.transformer.transformer import TransformerLayer
from fast_llm.utils import Assert
from tests.utils.model_configs import ModelTestingGroup
from tests.utils.utils import requires_cuda


def _get_trainer_from_args(args: list[str], model_type: str = "gpt") -> Trainer:
    cls = TrainerConfig.get_subclass(model_type)
    parsed, unparsed = cls._get_parser().parse_known_args(args)
    config: TrainerConfig = cls._from_parsed_args(parsed, unparsed)
    distributed = Distributed(config.model.distributed)
    trainer = config.get_trainer_class()(config=config)
    trainer.setup(distributed, config.get_run(distributed))
    return trainer


@requires_cuda
@pytest.mark.model_testing_group(ModelTestingGroup.basic)
def test_frozen_weights(model_testing_config):
    args = model_testing_config.config_args + ["run.tensor_logs.save=False"]
    model_ref = _get_trainer_from_args(args, model_testing_config.model_type)._multi_stage
    model_frozen = _get_trainer_from_args(
        args
        + [
            f"model.base_model.transformer.mlp_lr_scale={[0]*model_ref.config.base_model.transformer.num_experts}",
            f"model.base_model.transformer.router_lr_scale=0",
        ],
        model_testing_config.model_type,
    )._multi_stage

    Assert.eq(
        model_ref._num_stages,
        model_frozen._num_stages,
    )
    frozen_parameter_counts = [
        sum(p.numel() for p in layer.mlp.parameters()) if isinstance(layer, (TransformerLayer, LlambaBlock)) else 0
        for layer in model_ref.base_model.layers
    ]
    for weight_buffer_ref, weight_buffer_frozen in zip(
        model_ref._weight_buffers, model_frozen._weight_buffers, strict=True
    ):
        Assert.eq(weight_buffer_ref.numel() == weight_buffer_frozen.numel())

    for grad_buffer_ref, grad_buffer_frozen, frozen_parameter_count in zip(
        model_ref._grad_buffers, model_frozen._grad_buffers, frozen_parameter_counts, strict=True
    ):
        Assert.eq(grad_buffer_ref.numel() - grad_buffer_frozen.numel() == frozen_parameter_count)

    for shard_name, shard_frozen_count in zip(
        model_ref._shard_names,
        [0] + [sum(frozen_parameter_counts)] * (len(model_ref._all_shard_names) - 1),
        strict=True,
    ):
        Assert.eq(
            model_ref.get_shard(shard_name).numel() - model_frozen.get_shard(shard_name).numel(), shard_frozen_count
        )
