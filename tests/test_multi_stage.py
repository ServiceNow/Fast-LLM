from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.engine.training.trainer import Trainer
from fast_llm.layers.transformer.transformer import TransformerLayer
from fast_llm.tools.train import CliTrainingConfig
from fast_llm.utils import Assert
from tests.common import CONFIG_COMMON, requires_cuda


def _get_trainer_from_args(args: list[str], model_type: str = "gpt") -> Trainer:
    parsed, unparsed = CliTrainingConfig._get_parser().parse_known_args([model_type] + args)
    config: TrainerConfig = CliTrainingConfig._from_parsed_args(parsed, unparsed)
    distributed = Distributed(config.model.distributed)
    trainer = config.get_trainer_class()(config=config)
    trainer.setup(distributed, config.get_run(distributed))
    return trainer


@requires_cuda
def test_frozen_weights():
    args = CONFIG_COMMON + ["run.tensor_logs.save=False"]
    model_ref = _get_trainer_from_args(args)._multi_stage
    model_frozen = _get_trainer_from_args(args + ["model.base_model.transformer.mlp_lr_scale=[0]"])._multi_stage

    Assert.eq(
        model_ref._num_stages,
        model_frozen._num_stages,
    )
    diff_by_layer = [
        sum(p.numel() for p in layer.mlp.parameters()) if isinstance(layer, TransformerLayer) else 0
        for layer in model_ref.base_model.layers
    ]
    assert all((diff_by_layer[i] == 0) == (i in (0, len(diff_by_layer) - 1)) for i in range(len(diff_by_layer)))
    total_diff = sum(diff_by_layer)

    for weight_buffer_ref, weight_buffer_frozen in zip(
        model_ref._weight_buffers, model_frozen._weight_buffers, strict=True
    ):
        assert weight_buffer_ref.numel() == weight_buffer_frozen.numel()

    for grad_buffer_ref, grad_buffer_frozen, diff in zip(
        model_ref._grad_buffers, model_frozen._grad_buffers, diff_by_layer, strict=True
    ):
        Assert.eq(grad_buffer_ref.numel() - grad_buffer_frozen.numel() == diff)

    for shard_name, shard_diff in zip(
        model_ref._shard_names, [0] + [total_diff] * (len(model_ref._all_shard_names) - 1), strict=True
    ):
        Assert.eq(model_ref.get_shard(shard_name).numel() - model_frozen.get_shard(shard_name).numel(), shard_diff)
