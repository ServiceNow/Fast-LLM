import copy

import pytest

from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.layers.decoder.block import DecoderBlock
from fast_llm.utils import Assert
from tests.utils.dataset import get_model_test_dataset
from tests.utils.model_configs import ModelTestingGroup
from tests.utils.utils import requires_cuda


def _get_model(config_dict: dict, model_type: str = "gpt") -> FastLLMModel:
    cls = FastLLMModelConfig.get_subclass(model_type)
    config: FastLLMModelConfig = cls.from_dict(config_dict)
    model = config.get_model_class()(config)
    model.setup(Distributed(config.distributed))
    return model


@requires_cuda
@pytest.mark.model_testing_group(ModelTestingGroup.basic)
def test_frozen_weights(model_testing_config):
    get_model_test_dataset()
    frozen_config_dict = copy.deepcopy(model_testing_config.config_dict)
    decoder_config = frozen_config_dict["model"]["base_model"]["decoder"]
    if (decoder_type := decoder_config.get("type", "fixed")) == "fixed":
        decoder_config["block"]["mlp"]["lr_scale"] = 0
    elif decoder_type == "pattern":
        for block_config in decoder_config["blocks"].values():
            block_config["mlp"]["lr_scale"] = 0
    else:
        raise NotImplementedError(decoder_type)

    model_ref = _get_model(model_testing_config.config_dict["model"], model_testing_config.model_type)
    model_frozen = _get_model(frozen_config_dict["model"], model_testing_config.model_type)

    Assert.eq(
        model_ref._num_stages,
        model_frozen._num_stages,
    )
    frozen_parameter_counts = [
        sum(p.numel() for p in layer.mlp.parameters()) if isinstance(layer, DecoderBlock) else 0
        for layer in model_ref.base_model.layers
    ]

    # Make sure each layer has its own buffer so the check below works.
    Assert.eq(
        num_stages := len(model_ref.base_model.layers),
        len(model_frozen.base_model.layers),
        len(model_ref.stages),
        len(model_frozen.stages),
    )
    for stage_index in range(num_stages):
        # Weight buffers are the same.
        Assert.eq(
            model_ref._weight_buffers[model_ref._weight_buffer_indices[stage_index]].numel(),
            model_frozen._weight_buffers[model_frozen._weight_buffer_indices[stage_index]].numel(),
        )
        # Weight buffers exclude frozen weights.
        Assert.eq(
            model_ref._grad_buffers[model_ref._grad_buffer_indices[stage_index]].numel()
            - model_frozen._grad_buffers[model_frozen._grad_buffer_indices[stage_index]].numel(),
            frozen_parameter_counts[stage_index],
        )

    for shard_name, shard_frozen_count in zip(
        model_ref._shard_names,
        [0] + [sum(frozen_parameter_counts)] * (len(model_ref._all_shard_names) - 1),
        strict=True,
    ):
        # Same with shards.
        Assert.eq(
            model_ref.get_shard(shard_name).numel() - model_frozen.get_shard(shard_name).numel(),
            shard_frozen_count,
            msg=shard_name,
        )
