import copy

import pytest

from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.utils import Assert
from tests.utils.model_configs import ModelTestingGroup


def _get_model(config_dict: dict, model_type: str = "gpt") -> FastLLMModel:
    cls = FastLLMModelConfig.get_subclass(model_type)
    config: FastLLMModelConfig = cls.from_dict(config_dict)
    model = config.get_model_class()(config)
    model.setup(Distributed(config.distributed))
    return model


@pytest.mark.model_testing_group(ModelTestingGroup.basic)
def test_frozen_weights(model_testing_config):
    model_testing_config.get_dataset()
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
        (
            sum(p.numel() for p in layer.unwrap().mlp.parameters())
            if layer.module_name.startswith("decoder") or layer.module_name.startswith("multi_token_prediction.block")
            else 0
        )
        for layer in model_ref.base_model.get_layers()
    ]

    # Make sure each layer has its own buffer so the check below works.
    Assert.eq(
        num_stages := len(model_ref.base_model.get_layers()),
        len(model_frozen.base_model.get_layers()),
        len(model_ref.stages),
        len(model_frozen.stages),
    )
    # Compare unpadded `parameter_count` rather than buffer `numel()`. Each FSDP independently pads to
    # `SHARD_PAD_TO_MULTIPLE`, so moving parameters between trainable and frozen FSDPs can shift total
    # padding even though no parameter changed — buffer-level equality is incidental to the alignment of
    # each fixture's MLP parameter counts and not the invariant we want to assert.
    for stage_index in range(num_stages):
        ref_stage = model_ref.stages[stage_index]
        frozen_stage = model_frozen.stages[stage_index]
        # Total parameter count is invariant: `lr_scale=0` does not add or remove parameters.
        Assert.eq(ref_stage.parameter_count, frozen_stage.parameter_count)
        # Frozen MLP parameters drop out of the trainable set.
        Assert.eq(
            sum(fsdp.parameter_count for fsdp in ref_stage.fsdps if fsdp.requires_grad)
            - sum(fsdp.parameter_count for fsdp in frozen_stage.fsdps if fsdp.requires_grad),
            frozen_parameter_counts[stage_index],
        )
