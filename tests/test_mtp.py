import typing

import pytest
import torch

from fast_llm.config import UpdateType
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.language_model.config import LanguageModelKwargs, LanguageModelLossNames
from fast_llm.layers.language_model.embedding import WORD_EMBEDDINGS_WEIGHT
from fast_llm.layers.language_model.head import OUTPUT_WEIGHTS
from fast_llm.layers.transformer.config import TransformerKwargs
from fast_llm.models.gpt.config import GPTBaseModelConfig
from fast_llm.models.gpt.model import GPTBaseModel
from fast_llm.utils import Assert
from tests.common import requires_cuda


SEQUENCE_LENGTH = 200
BATCH_SIZE = 4
HIDDEN_SIZE = 256
VOCAB_SIZE = 500


def materialize_meta_tensors(model, tensor_space):
    # Materialize parameters that are on meta device
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            # Check if the parameter is a custom tensor type
            if hasattr(param, "tensor_name") and hasattr(param, "init_parameter"):
                param_data = param.new_empty(param.shape, device="cuda")
                # Initialize param_data
                param.init_parameter(param_data, tensor_space.distributed)
                # Replace the parameter in the module
                module_path, param_name = name.rsplit(".", 1) if "." in name else (None, name)
                module = model
                if module_path is not None:
                    for part in module_path.split("."):
                        module = getattr(module, part)
                param = torch.nn.Parameter(param_data, requires_grad=param.requires_grad)
                # TODO: add param_grad_is_zero etc., grad_buffer, etc., see test_mlp_recomputation
                param.grad = None
                param.grad_buffer = torch.empty_like(param)
                param.param_grad_is_zero = True
                module._parameters[param_name] = param
    return model


@pytest.fixture
def distributed_config():
    return DistributedConfig(
        tensor_parallel=1,
        pipeline_parallel=1,
        sequence_data_parallel=1,
        local_world_size=1,
        world_size=1,
    )


@pytest.fixture
def distributed(distributed_config):
    return Distributed(config=distributed_config)


@requires_cuda
@pytest.mark.parametrize(
    "config_dict",
    (
        {"prediction_heads": 1},
        {"prediction_heads": 2, "tie_word_embeddings": False},
        {"prediction_heads": 5, "tie_word_embeddings": False},
    ),
)
def test_transformer_mtp(config_dict: dict[str, typing.Any]):
    config = GPTBaseModelConfig.from_dict(
        {
            "transformer": {
                "hidden_size": HIDDEN_SIZE,
                "num_layers": 2,
            },
            "vocab_size": VOCAB_SIZE,
        },
        config_dict,
        update_type=UpdateType.update,
    )
    distributed_config = DistributedConfig.from_dict({})
    distributed = Distributed(distributed_config)
    model = GPTBaseModel(config, distributed_config)
    model.setup(distributed)
    materialize_meta_tensors(model, model._tensor_space)
    model.to("cuda")

    sequence_first = config.sequence_first or (
        config.cross_entropy_splits is not None and config.cross_entropy_splits > 1
    )
    target = torch.randint(
        0,
        VOCAB_SIZE,
        (
            (SEQUENCE_LENGTH + config.prediction_heads - 1, BATCH_SIZE)
            if sequence_first
            else (BATCH_SIZE, SEQUENCE_LENGTH + config.prediction_heads - 1)
        ),
        dtype=torch.int64,
        device=distributed.device,
    )
    input_ = torch.randint(
        0,
        VOCAB_SIZE,
        (SEQUENCE_LENGTH, BATCH_SIZE) if sequence_first else (BATCH_SIZE, SEQUENCE_LENGTH),
        device=distributed.device,
    )
    attention_mask = torch.ones((1, 1, 1, 1), device="cuda", dtype=torch.bool)
    position_ids = torch.arange(SEQUENCE_LENGTH, device="cuda", dtype=torch.int64)
    kwargs = {
        "position_ids": position_ids,
        TransformerKwargs.sequence_first: sequence_first,
        TransformerKwargs.attention_mask: attention_mask,
        TransformerKwargs.attention_mask_value: -100,
        TransformerKwargs.grad_output: 1.0,
        LanguageModelKwargs.labels: target,
    }
    if config.tie_word_embeddings:
        kwargs[WORD_EMBEDDINGS_WEIGHT] = model.embedding.word_embeddings_weight
    else:
        kwargs[OUTPUT_WEIGHTS] = model.model_head.output_weights
    losses = {LanguageModelLossNames.multi_token_prediction_loss(i): [] for i in range(model._config.prediction_heads)}
    _ = model(input_, kwargs, losses=losses)
    for loss_name, loss_values in losses.items():
        Assert.gt(len(loss_values), 0)
    loss = sum(
        [
            sum(losses[LanguageModelLossNames.multi_token_prediction_loss(i)])
            for i in range(model._config.prediction_heads)
        ]
    )
    loss.backward()
